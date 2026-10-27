"""Durable CPU-only post-recording review, isolated from the control process.

Only sealed small JSON is read on submit. Geometry, archive hashes, collection
locks and numbering run in a separate low-priority process. A pending result is
never a demonstration admission. Original episode/metadata bytes stay immutable.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import signal
import copy
import sys
import tempfile
import time
import uuid

STATUS_CONTRACT = "cvpr-pico-review-queue-status-v1"
JOB_CONTRACT = "cvpr-pico-review-job-v1"
SUBMISSION_CONTRACT = "cvpr-pico-review-submission-v1"
COMPLETION_CONTRACT = "cvpr-pico-quality-geometry-completion-v1"
MAX_STATUS_BYTES = 65536


def _now():
    return datetime.now(timezone.utc).isoformat()


def _read(path, limit=None):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("拒绝符号链接JSON：" + str(path))
    with path.open("rb") as stream:
        value = stream.read() if limit is None else stream.read(limit + 1)
    if limit is not None and len(value) > limit:
        raise ValueError("状态JSON超过大小限制")
    return json.loads(value, parse_constant=lambda item: (_ for _ in ()).throw(ValueError("非有限JSON：" + item)))


def _sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _atomic(path, value):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("拒绝覆盖符号链接")
    data = (json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode()
    fd, temporary = tempfile.mkstemp(prefix=".review-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if os.path.lexists(temporary):
            os.unlink(temporary)


def _identity(value):
    if not isinstance(value, str) or str(uuid.UUID(value)) != value:
        raise ValueError("示例UUID格式无效")
    return value


def _summary(value):
    """Bound results before exposing them to the real-time process/browser."""
    keys = ("uuid", "status", "number", "disposition", "available", "pending", "path",
            "source_path", "sequence", "collection_enabled", "pendingRTX", "discarded", "ended_time_ns",
            "selection_policy", "selection_status", "official_task_success_observed", "acceptance_gate_version")
    result = {key: value.get(key) for key in keys}
    result["reasons"] = [str(reason)[:320] for reason in value.get("reasons", [])[:8]]
    result["error"] = str(value["error"])[:640] if value.get("error") else None
    result["accepted_new_dataset"] = False
    result["eligible_for_packing"] = False
    return result


def _error(error, identity=None):
    return _summary({"uuid": identity, "status": "error", "available": False,
        "pending": True, "reasons": ["后台复核未完成，原始数据保留：" + str(error)],
        "error": f"{type(error).__name__}: {error}"})


def _within(path, root):
    path, root = Path(path).absolute(), Path(root).resolve()
    if not path.is_relative_to(root) or not path.resolve().is_relative_to(root):
        raise ValueError("示例路径不能离开本集合")
    # Only the leaf UUID alias created by EpisodeHistory is permitted.
    for parent in path.parents:
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError("示例上级目录不能是符号链接")
    if path.is_symlink():
        record = _read(path / "collection_record.json")
        if record.get("source_episode_directory") != str(path) or record.get("status") != "committed":
            raise ValueError("拒绝未登记的目录符号链接")
    return path


def discard_requested(directory):
    """Admission-side fail-closed check, including an intent not yet classified.

    A malformed submission marker raises rather than hiding a possible request.
    This function never acquires the history lock or loads capture/state arrays.
    """
    directory = Path(directory)
    marker_path = directory / "review_queue_submission.json"
    if marker_path.is_symlink():
        raise ValueError("后台登记标记不能是符号链接")
    if not marker_path.exists():
        seal_path = directory / "episode.json"
        if seal_path.exists() and _read(seal_path).get("geometry_pending") is True:
            raise ValueError("后台几何待复核示例缺少队列登记标记")
        # Historical rows predate queue markers. Their canonical collection
        # record identifies the default spool without loading a history index.
        record_path = directory / "collection_record.json"
        if not record_path.exists():
            seal_path = directory / "episode.json"
            if seal_path.exists() and _read(seal_path).get("geometry_pending") is True:
                raise ValueError("后台几何待复核示例缺少队列登记标记")
            return False
        record = _read(record_path, MAX_STATUS_BYTES)
        actual = directory.resolve()
        if actual.parent.name not in ("kept", "discarded") or record.get("status") != "committed":
            raise ValueError("历史归档位置或状态无效")
        marker = {"contract": SUBMISSION_CONTRACT, "uuid": record["human_demo_uuid"],
            "episode_json_sha256": record["episode_json_sha256"],
            "collection_root": str(actual.parent.parent),
            "spool_directory": str(actual.parent.parent / ".pico_episode_review")}
    else:
        marker = _read(marker_path, MAX_STATUS_BYTES)
    if marker.get("contract") != SUBMISSION_CONTRACT:
        raise ValueError("后台复核登记标记无效")
    identity = _identity(marker["uuid"])
    spool = Path(marker["spool_directory"])
    root = Path(marker["collection_root"]).resolve()
    if not spool.is_absolute() or not spool.resolve().is_relative_to(root):
        raise ValueError("后台复核队列路径离开集合")
    _within(directory, root)
    sealed = _read(directory / "episode.json", 4 * 1024 * 1024)
    sealed_uuid = sealed.get("state", {}).get("human_demo_uuid") or directory.name
    if sealed_uuid != identity or _sha(directory / "episode.json") != marker.get("episode_json_sha256"):
        raise ValueError("后台队列标记与原始结束回执身份不一致")
    intent_path = spool / "discard_requests" / (identity + ".json")
    if intent_path.is_symlink():
        raise ValueError("后台丢弃意图不能是符号链接")
    if not intent_path.exists():
        return False
    intent = _read(intent_path, MAX_STATUS_BYTES)
    if intent.get("uuid") != identity or intent.get("operator_discard_requested") is not True:
        raise ValueError("后台丢弃请求身份无效")
    return True


class ReviewQueue:
    def __init__(self, index_path, *, worker_python=None, spool_directory=None):
        self.index_path = Path(index_path).absolute()
        self.root = self.index_path.parent.resolve()
        self.spool = Path(spool_directory or self.root / ".pico_episode_review").absolute()
        if not self.spool.resolve().is_relative_to(self.root):
            raise ValueError("复核队列必须位于当前集合内")
        self.worker_python = str(worker_python or sys.executable)
        for path in (self.spool, self.spool / "jobs", self.spool / "discard_requests"):
            path.mkdir(parents=True, exist_ok=True)
            if path.is_symlink():
                raise ValueError("复核队列目录不能是符号链接")
        self._process = None
        self._auto_start = False

    @contextmanager
    def _locked(self):
        with (self.spool / "queue.lock").open("a+b") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def _load(self):
        path = self.spool / "queue.json"
        if not path.exists():
            return {"version": 1, "next_sequence": 1, "latest_finished_uuid": None,
                "latest_episode": None, "latest_completed": None, "pending": [],
                "latest_results": [], "worker": {"ready": False}, "baseline_seeded": False,
                "pending_discards": []}
        value = _read(path)
        if value.get("version") != 1 or not isinstance(value.get("pending"), list):
            raise ValueError("后台队列索引无效")
        return value

    def _publish(self, control, *, status=None):
        pending = list(dict.fromkeys(control["pending"] + control.get("pending_discards", [])))
        snapshot = {"contract": STATUS_CONTRACT, "status": status or
            ("working" if pending else "idle"), "worker": control.get("worker", {}),
            "latest_finished_uuid": control.get("latest_finished_uuid"),
            "latest_sealed_mtime_ns": control.get("latest_sealed_mtime_ns", 0),
            "latest_episode": control.get("latest_episode"), "latest_completed": control.get("latest_completed"),
            "pending_count": len(pending), "pending_uuids": pending[-8:],
            "latest_results": control.get("latest_results", [])[-8:],
            "updated_utc": _now(), "accepted_new_dataset": False}
        if len(json.dumps(snapshot, ensure_ascii=False).encode()) > MAX_STATUS_BYTES:
            raise ValueError("复核轻量状态超过大小限制")
        _atomic(self.spool / "queue.json", control)
        _atomic(self.spool / "status.json", snapshot)
        return snapshot

    def start(self):
        first_start = not self._auto_start
        self._auto_start = True
        if first_start and self.spool == self.root / ".pico_episode_review":
            _atomic(self.spool / "startup_scan_required.json", {"requested_utc": _now()})
        if self._process is not None and self._process.poll() is None:
            return self
        # A finisher/new collector may share a still-running worker. Do not
        # spawn a stream of losing singleton processes while waiting for it.
        status_path = self.spool / "status.json"
        if status_path.exists():
            try:
                worker = _read(status_path, MAX_STATUS_BYTES).get("worker", {})
                if worker.get("ready") is True:
                    pid = int(worker["pid"])
                    os.kill(pid, 0)
                    command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
                    if (os.fsencode(Path(__file__).resolve()) in command and b"--worker" in command
                            and os.fsencode(self.index_path) in command and os.fsencode(self.spool) in command):
                        return self
            except (OSError, ValueError, KeyError, TypeError):
                pass
        environment = os.environ.copy()
        environment.update({name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")})
        environment.update(CUDA_VISIBLE_DEVICES="", PYTHONNOUSERSITE="1")
        with (self.spool / "worker.log").open("ab", buffering=0) as output:
            self._process = subprocess.Popen([self.worker_python, str(Path(__file__).resolve()),
                "--worker", "--index", str(self.index_path), "--spool", str(self.spool)],
                stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                env=environment, start_new_session=True, close_fds=True)
        return self

    def _terminate_owned_worker(self, pid):
        """Caller holds queue/history locks and has excluded any running job."""
        if type(pid) is not int or pid <= 1:
            raise ValueError("Invalid owned review worker PID")
        descriptor = None
        try:
            if hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"):
                descriptor = os.pidfd_open(pid)
            command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            if not (os.fsencode(Path(__file__).resolve()) in command and b"--worker" in command
                    and os.fsencode(self.index_path) in command and os.fsencode(self.spool) in command):
                raise ValueError("Refusing to signal a process outside this review queue")
            if descriptor is not None:
                signal.pidfd_send_signal(descriptor, signal.SIGTERM)
            else:
                os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return False
        except FileNotFoundError:
            return False
        finally:
            if descriptor is not None:
                os.close(descriptor)
        return True

    def rotate_worker_at_boundary(self, *, start_new=True):
        """Reload review code only between jobs; never signal the collector.

        Queue lock excludes a new job claim, history lock excludes a pending
        history mutation before that claim. An active job is never interrupted.
        Queued jobs and original evidence remain unchanged.
        """
        from pico_episode_history import EpisodeHistory
        with self._locked():
            if any((self.spool / name).exists() for name in
                    ("recovery_required.json", "startup_scan_required.json")):
                return {"rotated":False,"reason":"queue_recovery_pending"}
            jobs = [_read(path) for path in (self.spool / "jobs").glob("*.json")]
            if any(job.get("status") == "running" for job in jobs):
                return {"rotated":False,"reason":"active_review_job"}
            history = EpisodeHistory(self.index_path)
            with history._locked():
                control = self._load()
                old_worker = copy.deepcopy(control.get("worker", {}))
                pid = old_worker.get("pid")
                if pid is not None:
                    self._terminate_owned_worker(pid)
                control["worker"] = {**old_worker,"ready":False,
                    "retired_at_job_boundary_utc":_now(),"reload_requested":True}
                self._publish(control)
                self._process = None
        if start_new:
            # SIGTERM delivery is asynchronous. Wait outside both data locks
            # until the old singleton releases its worker lock, otherwise the
            # replacement can lose that lock and exit before becoming ready.
            deadline = time.monotonic() + 2.
            with (self.spool / "worker.lock").open("a+b") as lock:
                while True:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if time.monotonic() >= deadline:
                            raise TimeoutError("Retired review worker has not released its singleton lock; replacement not started")
                        time.sleep(.02)
                fcntl.flock(lock, fcntl.LOCK_UN)
            self.start()
        return {"rotated":True,"previous_worker":old_worker,"collector_modified":False}

    @staticmethod
    def _archive_bytes(source, destination):
        """Preserve exact previous bytes once, without replacing an archive."""
        if source.is_symlink() or destination.is_symlink():
            raise ValueError("Reselection archive cannot use symlinks")
        data = source.read_bytes()
        if destination.exists():
            if destination.read_bytes() != data:
                raise ValueError("Existing reselection backup differs; refusing overwrite")
            return
        with destination.open("xb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())

    def refresh_reselection_results(self, identities, revision_id):
        """Publish revised History views for explicit already-completed UUIDs.

        Old job/queue/status bytes are archived. A recovery barrier makes a
        partially written refresh recoverable without re-enqueueing, renumbering
        or changing raw capture files. Pending/latest-newer work is preserved.
        """
        from pico_episode_history import EpisodeHistory
        revision_id = _identity(revision_id)
        identities = [_identity(value) for value in identities]
        if not identities or len(set(identities)) != len(identities):
            raise ValueError("Explicit unique completed UUIDs required")
        archive = self.spool / "reselections" / revision_id
        done = archive / "refresh_complete.json"
        with self._locked():
            if (self.spool / "recovery_required.json").exists():
                raise ValueError("Queue transaction is awaiting recovery")
            if done.exists():
                previous = _read(done)
                if previous.get("uuids") != identities:
                    raise ValueError("Revision ID already belongs to another UUID set")
                return {**previous,"already_refreshed":True}
            control = self._load()
            for identity in identities:
                intent_path = self.spool / "discard_requests" / (identity + ".json")
                if (identity in control.get("pending", []) or identity in control.get("pending_discards", [])
                        or (intent_path.exists() and _read(intent_path).get("completed") is not True)):
                    raise ValueError("Cannot refresh a review with pending work or operator discard intent")
            jobs = {}
            for identity in identities:
                job = _read(self.spool / "jobs" / (identity + ".json"))
                if job.get("status") != "completed":
                    raise ValueError("Cannot refresh a pending or running review job")
                jobs[identity] = job
            history = EpisodeHistory(self.index_path)
            with history._locked():
                index = history._load()
                views = {identity:history._view({**index,"latest_uuid":identity}) for identity in identities}
            results = {}
            for identity, job in jobs.items():
                view = views[identity]
                result = _summary({**view,"uuid":identity,"sequence":job["sequence"],
                    "ended_time_ns":job.get("sealed_mtime_ns"),"collection_enabled":job["collection_enabled"],
                    "pending":False})
                results[identity] = result
                job.update(result=result,reselection_revision=revision_id,reselected_utc=_now())
            for field in ("latest_episode","latest_completed"):
                old = control.get(field) or {}
                if old.get("uuid") in results:
                    control[field] = results[old["uuid"]]
            control["latest_results"] = [results.get(item.get("uuid"),item)
                for item in control.get("latest_results",[])]
            for directory in (archive.parent,archive,archive / "jobs"):
                directory.mkdir(exist_ok=True)
                if directory.is_symlink():
                    raise ValueError("Unsafe reselection archive path")
            for name in ("queue.json","status.json"):
                if (self.spool / name).exists():
                    self._archive_bytes(self.spool / name,archive / name)
            for identity in identities:
                self._archive_bytes(self.spool / "jobs" / (identity + ".json"),
                    archive / "jobs" / (identity + ".json"))
            receipt = {"refreshed":True,"revision_id":revision_id,"uuids":identities,
                "updated_results":len(results),"completed_utc":_now(),"accepted_new_dataset":False}
            barrier = {"operation":"selection_refresh","jobs":list(jobs.values()),
                "control":control,"receipt":receipt,"receipt_relative_path":str(done.relative_to(self.spool))}
            _atomic(self.spool / "recovery_required.json",barrier)
            for identity,job in jobs.items():
                _atomic(self.spool / "jobs" / (identity + ".json"),job)
            self._publish(control)
            _atomic(done,receipt)
            (self.spool / "recovery_required.json").unlink()
            return receipt

    def _marker(self, job):
        return {"contract": SUBMISSION_CONTRACT, "uuid": job["uuid"],
            "episode_json_sha256": job["episode_json_sha256"], "sequence": job["sequence"],
            "metadata_sha256": job.get("metadata_sha256"),
            "collection_root": str(self.root), "spool_directory": str(self.spool),
            "collection_enabled": job["collection_enabled"], "accepted_new_dataset": False}

    def _register(self, control, job):
        identity = job["uuid"]
        job_path = self.spool / "jobs" / (identity + ".json")
        if job_path.exists():
            old = _read(job_path)
            if any(old.get(key) != job.get(key) for key in (
                    "uuid", "source_path", "episode_json_sha256", "metadata_sha256", "collection_enabled")):
                raise ValueError("同UUID不能提交不同来源或不同封存内容")
            job = old
        else:
            _atomic(job_path, job)
        if job["status"] != "completed" and identity not in control["pending"]:
            control["pending"].append(identity)
        stamp = job.get("sealed_mtime_ns", 0)
        if (stamp, job["sequence"]) >= (control.get("latest_sealed_mtime_ns", 0), control.get("latest_sequence", 0)):
            control.update(latest_sequence=job["sequence"], latest_sealed_mtime_ns=stamp, latest_finished_uuid=identity,
                latest_episode=job.get("result") or self._pending(job))
        control["next_sequence"] = max(control["next_sequence"], job["sequence"] + 1)
        return job

    @staticmethod
    def _pending(job, *, discarded=False, running=False):
        return _summary({"uuid": job["uuid"], "status": "pending_discard" if discarded else
            ("running" if running else "pending"), "pending": True, "available": False,
            "source_path": job["source_path"], "path": job["source_path"],
            "sequence": job["sequence"], "collection_enabled": job["collection_enabled"],
            "ended_time_ns": job.get("sealed_mtime_ns"), "reasons": ["已请求丢弃，等待后台归档"] if discarded else ["等待独立物理复核，尚未接纳"]})

    def submit(self, directory, *, collection_enabled=True):
        identity = None
        try:
            path = _within(directory, self.root)
            seal = _read(path / "episode.json", 4 * 1024 * 1024)
            state = seal.get("state", {})
            identity = _identity(state.get("human_demo_uuid") or path.name)
            if path.name != identity or state.get("status") not in ("pending_audit", "rejected", "discarded"):
                raise ValueError("只能提交已结束且具备原始UUID的录制")
            if state.get("recording") is True or state.get("warmup") is True:
                raise ValueError("不能提交仍在录制的示例")
            if collection_enabled and seal.get("source_kind") != "live_pico_human":
                raise ValueError("合成或重放来源不得加入真人集合")
            seal_sha = _sha(path / "episode.json")
            with self._locked():
                if (self.spool / "recovery_required.json").exists():
                    raise ValueError("队列登记事务正在恢复，请保留原件后重试")
                control = self._load()
                existing_path = self.spool / "jobs" / (identity + ".json")
                job = {"contract": JOB_CONTRACT, "uuid": identity, "source_path": str(path),
                    "episode_json_sha256": seal_sha, "collection_enabled": bool(collection_enabled),
                    "metadata_sha256": _sha(path / "metadata.json") if (path / "metadata.json").exists() else None,
                    "sequence": control["next_sequence"], "status": "pending", "submitted_utc": _now(),
                    "sealed_mtime_ns": (path / "episode.json").stat().st_mtime_ns,
                    "source_kind": seal.get("source_kind"), "attempts": 0, "result": None}
                if existing_path.exists():
                    previous = _read(existing_path)
                    if any(previous.get(key) != job[key] for key in (
                            "source_path", "episode_json_sha256", "metadata_sha256", "collection_enabled")):
                        raise ValueError("同UUID的封存文件或来源已变化，拒绝覆盖")
                    return previous.get("result") or self._pending(previous,
                        discarded=(self.spool / "discard_requests" / (identity + ".json")).exists())
                _atomic(self.spool / "recovery_required.json", {"operation": "submit", "job": job})
                marker_path = path / "review_queue_submission.json"
                marker = self._marker(job)
                if marker_path.exists() and _read(marker_path) != marker:
                    raise ValueError("原件已有不同后台队列登记标记")
                if not marker_path.exists():
                    _atomic(marker_path, marker)
                self._register(control, job)
                self._publish(control)
                (self.spool / "recovery_required.json").unlink()
            if self._auto_start:
                self.start()
            return self._pending(job)
        except Exception as error:
            return _error(error, identity)

    enqueue = submit

    def snapshot(self):
        """Constant-size cached read. No flock, capture files or directory scans."""
        try:
            barrier = self.spool / "recovery_required.json"
            startup = self.spool / "startup_scan_required.json"
            if startup.exists() and not barrier.exists():
                return {"contract": STATUS_CONTRACT, "status": "recovering", "worker": {"ready": False},
                    "latest_finished_uuid": None, "latest_episode": _error("正在恢复未登记的新结束记录"),
                    "latest_completed": None, "pending_count": 1, "pending_uuids": [],
                    "latest_results": [], "updated_utc": _now(), "accepted_new_dataset": False}
            if barrier.exists():
                value = _read(barrier, MAX_STATUS_BYTES)
                identity, previous_time = None, 0
                previous_path = self.spool / "status.json"
                if previous_path.exists():
                    previous = _read(previous_path, MAX_STATUS_BYTES)
                    identity = previous.get("latest_finished_uuid")
                    previous_time = previous.get("latest_sealed_mtime_ns") or (
                        previous.get("latest_episode") or {}).get("ended_time_ns") or 0
                if (value.get("operation") == "submit"
                        and value.get("job", {}).get("sealed_mtime_ns", 0) >= previous_time):
                    identity = value.get("job", {}).get("uuid")
                return {"contract": STATUS_CONTRACT, "status": "recovering", "worker": {"ready": False},
                    "latest_finished_uuid": identity, "latest_episode": _error("队列事务等待恢复", identity),
                    "latest_completed": None, "pending_count": 1, "pending_uuids": [identity] if identity else [],
                    "latest_results": [], "updated_utc": _now(), "accepted_new_dataset": False}
            path = self.spool / "status.json"
            if not path.exists():
                return {"contract": STATUS_CONTRACT, "status": "starting", "worker": {"ready": False},
                    "latest_finished_uuid": None, "latest_episode": None, "latest_completed": None,
                    "pending_count": 0, "pending_uuids": [], "latest_results": [],
                    "updated_utc": _now(), "accepted_new_dataset": False}
            result = _read(path, MAX_STATUS_BYTES)
            if result.get("contract") != STATUS_CONTRACT:
                raise ValueError("轻量后台状态协议无效")
            worker = result.get("worker", {})
            if worker.get("ready") is True:
                try:
                    os.kill(int(worker["pid"]), 0)
                except (OSError, KeyError, TypeError, ValueError):
                    result["worker"] = {**worker, "ready": False, "stale": True}
                    if result.get("pending_count"):
                        result["status"] = "recovering"
            return result
        except Exception as error:
            return {"contract": STATUS_CONTRACT, "status": "error", "worker": {"ready": False},
                "latest_finished_uuid": None, "latest_episode": _error(error), "latest_completed": None,
                "pending_count": 1, "pending_uuids": [], "latest_results": [],
                "error": str(error)[:640], "updated_utc": _now(), "accepted_new_dataset": False}

    light_status = snapshot
    poll = snapshot

    def request_discard(self, identity):
        try:
            identity = _identity(identity)
            with self._locked():
                if (self.spool / "recovery_required.json").exists():
                    raise ValueError("队列事务尚待恢复；不能回退丢弃更早记录")
                control = self._load()
                intent = {"uuid": identity, "operator_discard_requested": True,
                    "requested_utc": _now(), "accepted_new_dataset": False}
                intent_path = self.spool / "discard_requests" / (identity + ".json")
                if intent_path.exists() and _read(intent_path).get("completed") is True:
                    return _read(intent_path)["result"]
                if not intent_path.exists():
                    _atomic(self.spool / "recovery_required.json", {"operation": "discard", "uuid": identity, "intent": intent})
                    _atomic(intent_path, intent)
                if identity not in control.setdefault("pending_discards", []):
                    control["pending_discards"].append(identity)
                # Intent is permanent; repeated clicks cannot target an older UUID.
                result = _summary({"uuid": identity, "status": "pending_discard", "pending": True,
                    "available": False, "reasons": ["已请求丢弃，等待后台归档"]})
                if control.get("latest_finished_uuid") == identity:
                    result.update({key: control["latest_episode"].get(key) for key in
                        ("path", "source_path", "number", "sequence", "collection_enabled")})
                    control["latest_episode"] = result
                self._publish(control)
                barrier = self.spool / "recovery_required.json"
                if barrier.exists():
                    barrier.unlink()
            if self._auto_start:
                self.start()
            return result
        except Exception as error:
            return _error(error, identity)

    discard = request_discard

    def close(self, wait=False):
        # Finishing CPU reviews survive the collector and next scene startup.
        if wait:
            self.wait_session(self.root)

    def wait_session(self, session_directory, timeout=None, poll_interval=.2):
        """Only for a stopped collector/finisher; never call in the control loop."""
        session = Path(session_directory).absolute()
        _within(session, self.root)
        started = time.monotonic()
        self.start()
        while True:
            jobs = [_read(path) for path in (self.spool / "jobs").glob("*.json")]
            selected = [job for job in jobs if Path(job["source_path"]).is_relative_to(session)]
            pending = [job["uuid"] for job in selected if job["status"] != "completed"]
            if (not pending and not (self.spool / "recovery_required.json").exists()
                    and not (self.spool / "startup_scan_required.json").exists()):
                return {"completed": True, "pending_uuids": [], "results": [job.get("result") for job in selected]}
            if timeout is not None and time.monotonic() - started >= timeout:
                return {"completed": False, "pending_uuids": pending, "error": "后台物理复核等待超时"}
            if self._process is None or self._process.poll() is not None:
                self.start()
            time.sleep(max(.02, min(float(poll_interval), 1.)))

    def _recover(self):
        with self._locked():
            path = self.spool / "recovery_required.json"
            if not path.exists():
                return
            barrier = _read(path)
            control = self._load()
            if barrier["operation"] == "submit":
                job = barrier["job"]
                directory = _within(job["source_path"], self.root)
                if _sha(directory / "episode.json") != job["episode_json_sha256"]:
                    raise ValueError("待恢复记录的封存SHA已变化")
                marker = directory / "review_queue_submission.json"
                if not marker.exists():
                    _atomic(marker, self._marker(job))
                elif _read(marker) != self._marker(job):
                    raise ValueError("待恢复记录的队列标记不一致")
                self._register(control, job)
            elif barrier["operation"] == "discard":
                identity = _identity(barrier["uuid"])
                intent_path = self.spool / "discard_requests" / (identity + ".json")
                if not intent_path.exists():
                    _atomic(intent_path, barrier["intent"])
                if identity not in control.setdefault("pending_discards", []):
                    control["pending_discards"].append(identity)
                if control.get("latest_finished_uuid") == identity:
                    control["latest_episode"] = {**control["latest_episode"], "available": False,
                        "pending": True, "status": "pending_discard"}
            elif barrier["operation"] == "manual_result":
                identity, result = barrier["uuid"], barrier["result"]
                self._complete_intent(identity, result)
                self._merge_manual_result(control, identity, result)
            elif barrier["operation"] == "selection_refresh":
                for job in barrier["jobs"]:
                    _atomic(self.spool / "jobs" / (_identity(job["uuid"]) + ".json"),job)
                control = barrier["control"]
                relative = Path(barrier["receipt_relative_path"])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("Invalid reselection recovery receipt path")
                _atomic(self.spool / relative,barrier["receipt"])
            elif barrier["operation"] == "result":
                job = barrier["job"]
                _atomic(self.spool / "jobs" / (job["uuid"] + ".json"), job)
                self._merge_result(control, job)
            else:
                raise ValueError("未知队列恢复事务")
            self._publish(control)
            path.unlink()

    @staticmethod
    def _merge_result(control, job):
        identity = job["uuid"]
        control["pending"] = [item for item in control["pending"] if item != identity]
        control["latest_completed"] = job["result"]
        if control.get("latest_finished_uuid") == identity:
            control["latest_episode"] = job["result"]
        control["latest_results"] = [item for item in control["latest_results"] if item["uuid"] != identity][-7:] + [job["result"]]

    def _publish_result(self, job, result):
        identity = job["uuid"]
        with self._locked():
            control = self._load()
            intent_path = self.spool / "discard_requests" / (identity + ".json")
            if intent_path.exists() and result.get("disposition") == "kept":
                return False  # Worker must consume late intent before publishing.
            job.update(status="completed", result=_summary(result), completed_utc=_now(), retry_after=0)
            _atomic(self.spool / "recovery_required.json", {"operation": "result", "job": job})
            _atomic(self.spool / "jobs" / (identity + ".json"), job)
            self._merge_result(control, job)
            self._publish(control)
            (self.spool / "recovery_required.json").unlink()
            return True

    def _discover_orphans(self):
        """Startup worker-only recovery, never part of light status polling."""
        if self.spool != self.root / ".pico_episode_review":
            return
        candidates = []
        for directory in self.root.glob("*/episodes/*"):
            if directory.is_symlink() or not directory.is_dir():
                continue
            if (self.spool / "jobs" / (directory.name + ".json")).exists():
                continue
            path = directory / "episode.json"
            if not path.exists() or path.is_symlink():
                continue
            try:
                outcome = _read(path, 4 * 1024 * 1024)
            except (OSError, ValueError):
                continue  # Unidentified legacy/corrupt files cannot be auto-imported.
            if (outcome.get("source_kind") == "live_pico_human" and outcome.get("capture_started") is True
                    and outcome.get("geometry_pending") is True
                    and outcome.get("state", {}).get("status") in ("pending_audit", "rejected", "discarded")
                    and not (self.spool / "jobs" / (directory.name + ".json")).exists()):
                candidates.append((path.stat().st_mtime_ns, str(directory), directory))
        for _, _, directory in sorted(candidates):
            result = self.submit(directory)
            if result.get("status") == "error":
                raise RuntimeError(result["error"])
        marker = self.spool / "startup_scan_required.json"
        if marker.exists():
            marker.unlink()

    def _process_one(self):
        """Worker-only heavy path; separated for deterministic CPU regression tests."""
        from pico_episode_history import EpisodeHistory
        self._recover()
        if (self.spool / "startup_scan_required.json").exists():
            self._discover_orphans()
        history = EpisodeHistory(self.index_path)
        jobs = [_read(path) for path in (self.spool / "jobs").glob("*.json")]
        if any(job["status"] != "completed" for job in jobs):
            # A preceding failed submit might not have managed even its first
            # spool write. Discover sealed new attempts before numbering a later
            # one; all scans run in this CPU worker, never in submit/poll.
            self._discover_orphans()
            jobs = [_read(path) for path in (self.spool / "jobs").glob("*.json")]
        jobs.sort(key=lambda job: (job.get("sealed_mtime_ns", 0), job["sequence"]))
        unfinished = [job for job in jobs if job["status"] != "completed"]
        if unfinished:
            job = unfinished[0]  # Never bypass a failed earlier ended attempt.
            if job.get("retry_after", 0) > time.time():
                return False
            identity = job["uuid"]
            try:
                # Recover History after an interrupted rename before requiring
                # the original UUID alias to be present again.
                if job["collection_enabled"]:
                    history.status()
                directory = _within(job["source_path"], self.root)
                if _sha(directory / "episode.json") != job["episode_json_sha256"]:
                    raise ValueError("封存episode.json已变化，拒绝重新解释记录")
                with self._locked():
                    control = self._load()
                    job.update(status="running", attempts=job.get("attempts", 0) + 1)
                    _atomic(self.spool / "jobs" / (identity + ".json"), job)
                    if control.get("latest_finished_uuid") == identity:
                        control["latest_episode"] = self._pending(job, running=True,
                            discarded=(self.spool / "discard_requests" / (identity + ".json")).exists())
                    self._publish(control)
                geometry = complete_geometry(directory, job)
                requested = (self.spool / "discard_requests" / (identity + ".json")).exists()
                if job["collection_enabled"]:
                    result = history.remember(directory)
                    if result.get("status") == "error":
                        raise RuntimeError(result.get("error") or result)
                    if requested:
                        result = history.discard_uuid(identity)
                        if result.get("status") == "error":
                            raise RuntimeError(result.get("error") or result)
                else:
                    result = {"uuid": identity, "status": "diagnostic_review_completed", "available": False,
                        "pending": False, "disposition": "diagnostic", "path": str(directory),
                        "source_path": str(directory), "reasons": ["诊断来源：只完成几何，不纳入真人集合"]}
                    if geometry and geometry.get("status") == "error":
                        result.update(status="diagnostic_review_error", error=geometry.get("error"),
                            reasons=["诊断几何复核失败：" + str(geometry.get("error"))])
                result.update(uuid=identity, sequence=job["sequence"], ended_time_ns=job.get("sealed_mtime_ns"),
                    collection_enabled=job["collection_enabled"], pending=False)
                while not self._publish_result(job, result):
                    result = history.discard_uuid(identity)
                    if result.get("status") == "error":
                        raise RuntimeError(result.get("error") or result)
                    result.update(sequence=job["sequence"], ended_time_ns=job.get("sealed_mtime_ns"), collection_enabled=True, pending=False)
                if (self.spool / "discard_requests" / (identity + ".json")).exists():
                    self._publish_manual_result(identity, result)
                return True
            except Exception as error:
                with self._locked():
                    control = self._load()
                    job.update(status="error", retry_after=time.time() + 2., result=_error(error, identity))
                    _atomic(self.spool / "jobs" / (identity + ".json"), job)
                    if control.get("latest_finished_uuid") == identity:
                        control["latest_episode"] = job["result"]
                    self._publish(control, status="error")
                return True
        # Old historical UUIDs and late requests have no pending queue job.
        for path in sorted((self.spool / "discard_requests").glob("*.json")):
            intent = _read(path)
            if intent.get("completed") is True or intent.get("retry_after", 0) > time.time():
                continue
            identity = _identity(intent["uuid"])
            result = history.discard_uuid(identity)
            if result.get("status") == "error":
                intent.update(error=result.get("error"), retry_after=time.time() + 2.)
                _atomic(path, intent)
                return True
            result.update(pending=False)
            self._publish_manual_result(identity, result)
            return True
        return False

    @staticmethod
    def _merge_manual_result(control, identity, result):
        summary = _summary(result)
        if summary.get("ended_time_ns") is None and control.get("latest_finished_uuid") == identity:
            summary["ended_time_ns"] = control.get("latest_sealed_mtime_ns")
        control["pending_discards"] = [item for item in control.get("pending_discards", []) if item != identity]
        if control.get("latest_finished_uuid") == identity:
            control["latest_episode"] = summary
        if (control.get("latest_completed") or {}).get("uuid") == identity:
            control["latest_completed"] = summary
        control["latest_results"] = [summary if item["uuid"] == identity else item for item in control["latest_results"]]

    def _publish_manual_result(self, identity, result):
        with self._locked():
            control = self._load()
            _atomic(self.spool / "recovery_required.json", {"operation": "manual_result", "uuid": identity, "result": _summary(result)})
            self._complete_intent(identity, result)
            self._merge_manual_result(control, identity, result)
            self._publish(control)
            (self.spool / "recovery_required.json").unlink()

    def _complete_intent(self, identity, result):
        job_path = self.spool / "jobs" / (identity + ".json")
        if job_path.exists():
            job = _read(job_path)
            if job.get("status") == "completed":
                result = {**result, "sequence": job["sequence"], "ended_time_ns": job.get("sealed_mtime_ns"),
                    "collection_enabled": job["collection_enabled"]}
                job["result"] = _summary(result)
                _atomic(job_path, job)
        path = self.spool / "discard_requests" / (identity + ".json")
        if path.exists():
            intent = _read(path)
            intent.update(completed=True, completed_utc=_now(), result=_summary(result))
            _atomic(path, intent)


def complete_geometry(directory, job):
    """Preserve raw solved forces; derive only geometry from recorded poses."""
    from pico_simple_teleop import complete_quality_geometry
    directory = Path(directory)
    episode_path, metadata_path = directory / "episode.json", directory / "metadata.json"
    if _sha(episode_path) != job["episode_json_sha256"]:
        raise ValueError("复核前封存回执SHA不一致")
    outcome = _read(episode_path)
    metadata = _read(metadata_path) if metadata_path.exists() else {}
    current_metadata_sha = _sha(metadata_path) if metadata_path.exists() else None
    if current_metadata_sha != job.get("metadata_sha256"):
        raise ValueError("封存时绑定的metadata SHA已变化")
    if metadata.get("task_id") == "simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0":
        return {"status":"not_required","selection_policy":"native_only",
            "reason":"Official native success is already recorded in original solved samples; extra geometry gate disabled"}
    if outcome.get("geometry_pending") is not True:
        return  # Compatible older completed geometry is rechecked by History.
    receipt_path = directory / "quality_geometry_completion.json"
    receipt = {"contract": COMPLETION_CONTRACT, "status": "error", "human_demo_uuid": job["uuid"],
        "source_kind": job["source_kind"], "episode_json_sha256": job["episode_json_sha256"],
        "metadata_sha256": _sha(metadata_path) if metadata_path.exists() else None,
        "accepted_new_dataset": False, "reviewed_utc": _now(),
        "algorithm_source_sha256": _sha(Path(__file__).with_name("pico_simple_teleop.py"))}
    try:
        raw_path, cache_path = directory / "raw_quality_samples.jsonl", directory / "container_geometry_cache.json"
        if any(path.is_symlink() or not path.is_file() for path in (raw_path, cache_path)):
            raise ValueError("原始质量采样或几何缓存缺失/是符号链接")
        cache_sha = _sha(cache_path)
        if cache_sha != metadata.get("container_geometry_cache_sha256"):
            raise ValueError("几何缓存与起录前身份SHA不一致")
        expected_cache = outcome.get("quality_geometry_expected_cache_sha256")
        if expected_cache is not None and cache_sha != expected_cache:
            raise ValueError("几何缓存与封存回执中的起录前身份SHA不一致")
        raw = [json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()]
        completed = complete_quality_geometry(raw, _read(cache_path))
        completed_path = directory / "quality_samples.jsonl"
        if completed_path.exists():
            previous = [json.loads(line) for line in completed_path.read_text().splitlines() if line.strip()]
            if completed_path.is_symlink() or previous != completed:
                raise ValueError("已存在的质量结果与原始几何重算不一致，拒绝覆盖")
        else:
            encoded = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n" for row in completed)
            fd, temporary = tempfile.mkstemp(prefix=".quality-", dir=directory)
            try:
                with os.fdopen(fd, "w") as stream:
                    stream.write(encoded)
                    stream.flush()
                    os.fsync(stream.fileno())
                # Do not replace an independently created differing result.
                os.link(temporary, completed_path)
            finally:
                os.unlink(temporary)
        receipt.update(status="completed", quality_geometry_sources={
            "raw_samples_sha256": _sha(raw_path), "cache_sha256": cache_sha,
            "completed_samples_sha256": _sha(completed_path)})
    except Exception as error:
        receipt.update(status="error", error=f"{type(error).__name__}: {error}")
    if _sha(episode_path) != job["episode_json_sha256"] or (metadata_path.exists() and _sha(metadata_path) != receipt["metadata_sha256"]):
        raise ValueError("复核期间原始回执或metadata发生变化")
    if receipt_path.exists():
        previous = _read(receipt_path)
        comparable = lambda value: {key: item for key, item in value.items() if key != "reviewed_utc"}
        if comparable(previous) != comparable(receipt):
            raise ValueError("已存在的独立几何回执与重算不一致，保留原件等待人工检查")
    else:
        _atomic(receipt_path, receipt)
    return receipt


PicoReviewQueue = ReviewQueue


def wait_session(session_directory, *, index_path=None, spool_directory=None, timeout=None):
    """Finisher convenience wrapper; the collector must already have stopped."""
    session_directory = Path(session_directory).absolute()
    index_path = Path(index_path) if index_path is not None else session_directory.parent / "pico_episode_history.json"
    return ReviewQueue(index_path, spool_directory=spool_directory).wait_session(session_directory, timeout=timeout)


def _cpu_policy():
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.nice(max(0, 19 - os.getpriority(os.PRIO_PROCESS, 0)))
    cpus = sorted(os.sched_getaffinity(0))[-1:]
    os.sched_setaffinity(0, cpus)
    try:
        subprocess.run(["ionice", "-c", "3", "-p", str(os.getpid())], timeout=2,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    except (OSError, subprocess.TimeoutExpired):
        pass
    return {"pid": os.getpid(), "ready": True, "nice": os.getpriority(os.PRIO_PROCESS, 0),
        "cpus": cpus, "cpu_only": True, "threads_per_library": 1}


def _worker_main(queue, *, once=False, idle_timeout=300.):
    with (queue.spool / "worker.lock").open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0
        policy = _cpu_policy()
        # Warm CPU imports before declaring ready; no SDK/simulator construction.
        from pico_episode_history import EpisodeHistory
        from pico_simple_teleop import complete_quality_geometry as _prewarmed
        from pico_mobile_quality import GATE_VERSION as mobile_selection_gate
        policy.update(mobile_selection_gate_version=mobile_selection_gate,
            review_code_sha256=_sha(Path(__file__)),
            history_code_sha256=_sha(Path(__file__).with_name("pico_episode_history.py")),
            mobile_quality_code_sha256=_sha(Path(__file__).with_name("pico_mobile_quality.py")))
        queue._recover()
        queue._discover_orphans()
        with queue._locked():
            control = queue._load()
        if not control.get("baseline_seeded"):
            # Isolated diagnostic spools must never touch human collection
            # recovery/locks, even when started before their first job arrives.
            baseline = (EpisodeHistory(queue.index_path).status()
                if queue.spool == queue.root / ".pico_episode_review" else {})
            with queue._locked():
                control = queue._load()
                if not control.get("latest_finished_uuid") and baseline.get("uuid"):
                    control.update(latest_finished_uuid=baseline["uuid"], latest_episode=_summary(baseline))
                if baseline.get("uuid") and not control.get("latest_results"):
                    summary = _summary(baseline)
                    control["latest_results"] = [summary]
                    if not control.get("latest_completed"):
                        control["latest_completed"] = summary
                control["baseline_seeded"] = True
                queue._publish(control)
        with queue._locked():
            control = queue._load()
            control["worker"] = policy
            queue._publish(control)
        last_work = time.monotonic()
        while True:
            worked = queue._process_one()
            if once:
                break
            if worked:
                last_work = time.monotonic()
            elif time.monotonic() - last_work > idle_timeout:
                break
            time.sleep(.15)
        with queue._locked():
            control = queue._load()
            control["worker"] = {**policy, "ready": False, "exited_utc": _now()}
            queue._publish(control)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--spool", type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--wait-session", type=Path)
    parser.add_argument("--timeout", type=float)
    options = parser.parse_args()
    queue = ReviewQueue(options.index, spool_directory=options.spool)
    if options.worker:
        return _worker_main(queue, once=options.once)
    if options.wait_session:
        result = queue.wait_session(options.wait_session, timeout=options.timeout)
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result["completed"] else 1
    parser.error("需要 --worker 或 --wait-session")


if __name__ == "__main__":
    raise SystemExit(main())
