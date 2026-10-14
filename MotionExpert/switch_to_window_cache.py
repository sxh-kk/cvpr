"""Stop at the authorized natural epoch boundary, then await the window cache.

Only last.ckpt publication and its target filename are inspected. Checkpoint
contents are never loaded here. The stopped run resumes exactly once from that
same saved path after full independent cache acceptance.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time

import yaml

from .watch_dino import (REPO_ROOT, atomic_json, command_line, locked, now,
                         process_alive, process_identity, read_json)

CONTRACT = "motionexpert-window-cache-switch-v1"
CACHE_CONTRACT = "smplh245-window-cache-v1"
ACCEPTANCE_FORMAT = "motionexpert-window-cache-acceptance-v1"
DEFAULT_POINTER = Path("/gaozt-test1/datasets/MixEgoMotion/training/current_run.json")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _process_fields(pid):
    try:
        values = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
        if values[0] in {"Z", "X"}:
            return None
        return {"parent": int(values[1]), "group": int(values[2])}
    except (FileNotFoundError, ProcessLookupError):
        return None


def _local_rank(pid):
    try:
        # Read only LOCAL_RANK; never publish environment contents or credentials.
        for item in (Path("/proc") / str(pid) / "environ").read_bytes().split(b"\0"):
            if item.startswith(b"LOCAL_RANK="):
                return int(item.split(b"=", 1)[1])
    except (FileNotFoundError, ProcessLookupError, ValueError):
        pass
    return None


def training_ranks(pid):
    """Lightning's rank zero owns a session and directly launches ranks 1..7."""
    ranks = {0: process_identity(pid)}
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        child = int(path.name)
        fields = _process_fields(child)
        if fields is None or fields["parent"] != pid or fields["group"] != pid:
            continue
        rank = _local_rank(child)
        argv = command_line(child)
        if rank in range(1, 8) and "MotionExpert.train" in argv:
            if rank in ranks:
                raise RuntimeError("More than one child claims the same training rank")
            ranks[rank] = process_identity(child)
    if set(ranks) != set(range(8)) or any(value is None for value in ranks.values()):
        raise RuntimeError("Cannot freeze the identities of all eight Lightning ranks")
    return ranks


def _settings(config):
    expected = {("DATA", "BATCH_SIZE"): 256, ("DATA", "NUM_WORKERS"): 24,
                ("DATA", "FPS"): 5, ("DATA", "WINDOW"): 11,
                ("DATA", "FORECAST_PREFIX"): 6, ("TRAIN", "NUM_GPUS"): 8,
                ("TRAIN", "LR"): 1e-4, ("TRAIN", "PRECISION"): "bf16-mixed",
                ("TRAIN", "NUM_EPOCHS"): 200, ("TRAIN", "SCHEDULER_TOTAL_EPOCHS"): 200,
                ("TRAIN", "WARMUP_EPOCHS"): 20, ("TRAIN", "MIN_LR_RATIO"): .1,
                ("TRAIN", "SAVE_EVERY_N_EPOCHS"): 20, ("TRAIN", "USE_CKPT_LR"): True}
    for (section, name), value in expected.items():
        if config.get(section, {}).get(name) != value:
            raise ValueError(f"Original training requires {section}.{name}={value!r}")
    if config.get("MODEL", {}).get("CKPT_PATH") or config.get("MODEL", {}).get("INIT_CKPT_PATH"):
        raise ValueError("Current source configuration must retain its original initialization")


def validate_resume_config(source, resumed, cache_root):
    """Cache selection is the sole approved config change, including nested fields."""
    if not isinstance(source, dict) or not isinstance(resumed, dict):
        raise ValueError("Training configurations must be YAML mappings")
    _settings(source)
    _settings(resumed)
    if Path(resumed.get("DATA", {}).get("WINDOW_CACHE_ROOT", "")).resolve() != Path(cache_root).resolve():
        raise ValueError("Resume WINDOW_CACHE_ROOT differs from --cache-root")
    left, right = deepcopy(source), deepcopy(resumed)
    left.get("DATA", {}).pop("WINDOW_CACHE_ROOT", None)
    right.get("DATA", {}).pop("WINDOW_CACHE_ROOT", None)
    if left != right:
        raise ValueError("Resume config may change only DATA.WINDOW_CACHE_ROOT")


def log_progress(path):
    """Read a bounded log tail; Lightning epochs are zero based."""
    with Path(path).open("rb") as stream:
        stream.seek(max(0, Path(path).stat().st_size - (1 << 16)))
        tail = stream.read().decode(errors="replace")
    matches = list(re.finditer(r"\bEpoch\s+(\d+)\s*:[^\r\n]*?\b(\d+)/(\d+)\b", tail))
    if not matches:
        raise ValueError("Cannot determine the current training epoch from its log")
    epoch, batch, total = map(int, matches[-1].groups())
    if total < 1 or not 0 <= batch <= total:
        raise ValueError("Training progress contains invalid batch counters")
    return {"epoch": epoch, "batch": batch, "steps_per_epoch": total}


def completed_checkpoint(output):
    """A new natural last symlink is published only after full serialization."""
    output = Path(output).resolve()
    last = output / "last.ckpt"
    if not last.is_symlink():
        return None
    try:
        target = last.resolve(strict=True)
    except FileNotFoundError:
        return None
    match = re.fullmatch(r"epoch=(\d+)-step=(\d+)(?:-v\d+)?\.ckpt", target.name)
    if not target.is_file() or target.parent != output or match is None:
        return None
    epoch, step = map(int, match.groups())
    if (epoch + 1) % 20 or step < 1:
        raise ValueError("last.ckpt is not a natural twenty-epoch training checkpoint")
    return {"path": str(target), "epoch": epoch, "global_step": step,
            "completed_epochs": epoch + 1}


def validate_boundary(value):
    if type(value) is not int or not 0 < value < 200 or value % 20:
        raise ValueError("--completed-epochs must name a natural twenty-epoch save before epoch200")


def validate_acceptance(plan):
    """Only independently bound full acceptance can authorize the saved-run resume."""
    root = Path(plan["cache_root"])
    manifest_path = root / "manifest.json"
    acceptance_path = Path(plan["acceptance"])
    acceptance = None
    if acceptance_path.is_file():
        report = read_json(acceptance_path)
        if report.get("format") != ACCEPTANCE_FORMAT:
            raise ValueError("Window cache acceptance has an unsupported format")
        if report.get("passed") is True:
            if report.get("state") == "failed" or report.get("error"):
                raise ValueError("Window cache independent acceptance has failed")
            acceptance = report
        elif not (report.get("passed") is False
                  and report.get("state") in (None, "waiting_cache", "validating")
                  and not report.get("finished_at") and not report.get("error")):
            # An interrupted builder can leave a building manifest behind.
            # A finished validator failure must win over that pending state.
            raise ValueError("Window cache independent acceptance has not passed")
    if not manifest_path.is_file():
        return None
    manifest = read_json(manifest_path)
    if manifest.get("state") == "failed":
        raise RuntimeError("Window cache build failed; no cache-based training will be launched")
    if manifest.get("state") in {"building", "extracting", "pending"}:
        return None
    if (manifest.get("contract") != CACHE_CONTRACT or manifest.get("state") != "complete"
            or manifest.get("scope") != "all" or manifest.get("diagnostic") is not False
            or manifest.get("splits") != ["train", "val", "test"]):
        raise ValueError("Only a completed full non-diagnostic window cache can be switched in")
    if acceptance is None:
        return None
    if Path(acceptance.get("cache_root", "")).resolve() != root:
        raise ValueError("Window cache acceptance names a different cache")
    expected = {"cache_manifest_sha256": file_hash(manifest_path),
                "source_config_sha256": plan["source_config_sha256"],
                "resume_config_sha256": plan["resume_config_sha256"]}
    for key, value in expected.items():
        if acceptance.get(key) != value:
            raise ValueError(f"Window cache acceptance binding differs: {key}")
    if file_hash(plan["source_config"]) != plan["source_config_sha256"]:
        raise ValueError("Original frozen configuration changed")
    if file_hash(plan["resume_config"]) != plan["resume_config_sha256"]:
        raise ValueError("Validated resume configuration changed")
    return {**expected, "acceptance_sha256": file_hash(acceptance_path)}


def assert_original(plan):
    current = read_json(plan["current_run"])
    if current.get("trainer") != plan["trainer"] or current.get("pid") != plan["trainer"]["pid"]:
        raise RuntimeError("Current run was replaced; refusing to touch the frozen trainer")
    if not process_alive(plan["trainer"]) or os.getpgid(plan["trainer"]["pid"]) != plan["pgid"]:
        raise RuntimeError("Original training identity/session changed")
    for identity in plan["ranks"].values():
        if not process_alive(identity) or os.getpgid(identity["pid"]) != plan["pgid"]:
            raise RuntimeError("A frozen Lightning rank exited or changed identity/session")
    guardian = plan.get("guardian")
    if guardian is not None and process_alive(guardian):
        if current.get("watcher") != guardian:
            raise RuntimeError("Original guardian identity changed")


def _group_alive(pgid):
    return any(path.name.isdigit() and (_process_fields(int(path.name)) or {}).get("group") == pgid
               for path in Path("/proc").iterdir())


def stop_original(plan, timeout=180):
    """Terminate only the verified Lightning session, then wait for its guardian."""
    assert_original(plan)
    os.killpg(plan["pgid"], signal.SIGTERM)
    try:
        os.killpg(plan["pgid"], signal.SIGCONT)
    except ProcessLookupError:
        # All processes may already have honored TERM before CONT is sent.
        pass
    deadline = time.monotonic() + timeout
    while _group_alive(plan["pgid"]):
        if time.monotonic() >= deadline:
            raise RuntimeError("Original training session did not exit; no new training was launched")
        time.sleep(.1)
    guardian = plan.get("guardian")
    while guardian is not None and process_alive(guardian):
        if time.monotonic() >= deadline:
            raise RuntimeError("Original guardian has not completed its final pointer write")
        time.sleep(.1)


def prepare_plan(pointer, cache_root, acceptance, job_dir, resume_config=None, completed_epochs=40):
    validate_boundary(completed_epochs)
    current = read_json(pointer)
    if current.get("training_running") is not True:
        raise ValueError("Current run is not an active training session")
    trainer = current.get("trainer")
    if trainer is None or trainer != process_identity(current.get("pid", -1)):
        raise ValueError("Current trainer PID identity is no longer valid")
    pid = trainer["pid"]
    if os.getpgid(pid) != pid:
        raise ValueError("Current Lightning trainer must own an isolated session")
    guardian = current.get("watcher")
    if guardian is not None and not process_alive(guardian):
        raise ValueError("Current guardian PID identity is no longer valid")
    gpus = current.get("gpus")
    if not isinstance(gpus, list) or len(set(gpus)) != 8 or any(type(g) is not int or g < 0 for g in gpus):
        raise ValueError("Current run must use eight distinct GPU IDs")
    source = Path(current["config"]).resolve(strict=True)
    old_config = yaml.safe_load(source.read_text())
    _settings(old_config)
    if Path(old_config["TRAIN"]["EXP_PATH"]).resolve() != Path(current["output"]).resolve():
        raise ValueError("Current frozen config and output directory disagree")
    snapshot = Path(job_dir) / "resume-config.yaml"
    if resume_config is None:
        new_config = deepcopy(old_config)
        new_config["DATA"]["WINDOW_CACHE_ROOT"] = str(Path(cache_root).resolve())
        content = yaml.safe_dump(new_config, sort_keys=False).encode()
    else:
        content = Path(resume_config).read_bytes()
        new_config = yaml.safe_load(content)
    validate_resume_config(old_config, new_config, cache_root)
    with snapshot.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    command = current.get("command", [])
    if not command or not Path(command[0]).is_file():
        raise ValueError("Current run has no existing training Python executable")
    output = Path(current["output"]).resolve(strict=True)
    root = Path(pointer).resolve().parent.parent
    if (output / "window_cache_launch.json").exists():
        raise FileExistsError("This training run already consumed its one-shot cache transition")
    plan = {"contract": CONTRACT, "current_run": str(Path(pointer).resolve()),
            "root": str(root), "output": str(output), "old_run": deepcopy(current),
            "trainer": trainer, "guardian": guardian, "pgid": pid, "ranks": training_ranks(pid),
            "old_log": current["log"], "gpus": gpus, "python": command[0],
            "cache_root": str(Path(cache_root).resolve()), "acceptance": str(Path(acceptance).resolve()),
            "source_config": str(source), "source_config_sha256": file_hash(source),
            "resume_config": str(snapshot.resolve()), "resume_config_sha256": file_hash(snapshot),
            "job_dir": str(Path(job_dir).resolve()), "created_at": now(),
            "completed_epochs": completed_epochs}
    log_progress(plan["old_log"])
    assert_original(plan)
    atomic_json(Path(job_dir) / "plan.json", plan)
    return plan


def run_switch(plan, *, poll_seconds=.5, stop=None, launch=subprocess.Popen,
               shutdown=stop_original):
    stop = stop or threading.Event()
    root, job = Path(plan["root"]), Path(plan["job_dir"])
    boundary = plan.get("completed_epochs", 40)
    validate_boundary(boundary)
    state = {**deepcopy(plan), "watcher": process_identity(os.getpid()), "state": "armed",
             "automatic_restart_enabled": False, "training_running": True,
             "checkpoint": None, "pid": plan["trainer"]["pid"], "boundary": boundary,
             "log": str(job / "training.log"), "daemon_log": str(job / "daemon.log"),
             "checkpoint_inspection": "natural_last_symlink_and_filename_only"}
    last_publish = 0.
    owns_pointer, frozen, old_stopped = False, False, False
    child = None
    acceptance_stamps, acceptance_result = None, None

    def accepted(*, force=False):
        nonlocal acceptance_stamps, acceptance_result
        stamps = []
        for path in (Path(plan["cache_root"]) / "manifest.json", Path(plan["acceptance"]),
                     Path(plan["source_config"]), Path(plan["resume_config"])):
            try:
                info = path.stat()
                stamps.append((info.st_ino, info.st_size, info.st_mtime_ns))
            except FileNotFoundError:
                stamps.append(None)
        if force or stamps != acceptance_stamps:
            acceptance_result = validate_acceptance(plan)
            acceptance_stamps = stamps
        return acceptance_result

    def publish(phase, **fields):
        nonlocal last_publish
        changed = state.get("state") != phase
        state.update(state=phase, updated_at=now(), **fields)
        atomic_json(job / "state.json", state)
        atomic_json(root / ".pipeline/current_cache_transition.json", state)
        if owns_pointer:
            for pointer in (Path(plan["current_run"]), root / ".pipeline/current_dino_to_motion.json"):
                atomic_json(pointer, state)
        last_publish = time.monotonic()
        if changed:
            print(f"[{now()}] {phase}: {fields.get('reason', '')}", flush=True)

    def check_configuration():
        if (file_hash(plan["source_config"]) != plan["source_config_sha256"]
                or file_hash(plan["resume_config"]) != plan["resume_config_sha256"]):
            raise ValueError("Frozen training configuration changed before the transition")

    def checkpoint_identity(path):
        # stat only: never open or deserialize checkpoint contents.
        value = Path(path).stat()
        return value.st_ino, value.st_size, value.st_mtime_ns

    try:
        check_configuration()
        config = yaml.safe_load(Path(plan["resume_config"]).read_text())
        vision = config["DATA"]["VISION_PROFILE_ROOT"]
        log = job / "training.log"
        if log.exists() or (job / "launch-marker.json").exists():
            raise FileExistsError("A previous transition launch must never be retried")
        publish("waiting_checkpoint", reason=f"Stop after {boundary} completed epochs regardless of cache readiness")
        while not stop.is_set():
            assert_original(plan)
            checkpoint = completed_checkpoint(plan["output"])
            if checkpoint is not None and checkpoint["completed_epochs"] >= boundary:
                if checkpoint["completed_epochs"] != boundary:
                    raise RuntimeError(f"The fixed {boundary}-epoch last checkpoint was already superseded")
                break
            if time.monotonic() - last_publish > 30:
                publish("waiting_checkpoint")
            stop.wait(poll_seconds)
        if stop.is_set():
            publish("cancelled", reason="Transition cancelled before the boundary; original training continues")
            return 0
        command = [plan["python"], "-u", "-m", "MotionExpert.train", "--config", plan["resume_config"],
                   "--resume", checkpoint["path"], "--output-dir", plan["output"]]
        env = os.environ.copy()
        env.pop("MOTIONEXPERT_DATA_DIR", None)
        for name in ("LOCAL_RANK", "RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
            env.pop(name, None)
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, plan["gpus"]))
        env["PYTHONUNBUFFERED"] = "1"
        saved_identity = checkpoint_identity(checkpoint["path"])
        state.update(checkpoint=checkpoint["path"], command=command,
                     saved_checkpoint=checkpoint, checkpoint_file_identity=list(saved_identity))
        # Keep the visual read lease, and reserve the cooperating launchers'
        # handoff while the original guardian releases its existing run lock.
        with ExitStack() as resume_locks, locked(Path(vision) / ".extraction.lock", shared=True) as vision_lock:
            with locked(root / "training/.launcher-reservation.lock"):
                check_configuration()
                assert_original(plan)
                if completed_checkpoint(plan["output"]) != checkpoint:
                    raise ValueError("Natural last checkpoint changed before stopping")
                marker = {"contract": CONTRACT, "created_at": now(), "checkpoint": checkpoint,
                          "command": command, "cache_acceptance_pending": True,
                          "checkpoint_inspection": state["checkpoint_inspection"],
                          "automatic_restart_enabled": False}
                for path in (job / "launch-marker.json", Path(plan["output"]) / "window_cache_launch.json"):
                    with path.open("x") as stream:
                        json.dump(marker, stream, indent=2)
                        stream.flush()
                        os.fsync(stream.fileno())
                if stop.is_set():
                    publish("cancelled", reason="Transition cancelled before handoff; original training continues")
                    return 0
                assert_original(plan)
                os.killpg(plan["pgid"], signal.SIGSTOP)
                frozen = True
                assert_original(plan)
                if stop.is_set():
                    os.killpg(plan["pgid"], signal.SIGCONT)
                    frozen = False
                    publish("cancelled", reason="Original training continues without switching")
                    return 0
                publish("stopping_original", reason=f"Natural {boundary}-epoch last symlink published; stopping the owned session")
                shutdown(plan)
                old_stopped, frozen = True, False
                atomic_json(job / "previous-run-final.json", read_json(plan["current_run"]))
                run_lock = resume_locks.enter_context(locked(root / "training/.dino-to-motion.lock"))
            # The run lock now covers the complete stopped waiting period.
            # No alive-original checks are meaningful after this handoff.
            owns_pointer = True
            state["original_trainer"] = state["trainer"]
            state["original_ranks"] = state.pop("ranks")
            publish("waiting_cache_after_stop", pid=None, trainer=None, pgid=None,
                    config=plan["resume_config"], training_running=False, stopped_at=now(),
                    cache_acceptance_pending=True,
                    reason=f"Training stopped at the saved {boundary}-epoch boundary; cache is not yet accepted")
            while not stop.is_set():
                acceptance = accepted()
                if acceptance is not None:
                    break
                if time.monotonic() - last_publish > 30:
                    publish("waiting_cache_after_stop", training_running=False)
                stop.wait(poll_seconds)
            if stop.is_set():
                publish("supervisor_stopped", training_running=False,
                        reason=f"Supervisor stopped; training remains stopped at the saved {boundary}-epoch checkpoint")
                return 0
            check_configuration()
            if accepted(force=True) != acceptance:
                raise ValueError("Window cache acceptance changed before the saved-run resume")
            if checkpoint_identity(checkpoint["path"]) != saved_identity:
                raise ValueError("The frozen natural checkpoint file was replaced before resume")
            # No checkpoint contents are inspected; train.py performs its
            # ordinary --resume loading only when launching the actual resume.
            state["acceptance_binding"] = acceptance
            with log.open("x") as stream:
                publish("starting_resume", training_running=False, cache_acceptance_pending=False,
                        reason=f"Cache accepted; launching once from the saved {boundary}-epoch checkpoint")
                child = launch(command, cwd=REPO_ROOT, env=env, stdin=subprocess.DEVNULL,
                               stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                               pass_fds=(run_lock.fileno(), vision_lock.fileno()))
            publish("training", pid=child.pid, trainer=process_identity(child.pid), pgid=child.pid,
                    training_running=True, resumed_at=now(), initialization="resume",
                    automatic_start_enabled=False, reason="Window-cache resume launched; no automatic retries")
            while child.poll() is None and not stop.wait(poll_seconds):
                if time.monotonic() - last_publish > 30:
                    publish("training", training_running=True)
            if child.poll() is None:
                publish("supervisor_stopped", training_running=True,
                        reason="Supervisor stopped; resumed training continues independently")
                return 0
            code = child.returncode
            publish("completed" if code == 0 else "failed", training_running=False,
                    exit_code=code, finished_at=now(), reason="Resume ended; automatic retry remains disabled")
            return 0 if code == 0 else 1
    except Exception as error:
        if frozen and any(process_alive(identity) for identity in plan["ranks"].values()):
            try:
                os.killpg(plan["pgid"], signal.SIGCONT)
            except ProcessLookupError:
                pass
        publish("failed", error=f"{type(error).__name__}: {error}",
                training_running=(child is not None and child.poll() is None) or
                                 (not old_stopped and process_alive(plan["trainer"])),
                reason=("Transition failed; training remains stopped at its saved boundary" if old_stopped
                        else "Transition failed before stopping; original training remains independent"))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-run", type=Path, default=DEFAULT_POINTER)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--resume-config", type=Path)
    parser.add_argument("--job-dir", type=Path)
    parser.add_argument("--poll-seconds", type=float, default=.5)
    parser.add_argument("--completed-epochs", type=int, default=40)
    parser.add_argument("--background", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.poll_seconds) or args.poll_seconds <= 0:
        parser.error("--poll-seconds must be finite and positive")
    try:
        validate_boundary(args.completed_epochs)
    except ValueError as error:
        parser.error(str(error))
    pointer = args.current_run.resolve(strict=True)
    root = pointer.parent.parent
    job = (args.job_dir or root / ".pipeline/jobs" / (
        "motion-window-cache-switch-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))).resolve()
    if job.is_relative_to(REPO_ROOT) or not job.is_relative_to(root / ".pipeline/jobs"):
        parser.error("--job-dir must be owned by MixEgoMotion/.pipeline/jobs outside the repository")
    job.mkdir(parents=True, exist_ok=True)
    if args.background:
        if any(job.iterdir()):
            parser.error("Use an empty job directory for a new transition")
        command = [sys.executable, "-u", "-m", "MotionExpert.switch_to_window_cache",
                   "--current-run", str(pointer), "--cache-root", str(args.cache_root.resolve()),
                   "--acceptance", str(args.acceptance.resolve()), "--job-dir", str(job),
                   "--poll-seconds", str(args.poll_seconds),
                   "--completed-epochs", str(args.completed_epochs)]
        if args.resume_config:
            command.extend(["--resume-config", str(args.resume_config.resolve())])
        with (job / "daemon.log").open("x") as stream:
            child = subprocess.Popen(command, cwd=REPO_ROOT, stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if (job / "state.json").is_file():
                state = read_json(job / "state.json")
                print(json.dumps({"pid": child.pid, "state": state["state"], "job_dir": str(job),
                                  "state_file": str(job / "state.json")}), flush=True)
                return 1 if state["state"] == "failed" else 0
            if child.poll() is not None:
                print((job / "daemon.log").read_text(), file=sys.stderr)
                return 1
            time.sleep(.1)
        print(json.dumps({"pid": child.pid, "state": "starting", "job_dir": str(job)}), flush=True)
        return 0
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    try:
        with locked(root / "training/.window-cache-transition.lock"):
            if any((job / name).exists() for name in ("plan.json", "state.json", "resume-config.yaml", "launch-marker.json")):
                raise FileExistsError("Existing transition jobs cannot be restarted")
            plan = prepare_plan(pointer, args.cache_root, args.acceptance, job, args.resume_config,
                                args.completed_epochs)
            return run_switch(plan, poll_seconds=args.poll_seconds, stop=stop)
    except Exception as error:
        print(f"[{now()}] Transition failed: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
