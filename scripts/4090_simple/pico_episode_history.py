"""Locked, recoverable numbering and physical precheck of ended human attempts.

The collection keeps original evidence byte-for-byte. ``kept`` means only the
unchanged physical precheck passed; RTX, independent admission and training
packaging remain pending. No SDK, simulator or inference is initialized here.
"""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import uuid

import numpy as np


INDEX_CONTRACT = "cvpr-pico-episode-collection-index-v1"
RECORD_CONTRACT = "cvpr-pico-episode-collection-record-v1"
DISCARD_CONTRACT = "cvpr-pico-operator-discard-v1"


def _now():
    return datetime.now(timezone.utc).isoformat()


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path, *, lines=False):
    if path.is_symlink() or not path.is_file():
        raise ValueError("证据文件缺失或是符号链接：" + path.name)
    text = path.read_text()
    def decode(value):
        return json.loads(value, parse_constant=lambda word: (_ for _ in ()).throw(ValueError("非有限JSON：" + word)))
    return [decode(line) for line in text.splitlines() if line.strip()] if lines else decode(text)


def _atomic(path, value, *, json_lines=False):
    if path.is_symlink():
        raise ValueError("拒绝覆盖符号链接：" + str(path))
    encoded = (("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in value)) if json_lines
               else json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode()
    descriptor, temporary = tempfile.mkstemp(prefix=".pico-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.lexists(temporary):
            os.unlink(temporary)


def _rename_no_replace(source, destination):
    """Linux atomic directory rename that never replaces another numbered item."""
    library = ctypes.CDLL(None, use_errno=True)
    operation = getattr(library, "renameat2", None)
    if operation is None:
        raise OSError("当前系统缺少不覆盖目标的原子目录rename；原数据保留")
    operation.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    operation.restype = ctypes.c_int
    if operation(-100, os.fsencode(source), -100, os.fsencode(destination), 1):
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(destination))


def _physical_precheck(directory, outcome, metadata):
    """Recompute the existing physical gates; do not trust an outcome boolean."""
    from handover_contact_quality import evaluate_capture
    reasons, report = [], {}
    if os.path.lexists(directory / "operator_discard.json"):
        reasons.append("操作员已标记丢弃此记录")
    if "physical_gate_error" in outcome:
        reasons.append("收尾物理验收失败：" + str(outcome["physical_gate_error"]))
    if "finalization_errors" in outcome:
        reasons.append("录制收尾失败：" + str(outcome["finalization_errors"]))
    state = outcome["state"]
    if state.get("status") != "pending_audit" or state.get("pending_audit") is not True:
        why = "；".join(str(item) for item in state.get("reasons", []))
        reasons.append("录制中断：" + (why or "未完整保存待验收"))
    try:
        deferred = (directory / "deferred_capture").exists()
        capture_dir = directory / ("deferred_capture" if deferred else "capture")
        if capture_dir.is_symlink():
            raise ValueError("capture目录不能是符号链接")
        capture = _json(capture_dir / "capture.json")
        expected = "raw_physics_capture_completed_pending_rtx" if deferred else "raw_capture_completed_pending_independent_audit"
        if (capture.get("status") != expected or capture.get("failure") is not None
                or capture.get("accepted_new_dataset") is not False):
            raise ValueError("capture未完整结束或存在写入失败")
        if deferred:
            if capture.get("requested_completed") is not True or outcome.get("capture") != capture:
                raise ValueError("原始状态归档与结束回执不一致")
            for kind in ("states", "model"):
                path = capture_dir / ("states.npy" if kind == "states" else "model.mjb")
                if path.is_symlink() or not path.is_file() or _sha(path) != capture[kind + "_sha256"]:
                    raise ValueError("完整物理状态或模型文件校验失败")
            states = np.load(capture_dir / "states.npy", mmap_mode="r", allow_pickle=False)
            if states.shape != (capture["event_count"], capture["state_width"]) or not np.isfinite(states).all():
                raise ValueError("完整物理状态矩阵缺帧或非有限")
        else:
            receipt_path = capture_dir / "async_capture.json"
            receipt = _json(receipt_path)
            details = receipt["async_capture"]
            if (outcome.get("capture") != receipt or details.get("completed") is not True
                    or details.get("writer_joined") is not True or details.get("failure") is not None
                    or details.get("written_events") != capture["event_count"]
                    or details.get("enqueued_events") != capture["event_count"]
                    or {key:value for key,value in receipt.items() if key != "async_capture"} != capture):
                raise ValueError("图像写入器未完整排空或回执不一致")
        event_path = capture_dir / "events.jsonl"
        events = _json(event_path, lines=True)
        if not events or len(events) != capture["event_count"] or _sha(event_path) != capture["events_sha256"]:
            raise ValueError("capture事件缺帧或SHA不一致")
        warmup = round(float(capture["warmup_seconds"]) * 50)
        if warmup < 50 or warmup % 5:
            raise ValueError("缺少真实一秒起始历史")
        start = float(events[0]["sim_time_s"])
        for index, event in enumerate(events):
            tick = (index * 50 + 29) // 30
            if (event["event_index"] != index or event["physical_tick50"] != tick
                    or not math.isclose(event["sim_time_s"], start + tick / 50, rel_tol=0, abs_tol=1e-6)):
                raise ValueError("30Hz事件时钟不连续")
            if not deferred:
                relative = Path(event["rgb"]["path"])
                image = capture_dir / relative
                if (relative.is_absolute() or ".." in relative.parts or image.is_symlink() or not image.is_file()
                        or not image.resolve().is_relative_to(capture_dir.resolve())
                        or _sha(image) != event["rgb"]["file_sha256"]):
                    raise ValueError("RGB图像缺失、路径越界或SHA不一致")
        initial = np.asarray(events[0]["objects"]["target"]["pose7"][:3], dtype=float)
        if "initial_target_z_m" not in metadata:
            raise ValueError("缺少绑定的初始物体高度元数据")
        initial_z = metadata["initial_target_z_m"]
        if (initial.shape != (3,) or not np.isfinite(initial).all() or type(initial_z) not in (int, float)
                or not math.isfinite(initial_z) or abs(initial_z - initial[2]) > 1e-9):
            raise ValueError("初始物体高度未绑定首个实际状态")
        qualities = _json(directory / "quality_samples.jsonl", lines=True)
        if metadata.get("quality_geometry_mode") == "deferred_geometry_with_original_solved_support":
            # finalize can write completed geometry and subsequently fail its
            # source/cache check. A readable quality file is not proof that the
            # unchanged original solved-support evidence was actually retained.
            from pico_simple_teleop import complete_quality_geometry
            sources = outcome.get("quality_geometry_sources")
            expected_sources = {"raw_samples_sha256": "raw_quality_samples.jsonl",
                "cache_sha256": "container_geometry_cache.json", "completed_samples_sha256": "quality_samples.jsonl"}
            if not isinstance(sources, dict) or set(sources) != set(expected_sources):
                raise ValueError("缺少完整的原始物理/几何/结果三项SHA回执")
            actual_sources = {}
            for key, filename in expected_sources.items():
                evidence = directory / filename
                if evidence.is_symlink() or not evidence.is_file():
                    raise ValueError("原始物理或几何证据缺失/符号链接：" + filename)
                actual_sources[key] = _sha(evidence)
            if actual_sources != sources or metadata.get("container_geometry_cache_sha256") != actual_sources["cache_sha256"]:
                raise ValueError("原始物理/几何/结果SHA与收尾回执或元数据不一致")
            raw = _json(directory / "raw_quality_samples.jsonl", lines=True)
            cache = _json(directory / "container_geometry_cache.json")
            if complete_quality_geometry(raw, cache) != qualities:
                raise ValueError("几何补算与原始已解算接触/支撑记录不一致")
        if len(qualities) <= warmup or len(events) != (len(qualities) - 1) * 30 // 50 + 1:
            raise ValueError("50Hz物理记录或30Hz状态记录不完整")
        for index, row in enumerate(qualities):
            if row["completed_steps"] != index + 1 or not math.isclose(row["sim_time_s"], start + (index + 1) / 50, rel_tol=0, abs_tol=1e-6):
                raise ValueError("50Hz物理记录时钟不连续")
        if state.get("local_tick50") != len(qualities) - 1:
            raise ValueError("结束步数与物理记录不一致")
        if len(events) - warmup * 30 // 50 < 30:
            raise ValueError("有效动作记录不足一秒")
        for row in qualities[:warmup]:
            if (row["contacts"].get("left_hand") is not False or row["contacts"].get("right_hand") is not False
                    or np.linalg.norm(np.asarray(row["actual_target_xyz_m"]) - initial) > .01):
                reasons.append("起始无监督历史中已开始抓取或移动物体")
                break
        supervised = [{**row, "completed_steps": index + 1} for index, row in enumerate(qualities[warmup:])]
        report = evaluate_capture(supervised, initial_target_z_m=initial_z)
        report["excluded_unsupervised_warmup_ticks50"] = warmup
        checks = (report.get("contact_evidence") or {}).get("checks", {})
        labels = {"contact_data_complete":"接触记录不完整", "ordered_lifted_right_both_left":"未完成右手→双手→左手交接",
            "object_lifted":"物体抬升不足10厘米", "both_hands_released_final_window":"末段未完全松手",
            "final_container_contact_sustained":"末段未持续接触篮筐", "final_object_stable":"末段物体仍在晃动",
            "pelvis_above_fall_threshold":"机器人跌倒", "final_window_complete":"末段稳定观察不足一秒",
            "stability_window_complete":"末段稳定观察不足一秒"}
        for key, text in labels.items():
            if checks.get(key) is False:
                reasons.append(text)
        geometry = (report.get("final_placement_geometric_validation") or {}).get("v2_checks", {})
        for key, text in (("container_upright", "篮筐倾倒"), ("collision_vertical_ranges_overlap", "目标未落在篮筐有效范围内"),
                          ("basket_supports_target_weight", "篮筐支撑不足")):
            if geometry.get(key) is False:
                reasons.append(text)
        placement = report.get("final_placement_geometric_validation") or {}
        if placement.get("legacy_outer_footprint_passed") is False:
            reasons.append("目标未落在篮筐有效范围内")
        if report.get("eligible_physics") is not True and not reasons:
            reasons.append("物理交接与放置验收未通过")
        forces = _json(directory / "applied_force_samples.jsonl", lines=True)
        if len(forces) != len(qualities) * 4:
            raise ValueError("缺少原始200Hz外力记录")
        for index, row in enumerate(forces):
            if (not math.isclose(row["sim_time_s"], start + (index + 1) / 200, rel_tol=0, abs_tol=1e-6)
                    or any(row.get(key) is not True for key in ("all_zero", "qfrc_applied_all_zero", "xfrc_applied_all_zero"))
                    or any(row.get(key) != 0 for key in ("qfrc_applied_nonzero_count", "xfrc_applied_nonzero_count",
                        "qfrc_applied_max_abs", "xfrc_applied_max_abs", "max_abs_qfrc_applied", "max_abs_xfrc_applied"))):
                raise ValueError("原始200Hz外力记录非零或时钟缺失")
    except Exception as error:
        reasons.append("记录完整性验收失败：" + str(error))
    return list(dict.fromkeys(reasons)), report


class EpisodeHistory:
    """Three controlled APIs; no operation removes an original evidence file."""
    def __init__(self, index_path):
        self.index_path = Path(os.path.abspath(os.fspath(index_path)))
        self.root = self.index_path.parent
        self.lock_path = self.index_path.with_name(self.index_path.name + ".lock")
        self.log_path = self.root / "collection_records.jsonl"

    def _safe(self, path, *, allow_leaf_link=False):
        path = Path(path)
        if not path.is_absolute():
            path = self.root / path
        if ".." in path.parts or not path.is_relative_to(self.root):
            raise ValueError("示例路径超出本次集合目录")
        for item in (self.root, *reversed(path.relative_to(self.root).parents)):
            item = self.root / item if not item.is_absolute() else item
            if item.is_symlink():
                raise ValueError("拒绝目录符号链接逃逸")
        if path.is_symlink() and not allow_leaf_link:
            raise ValueError("拒绝未登记的目录符号链接")
        return path

    @contextmanager
    def _locked(self):
        for item in (self.root, *self.root.parents):
            if item.is_symlink():
                raise ValueError("索引目录不能经过符号链接")
        if not self.root.is_dir() or self.index_path.is_symlink():
            raise ValueError("索引目录缺失或索引是符号链接")
        descriptor = os.open(self.lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _load(self):
        if not self.index_path.exists():
            return {"contract": INDEX_CONTRACT, "version": 1, "next_number": 1, "latest_uuid": None, "entries": {}}
        value = _json(self.index_path)
        if (value.get("contract") != INDEX_CONTRACT or value.get("version") != 1
                or type(value.get("next_number")) is not int or value["next_number"] < 1
                or not isinstance(value.get("entries"), dict)):
            raise ValueError("编号索引格式无效，未分配或移动任何新示例")
        numbers = []
        for identity, entry in value["entries"].items():
            if str(uuid.UUID(identity)) != identity or type(entry.get("number")) is not int or entry["number"] < 1:
                raise ValueError("索引中的UUID或编号无效")
            numbers.append(entry["number"])
            self._target(entry["disposition"], entry["number"])
            self._safe(entry["source_relative_path"], allow_leaf_link=True)
        if len(set(numbers)) != len(numbers) or value["next_number"] <= max(numbers, default=0):
            raise ValueError("索引存在重复编号或编号倒退，拒绝覆盖")
        return value

    def _target(self, disposition, number):
        if disposition not in ("kept", "discarded") or type(number) is not int or number < 1:
            raise ValueError("编号或分流目录无效")
        return self._safe(self.root / disposition / f"{number:03d}")

    def _episode(self, directory):
        outcome = _json(directory / "episode.json")
        state = outcome.get("state", {})
        if outcome.get("source_kind") != "live_pico_human":
            raise ValueError("仅能登记本机真人结束记录，拒绝合成或外来来源")
        if (state.get("status") not in ("pending_audit", "rejected", "discarded")
                or state.get("recording") is True or state.get("warmup") is True
                or outcome.get("status") in ("following", "recording", "warmup")):
            raise ValueError("不能登记仍在进行的录制")
        metadata = _json(directory / "metadata.json") if (directory / "metadata.json").exists() else {}
        # A failed finalization may have only {status: rejected}; it still
        # belongs in the numbered rejected collection, preserving its UUID.
        supplied_identity = state.get("human_demo_uuid") or metadata.get("human_demo_uuid") or directory.name
        identity = str(uuid.UUID(supplied_identity))
        if metadata and (metadata.get("source_kind") != "live_pico_human"
                         or str(uuid.UUID(metadata["human_demo_uuid"])) != identity):
            raise ValueError("示例元数据的真人来源或身份不一致")
        return identity, outcome, metadata

    def _record(self, entry, **changes):
        value = dict(entry["planned_record"])
        value.update(changes)
        return value

    def _alias(self, source, target, previous):
        source = self._safe(source, allow_leaf_link=True)
        if os.path.lexists(source):
            if not source.is_symlink() or source.resolve(strict=False) not in (target, previous):
                raise ValueError("原UUID路径被其他数据占用，实际记录已保留可恢复")
            temporary = source.with_name(".pico-alias-" + str(uuid.uuid4()))
            try:
                os.symlink(os.path.relpath(target, source.parent), temporary, target_is_directory=True)
                os.replace(temporary, source)
            finally:
                if os.path.lexists(temporary):
                    temporary.unlink()
        else:
            os.symlink(os.path.relpath(target, source.parent), source, target_is_directory=True)

    def _log(self, record, operation_id):
        rows = _json(self.log_path, lines=True) if self.log_path.exists() else []
        if not any(row.get("operation_id") == operation_id for row in rows):
            _atomic(self.log_path, rows + [{**record, "operation_id": operation_id}], json_lines=True)

    def _finish(self, index, identity):
        entry = index["entries"][identity]
        if entry["status"] == "committed":
            return
        target = self._target(entry["disposition"], entry["number"])
        source = self._safe(entry["source_relative_path"], allow_leaf_link=True)
        previous = self._safe(entry["from_relative_path"], allow_leaf_link=True)
        if previous.is_symlink() and (previous != source or previous.resolve(strict=False) != target):
            raise ValueError("待恢复目录出现未登记符号链接")
        target.parent.mkdir(exist_ok=True)
        if target.exists():
            if previous.exists() and not previous.is_symlink():
                raise ValueError("目标编号已存在，保留原数据，拒绝覆盖")
            existing = _json(target / "collection_record.json")
            if existing.get("human_demo_uuid") != identity or existing.get("number") != entry["number"]:
                raise ValueError("目标编号属于其他记录，拒绝覆盖")
        else:
            if not previous.is_dir() or previous.is_symlink():
                raise ValueError("待恢复的原始示例目录缺失；编号事务保留")
            if _sha(previous / "episode.json") != entry["episode_json_sha256"]:
                raise ValueError("结束回执已变化，拒绝移动")
            record_path = previous / "collection_record.json"
            if record_path.exists() and _json(record_path).get("human_demo_uuid") != identity:
                raise ValueError("已有集合记录身份冲突")
            _atomic(record_path, self._record(entry, status="prepared"))
            if entry.get("manual_discard"):
                tombstone = previous / "operator_discard.json"
                marker = {"contract": DISCARD_CONTRACT, "version": 1, "status": "operator_discarded",
                    "operator_discarded": True, "source_kind": "live_pico_human", "human_demo_uuid": identity,
                    "number": entry["number"], "episode_directory": str(target),
                    "episode_json_sha256": entry["episode_json_sha256"], "recorded_utc": _now(),
                    "reason": "operator_discard_previous_completed_episode", "accepted_new_dataset": False,
                    "eligible_for_packing": False}
                if os.path.lexists(tombstone):
                    old = _json(tombstone)
                    if old.get("human_demo_uuid") != identity or old.get("operator_discarded") is not True:
                        raise ValueError("丢弃标记无效，原始数据未覆盖")
                else:
                    _atomic(tombstone, marker)
            _rename_no_replace(previous, target)
        if _sha(target / "episode.json") != entry["episode_json_sha256"]:
            raise ValueError("归档后的原始结束回执变化")
        record = self._record(entry, status="committed")
        _atomic(target / "collection_record.json", record)
        self._alias(source, target, previous)
        self._log(record, entry["operation_id"])
        entry.update(status="committed", actual_relative_path=str(target.relative_to(self.root)))
        for key in ("planned_record", "from_relative_path", "manual_discard"):
            entry.pop(key, None)
        _atomic(self.index_path, index)

    def _recover(self, index):
        for identity, entry in index["entries"].items():
            if entry.get("status") != "committed":
                self._finish(index, identity)

    def _view(self, index):
        identity = index.get("latest_uuid")
        if identity is None:
            return {"status": "empty", "available": False, "path": None, "source_path": None,
                "uuid": None, "number": None, "disposition": None, "reasons": [], "discarded": False, "error": None}
        entry = index["entries"][identity]
        target = self._target(entry["disposition"], entry["number"])
        if entry["status"] != "committed" or not target.is_dir():
            raise ValueError("上一条编号事务尚未完成，原始数据保留可恢复")
        record = _json(target / "collection_record.json")
        if (record.get("human_demo_uuid") != identity or record.get("number") != entry["number"]
                or record.get("disposition") != entry["disposition"] or record.get("status") != "committed"):
            raise ValueError("上一条编号记录身份不一致")
        if entry["disposition"] == "kept" and os.path.lexists(target / "operator_discard.json"):
            raise ValueError("发现丢弃标记但实际目录尚未分流，保留记录等待恢复")
        discarded = entry["disposition"] == "discarded" or os.path.lexists(target / "operator_discard.json")
        return {"status": "discarded" if discarded else "ready", "available": not discarded,
            "path": str(target), "source_path": str(self._safe(entry["source_relative_path"], allow_leaf_link=True)),
            "uuid": identity, "number": entry["number"], "disposition": entry["disposition"],
            "reasons": entry["reasons"], "discarded": discarded, "error": None,
            "pendingRTX": entry.get("pendingRTX", False), "accepted_new_dataset": False}

    def _guard_unregistered_attempts(self, index):
        """Never silently fall back to an older pointer after index I/O loss.

        Only the current collector's explicit capture_started marker opts into
        this recovery barrier. Historical diagnostic/legacy folders are not
        automatically imported or made into new collection attempts.
        """
        for directory in self.root.glob("*/episodes/*"):
            if (directory.is_symlink() or directory.parent.is_symlink()
                    or directory.parent.parent.is_symlink() or not directory.is_dir()):
                continue
            receipt = directory / "episode.json"
            if not receipt.is_file() or receipt.is_symlink():
                continue
            try:
                outcome = _json(receipt)
            except (ValueError, OSError):
                continue  # Cannot identify an explicit new-workflow marker.
            state = outcome.get("state", {})
            if (outcome.get("source_kind") != "live_pico_human" or outcome.get("capture_started") is not True
                    or state.get("status") not in ("pending_audit", "rejected", "discarded")):
                continue
            identity = state.get("human_demo_uuid") or directory.name
            entry = index["entries"].get(identity)
            if entry is None or self.root / entry["source_relative_path"] != directory:
                raise ValueError("发现未登记的新结束记录，禁止回退丢弃旧记录；请先恢复编号归档：" + str(directory))

    def _run(self, operation):
        try:
            with self._locked():
                index = self._load()
                self._recover(index)
                return operation(index)
        except Exception as error:
            return {"status": "error", "available": False, "path": None, "source_path": None,
                "uuid": None, "number": None, "disposition": None, "reasons": ["归档失败，原始数据保留可恢复：" + str(error)],
                "discarded": False, "error": f"{type(error).__name__}: {error}", "accepted_new_dataset": False}

    def status(self):
        def operation(index):
            self._guard_unregistered_attempts(index)
            return self._view(index)
        return self._run(operation)

    def remember(self, episode_directory):
        def operation(index):
            path = self._safe(Path(episode_directory), allow_leaf_link=True)
            if path.is_symlink():
                known = [entry for entry in index["entries"].values()
                         if self.root / entry["source_relative_path"] == path]
                if len(known) != 1 or path.resolve(strict=True) != self._target(known[0]["disposition"], known[0]["number"]):
                    raise ValueError("拒绝未登记或指向集合外的符号链接")
                path = path.resolve(strict=True)
            identity, outcome, metadata = self._episode(path)
            if identity in index["entries"]:
                entry = index["entries"][identity]
                if path != self._target(entry["disposition"], entry["number"]) or _sha(path / "episode.json") != entry["episode_json_sha256"]:
                    raise ValueError("同UUID不能登记另一份外来记录")
                return self._view(index)
            if path.name != identity:
                raise ValueError("未登记示例目录名必须是原始human_demo_uuid")
            reasons, physical = _physical_precheck(path, outcome, metadata)
            disposition = "discarded" if reasons else "kept"
            number = index["next_number"]
            target = self._target(disposition, number)
            if any(os.path.lexists(self._target(kind, number)) for kind in ("kept", "discarded")):
                raise ValueError("待分配编号目标已存在，未覆盖或消耗新编号")
            pending_rtx = outcome.get("status") == "pending_rtx"
            record = {"contract": RECORD_CONTRACT, "version": 1, "number": number,
                "human_demo_uuid": identity, "source_kind": "live_pico_human", "disposition": disposition,
                "status": "prepared", "reasons": reasons, "physical_report": physical,
                "source_episode_directory": str(path), "episode_directory": str(target),
                "episode_json_sha256": _sha(path / "episode.json"), "recorded_utc": _now(),
                "pendingRTX": pending_rtx, "accepted_new_dataset": False, "eligible_for_packing": False,
                "scope": "仅现有物理门禁预验收；保留不代表RTX、独立验收或训练数据已通过"}
            index["entries"][identity] = {"number": number, "disposition": disposition, "status": "prepared",
                "source_relative_path": str(path.relative_to(self.root)), "from_relative_path": str(path.relative_to(self.root)),
                "episode_json_sha256": record["episode_json_sha256"], "reasons": reasons, "pendingRTX": pending_rtx,
                "planned_record": record, "operation_id": "remember:" + identity}
            index.update(next_number=number + 1, latest_uuid=identity)
            _atomic(self.index_path, index)
            self._finish(index, identity)
            return self._view(index)
        return self._run(operation)

    def discard_previous(self):
        def operation(index):
            self._guard_unregistered_attempts(index)
            current = self._view(index)
            if current["status"] == "empty":
                return {**current, "already_discarded": False}
            if current["discarded"]:
                return {**current, "already_discarded": True}
            identity = current["uuid"]
            entry = index["entries"][identity]
            previous = self._target(entry["disposition"], entry["number"])
            actual_identity, _, _ = self._episode(previous)
            if actual_identity != identity or _sha(previous / "episode.json") != entry["episode_json_sha256"]:
                raise ValueError("上一条原始记录已变化，未执行丢弃")
            record = _json(previous / "collection_record.json")
            reasons = list(dict.fromkeys(entry["reasons"] + ["操作员手动丢弃上一条已结束记录"]))
            target = self._target("discarded", entry["number"])
            if os.path.lexists(target):
                raise ValueError("discarded同编号目录已存在，未覆盖")
            record.update(disposition="discarded", reasons=reasons, episode_directory=str(target), status="prepared")
            entry.update(disposition="discarded", reasons=reasons, status="prepared", manual_discard=True,
                from_relative_path=str(previous.relative_to(self.root)), planned_record=record,
                operation_id="operator_discard:" + identity)
            _atomic(self.index_path, index)
            self._finish(index, identity)
            return {**self._view(index), "already_discarded": False}
        return self._run(operation)
