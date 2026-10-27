"""Locked numbering and recoverable selection of ended human attempts.

The collection keeps original evidence byte-for-byte. Handover retains its
physical precheck; Mobile uses captured official native success with bound task
identity and recording integrity. ``kept`` still leaves RTX, independent
admission and training packaging pending. No SDK, simulator or inference starts.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
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
MOBILE_TASK_ID = "simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0"
MOBILE_OFFICIAL_SOURCE_SHA256 = "fee537d56e134b0399ff147013438168601498ab9b3ee068658dcad5fb8ca521"
MOBILE_NATIVE_SUCCESS_CRITERIA = .9
RESELECT_CONTRACT = "cvpr-pico-mobile-native-reselection-v1"


def _now():
    return datetime.now(timezone.utc).isoformat()


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def _selection_fields(report):
    return {key: deepcopy(report[key]) for key in ("selection_policy", "selection_status",
        "acceptance_gate_version", "official_task_success_observed", "official_task_success",
        "native_success_evidence_complete", "technical_blocked", "evaluator_source_sha256") if key in report}


def _preserve_bytes(source, target):
    """Exclusive durable evidence backup; an existing different backup is fatal."""
    data = source.read_bytes()
    if target.exists():
        if target.is_symlink() or target.read_bytes() != data:
            raise ValueError("旧证据备份已存在但内容不同：" + str(target))
        return
    with target.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _file_inventory(directory):
    """Original evidence bindings; collection decisions are backed up separately."""
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("原始示例内不允许符号链接：" + str(path))
        if path.is_file() and path.name not in ("collection_record.json", "operator_discard.json"):
            result[str(path.relative_to(directory))] = {"bytes": path.stat().st_size, "sha256": _sha(path)}
    return result


def _verify_inventory(directory, inventory):
    # Concurrent later review may add a separate receipt. Every pre-existing
    # evidence file must still exist unchanged; directory rename preserves extras.
    for relative, expected in inventory.items():
        path = directory / relative
        if (Path(relative).is_absolute() or ".." in Path(relative).parts or path.is_symlink()
                or any(parent.is_symlink() for parent in path.parents if parent != directory and parent.is_relative_to(directory))
                or not path.is_file() or path.stat().st_size != expected["bytes"]
                or _sha(path) != expected["sha256"]):
            raise ValueError("重筛迁移中的原始证据身份变化：" + relative)


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



def quality_geometry_outcome(directory, outcome):
    """Read a separately bound background result, leaving sealed evidence intact."""
    if outcome.get("geometry_pending") is not True:
        return outcome
    directory = Path(directory)
    receipt = _json(directory / "quality_geometry_completion.json")
    metadata = _json(directory / "metadata.json")
    if (metadata.get("quality_geometry_mode") != "deferred_geometry_with_original_solved_support"
            or outcome.get("quality_geometry_expected_cache_sha256") != metadata.get("container_geometry_cache_sha256")
            or not isinstance(outcome.get("quality_geometry_expected_cache_sha256"), str)):
        raise ValueError("后台几何缓存未绑定起录前的原始身份")
    if (receipt.get("contract") != "cvpr-pico-quality-geometry-completion-v1"
            or receipt.get("status") not in ("completed", "error")
            or receipt.get("episode_json_sha256") != _sha(directory / "episode.json")
            or receipt.get("metadata_sha256") != _sha(directory / "metadata.json")
            or receipt.get("human_demo_uuid") != metadata.get("human_demo_uuid")
            or receipt.get("source_kind") != metadata.get("source_kind")
            or receipt.get("accepted_new_dataset") is not False):
        raise ValueError("后台几何复核回执未绑定原始示例或状态无效")
    effective = dict(outcome)
    if receipt["status"] == "error":
        if not receipt.get("error"):
            raise ValueError("后台几何失败回执缺少原因")
        effective["physical_gate_error"] = str(receipt["error"])
        return effective
    sources = receipt.get("quality_geometry_sources")
    if receipt.get("error") or not isinstance(sources, dict) or set(sources) != {
            "raw_samples_sha256", "cache_sha256", "completed_samples_sha256"}:
        raise ValueError("后台几何结果缺少完整来源SHA")
    for digest in sources.values():
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("后台几何来源SHA格式无效")
    effective["quality_geometry_sources"] = sources
    effective["quality_geometry_mode"] = "deferred_geometry_with_original_solved_support"
    return effective


def _physical_precheck(directory, outcome, metadata):
    """Check raw recording integrity plus the explicitly dispatched task policy.

    Mobile selects any captured native success from the fixed official task;
    Handover retains its existing physical/geometry gates. Neither trusts an
    outcome summary boolean or relabels technical failure as official failure.
    """
    reasons, report = [], {}
    mobile = metadata.get("task_id") == MOBILE_TASK_ID
    technical = []
    native_identity = None
    if mobile:
        supplied_source = metadata.get("actual_task_source_sha256")
        supplied_criteria = metadata.get("actual_native_success_criteria")
        source_valid = supplied_source == MOBILE_OFFICIAL_SOURCE_SHA256
        criteria_valid = type(supplied_criteria) in (int, float) and supplied_criteria == MOBILE_NATIVE_SUCCESS_CRITERIA
        native_identity = {"expected_task_source_sha256": MOBILE_OFFICIAL_SOURCE_SHA256,
            "recorded_task_source_sha256": supplied_source, "source_sha256_matches": source_valid,
            "expected_success_criteria": MOBILE_NATIVE_SUCCESS_CRITERIA,
            "recorded_success_criteria": supplied_criteria, "success_criteria_matches": criteria_valid,
            "passed": source_valid and criteria_valid,
            "identity_scope": "fixed audited official task, independent of currently installed source or old collection gates"}
        if not source_valid:
            technical.append("官方Mobile任务模块SHA缺失或不匹配固定官方实现")
        if not criteria_valid:
            technical.append("官方Mobile success_criteria缺失或不等于0.9")
        reasons.extend(technical)
    if os.path.lexists(directory / "operator_discard.json"):
        reasons.append("操作员已标记丢弃此记录")
    if "physical_gate_error" in outcome:
        reasons.append("收尾物理验收失败：" + str(outcome["physical_gate_error"]))
        technical.append(reasons[-1])
    if "finalization_errors" in outcome:
        reasons.append("录制收尾失败：" + str(outcome["finalization_errors"]))
        technical.append(reasons[-1])
    if not mobile:
        try:
            outcome = quality_geometry_outcome(directory, outcome)
            if "physical_gate_error" in outcome:
                reasons.append("后台物理几何复核失败：" + str(outcome["physical_gate_error"]))
        except Exception as error:
            reasons.append("后台几何复核未完成或回执无效：" + str(error))
    state = outcome["state"]
    if state.get("status") != "pending_audit" or state.get("pending_audit") is not True:
        why = "；".join(str(item) for item in state.get("reasons", []))
        reasons.append("录制中断：" + (why or "未完整保存待验收"))
        technical.append(reasons[-1])
    try:
        task_id = metadata.get("task_id")
        if task_id == "simple/G1WholebodyHandoverTeleop-v0":
            from handover_contact_quality import evaluate_capture
        elif task_id == MOBILE_TASK_ID:
            from pico_mobile_quality import evaluate_mobile_capture as evaluate_capture
        else:
            raise ValueError("缺少或不支持的任务身份，不能套用交接验收：" + str(task_id))
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
        if warmup < (0 if mobile else 50) or warmup % 5:
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
        if not mobile and "initial_target_z_m" not in metadata:
            raise ValueError("缺少绑定的初始物体高度元数据")
        initial_z = metadata.get("initial_target_z_m")
        if not mobile and (initial.shape != (3,) or not np.isfinite(initial).all() or type(initial_z) not in (int, float)
                or not math.isfinite(initial_z) or abs(initial_z - initial[2]) > 1e-9):
            raise ValueError("初始物体高度未绑定首个实际状态")
        quality_path = directory / ("raw_quality_samples.jsonl" if mobile and os.path.lexists(
            directory / "raw_quality_samples.jsonl") else "quality_samples.jsonl")
        qualities = _json(quality_path, lines=True)
        if mobile:
            raw_digest = _sha(quality_path)
            prior_record_path = directory / "collection_record.json"
            if prior_record_path.exists():
                prior_report = _json(prior_record_path).get("physical_report", {})
                if (prior_report.get("native_flag_source_file") == quality_path.name
                        and prior_report.get("native_flag_source_sha256") not in (None, raw_digest)):
                    raise ValueError("原始native记录与此前筛选回执SHA不一致")
            old_sources = outcome.get("quality_geometry_sources", {})
            geometry_receipt = directory / "quality_geometry_completion.json"
            if geometry_receipt.exists():
                old_receipt = _json(geometry_receipt)
                if (old_receipt.get("episode_json_sha256") != _sha(directory / "episode.json")
                        or old_receipt.get("metadata_sha256") != _sha(directory / "metadata.json")):
                    raise ValueError("旧几何回执的原始身份已变化")
                old_sources = old_receipt.get("quality_geometry_sources", old_sources)
            if quality_path.name == "raw_quality_samples.jsonl" and isinstance(old_sources, dict):
                bound_raw = old_sources.get("raw_samples_sha256")
                if bound_raw is not None and bound_raw != raw_digest:
                    raise ValueError("原始native物理记录与旧SHA绑定不一致")
        if not mobile and metadata.get("quality_geometry_mode") == "deferred_geometry_with_original_solved_support":
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
        if len(qualities) <= (0 if mobile else warmup) or len(events) != (len(qualities) - 1) * 30 // 50 + 1:
            raise ValueError("50Hz物理记录或30Hz状态记录不完整")
        for index, row in enumerate(qualities):
            if row["completed_steps"] != index + 1 or not math.isclose(row["sim_time_s"], start + (index + 1) / 50, rel_tol=0, abs_tol=1e-6):
                raise ValueError("50Hz物理记录时钟不连续")
        if state.get("local_tick50") != len(qualities) - 1:
            raise ValueError("结束步数与物理记录不一致")
        if not mobile and len(events) - warmup * 30 // 50 < 30:
            raise ValueError("有效动作记录不足一秒")
        for row in ([] if mobile else qualities[:warmup]):
            if (row["contacts"].get("left_hand") is not False or row["contacts"].get("right_hand") is not False
                    or np.linalg.norm(np.asarray(row["actual_target_xyz_m"]) - initial) > .01):
                reasons.append("起始无监督历史中已开始抓取或移动物体")
                break
        supervised = qualities if mobile else [{**row, "completed_steps": index + 1}
                                               for index, row in enumerate(qualities[warmup:])]
        report = evaluate_capture(supervised, initial_target_z_m=initial_z)
        report["excluded_unsupervised_warmup_ticks50"] = 0 if mobile else warmup
        if mobile:
            report["native_flag_source_file"] = quality_path.name
            report["native_flag_source_sha256"] = _sha(quality_path)
        checks = (report.get("contact_evidence") or {}).get("checks", {})
        labels = {"contact_data_complete":"接触记录不完整", "ordered_lifted_right_both_left":"未完成右手→双手→左手交接",
            "object_lifted":"物体抬升不足10厘米", "both_hands_released_final_window":"末段未完全松手",
            "final_container_contact_sustained":"末段未持续接触篮筐", "final_object_stable":"末段物体仍在晃动",
            "pelvis_above_fall_threshold":"机器人跌倒", "final_window_complete":"末段稳定观察不足一秒",
            "stability_window_complete":"末段稳定观察不足一秒",
            "root_position_evidence_complete":"缺少实际机器人根部位置证据",
            "lifted_hand_hold_sustained":"未形成持续的单手或双手抬起抓持",
            "target_and_root_carried_075m":"抓持搬运期间物体与机器人未同时移动至少0.75米（本次数采附加门）",
            "target_final_transport_075m":"最终物体水平搬运不足0.75米（本次数采附加门）"}
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
        if not mobile and report.get("eligible_physics") is not True and not reasons:
            reasons.append("移动抓取与放置验收未通过" if task_id == "simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0"
                           else "物理交接与放置验收未通过")
        for error in report.get("capture_validation_errors", []):
            if mobile:
                technical.append("原始官方成功证据无效：" + str(error))
                reasons.append(technical[-1])
                continue
            if "actual_root_xyz_m" in str(error):
                reasons.append("缺少或无效的实际机器人根部三维位置证据")
            else:
                reasons.append("原始物理证据缺失或无效：" + str(error))
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
        technical.append(reasons[-1])
    if mobile:
        from pico_mobile_quality import GATE_VERSION
        report.update(selection_policy="native_only", technical_errors=list(dict.fromkeys(technical)),
                      technical_blocked=bool(technical), acceptance_gate_version=GATE_VERSION, evaluator_source_sha256={
                          "pico_episode_history.py": _sha(Path(__file__)),
                          "pico_mobile_quality.py": _sha(Path(__file__).with_name("pico_mobile_quality.py"))})
        report["task_identity_validation"] = native_identity
        if technical:
            report.setdefault("official_task_success_observed", None)
            report.setdefault("native_success_evidence_complete", False)
            report.update(selection_status="technical_evidence_incomplete", eligible_physics=False,
                          official_task_success=None)
        elif report.get("eligible_physics") is not True:
            reasons.append("官方任务成功标志未出现")
    return list(dict.fromkeys(reasons)), report


class EpisodeHistory:
    """Three controlled APIs; no operation removes an original evidence file."""
    def __init__(self, index_path):
        self.index_path = Path(os.path.abspath(os.fspath(index_path)))
        self.root = self.index_path.parent
        self.lock_path = self.index_path.with_name(self.index_path.name + ".lock")
        self.log_path = self.root / "collection_records.jsonl"
        self.reselection_path = self.root / ".pico_mobile_native_reselection_pending.json"

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
        if os.path.lexists(self.reselection_path):
            self._finish_mobile_reselection(index, _json(self.reselection_path))

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
        selection = _selection_fields({**record.get("physical_report", {}), **entry})
        blocked = selection.get("technical_blocked") is True
        return {"status": "technical_blocked" if blocked else "discarded" if discarded else "ready", "available": not discarded and not blocked,
            "path": str(target), "source_path": str(self._safe(entry["source_relative_path"], allow_leaf_link=True)),
            "uuid": identity, "number": entry["number"], "disposition": entry["disposition"],
            "reasons": entry["reasons"], "discarded": discarded, "error": None,
            "pendingRTX": entry.get("pendingRTX", False), "accepted_new_dataset": False, **selection}

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
                return self._view({**index, "latest_uuid": identity})
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
            record.update(_selection_fields(physical))
            if metadata.get("task_id") == MOBILE_TASK_ID:
                record["scope"] = "仅官方native成功筛选及原始记录完整性；RTX、独立验收和训练准入仍待完成"
            index["entries"][identity] = {"number": number, "disposition": disposition, "status": "prepared",
                "source_relative_path": str(path.relative_to(self.root)), "from_relative_path": str(path.relative_to(self.root)),
                "episode_json_sha256": record["episode_json_sha256"], "reasons": reasons, "pendingRTX": pending_rtx,
                "planned_record": record, "operation_id": "remember:" + identity}
            index["entries"][identity].update(_selection_fields(physical))
            index.update(next_number=number + 1, latest_uuid=identity)
            _atomic(self.index_path, index)
            self._finish(index, identity)
            return self._view(index)
        return self._run(operation)

    def _mobile_snapshot(self, index, identities):
        if not isinstance(identities, (list, tuple)) or not identities or len(set(identities)) != len(identities):
            raise ValueError("重筛必须提供非空且不重复的显式Mobile UUID集合")
        selected = []
        for identity in identities:
            if str(uuid.UUID(identity)) != identity or identity not in index["entries"]:
                raise ValueError("重筛UUID尚未登记或格式无效：" + str(identity))
            entry = index["entries"][identity]
            target = self._target(entry["disposition"], entry["number"])
            actual, outcome, metadata = self._episode(target)
            if actual != identity or metadata.get("task_id") != MOBILE_TASK_ID or entry.get("status") != "committed":
                raise ValueError("重筛仅支持已提交的Mobile真人记录")
            if _sha(target / "episode.json") != entry["episode_json_sha256"]:
                raise ValueError("原始结束回执与索引SHA不一致")
            record = _json(target / "collection_record.json")
            if (record.get("human_demo_uuid") != identity or record.get("number") != entry["number"]
                    or record.get("disposition") != entry["disposition"] or record.get("status") != "committed"):
                raise ValueError("原集合回执与索引身份不一致")
            source = self._safe(entry["source_relative_path"], allow_leaf_link=True)
            if not source.is_symlink() or source.resolve(strict=True) != target:
                raise ValueError("原UUID alias未指向已登记编号目录")
            from pico_episode_review import discard_requested
            operator = target / "operator_discard.json"
            if operator.is_symlink():
                raise ValueError("原手动丢弃标记不能为符号链接")
            manual = bool(operator.exists() or discard_requested(target))
            if operator.exists():
                marker = _json(operator)
                if marker.get("human_demo_uuid") != identity or marker.get("operator_discarded") is not True:
                    raise ValueError("原手动丢弃标记身份无效")
            samples = target / ("raw_quality_samples.jsonl" if os.path.lexists(
                target / "raw_quality_samples.jsonl") else "quality_samples.jsonl")
            force = target / "applied_force_samples.jsonl"
            if samples.is_symlink() or force.is_symlink():
                raise ValueError("原始native/外力记录不能为符号链接")
            selected.append({"uuid": identity, "number": entry["number"],
                "entry_sha256": _json_sha(entry), "actual_relative_path": str(target.relative_to(self.root)),
                "source_relative_path": entry["source_relative_path"],
                "episode_json_sha256": entry["episode_json_sha256"],
                "metadata_sha256": _sha(target / "metadata.json"),
                "collection_record_sha256": _sha(target / "collection_record.json"),
                "native_source_file": samples.name,
                "native_source_sha256": _sha(samples) if samples.is_file() else None,
                "force_source_sha256": _sha(force) if force.is_file() else None,
                "operator_discard_sha256": _sha(operator) if operator.exists() else None,
                "manual_discard_intent": manual})
        selected.sort(key=lambda item: item["number"])
        return {"contract": RESELECT_CONTRACT, "selection_policy": "native_only", "root": str(self.root),
            "index_sha256": _sha(self.index_path), "next_number": index["next_number"],
            "latest_uuid": index["latest_uuid"], "selected": selected}

    def snapshot_mobile_native(self, identities):
        """Freeze caller-selected completed UUIDs; never silently expand a batch."""
        return self._run(lambda index: self._mobile_snapshot(index, identities))

    def _finish_mobile_reselection(self, index, transaction):
        if transaction.get("contract") != RESELECT_CONTRACT or transaction.get("root") != str(self.root):
            raise ValueError("重筛恢复事务身份无效")
        operation = str(uuid.UUID(transaction["operation_uuid"]))
        archive = self._safe(self.root / "reclassification_archive" / operation)
        saved = _json(archive / "transaction.json")
        if saved != transaction:
            raise ValueError("重筛事务与不可变归档不一致")
        results = []
        for plan in transaction["plans"]:
            identity = plan["uuid"]
            entry = index["entries"][identity]
            if entry["number"] != plan["number"] or entry["episode_json_sha256"] != plan["episode_json_sha256"]:
                raise ValueError("恢复重筛时原编号或UUID身份已变化")
            previous = self._safe(plan["previous_relative_path"])
            target = self._target(plan["disposition"], plan["number"])
            target.parent.mkdir(exist_ok=True)
            if previous != target and target.exists() and previous.exists():
                raise ValueError("重筛目标编号已有另一份数据，拒绝覆盖")
            location = target if target.exists() else previous
            actual_identity, _, _ = self._episode(location)
            if actual_identity != identity or _sha(location / "episode.json") != plan["episode_json_sha256"]:
                raise ValueError("重筛恢复目标的原始结束回执身份不同")
            _verify_inventory(location, plan["evidence_inventory"])
            if previous != target and not target.exists():
                _rename_no_replace(previous, target)
                for parent in (previous.parent, target.parent):
                    descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
                    try:
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)
            record = plan["new_record"]
            _atomic(target / "collection_record.json", record)
            self._alias(self._safe(entry["source_relative_path"], allow_leaf_link=True), target, previous)
            self._log(record, plan["operation_id"])
            entry.update(disposition=plan["disposition"], reasons=plan["reasons"], status="committed",
                actual_relative_path=str(target.relative_to(self.root)), operation_id=plan["operation_id"],
                reclassification_archive=str(archive.relative_to(self.root)), **_selection_fields(plan["physical_report"]))
            _atomic(self.index_path, index)
            results.append({"uuid": identity, "number": plan["number"], "previous_disposition": plan["previous_disposition"],
                "disposition": plan["disposition"], "path": str(target), "reasons": plan["reasons"],
                "manual_discard_preserved": plan["manual_discard_intent"], **_selection_fields(plan["physical_report"])})
        receipt = {"contract": RESELECT_CONTRACT, "status": "committed", "operation_uuid": operation,
            "selection_policy": "native_only", "snapshot_sha256": transaction["snapshot_sha256"],
            "selected_uuids": [plan["uuid"] for plan in transaction["plans"]], "results": results,
            "original_index_sha256": transaction["snapshot"]["index_sha256"],
            "committed_index_sha256": _sha(self.index_path), "archive": str(archive),
            "numbering_preserved": True, "manual_discard_policy": "preserve_independent_operator_intent",
            "accepted_new_dataset": False, "eligible_for_packing": False,
            "scope": "官方native筛选重新分类；保留不代表RTX、独立验收和训练准入通过"}
        receipt_path = archive / "receipt.json"
        if receipt_path.exists() and _json(receipt_path) != receipt:
            raise ValueError("已存在不同的重筛完成回执，保留事务等待核对")
        if not receipt_path.exists():
            _atomic(receipt_path, receipt)
        self.reselection_path.unlink()
        descriptor = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return receipt

    def reselect_mobile_native(self, snapshot, *, operation_uuid):
        """Explicit recoverable reclassification; never overrides manual discard.

        Caller retries a changed global index with a NEW snapshot. Once the
        prepared transaction exists, ordinary History operations recover it
        first. Old decisions/index/journal are retained byte-for-byte and no
        original episode, metadata, capture, force or geometry file is edited.
        """
        def operation(index):
            operation_id = str(uuid.UUID(operation_uuid))
            if operation_id != operation_uuid or snapshot.get("contract") != RESELECT_CONTRACT:
                raise ValueError("重筛操作UUID或快照格式无效")
            archive = self._safe(self.root / "reclassification_archive" / operation_id)
            receipt_path = archive / "receipt.json"
            if receipt_path.exists():
                receipt = _json(receipt_path)
                if receipt.get("snapshot_sha256") != _json_sha(snapshot):
                    raise ValueError("同一重筛操作不能绑定不同集合快照")
                return receipt
            current = self._mobile_snapshot(index, [item["uuid"] for item in snapshot["selected"]])
            if current != snapshot:
                raise ValueError("重筛快照已过期，索引或记录身份变化；请重新快照，未静默追加新记录")
            plans = []
            created = _now()
            for item in snapshot["selected"]:
                identity = item["uuid"]
                entry = index["entries"][identity]
                previous = self._target(entry["disposition"], entry["number"])
                _, outcome, metadata = self._episode(previous)
                reasons, report = _physical_precheck(previous, outcome, metadata)
                manual = item["manual_discard_intent"]
                if manual:
                    reasons = list(dict.fromkeys(reasons + ["操作员手动丢弃意图独立保留"])); disposition = "discarded"
                    report["selection_status"] = "operator_discarded"
                elif report.get("technical_blocked"):
                    disposition = entry["disposition"]
                else:
                    disposition = "kept" if report.get("eligible_physics") is True else "discarded"
                target = self._target(disposition, entry["number"])
                if target != previous and os.path.lexists(target):
                    raise ValueError("重筛目标编号已存在；未覆盖任何记录")
                record = _json(previous / "collection_record.json")
                record.update(disposition=disposition, reasons=reasons, physical_report=report,
                    episode_directory=str(target), status="committed", reclassified_utc=created,
                    reclassification_receipt=str(archive / "receipt.json"),
                    scope="官方native成功筛选及原始记录完整性；RTX、独立验收和训练准入仍待完成",
                    **_selection_fields(report))
                plans.append({"uuid": identity, "number": entry["number"], "episode_json_sha256": entry["episode_json_sha256"],
                    "previous_relative_path": str(previous.relative_to(self.root)),
                    "previous_disposition": entry["disposition"], "disposition": disposition,
                    "manual_discard_intent": manual, "reasons": reasons, "physical_report": report,
                    "new_record": record, "evidence_inventory": _file_inventory(previous),
                    "operation_id": "reselect_mobile_native:" + operation_id + ":" + identity})
            archive.mkdir(parents=True, exist_ok=True)
            _preserve_bytes(self.index_path, archive / "before_index.json")
            if self.log_path.exists():
                _preserve_bytes(self.log_path, archive / "before_collection_records.jsonl")
            for plan in plans:
                previous = self._safe(plan["previous_relative_path"])
                item_archive = self._safe(archive / plan["uuid"])
                item_archive.mkdir(exist_ok=True)
                _preserve_bytes(previous / "collection_record.json", item_archive / "collection_record.before.json")
                if (previous / "operator_discard.json").exists():
                    _preserve_bytes(previous / "operator_discard.json", item_archive / "operator_discard.before.json")
            transaction = {"contract": RESELECT_CONTRACT, "root": str(self.root), "operation_uuid": operation_id,
                "prepared_utc": created, "snapshot": snapshot, "snapshot_sha256": _json_sha(snapshot), "plans": plans}
            transaction_path = archive / "transaction.json"
            if transaction_path.exists() and _json(transaction_path) != transaction:
                raise ValueError("重筛归档已有不同事务，请使用新操作UUID")
            _atomic(transaction_path, transaction)
            _atomic(self.reselection_path, transaction)
            return self._finish_mobile_reselection(index, transaction)
        return self._run(operation)

    def _discard(self, index, identity):
        if identity not in index["entries"]:
            raise ValueError("指定UUID尚未编号，不能退回丢弃更早记录")
        current = self._view({**index, "latest_uuid": identity})
        if current["discarded"]:
            return {**current, "already_discarded": True}
        entry = index["entries"][identity]
        previous = self._target(entry["disposition"], entry["number"])
        actual_identity, _, _ = self._episode(previous)
        if actual_identity != identity or _sha(previous / "episode.json") != entry["episode_json_sha256"]:
            raise ValueError("指定原始记录已变化，未执行丢弃")
        record = _json(previous / "collection_record.json")
        reasons = list(dict.fromkeys(entry["reasons"] + ["操作员手动丢弃已结束记录"]))
        target = self._target("discarded", entry["number"])
        if os.path.lexists(target):
            raise ValueError("discarded同编号目录已存在，未覆盖")
        record.update(disposition="discarded", reasons=reasons, episode_directory=str(target), status="prepared")
        if record.get("selection_policy") == "native_only":
            record["selection_status"] = "operator_discarded"
        entry.update(disposition="discarded", reasons=reasons, status="prepared", manual_discard=True,
            from_relative_path=str(previous.relative_to(self.root)), planned_record=record,
            operation_id="operator_discard:" + identity)
        if record.get("selection_policy") == "native_only":
            entry["selection_status"] = "operator_discarded"
        _atomic(self.index_path, index)
        self._finish(index, identity)
        return {**self._view({**index, "latest_uuid": identity}), "already_discarded": False}

    def discard_uuid(self, identity):
        """Worker-only exact identity operation; never follow a changing pointer."""
        def operation(index):
            normalized = str(uuid.UUID(identity))
            if normalized != identity:
                raise ValueError("丢弃UUID格式无效")
            return self._discard(index, identity)
        return self._run(operation)

    def discard_previous(self):
        def operation(index):
            self._guard_unregistered_attempts(index)
            current = self._view(index)
            if current["status"] == "empty":
                return {**current, "already_discarded": False}
            return self._discard(index, current["uuid"])
        return self._run(operation)
