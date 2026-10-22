"""CPU lossless LeRobot packaging of independently admitted real RTX captures.

Raw warmup action=null stays unavailable: packed labels are NaN, validity and
all80 mask dimensions are false. Actual warmup RGB/state remain valid history.
Only currents >=30 with30 real future labels are eligible. This does not run
DINO, training, simulation, or turn a pending independent TEST into readiness.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sys
from types import ModuleType
import uuid

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
VIDEO_KEY = "observation.images.egocentric"
CONTRACT = "cvpr-handover-quality100-lerobot-pack-v1"
CAPTURE_CONTRACT = "cvpr-handover-teacher-rtx-capture-v1"
SOURCE38_SHA = "12a71e046e221b83f0bdb0efb50f9ba82442045aca0efabef3ae337f185320b7"
HELDOUT = {39, 65, 80, 92, 94}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def owned(root, relative):
    root, relative = Path(root).resolve(strict=True), Path(relative)
    require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe relative input path")
    target = root / relative
    require(target.is_file() and not target.is_symlink() and target.resolve().is_relative_to(root),
            "Input escapes its declared capture directory")
    return target


def vector(value, width, label):
    array = np.asarray(value, dtype=np.float64)
    require(array.shape == (width,) and np.isfinite(array).all(), f"Require finite {label}[{width}]")
    return array


def checkpoint_contract(preparation, run_config, statistics, state_info):
    preparation, run_config, statistics, state_info = map(lambda p: Path(p).resolve(strict=True),
                                                        (preparation, run_config, statistics, state_info))
    receipt, config, info, stats = read(preparation), read(run_config), read(state_info), read(statistics)
    require(receipt.get("source_step") == 38000 and receipt.get("source_checkpoint_sha256") == SOURCE38_SHA,
            "Require the actual protected UnifoLM38k initialization receipt")
    require(sha(run_config) == receipt["config_sha256"], "Bound training config changed")
    norm = receipt["normalization"]
    require(sha(statistics) == norm["stat_file_sha256"], "Inherited normalization bytes changed")
    require(sha(state_info) == receipt["source_metadata_sha256"][receipt["state_contract"]],
            "Actual38k named state schema changed")
    field = config["data"]["transform"]["field"]
    require(field["stat_action_keys"] == ["action.body_token_v1_1", "action[:14]"]
            and field["stat_state_keys"] == ["observation.state"]
            and field["pad_action_dim"] == 80 and field["pad_state_dim"] == 45
            and field["normalize_state"] is True and field["action_norm_type"] == "bounds",
            "Checkpoint normalization/repack layout differs")
    bounds = {}
    for key in ("action_min", "action_max", "state_min", "state_max"):
        digest = hashlib.sha256(json.dumps(field[key], separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        require(digest == norm["bounds_sha256"][key], "Inherited normalization bounds changed")
        bounds[key] = digest
    state = info["features"]["observation.state"]
    names = state["names"]
    require(state["shape"] == [43] and len(names) == len(set(names)) == 43, "Require43 named physical state channels")
    hand_names = names[-14:]
    require(all("hand" in name for name in hand_names), "State contract must end in14 named hand joints")
    require(info["features"]["action"]["names"][:14] == hand_names,
            "Original action[:14] normalization must use the same named hand order")
    for stat_key, suffix in (("min", "min"), ("max", "max")):
        expected_action = np.r_[stats["action.body_token_v1_1"][stat_key], stats["action"][stat_key][:14], 0., 0.].astype(np.float32)
        expected_state = np.r_[stats["observation.state"][stat_key], 0., 0.].astype(np.float32)
        require(np.array_equal(expected_action, np.asarray(field[f"action_{suffix}"], np.float32))
                and np.array_equal(expected_state, np.asarray(field[f"state_{suffix}"], np.float32)),
                "Bound field arrays do not derive from the unchanged checkpoint statistics")
    inherited = {"observation.state": deepcopy(stats["observation.state"]),
                 "action.body_token_v1_1": deepcopy(stats["action.body_token_v1_1"])}
    # The original action statistics contain more than hands. Preserve the exact
    # inherited first14 values used by the actual training field, without fit.
    inherited["action"] = {key: values[:14] if key != "count" else values
                           for key, values in stats["action"].items()}
    identity = {"source_step": 38000, "source_checkpoint_sha256": SOURCE38_SHA,
        "preparation": {"path": str(preparation), "sha256": sha(preparation)},
        "run_config": {"path": str(run_config), "sha256": sha(run_config)},
        "statistics": {"path": str(statistics), "sha256": sha(statistics)},
        "state_info": {"path": str(state_info), "sha256": sha(state_info)},
        "bounds_sha256": bounds, "bounds_refitted": False, "state_joint_names": names,
        "hand_joint_names": hand_names}
    return identity, inherited


def collect_admitted(queue_root, limit=None, *, collector=None, candidate_ids=None):
    """Recheck collector admission, every input SHA and independent CPU audit."""
    if collector is None:
        import collect_handover_quality100 as collector
    queue_root = Path(queue_root).resolve(strict=True)
    queue_bytes = (queue_root / "queue.json").read_bytes()
    queue_sha = hashlib.sha256(queue_bytes).hexdigest()
    queue, plan, candidates = verify_registry_snapshot(collector, queue_root, queue_bytes)
    verify_append_only_registry(json.loads(queue_bytes), read(queue_root / "queue.json"))
    require(queue.get("contract") == collector.QUEUE_CONTRACT
            and queue.get("authority_revision") == collector.AUTHORITY_REVISION,
            "Unknown collector registry or obsolete audit authority")
    accepted = [a for a in queue["attempts"] if a["status"] == "accepted"]
    require(bool(accepted), "No independently admitted real RTX capture exists")
    if candidate_ids is not None:
        require(isinstance(candidate_ids, list) and bool(candidate_ids)
                and all(isinstance(value, str) for value in candidate_ids)
                and len(set(candidate_ids)) == len(candidate_ids), "Candidate selection must be an explicit unique list")
        by_id = {a["candidate_id"]: a for a in accepted}
        require(set(candidate_ids) <= set(by_id), "Selection contains an unaccepted candidate")
        accepted = [by_id[value] for value in candidate_ids]
    if limit is not None:
        require(type(limit) is int and limit > 0, "Pilot limit must be a positive integer")
        accepted = accepted[:limit]
    counts = Counter(a["source_episode_index"] for a in accepted)
    allowed = {a["source_episode_index"] for a in plan["original_ancestors"]}
    require(len(allowed) == 24 and not allowed & HELDOUT and set(counts) <= allowed
            and len(accepted) <= 100 and max(counts.values()) <= 5,
            "TRAIN100 requires24 original ancestors, excludes VAL and caps each family at5")
    recipe = queue.get("recipe")
    require(isinstance(recipe, dict), "Missing verified frozen warmup recipe")
    schedule = recipe.get("warmup_schedule")
    require(isinstance(schedule, dict), "Missing verified frozen warmup schedule")
    controls = schedule.get("controls")
    require(isinstance(controls, dict) and set(controls) == {str(source) for source in allowed}
            and all(control in collector.WARMUP_KINDS for control in controls.values()),
            "Frozen warmup controls must cover every original TRAIN ancestor")
    proposals = {(c["candidate_id"], digest): c for c, _, digest in candidates}
    packed, ids, content = [], set(), set()
    for attempt in accepted:
        candidate = proposals[(attempt["candidate_id"], attempt["plan_sha256"])]
        require(candidate["candidate_spec_sha256"] == attempt["candidate_spec_sha256"], "Accepted candidate changed")
        source_id = candidate["ancestor"]["source_episode_index"]
        require(source_id == attempt["source_episode_index"], "Accepted source differs from its frozen candidate")
        require(attempt["candidate_id"] not in ids, "Repeated accepted candidate")
        ids.add(attempt["candidate_id"])
        audit_path = collector.local_file(queue_root, attempt["audit_path"])
        require(sha(audit_path) == attempt["audit_sha256"], "Independent admission receipt changed")
        audit = read(audit_path)
        require(audit.get("contract") == collector.AUDIT_CONTRACT
                and audit.get("authority_revision") == collector.AUTHORITY_REVISION
                and audit.get("status") == "independent_capture_audit_passed"
                and audit.get("eligible_dataset") is True, "Physics-only reports are not dataset admissions")
        for group in (audit["bindings"], audit["source_sha256"]):
            for path, expected in group.items():
                require(Path(path).is_absolute() and sha(path) == expected, "Admitted input/source SHA changed")
        directory = Path(attempt["directory"])
        require(not directory.is_absolute() and ".." not in directory.parts, "Unsafe attempt directory")
        run = (queue_root / directory / "rtx").resolve(strict=True)
        require(run.is_relative_to(queue_root), "Run escapes collector root")
        capture = run / "capture"
        checked = collector.audit_capture(run, capture, candidate, attempt["plan_sha256"],
                                          expected_warmup=controls[str(source_id)],
                                          **({"expected_reference": schedule["teacher_references"][str(source_id)]}
                                             if str(source_id) in schedule.get("teacher_references", {}) else {}))
        for key in checked:
            if key == "generated_utc":
                continue
            require(checked[key] == audit[key], f"Independent admission no longer reproduces: {key}")
        require(audit["capture_content_sha256"] not in content, "Duplicate physical capture content")
        content.add(audit["capture_content_sha256"])
        manifest = read(capture / "capture.json")
        require(manifest.get("contract") == CAPTURE_CONTRACT, "Unknown real RTX event contract")
        events_path = owned(capture, manifest["events_path"])
        events = [json.loads(line) for line in events_path.read_text().splitlines() if line]
        packed.append({"attempt": attempt, "candidate": candidate, "capture": capture, "manifest": manifest,
                       "events": events, "audit": audit, "audit_path": audit_path, "result": read(run / "result.json")})
    verify_append_only_registry(json.loads(queue_bytes), read(queue_root / "queue.json"))
    return packed, plan, {"queue_path": str(queue_root / "queue.json"), "queue_sha256": queue_sha,
                          "collector_contract": collector.QUEUE_CONTRACT,
                          "audit_contract": collector.AUDIT_CONTRACT,
                          "authority_revision": collector.AUTHORITY_REVISION,
                          "warmup_schedule": schedule,
                          "_snapshot_utf8": queue_bytes.decode("utf-8"),
                          "accepted_selected": len(packed), "accepted_registry_total": sum(a["status"] == "accepted" for a in queue["attempts"])}


def verify_registry_snapshot(collector, root, registry_bytes):
    """Run the frozen auditor against one consistent initial registry read.

    Keep the actual capture paths and every validation function unchanged.
    Only this call's queue.json read uses its saved input bytes. This permits
    append-only collection to continue without repeatedly invalidating a long
    full pixel audit. The module globals and running collector are untouched.
    """
    from types import FunctionType
    from copy import deepcopy
    function = collector.verified_queue
    snapshot = json.loads(registry_bytes)
    if "read" not in function.__globals__:
        # Dependency-injected unit-test auditors do not read a registry file.
        return function(root)
    original_read = function.__globals__["read"]
    registry = (Path(root) / "queue.json").resolve(strict=True)
    def snapshot_read(path):
        if Path(path).resolve(strict=True) == registry:
            return deepcopy(snapshot)
        return original_read(path)
    globals_copy = {**function.__globals__, "read": snapshot_read}
    scoped = FunctionType(function.__code__, globals_copy, function.__name__, function.__defaults__, function.__closure__)
    scoped.__kwdefaults__ = function.__kwdefaults__
    return scoped(root)


def verify_append_only_registry(snapshot, live):
    """Running attempts may advance; admitted entries may never be rewritten."""
    for key in ("contract", "authority_revision", "plan_path", "plan_sha256", "recipe", "frozen_recipe_sha256"):
        require(live.get(key) == snapshot.get(key), "Live queue changed its frozen identity during packing")
    prior = snapshot.get("registrations", [])
    require(live.get("registrations", [])[:len(prior)] == prior, "Live queue rewrote prior registrations")
    old, current = snapshot["attempts"], live["attempts"]
    require(len(current) >= len(old), "Live queue removed previously registered attempts")
    identity_keys = ("candidate_id", "candidate_spec_sha256", "plan_path", "plan_sha256", "source_episode_index", "directory")
    for before, after in zip(old, current):
        require(all(before.get(key) == after.get(key) for key in identity_keys),
                "Live queue rewrote an existing attempt identity")
        if before["status"] == "accepted":
            require(before == after, "Live queue changed a previously admitted capture entry")


def collect_admitted_sources(manifest_path, limit=None):
    """Use each queue's exact frozen auditor without rewriting a running queue.

    This explicit local execution manifest is created by the operator. A
    declared SHA must match both the executable auditor and its already frozen
    queue recipe before import. Each auditor still performs its full independent
    source, physics, renderer and pixel checks; no version is upgraded in place.
    """
    import importlib.util
    manifest_path = Path(manifest_path).resolve(strict=True)
    document = read(manifest_path)
    require(document.get("contract") == "cvpr-handover-multi-queue-selection-v1"
            and document.get("scope") == "TRAIN_only"
            and isinstance(document.get("sources"), list) and bool(document["sources"]),
            "Use an explicit TRAIN-only multi-queue selection")
    items, bindings, snapshots, plan = [], [], [], None
    seen_queues, seen_ids, seen_contents = set(), set(), set()
    for index, source in enumerate(document["sources"]):
        queue_root = Path(source["queue_root"]).resolve(strict=True)
        require(queue_root not in seen_queues, "Repeated queue in collection selection")
        seen_queues.add(queue_root)
        auditor = Path(source["collector_path"]).resolve(strict=True)
        require(auditor.name == "collect_handover_quality100.py"
                and sha(auditor) == source["collector_sha256"], "Declared executable auditor identity changed")
        queue = read(queue_root / "queue.json")
        code_key = "scripts/4090_simple/collect_handover_quality100.py"
        require(queue.get("recipe", {}).get("code_sha256", {}).get(code_key) == sha(auditor)
                and sha(owned(queue_root / "frozen-code", code_key)) == sha(auditor),
                "Auditor must be the version frozen before queue execution")
        name = "handover_frozen_auditor_" + sha(auditor)
        spec = importlib.util.spec_from_file_location(name, auditor)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        selected, candidate_plan, binding = collect_admitted(queue_root, collector=module,
                                                            candidate_ids=source["candidate_ids"])
        if plan is None:
            plan = candidate_plan
        require(candidate_plan["original_ancestors"] == plan["original_ancestors"]
                and candidate_plan["source_manifest_sha256"] == plan["source_manifest_sha256"]
                and candidate_plan["selection_sha256"] == plan["selection_sha256"],
                "Queues must retain identical original TRAIN ancestry and release identity")
        snapshots.append(binding.pop("_snapshot_utf8"))
        bindings.append({**binding, "collector_path": str(auditor), "collector_sha256": sha(auditor)})
        for item in selected:
            candidate_id, content = item["candidate"]["candidate_id"], item["audit"]["capture_content_sha256"]
            require(candidate_id not in seen_ids and content not in seen_contents,
                    "Combined dataset repeats a candidate or complete physical capture")
            seen_ids.add(candidate_id); seen_contents.add(content)
            items.append(item)
    if limit is not None:
        require(type(limit) is int and limit > 0, "Pilot limit must be a positive integer")
        items = items[:limit]
    counts = Counter(item["candidate"]["ancestor"]["source_episode_index"] for item in items)
    require(bool(items) and len(items) <= 100 and max(counts.values()) <= 5,
            "Combined TRAIN selection must contain at most100 captures and at most5 per original ancestor")
    return items, plan, {"contract": document["contract"], "selection_path": str(manifest_path),
        "selection_sha256": sha(manifest_path), "sources": bindings,
        "_snapshots_utf8": snapshots, "accepted_selected": len(items)}


def eligible_currents(events):
    """Use the production history selector and complete-future validity check."""
    package_name = "_handover_pack_history"
    if package_name not in sys.modules:
        package = ModuleType(package_name)
        package.__path__ = [str(REPO / "VLMExpert/src/psi/data/lerobot")]
        sys.modules[package_name] = package
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    history = importlib.import_module(f"{package_name}.motion_history")
    from finetune_carrybox import future_is_complete
    n = len(events); frame = np.arange(n, dtype=np.int64)
    nominal = np.asarray([e["nominal_elapsed_s"] for e in events], dtype=np.float32)
    ticks = np.asarray([e["physical_tick50"] for e in events], dtype=np.int64)
    valid = np.asarray([e["action_valid"] for e in events], dtype=bool)
    # DINO indexes the actual packed native30Hz stream; warmup RGB is valid.
    current, selected = history.history_rows(frame, nominal, frame, frame / 30, np.ones(n, bool), fps=30)
    accepted = [int(c) for c in current if c >= 30 and future_is_complete(int(c), n, frame, nominal,
        ticks, np.zeros(n, np.int64), source_frame_rule="ceil", label_valid=valid)]
    require(bool(accepted) and accepted[0] == 30, "Need real warmup history and a complete30-action teacher future")
    bycurrent = {int(c): s.tolist() for c, s in zip(current, selected)}
    return accepted, {str(c): bycurrent[c] for c in accepted}


def episode_values(events, episode, offset, state_names, hand_names):
    """No warmup command is invented; unavailable targets remain NaN."""
    n = len(events); values = {"observation.state": [], "observation.joint_velocities": [],
        "observation.base_pose": [], "observation.base_vel": [], "observation.object_poses": [],
        "action": [], "action.body_token_v1_1": [], "action.neck": [], "action_mask": []}
    require(n >= 60, "Too few real observations/actions")
    for index, event in enumerate(events):
        names = event["joint_names"]
        require(len(names) == len(set(names)) == 43 and set(names) == set(state_names), "Measured state joint identities differ")
        permutation = [names.index(name) for name in state_names]
        values["observation.state"].append(vector(event["state43"], 43, "state")[permutation])
        values["observation.joint_velocities"].append(vector(event["physical_joint_qvel43"], 43, "joint velocities")[permutation])
        values["observation.base_pose"].append(vector(event["root_pose7"], 7, "root pose"))
        values["observation.base_vel"].append(vector(event["root_qvel6"], 6, "root velocity"))
        values["observation.object_poses"].append(np.r_[vector(event["objects"]["container"]["pose7"], 7, "container"),
                                                         vector(event["objects"]["target"]["pose7"], 7, "target")])
        valid = event["action_valid"]
        require(type(valid) is bool and valid is (index >= 30), "Only the first30 real warmup rows may lack labels")
        if valid:
            action = event["action"]; supplied = action["hand_joint_names"]
            require(len(supplied) == len(set(supplied)) == 14 and set(supplied) == set(hand_names), "Named hand labels differ")
            values["action"].append(vector(action["hand14"], 14, "hand labels")[[supplied.index(name) for name in hand_names]])
            values["action.body_token_v1_1"].append(vector(action["body_token64"], 64, "executed token"))
            values["action.neck"].append([0., 0.]); values["action_mask"].append([True] * 78 + [False] * 2)
        else:
            require(event.get("action") is None, "Warmup must retain absent action labels")
            values["action"].append([float("nan")] * 14); values["action.body_token_v1_1"].append([float("nan")] * 64)
            values["action.neck"].append([float("nan")] * 2); values["action_mask"].append([False] * 80)
    for key in values:
        values[key] = np.asarray(values[key], dtype=bool if key == "action_mask" else np.float64 if key.startswith("observation.") and key != "observation.state" else np.float32)
    values.update(timestamp=np.arange(n, dtype=np.float32) / 30, frame_index=np.arange(n, dtype=np.int64),
        episode_index=np.full(n, episode, np.int64), index=np.arange(offset, offset + n, dtype=np.int64),
        task_index=np.zeros(n, np.int64), action_valid=np.asarray([e["action_valid"] for e in events], bool),
        source_frame_index=np.asarray([e["physical_tick50"] for e in events], np.int64),
        actual_physical_tick50=np.asarray([e["physical_tick50"] for e in events], np.int64),
        source_command_index=np.asarray([e["source_command_index"] if e["source_command_index"] is not None else -1 for e in events], np.int64),
        source_elapsed_timestamp=np.asarray([e["actual_elapsed_s"] for e in events], np.float64),
        source_timestamp=np.asarray([e["sim_time_s"] for e in events], np.float64),
        phase=np.asarray([e["phase"] for e in events]),
        **{"next.done": np.arange(n) == n - 1})
    return values


def encode_pngs(capture, events, output):
    """Encode raw RTX PNGs at30fps, then compare every decoded RGB byte."""
    import av
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    before = hashlib.sha256(); count = 0; shape = None
    with av.open(str(output), "w") as destination:
        stream = destination.add_stream("libx264rgb", rate=30)
        stream.options = {"crf": "0", "preset": "fast"}; stream.pix_fmt = "rgb24"; stream.time_base = Fraction(1, 30)
        for index, event in enumerate(events):
            rgb = event["rgb"]; path = owned(capture, rgb["path"])
            require(sha(path) == rgb["file_sha256"], "Accepted PNG bytes changed")
            with Image.open(path) as image:
                require(image.mode == "RGB" and image.size == (rgb["width"], rgb["height"]), "Raw PNG shape/mode differs")
                pixels = np.asarray(image)
            require(hashlib.sha256(pixels.tobytes()).hexdigest() == rgb["pixel_sha256"], "Accepted raw pixels changed")
            if shape is None:
                shape = list(pixels.shape); stream.width, stream.height = shape[1], shape[0]
            require(list(pixels.shape) == shape, "Mid-episode RGB geometry changed")
            before.update(pixels.tobytes()); frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = index; frame.time_base = Fraction(1, 30)
            for packet in stream.encode(frame): destination.mux(packet)
            count += 1
        for packet in stream.encode(): destination.mux(packet)
    after = hashlib.sha256(); actual = 0
    with av.open(str(output)) as encoded:
        require(encoded.streams.video[0].average_rate == 30, "Encoded video is not30fps")
        for frame in encoded.decode(video=0):
            require(frame.pts is not None and abs(float(frame.pts * frame.time_base) - actual / 30) < 1e-7, "Decoded video PTS differs")
            after.update(frame.to_ndarray(format="rgb24").tobytes()); actual += 1
    require(actual == count == len(events) and before.digest() == after.digest(), "Lossless raw RTX pixel verification failed")
    return {"frames": count, "shape": shape, "fps": 30, "codec": "libx264rgb", "crf": 0,
            "raw_rgb_sha256": before.hexdigest(), "decoded_rgb_sha256": after.hexdigest(), "pixels_exact": True,
            "sha256": sha(output)}


def features(state_names, hand_names, shape):
    def feature(width, names=None, dtype="float32"):
        return {"dtype": dtype, "shape": [width], "names": names}
    result = {"observation.state": feature(43, state_names), "observation.joint_velocities": feature(43, state_names, "float64"),
        "observation.base_pose": feature(7, dtype="float64"), "observation.base_vel": feature(6, dtype="float64"),
        "observation.object_poses": feature(14, [f"{r}.{k}" for r in ("container", "target") for k in ("pos_x", "pos_y", "pos_z", "quat_w", "quat_x", "quat_y", "quat_z")], "float64"),
        "action": feature(14, hand_names), "action.body_token_v1_1": feature(64), "action.neck": feature(2, ["neck_yaw", "neck_pitch"]),
        "action_mask": feature(80, dtype="bool"), "action_valid": feature(1, dtype="bool"), "phase": feature(1, dtype="string"),
        "task_description": feature(1, dtype="string"), "next.done": feature(1, dtype="bool"), "timestamp": feature(1)}
    for key in ("frame_index", "episode_index", "index", "task_index", "source_episode_index", "source_frame_index", "actual_physical_tick50", "source_command_index"):
        result[key] = feature(1, dtype="int64")
    for key in ("source_elapsed_timestamp", "source_timestamp"):
        result[key] = feature(1, dtype="float64")
    video = {"video.fps": 30., "video.codec": "h264", "video.pix_fmt": "gbrp", "video.is_depth_map": False, "has_audio": False}
    result[VIDEO_KEY] = {"dtype": "video", "shape": shape, "names": ["height", "width", "channel"], "info": video, "video_info": deepcopy(video)}
    return result


def arrow_table(values):
    import pyarrow as pa
    columns = {}
    for key, value in values.items():
        if not isinstance(value, np.ndarray):
            columns[key] = pa.array(value)
            continue
        dtype = pa.from_numpy_dtype(value.dtype) if value.dtype.kind not in "US" else pa.string()
        if value.ndim == 2:
            dtype = pa.list_(dtype)
        columns[key] = pa.array(value.tolist(), type=dtype)
    return pa.table(columns)


def check_parquet_roundtrip(original, reread):
    require(original.schema == reread.schema and original.num_rows == reread.num_rows,
            "Packed Parquet schema/row count changed")
    for key in original.column_names:
        a, b = (np.asarray(t[key].to_pylist()) for t in (original, reread))
        same = np.array_equal(a, b, equal_nan=True) if a.dtype.kind in "fc" else np.array_equal(a, b)
        require(same, f"Packed Parquet roundtrip changed {key}")


def verify_lerobot_reader(pack_root, episodes, raw_events):
    """Query real LeRobot future chunks/current video on CPU, without DINO/model."""
    import pyarrow.parquet as pq
    source_path = str(REPO / "VLMExpert/src")
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    from psi.data.lerobot.compat import LeRobotDataset
    delta = {VIDEO_KEY: [0.], "observation.state": [0.]}
    delta.update({key: [i / 30 for i in range(30)]
                  for key in ("action", "action.body_token_v1_1", "action_mask")})
    dataset = LeRobotDataset("train", root=Path(pack_root), delta_timestamps=delta, video_backend="pyav")
    checked = []; offset = 0
    for episode, events in zip(episodes, raw_events):
        table = pq.read_table(Path(pack_root) / f"data/chunk-000/episode_{episode['episode_index']:06d}.parquet")
        for current in sorted({episode["eligible_current_rows"][0], episode["eligible_current_rows"][-1]}):
            sample = dataset[offset + current]
            for key, width in (("action", 14), ("action.body_token_v1_1", 64), ("action_mask", 80)):
                actual = np.asarray(sample[key]).reshape(30, width)
                expected = np.asarray(table[key].slice(current, 30).to_pylist())
                require(np.isfinite(actual).all() and np.array_equal(actual, expected),
                        f"Real LeRobot future query changed/padded {key}")
            state = np.asarray(sample["observation.state"]).reshape(-1, 43)
            require(state.shape == (1, 43) and np.array_equal(state[0], table["observation.state"][current].as_py()),
                    "Real LeRobot current state differs")
            pixels = np.asarray(sample[VIDEO_KEY])
            if pixels.ndim == 4 and pixels.shape[0] == 1:
                pixels = pixels[0]
            require(pixels.ndim == 3 and pixels.shape[0] == 3 and np.isfinite(pixels).all(), "Real LeRobot video shape differs")
            rgb = np.rint(pixels.transpose(1, 2, 0) * 255).astype(np.uint8)
            require(hashlib.sha256(rgb.tobytes()).hexdigest() == events[current]["rgb"]["pixel_sha256"],
                    "Real LeRobot current RGB differs from actual accepted RTX pixels")
            checked.append({"episode_index": episode["episode_index"], "current_row": current,
                            "future_rows": 30, "current_raw_RTX_pixels_exact": True})
        offset += len(events)
    return {"status": "actual_cpu_LeRobot_numeric_video_queries_passed", "queries": checked,
            "scope": "Actual current RGB/state and complete future30 from LeRobot on CPU. Production MotionHistoryIndex history selector checked separately; real DINO extraction/joint-loader preflight remain pending.",
            "DINO_features_or_model_loaded": False, "GPU_used": False}


def pack(queue_root, output, preparation, run_config, statistics, state_info, *, limit=None, capture_sources=None):
    import pyarrow.parquet as pq
    identity, inherited_stats = checkpoint_contract(preparation, run_config, statistics, state_info)
    require((queue_root is None) != (capture_sources is None), "Select one queue or an explicit multi-queue manifest")
    if capture_sources is None:
        items, plan, queue_binding = collect_admitted(queue_root, limit)
        queue_roots = [Path(queue_root).resolve(strict=True)]
    else:
        items, plan, queue_binding = collect_admitted_sources(capture_sources, limit)
        queue_roots = [Path(row["queue_path"]).parent for row in queue_binding["sources"]]
    output = Path(output).absolute()
    require(not output.exists() and not output.is_symlink() and not output.resolve().is_relative_to(REPO)
            and not any(output.resolve().is_relative_to(root) for root in queue_roots), "Use a fresh external output outside the capture queue")
    staging = output.parent / f".{output.name}.partial-{uuid.uuid4().hex}"; staging.mkdir(parents=True)
    pack_root = staging / "train"; episodes = []; receipts = []; offset = 0; shape = None; task_ids = {}
    try:
        queue_snapshot = queue_binding.pop("_snapshot_utf8", None)
        if queue_snapshot is not None:
            destination = staging / "accepted_evidence/queue_snapshot.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(queue_snapshot.encode("utf-8"))
            require(sha(destination) == queue_binding["queue_sha256"], "Frozen collector snapshot changed")
            queue_binding["frozen_snapshot"] = "accepted_evidence/queue_snapshot.json"
        for index, snapshot in enumerate(queue_binding.pop("_snapshots_utf8", [])):
            relative = f"accepted_evidence/queue-{index:03d}-snapshot.json"
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(snapshot.encode("utf-8"))
            require(sha(destination) == queue_binding["sources"][index]["queue_sha256"], "Frozen queue snapshot changed")
            queue_binding["sources"][index]["frozen_snapshot"] = relative
        for episode, item in enumerate(items):
            events = item["events"]; candidate = item["candidate"]; manifest = item["manifest"]
            eligible, history = eligible_currents(events)
            values = episode_values(events, episode, offset, identity["state_joint_names"], identity["hand_joint_names"])
            instruction = manifest.get("instruction", item["result"].get("instruction"))
            require(isinstance(instruction, str) and instruction.strip(), "Actual task instruction must be recorded")
            task_id = task_ids.setdefault(instruction, len(task_ids))
            values["task_index"] = np.full(len(events), task_id, np.int64)
            values["task_description"] = [instruction] * len(events)
            values["source_episode_index"] = np.full(len(events), candidate["ancestor"]["source_episode_index"], np.int64)
            table = arrow_table(values)
            parquet = pack_root / f"data/chunk-000/episode_{episode:06d}.parquet"; parquet.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, parquet, compression="zstd")
            check_parquet_roundtrip(table, pq.read_table(parquet))
            video_path = pack_root / f"videos/chunk-000/{VIDEO_KEY}/episode_{episode:06d}.mp4"
            video = encode_pngs(item["capture"], events, video_path)
            if shape is None: shape = video["shape"]
            require(shape == video["shape"], "One dataset must retain a consistent camera geometry")
            provenance = staging / "accepted_evidence" / candidate["candidate_id"]; provenance.mkdir(parents=True)
            for name, original in (("admission.json", item["audit_path"]), ("capture.json", item["capture"] / "capture.json"),
                                   ("events.jsonl", item["capture"] / manifest["events_path"])):
                shutil.copyfile(original, provenance / name)
            episodes.append({"episode_index": episode, "tasks": [instruction], "length": len(events),
                "candidate_id": candidate["candidate_id"], "source_episode_index": candidate["ancestor"]["source_episode_index"],
                "source_dataset_key": candidate["ancestor"]["source_dataset_key"], "source_identity": candidate["ancestor"],
                "lineage_group_sources": [{"source_dataset_key": candidate["ancestor"]["source_dataset_key"], "source_episode_index": candidate["ancestor"]["source_episode_index"]}],
                "lineage_complete": True, "split": "train", "split_restriction": "train_only",
                "diversity_receipt": item["audit"].get("diversity_receipt"),
                "teacher_collection_warmup": item["audit"]["teacher_collection_warmup"],
                "teacher_reference": item["audit"].get("teacher_reference"),
                "startup_history_padding_audit": item["audit"].get("startup_history_padding_audit"),
                "history_only_prefix_rows": 30, "eligible_current_rows": eligible,
                "physics_validation": {"gate": "handover-physics-v2", "verified_handover_success": True}})
            receipts.append({"episode_index": episode, "candidate_id": candidate["candidate_id"],
                "admission_path": str(item["audit_path"]), "admission_sha256": sha(item["audit_path"]),
                "capture_content_sha256": item["audit"]["capture_content_sha256"], "source_sha256": manifest["source_sha256"],
                "teacher_collection_warmup": item["audit"]["teacher_collection_warmup"],
                "teacher_reference": item["audit"].get("teacher_reference"),
                "plan_sha256": manifest["plan_sha256"], "candidate_spec_sha256": manifest["candidate_spec_sha256"],
                "parquet_sha256": sha(parquet), "video": video, "eligible_current_rows": eligible,
                "history_rows": history, "frames": len(events)})
            offset += len(events)
        meta = pack_root / "meta"; meta.mkdir(parents=True)
        info = {"codebase_version": "v2.0", "robot_type": "g1_sonic", "fps": 30, "total_episodes": len(episodes),
            "total_frames": offset, "total_tasks": len(set(e["tasks"][0] for e in episodes)), "total_videos": len(episodes),
            "total_chunks": 1, "chunks_size": 1000, "splits": {"train": f"0:{len(episodes)}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": features(identity["state_joint_names"], identity["hand_joint_names"], shape),
            "warmup_label_policy": "NaN unavailable labels, action_valid=false and all80 action_mask=false; valid real RGB/state remain history",
            "normalization": {"refitted": False, "inherited_statistics_sha256": identity["statistics"]["sha256"]}}
        write(meta / "info.json", info); write(meta / "stats.json", inherited_stats); write(meta / "stats_psi0.json", inherited_stats)
        write(meta / "modality.json", {"state": {"joint_positions": {"start": 0, "end": 43}},
            "action": {"joint_positions": {"start": 0, "end": 14}}, "video": {"egocentric": {"original_key": VIDEO_KEY}}})
        (meta / "episodes.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in episodes))
        (meta / "tasks.jsonl").write_text("".join(json.dumps({"task_index": index, "task": task}, ensure_ascii=False) + "\n"
                                                for task, index in task_ids.items()))
        shutil.copyfile(statistics, staging / "source_dataset_statistics.json")
        shutil.copyfile(preparation, staging / "source_initialization_receipt.json")
        shutil.copyfile(run_config, staging / "source_run_config.json")
        shutil.copyfile(state_info, staging / "source_state_info.json")
        reader_check = verify_lerobot_reader(pack_root, episodes, [item["events"] for item in items])
        counts = Counter(e["source_episode_index"] for e in episodes)
        complete = len(episodes) == 100 and len(counts) == 24 and max(counts.values()) <= 5
        report = {"contract": CONTRACT, "status": "cpu_lossless_pack_complete_train_target_incomplete" if not complete else "cpu_lossless_pack_complete_train_target100",
            "created_utc": datetime.now(timezone.utc).isoformat(), "queue_binding": queue_binding, "checkpoint_contract": identity,
            "train_episodes": len(episodes), "target_train_episodes": 100, "total_frames": offset, "accepted_per_ancestor": dict(counts),
            "covered_ancestors": sorted(counts), "train_target_satisfied": complete, "all24_ancestors_required": True,
            "maximum_per_ancestor": 5, "existing_VAL_sources_excluded": sorted(HELDOUT), "train_payload_packed": True,
            "final_ready": False, "final_ready_blockers": ["independent_TEST_lineage_pending", "real_DINO_cache_pending", "joint_training_reader_with_real_DINO_preflight_pending"],
            "cpu_reader_validation": reader_check,
            "episodes": receipts, "normalization_refitted": False, "DINO_cache_generated": False, "training_started": False,
            "diversity_scope": "Only actual independently audited physical XY/yaw reset perturbations are claimed; material/generation proposal seeds are not evidence of applied visual augmentation",
            "boundary_policy": {"prefix_rows": 30, "warmup_actions": "unavailable NaN labels, never executed token claims",
                "first_eligible_current_row": 30, "history_offsets": [-30, -24, -18, -12, -6, 0],
                "future_action_rows": 30, "full_real_teacher_future_required": True, "tail_padding_permitted": False},
            "clock_policy": "nominal timestamp=row/30; source_timestamp and source_elapsed_timestamp preserve measured simulator50Hz ceil ticks; actual_physical_tick50 is the physical tick including warmup; parquet source_frame_index keeps the existing trainer ceil50Hz contract. DINO frames.npz source_frame_index must instead be the contiguous decoded30Hz frame ordinal produced by extract_unifolm_dino.py, never this physical tick.",
            "source_sha256": sha(__file__)}
        report["payload_sha256"] = {str(p.relative_to(staging)): sha(p) for p in sorted(staging.rglob("*")) if p.is_file()}
        write(staging / "PACK_READY.json", report); staging.rename(output)
        return report
    except Exception as error:
        write(staging / "FAILED.json", {"error": str(error), "output_requested": str(output), "final_ready": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--queue-root", type=Path)
    selection.add_argument("--capture-sources", type=Path, help="Explicit selection with each queue's frozen executable auditor")
    for key in ("output", "preparation", "run-config", "statistics", "state-info"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--limit", type=int, help="Package an admitted prefix as an explicitly incomplete pilot")
    args = parser.parse_args()
    report = pack(args.queue_root, args.output, args.preparation, args.run_config, args.statistics, args.state_info,
                  limit=args.limit, capture_sources=args.capture_sources)
    print(json.dumps({key: report[key] for key in ("status", "train_episodes", "train_target_satisfied", "final_ready")}))


if __name__ == "__main__": main()
