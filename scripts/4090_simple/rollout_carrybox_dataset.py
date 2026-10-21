"""Lossless command-event capture and CPU packing of new physical CarryBox runs.

This is a new commanded-rollout dataset, never a repair of measured-mode0 data.
Packing requires full production30Hz command / 50Hz decoder execution and
independently reconstructs the physical tabletop-release predicate. It does
not unlock training; the launcher remains blocked pending a separate review.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from prepare_carrybox_v11 import (TRAIN_STATE_NAMES, TRAIN_HAND_NAMES, VIDEO_KEY, ENCODER_SHA,
                                  output_features, vector_stats, sha256)

REPO = Path(__file__).resolve().parents[2]
CONTRACT = "cvpr-simple-commanded-rollout-v1"
OBSERVATION_PHASE = "post_step_mj_forward_same_qpos"
PILOT_SPLITS = {"train": (0, 1, 2), "val": (8,), "test": (4,)}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def owned_file(root, relative):
    path = (root / relative).resolve(strict=True)
    if not path.is_file() or not path.is_relative_to(root.resolve()):
        raise ValueError("Capture artifact must remain inside its physical run directory")
    return path


def raw_rgb_sha(rgb):
    value = np.asarray(rgb)
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3:
        raise ValueError("Require actual RGB uint8 HWC; no casting or preview-video extraction")
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def native_tick(event_index):
    if isinstance(event_index, bool) or not isinstance(event_index, (int, np.integer)) or event_index < 0:
        raise ValueError("Command event index must be a nonnegative integer")
    return (int(event_index) * 50 + 29) // 30


def validate_events(events):
    """Validate causal ceil-selected ticks and exact six-event history spacing."""
    if len(events) < 60:
        raise ValueError("A commanded episode must support real one-second history and thirty-action future")
    first_time = float(events[0]["sim_time_s"])
    if not np.isfinite(first_time):
        raise ValueError("Command event clocks must be finite")
    source_episode = int(events[0]["source_episode"])
    for index, row in enumerate(events):
        if not np.isfinite([row["sim_time_s"], row["source_time_s"]]).all():
            raise ValueError("Command event clocks must be finite")
        if row.get("event_index") != index or row.get("command_hz") != 30:
            raise ValueError("Require every actual30Hz command event, once, in chronological order")
        tick = native_tick(index)
        if row.get("native_frame_index") != tick or row.get("expected_native_frame_index") != tick:
            raise ValueError("Command event must use ceil(event_index*50/30), never floor subsampling")
        if (int(row["source_episode"]) != source_episode or int(row["source_ref_frame"]) != tick
                or abs(float(row["source_time_s"]) - tick / 50) > 1e-7
                or abs(float(row["sim_time_s"]) - first_time - tick / 50) > 1e-7):
            raise ValueError("Actual command/state/RGB source clock or episode identity differs")
        for name, width in (("state43", 43), ("executed_token64", 64), ("hand_targets14", 14), ("base_pose7", 7)):
            value = np.asarray(row[name])
            if value.shape != (width,) or not np.isfinite(value).all():
                raise ValueError(f"Invalid executed-command event feature: {name}")
        if not np.isclose(np.linalg.norm(np.asarray(row["base_pose7"])[3:]), 1, atol=1e-5):
            raise ValueError("Actual base pose must contain a unit wxyz quaternion")
        if index >= 6 and abs(float(row["sim_time_s"]) - float(events[index-6]["sim_time_s"]) - .2) > 1e-7:
            raise ValueError("Six command events must span an actual 0.2 simulator seconds")
    return source_episode


class LosslessCommandRecorder:
    """Optional native50 loop callback; closing only seals raw data, never accepts it."""
    def __init__(self, output, *, instruction, metadata=None):
        self.output = Path(output)
        if not self.output.is_absolute() or self.output.exists() or self.output.resolve().is_relative_to(REPO) or not instruction:
            raise ValueError("Use a fresh capture directory outside the repository and the real task instruction")
        (self.output / "rgb").mkdir(parents=True)
        self.instruction, self.metadata, self.events, self.closed = instruction, metadata or {}, [], False
        self.source_sha = sha256(__file__)
        shutil.copy2(__file__, self.output / "collector_source.py")

    def record(self, *, event_index, sim_time_s, rgb_uint8HWC, state43, executed_token64, hand_targets14,
               base_pose7, source_ref_frame, source_episode, source_time_s, box_pose7=None):
        if self.closed or event_index != len(self.events):
            raise ValueError("Capture must receive each new command once before close")
        expected = native_tick(event_index)
        if (abs(float(source_time_s) - expected / 50) > 1e-7
                or int(source_ref_frame) != expected):
            raise ValueError("Record only the real ceil-selected30Hz command update tick")
        rgb = np.asarray(rgb_uint8HWC)
        digest = raw_rgb_sha(rgb)
        path = Path("rgb") / f"event_{event_index:06d}.png"
        Image.fromarray(rgb).save(self.output / path)
        row = {"event_index": int(event_index), "command_hz": 30, "sim_time_s": float(sim_time_s),
               "source_time_s": float(source_time_s), "native_frame_index": native_tick(event_index),
               "expected_native_frame_index": native_tick(event_index), "source_ref_frame": int(source_ref_frame),
               "source_episode": int(source_episode), "state43": np.asarray(state43).tolist(),
               "executed_token64": np.asarray(executed_token64).tolist(), "hand_targets14": np.asarray(hand_targets14).tolist(),
               "base_pose7": np.asarray(base_pose7).tolist(), "rgb_sha256": digest, "rgb_shape": list(rgb.shape),
               "rgb_dtype": "uint8", "rgb_lossless_png": str(path), "rgb_png_file_sha256": sha256(self.output / path)}
        if box_pose7 is not None:
            row["box_pose7"] = np.asarray(box_pose7).tolist()
        self.events.append(row)
        with (self.output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")

    def close(self):
        if self.closed:
            return
        self.closed = True
        arrays = {key: np.asarray([row[key] for row in self.events]) for key in
                  ("event_index", "sim_time_s", "source_time_s", "native_frame_index", "source_ref_frame", "source_episode",
                   "state43", "executed_token64", "hand_targets14", "base_pose7")}
        np.savez_compressed(self.output / "events.npz", **arrays)
        write_json(self.output / "capture.json", {"contract": CONTRACT, "state": "capture_closed",
                   "instruction": self.instruction, "metadata": self.metadata, "event_count": len(self.events),
                   "state43_joint_names": list(TRAIN_STATE_NAMES), "hand14_joint_names": list(TRAIN_HAND_NAMES),
                   "events_jsonl_sha256": sha256(self.output / "events.jsonl") if self.events else None,
                   "events_npz_sha256": sha256(self.output / "events.npz"), "collector_sha256": self.source_sha,
                   "acceptance": "not_evaluated; raw capture cannot authorize packing or training"})


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _table_top(model, data):
    """Highest actual table-geometry AABB; fixed scene0's tabletop is axis aligned."""
    tops = []
    for geom in range(model.ngeom):
        body_name = model.body(int(model.geom_bodyid[geom])).name
        if "table" in body_name:
            center, half_size = model.geom_aabb[geom, :3], model.geom_aabb[geom, 3:]
            rotation = data.geom_xmat[geom].reshape(3, 3)
            tops.append(float((data.geom_xpos[geom] + rotation @ center)[2] + np.abs(rotation[2]) @ half_size))
    if not tops:
        raise ValueError("Exact replay physics model has no table geometry")
    return max(tops)


def reconstruct_placement(model_path, trajectory):
    """CPU-only geometric predicate at recorded actual states, independent of pass booleans."""
    import mujoco
    times = np.asarray(trajectory["sim_time_s"], dtype=np.float64)
    if (times.ndim != 1 or not len(times) or not np.isfinite(times).all()
            or not np.allclose(np.diff(times), .02, rtol=0, atol=1e-7)):
        raise ValueError("Physical trajectory requires finite, increasing genuine50Hz simulator times")
    model = mujoco.MjModel.from_binary_path(str(model_path))
    data = mujoco.MjData(model)
    if tuple(trajectory["state43_joint_names"]) != TRAIN_STATE_NAMES:
        raise ValueError("Recorded trajectory must preserve the named training joint layout")
    cube = model.body("cube").id
    pelvis = model.body("pelvis").id
    hold, best, previous, last_time, rows = 0., 0., False, None, []
    for index, now in enumerate(times):
        state, base, box = (trajectory[key][index] for key in ("actual_state43", "actual_base_pose", "actual_box_pose"))
        if any(not np.isfinite(value).all() for value in (state, base, box)):
            raise ValueError("Replay contains nonfinite actual state/pose")
        data.qpos[:] = model.qpos0
        data.qpos[:7] = base
        for name, value in zip(TRAIN_STATE_NAMES, state):
            data.joint(name).qpos[0] = value
        data.joint("cube_joint").qpos[:] = box
        mujoco.mj_forward(model, data)
        top = _table_top(model, data)
        hands, on_top = False, False
        for contact in data.contact[:data.ncon]:
            a, b = int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2])
            if cube not in (a, b):
                continue
            other = model.body(b if a == cube else a).name
            hands |= "hand" in other or "wrist_yaw" in other
            on_top |= "table" in other and abs(float(contact.pos[2]) - top) <= .05
        tilt = float(np.degrees(np.arccos(np.clip(data.xmat[pelvis].reshape(3, 3)[2, 2], -1, 1))))
        if data.xpos[pelvis, 2] < .35 or tilt > 75:
            raise ValueError("Independent physics reconstruction found a fall")
        placed = not hands and on_top and data.xpos[cube, 2] >= top
        delta = 0. if last_time is None else float(now - last_time)
        if delta < 0 or (last_time is not None and abs(delta - .02) > 1e-7):
            raise ValueError("Physical trajectory must preserve every genuine50Hz sample")
        hold = hold + delta if placed and previous else 0.
        best, previous, last_time = max(best, hold), placed, float(now)
        rows.append({"sim_time_s": float(now), "placed": bool(placed), "hold_s": hold, "table_top_z": top})
    if not np.isfinite(hold) or hold < .9 - 1e-7:
        raise ValueError(f"Full30Hz-command replay does not end with a genuine tabletop release held0.9s: {hold:.3f}s")
    return {"metric": "true_tabletop_continuous_simtime_v1", "final_reconstructed_hold_s": hold,
            "best_reconstructed_hold_s": best, "frames_reconstructed": len(rows), "table_top_z": rows[-1]["table_top_z"],
            "scope": "Exact saved MJB geometric contacts at recorded physical states; force magnitudes are not reconstructed"}


def audit_run(run_root, *, task_adapter=None):
    """Reject missing/failed/incomplete runs and compare execution evidence, not status alone."""
    root = run_root.resolve(strict=True)
    result = json.loads(owned_file(root, "result.json").read_text())
    if ((task_adapter is None and result.get("status") != "source_end") or result.get("command_hz") != 30
            or result.get("fell") is not False or result.get("finite_states") is not True):
        raise ValueError("Only complete, finite, non-fallen production30Hz full-source runs may be packed")
    if task_adapter is not None:
        task_adapter.validate_result(root, result)
    if result.get("observation_phase") != OBSERVATION_PHASE:
        raise ValueError("Require synchronized post-step mj_forward actual state and newly rendered same-qpos RGB")
    if result.get("teacher", {}).get("token_mode") != "feedback-heading-v11":
        raise ValueError("Require executed feedback-heading-v11 commands; measured-mode0/legacy data are ineligible")
    identity = result["controller_identity"]
    from simple.baselines.sonic_v1_1 import OFFICIAL_DECODER_SHA256
    if identity.get("decoder_sha256") != OFFICIAL_DECODER_SHA256:
        raise ValueError("Controller must be the fixed official1.1 decoder")
    model_path = Path(result["physics_model"]["path"]).resolve(strict=True)
    if not model_path.is_relative_to(root) or sha256(model_path) != result["physics_model"]["sha256"]:
        raise ValueError("Exact physical MJB bytes differ from the recorded run")
    trajectory_path, trace_path, event_path = (owned_file(root, name) for name in
                                             ("trajectory.npz", "trace.jsonl", "command_events.jsonl"))
    events, trace = read_jsonl(event_path), read_jsonl(trace_path)
    source_episode = validate_events(events)
    if source_episode != result["teacher"]["source_episode_index"]:
        raise ValueError("Command/source episode identity differs")
    active = [row for row in trace if row["phase"] == "oracle_teacher_token"]
    if any(not np.isfinite([row[key] for key in ("source_time_s", "sim_time_s", "next_sim_time_s")]).all()
           for row in active):
        raise ValueError("Physical execution clocks must be finite")
    if not active or any(abs(row["source_time_s"] - index / 50) > 1e-7 for index, row in enumerate(active)):
        raise ValueError("Physical trace skipped real native50Hz ticks")
    for row in active:
        if (abs(row["next_sim_time_s"] - row["sim_time_s"] - .02) > 1e-7
                or row["decoder_calls_after_step"] - row["decoder_calls_before_step"] != 1
                or row.get("command_event_index") != int(np.floor(row["source_time_s"] * 30 + 1e-7))):
            raise ValueError("Command hold/decoder/physical clock evidence differs from production rates")
        event = events[row["command_event_index"]]
        if (not np.array_equal(row["applied_body_token"], event["executed_token64"])
                or not np.array_equal([row["applied_hand_targets"][name] for name in TRAIN_HAND_NAMES], event["hand_targets14"])):
            raise ValueError("Archived command is not the command actually held/executed by the decoder")
    if len(events) != int(np.floor(active[-1]["source_time_s"] * 30 + 1e-7)) + 1:
        raise ValueError("Command capture does not cover the entire genuine source replay")
    trajectory = np.load(trajectory_path, allow_pickle=False)
    rows_by_time = {round(float(value), 8): index for index, value in enumerate(trajectory["sim_time_s"])}
    capture_root, captured = root, events
    if result.get("command_capture_root"):
        capture_root = Path(result["command_capture_root"]).resolve(strict=True)
        manifest = json.loads(owned_file(capture_root, "capture.json").read_text())
        if (manifest.get("contract") != CONTRACT or manifest.get("state") != "capture_closed"
                or manifest.get("event_count") != len(events)
                or Path(manifest["metadata"]["run_output"]).resolve() != root
                or manifest["metadata"].get("observation_phase") != OBSERVATION_PHASE
                or manifest.get("collector_sha256") != sha256(owned_file(capture_root, "collector_source.py"))
                or manifest.get("events_jsonl_sha256") != sha256(owned_file(capture_root, "events.jsonl"))
                or manifest.get("events_npz_sha256") != sha256(owned_file(capture_root, "events.npz"))):
            raise ValueError("External lossless capture is not bound to this completed physical run")
        captured = read_jsonl(owned_file(capture_root, "events.jsonl"))
        validate_events(captured)
        saved_numeric = np.load(owned_file(capture_root, "events.npz"), allow_pickle=False)
        for key in saved_numeric.files:
            if not np.array_equal(saved_numeric[key], np.asarray([row[key] for row in captured])):
                raise ValueError("Raw numeric capture differs between JSONL and NPZ")
    pngs, hashes = [], []
    for ordinal, event in enumerate(events):
        if event.get("observation_phase") != OBSERVATION_PHASE:
            raise ValueError("Command RGB/state must use the accepted same-qpos observation phase")
        index = rows_by_time.get(round(event["sim_time_s"], 8))
        if index is None or not all(np.array_equal(trajectory[key][index], event[field]) for key, field in
                                   (("actual_state43", "state43"), ("actual_base_pose", "base_pose7"), ("actual_box_pose", "box_pose7"))):
            raise ValueError("Actual RGB/state/command event differs from the independently saved physical trajectory")
        runtime = event.get("runtime_encoder", {})
        if (runtime.get("mode") != "feedback-heading-v11" or runtime.get("changed_input_slice") != [584, 644]
                or runtime.get("source_frame_index") != event["source_ref_frame"]
                or not runtime.get("encoder_input_sha256")):
            raise ValueError("Event lacks a bound actual-heading encoder input")
        raw_event = captured[ordinal]
        for key in ("event_index", "sim_time_s", "source_time_s", "source_ref_frame", "source_episode", "rgb_sha256",
                    "state43", "executed_token64", "hand_targets14", "base_pose7", "box_pose7"):
            if raw_event.get(key) != event.get(key):
                raise ValueError("Lossless recorder event differs from the actual execution command trace")
        path = owned_file(capture_root, raw_event["rgb_lossless_png"])
        with Image.open(path) as image:
            rgb = np.asarray(image)
        if (list(rgb.shape) != event["rgb_shape"] or event.get("rgb_dtype") != "uint8"
                or raw_rgb_sha(rgb) != event["rgb_sha256"]):
            raise ValueError("Lossless command RGB differs from the real pre-step observation")
        pngs.append(path)
        hashes.append(sha256(path))
    encoder_reproduction = verify_encoder_events(result, events, active, full_source=task_adapter is None)
    reconstructed = (reconstruct_placement(model_path, trajectory) if task_adapter is None else
                     task_adapter.reconstruct(model_path, trajectory, result, active))
    return events, pngs, {"source_episode": source_episode, "run_root": str(root), "seed": result["seed"],
        "instruction": _instruction(result) if task_adapter is None else task_adapter.instruction,
        "result_sha256": sha256(root / "result.json"),
        "trajectory_sha256": sha256(trajectory_path), "trace_sha256": sha256(trace_path),
        "command_events_sha256": sha256(event_path), "png_file_sha256": hashes,
        "physics_model_sha256": sha256(model_path), "controller_identity": identity,
        "source_parquet_sha256": result["teacher"]["source_parquet_sha256"],
        "source_archive_sha256": result["teacher"]["source_archive_sha256"],
        "encoder_sha256": result["teacher"]["encoder_sha256"], "encoder_reproduction": encoder_reproduction,
        "reconstructed_placement": reconstructed}


def validate_source_extent(native_frames, teacher, active, *, full_source=True):
    """Derive the whole usable prefix from actual source rows, not a run's claim."""
    replay_frames = native_frames - 45
    endpoint = (native_frames - 46) / 50
    if replay_frames < 2 or not active:
        raise ValueError("Actual source has no complete genuine future reference")
    reported_times = [teacher.get("replay_last_timestamp_s", np.nan),
                      teacher.get("comparison_valid_prefix_last_timestamp_s", np.nan)]
    if (not np.isfinite(reported_times).all()
            or teacher.get("source_native_frames") != native_frames
            or teacher.get("replay_frames") != replay_frames
            or teacher.get("excluded_future_tail_frames") != 45
            or any(abs(value - endpoint) > 1e-7 for value in reported_times)
            or not 0 < len(active) <= replay_frames - 1
            or (full_source and len(active) != replay_frames - 1)
            or not np.isfinite(active[-1]["source_time_s"])
            or abs(active[-1]["source_time_s"] - (len(active) - 1) / 50) > 1e-7
            or (full_source and abs(active[-1]["source_time_s"] + .02 - endpoint) > 1e-7)):
        raise ValueError("Replay does not cover the complete valid prefix derived from actual source row count")
    return {"actual_source_rows": native_frames, "valid_reference_rows": replay_frames,
            "physical_active_ticks": len(active), "valid_reference_endpoint_s": endpoint,
            "execution_scope": "full_source" if full_source else "actual_task_terminated_prefix"}


def verify_encoder_events(result, events, active, *, full_source=True):
    """Reproduce NEW executed feedback commands; never substitute them as labels."""
    import onnxruntime as ort
    import pyarrow.parquet as pq
    from replay_carrybox_native50 import feedback_encoder_input
    from prepare_carrybox_v11 import derived_velocity, matrix_wxyz
    source = Path(result["teacher"]["source_parquet"]).resolve(strict=True)
    if sha256(source) != result["teacher"]["source_parquet_sha256"]:
        raise ValueError("Original reference parquet bytes changed")
    table = pq.read_table(source, columns=["observation.state", "observation.base_pose", "timestamp"])
    q = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)[:, :29]
    source_extent = validate_source_extent(table.num_rows, result["teacher"], active, full_source=full_source)
    roots = np.asarray(table["observation.base_pose"].to_pylist(), dtype=np.float64)[:, 3:]
    timestamps = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)
    dq = derived_velocity(q, timestamps)
    resource_root = Path(result["controller_identity"]["provenance"]).resolve(strict=True).parent
    encoder_path = resource_root / "model_encoder.onnx"
    if sha256(encoder_path) != ENCODER_SHA:
        raise ValueError("Executed-command encoder does not match the fixed official1.1 bytes")
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(encoder_path), sess_options=options, providers=["CPUExecutionProvider"])
    for row in events:
        runtime = row["runtime_encoder"]
        rotation = matrix_wxyz(np.asarray(row["base_pose7"])[3:])
        actual_yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        if abs(np.arctan2(np.sin(actual_yaw-runtime["actual_yaw_rad"]), np.cos(actual_yaw-runtime["actual_yaw_rad"]))) > 1e-10:
            raise ValueError("Recorded encoder yaw differs from the actual same-tick physical root")
        value = feedback_encoder_input(q, dq, roots, row["source_ref_frame"], runtime["actual_yaw_rad"])
        if hashlib.sha256(value.tobytes()).hexdigest() != runtime["encoder_input_sha256"]:
            raise ValueError("Executed encoder input cannot be reproduced from its source and real root feedback")
        token = session.run(None, {session.get_inputs()[0].name: value})[0][0]
        if not np.array_equal(token, np.asarray(row["executed_token64"], dtype=np.float32)):
            raise ValueError("Archived actual command does not equal official1.1 encoding of the real feedback input")
    return {"events_reproduced": len(events), "input_sha256_matches": True, "tokens_bitwise_equal": True,
            "source_extent": source_extent,
            "provider": "CPUExecutionProvider", "scope": "Independent check; labels retain the originally executed command values"}


def _instruction(result):
    import pyarrow.parquet as pq
    source = Path(result["teacher"]["source_parquet"]).resolve(strict=True)
    if sha256(source) != result["teacher"]["source_parquet_sha256"] or result["teacher"]["encoder_sha256"] != ENCODER_SHA:
        raise ValueError("Source command reference or official encoder identity changed")
    instructions = set(pq.read_table(source, columns=["task_description"])["task_description"].to_pylist())
    if len(instructions) != 1:
        raise ValueError("Require one genuine task instruction per source episode")
    return instructions.pop()


def encode_video(pngs, expected_hashes, target):
    """Encode original PNG RGB with lossless x264rgb, then verify every decoded pixel."""
    import av
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(pngs[0]) as image:
        width, height = image.size
    with av.open(str(target), "w") as output:
        stream = output.add_stream("libx264rgb", rate=30)
        stream.width, stream.height, stream.pix_fmt = width, height, "rgb24"
        stream.options = {"crf": "0", "preset": "fast"}
        for index, path in enumerate(pngs):
            with Image.open(path) as image:
                rgb = np.asarray(image)
            if rgb.shape != (height, width, 3):
                raise ValueError("Camera resolution changed during one real rollout")
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts, frame.time_base = index, Fraction(1, 30)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    count = 0
    with av.open(str(target)) as video:
        for index, frame in enumerate(video.decode(video=0)):
            if index >= len(expected_hashes) or raw_rgb_sha(frame.to_ndarray(format="rgb24")) != expected_hashes[index]:
                raise ValueError("Packed RGB is not pixel-identical to the actual command-event observation")
            if abs(float(frame.pts * frame.time_base) - index / 30) > 1e-6:
                raise ValueError("Packed video PTS differs from the explicit nominal event clock")
            count += 1
    if count != len(pngs):
        raise ValueError("Lossless video changed the number of actual command observations")
    return {"codec": "libx264rgb", "crf": 0, "frames": count, "width": width, "height": height,
            "all_decoded_rgb_equal_actual_event": True, "sha256": sha256(target)}


def pack_episode(target, episode_id, offset, events, pngs, audit):
    import pyarrow as pa
    import pyarrow.parquet as pq
    count, source = len(events), audit["source_episode"]
    values = {"observation.state": np.asarray([row["state43"] for row in events], dtype=np.float32),
              "action.body_token_v1_1": np.asarray([row["executed_token64"] for row in events], dtype=np.float32),
              "action": np.asarray([row["hand_targets14"] for row in events], dtype=np.float32),
              "observation.base_pose": np.asarray([row["base_pose7"] for row in events], dtype=np.float64),
              "observation.object_poses": np.asarray([row["box_pose7"] for row in events], dtype=np.float64),
              "timestamp": np.arange(count, dtype=np.float32) / 30,
              "source_timestamp": np.asarray([row["sim_time_s"] for row in events], dtype=np.float64),
              "source_elapsed_timestamp": np.asarray([row["source_time_s"] for row in events], dtype=np.float64),
              "source_frame_index": np.asarray([row["native_frame_index"] for row in events], dtype=np.int64),
              "source_reference_frame_index": np.asarray([row["source_ref_frame"] for row in events], dtype=np.int64),
              "frame_index": np.arange(count, dtype=np.int64), "episode_index": np.full(count, episode_id, dtype=np.int64),
              "index": np.arange(offset, offset + count, dtype=np.int64), "task_index": np.zeros(count, dtype=np.int64),
              "source_episode_index": np.full(count, source, dtype=np.int64),
              "task_description": [audit["instruction"]] * count, "next.done": np.arange(count) == count - 1}
    arrays = {key: pa.array(value.tolist(), type=pa.list_(pa.float64() if value.dtype == np.float64 else pa.float32()))
              if isinstance(value, np.ndarray) and value.ndim == 2 else pa.array(value) for key, value in values.items()}
    parquet = target / f"data/chunk-000/episode_{episode_id:06d}.parquet"
    parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(arrays), parquet, compression="zstd")
    video_path = target / f"videos/chunk-000/{VIDEO_KEY}/episode_{episode_id:06d}.mp4"
    video = encode_video(pngs, [row["rgb_sha256"] for row in events], video_path)
    entry = {"episode_index": episode_id, "source_episode_index": source,
             "tasks": [audit["instruction"]], "length": count}
    stats = {"episode_index": episode_id, "stats": {key: vector_stats(values[key]) for key in
                                                   ("observation.state", "action.body_token_v1_1", "action")}}
    return entry, stats, {"episode_index": episode_id, "source_episode_index": source,
                         "parquet_sha256": sha256(parquet), "video": video, "audit": audit}


def pack(runs, output, base_run):
    if not output.is_absolute() or output.exists() or output.resolve().is_relative_to(REPO):
        raise ValueError("Use a fresh absolute dataset directory outside the repository")
    audited = {}
    for run in runs:
        events, pngs, receipt = audit_run(run)
        source = receipt["source_episode"]
        if source in audited:
            raise ValueError("Do not mix repeated variants of one source episode between splits")
        audited[source] = (events, pngs, receipt)
    if set(audited) != {source for sources in PILOT_SPLITS.values() for source in sources}:
        raise ValueError("Pilot requires successful TRAIN0/1/2, locked VAL8 and TEST4; no heldout selection")
    base_run = base_run.resolve(strict=True)
    statistics = base_run / "dataset_statistics.json"
    joint = json.loads((base_run / "LOCAL_RUN_READY.json").read_text())
    receipts = []
    for split, sources in PILOT_SPLITS.items():
        target = output / split
        episodes, episode_stats, offset = [], [], 0
        for episode_id, source in enumerate(sources):
            entry, stats, receipt = pack_episode(target, episode_id, offset, *audited[source])
            episodes.append(entry)
            episode_stats.append(stats)
            offset += entry["length"]
            receipts.append({"split": split, **receipt})
        features = output_features()
        for unused in ("observation.base_vel", "observation.body_joint_velocities"):
            features.pop(unused, None)
        features["source_elapsed_timestamp"] = {"dtype": "float64", "shape": [1], "names": None}
        features["source_reference_frame_index"] = {"dtype": "int64", "shape": [1], "names": None}
        width, height = receipts[-1]["video"]["width"], receipts[-1]["video"]["height"]
        features[VIDEO_KEY]["shape"] = [height, width, 3]
        info = {"codebase_version": "v2.1", "robot_type": "g1_sonic", "fps": 30,
                "total_episodes": len(episodes), "total_frames": offset, "total_tasks": 1,
                "total_videos": len(episodes), "total_chunks": 1, "chunks_size": 1000,
                "splits": {split: f"0:{len(episodes)}"}, "features": features,
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"}
        write_json(target / "meta/info.json", info)
        for filename, rows in (("episodes.jsonl", episodes), ("episodes_stats.jsonl", episode_stats),
                               ("tasks.jsonl", [{"task_index": 0, "task": episodes[0]["tasks"][0]}])):
            (target / "meta" / filename).write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))
    provenance = {"contract": CONTRACT, "splits": {key: list(value) for key, value in PILOT_SPLITS.items()},
                  "raw_event_selection": "ceil(event_index*50/30): actual causal production30Hz command update tick",
                  "timestamp": "event_index/30 derived nominal clock; source_timestamp retains actual simulator time",
                  "actual_history": "six-event offset=0.2 actual simulator seconds; no frame duplication or future observations",
                  "labels": "actual executed official1.1 token64 + hand14 commands, paired with new same-tick real RGB/state",
                  "source_rgb_used": False, "posthoc_measured_future_reencoding": False,
                  "normalization": "reuse original A100 bounds unchanged; episode stats are descriptive metadata only",
                  "source_joint_sha256": joint["joint_sha256"], "source_statistics_sha256": sha256(statistics),
                  "packer_sha256": sha256(__file__), "episodes": receipts,
                  "training_authorized": False, "dino_cache_generated": False}
    shutil.copy2(statistics, output / "source_dataset_statistics.json")
    write_json(output / "COMMANDED_ROLLOUT_READY.json", provenance)
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-run", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.is_absolute():
        parser.error("--output must be absolute and outside the repository")
    result = pack(args.runs, args.output, args.base_run)
    print(json.dumps({"contract": result["contract"], "episodes": len(result["episodes"]),
                      "output": str(args.output), "training_authorized": False}, indent=2))


if __name__ == "__main__":
    main()
