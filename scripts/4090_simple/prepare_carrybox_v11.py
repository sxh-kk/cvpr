"""CPU-only conversion of real SIMPLE CarryBox demonstrations to SONIC v1.1.

New tokens are encoded from native 50 Hz measured trajectories. The legacy
``action.body_token`` is never read as a label. Output RGB, state and hands all
select source frame floor(n*50/30); the last 45 native frames are excluded.
The three output directories are independent LeRobot v2.1 repositories.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform
import shutil
import time
import zipfile

import numpy as np


REPO = Path(__file__).resolve().parents[2]
ENCODER_SHA = "fb97de22819b2057b41459802128d91723d91a25f0ad73e7bfc41a9cf8365bae"
OBSERVATION_SHA = "4a67713b310932e50aca81f19188c8d76013148e98b15c8b5bbea995f12e59f0"
SOURCE_ARCHIVE_SHA = "c8b31cc1ea61a653210278894a33555b735dd33568c4c5ad97ee48ee4871d1af"
SOURCE_REVISION = "e78fb93cc28912a3031a10b8656d32d7f0a2b867"
SONIC_REVISION = "6733128a3d8a523b1418b06bca3cdf61c8b0987f"
CPP_DIR = REPO / "VLMExpert/third_party/GR00T-WholeBodyControl/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref"
CPP_SHA = "75cd7b102abc976c2e694d54d001c93c5a54014c708b232adacce062fb16e1bb"
PARAMETERS_SHA = "b9332adf07c2c9b75c9b1e0756e57c7a1c2a890d8bb0aa53c3f1905fb739b791"
POLICY_TO_BODY = np.array([0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22,
                           4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28])
BODY_NAMES = tuple(
    [f"{s}_{j}_joint" for s in ("left", "right")
     for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
    + [f"waist_{j}_joint" for j in ("yaw", "roll", "pitch")]
    + [f"{s}_{j}_joint" for s in ("left", "right")
       for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                 "wrist_roll", "wrist_pitch", "wrist_yaw")]
)


def hand_names(fingers):
    return tuple(f"{side}_hand_{finger}_{j}_joint" for side in ("left", "right")
                 for finger, count in fingers for j in range(count))


SIMPLE_HAND_NAMES = hand_names((("thumb", 3), ("index", 2), ("middle", 2)))
TRAIN_HAND_NAMES = hand_names((("thumb", 3), ("middle", 2), ("index", 2)))
SOURCE_STATE_NAMES = BODY_NAMES + SIMPLE_HAND_NAMES
TRAIN_STATE_NAMES = BODY_NAMES + TRAIN_HAND_NAMES
HAND_TO_TRAIN = np.array([SIMPLE_HAND_NAMES.index(n) for n in TRAIN_HAND_NAMES])
STATE_TO_TRAIN = np.array([SOURCE_STATE_NAMES.index(n) for n in TRAIN_STATE_NAMES])
VIDEO_KEY = "observation.images.egocentric"
FUTURE_OFFSETS = np.arange(10, dtype=np.int64) * 5
# observation_config.yaml order; offsets include inactive branches, not a compact vector.
ENCODER_LAYOUT = (
    ("encoder_mode_4", 0, 4),
    ("motion_joint_positions_10frame_step5", 4, 294),
    ("motion_joint_velocities_10frame_step5", 294, 584),
    ("motion_anchor_orientation_heading_10frame_step5", 584, 644),
    ("motion_anchor_orientation_heading", 644, 650),
    ("motion_joint_positions_lowerbody_10frame_step5", 650, 770),
    ("motion_joint_velocities_lowerbody_10frame_step5", 770, 890),
    ("vr_3point_local_target", 890, 899),
    ("vr_3point_local_orn_target", 899, 911),
    ("smpl_joints_10frame_step1", 911, 1631),
    ("smpl_anchor_orientation_heading_10frame_step1", 1631, 1691),
    ("motion_joint_positions_wrists_10frame_step1", 1691, 1751),
)


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def repair_video_metadata(root):
    """Finish older in-flight conversions without changing pixels or timestamps.

    Source info identities change, so existing DINO manifests must be rebuilt.
    A receipt records every changed info.json; it never edits cache manifests.
    """
    root = root.resolve(strict=True)
    updates = []
    for split in ("train", "val", "test"):
        path = root / split / "meta/info.json"
        if not path.is_file():
            continue
        value = json.loads(path.read_text())
        feature = value["features"][VIDEO_KEY]
        descriptor = feature["info"]
        if value["fps"] != 30 or descriptor["video.fps"] != 30:
            raise ValueError("Video descriptor is not the expected converted 30Hz source")
        if feature.get("video_info") == descriptor:
            continue
        before = sha256(path)
        feature["video_info"] = dict(descriptor)
        write_json(path, value)
        updates.append({"path": str(path), "before_sha256": before,
                        "after_sha256": sha256(path), "video_info_equal_info": True})
    report = {"purpose": "Add identical LeRobot info/DINO video_info descriptor aliases",
              "updates": updates, "pixels_or_timestamps_modified": False,
              "existing_dino_manifests_must_be_rebuilt": bool(updates),
              "repair_script_sha256": sha256(Path(__file__))}
    if updates:
        write_json(root / "VIDEO_METADATA_COMPATIBILITY_UPDATE.json", report)
    return report


def matrix_wxyz(quaternion):
    """Standard quaternion matrix; matches pinned C++ math_utils.hpp."""
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape[-1] != 4 or not np.isfinite(q).all():
        raise ValueError("Expected finite wxyz quaternions")
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(np.abs(norm - 1) > .001):
        raise ValueError("Measured root quaternions must be unit length")
    w, x, y, z = np.moveaxis(q / norm, -1, 0)
    result = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    result[..., 0, 0] = 1 - 2 * (y*y + z*z)
    result[..., 0, 1] = 2 * (x*y - w*z)
    result[..., 0, 2] = 2 * (x*z + w*y)
    result[..., 1, 0] = 2 * (x*y + w*z)
    result[..., 1, 1] = 1 - 2 * (x*x + z*z)
    result[..., 1, 2] = 2 * (y*z - w*x)
    result[..., 2, 0] = 2 * (x*z - w*y)
    result[..., 2, 1] = 2 * (y*z + w*x)
    result[..., 2, 2] = 1 - 2 * (x*x + y*y)
    return result


def encoder_input(body_q, body_dq, root_wxyz, current):
    """Exact G1/mode0 mask and layout for a measured offline reference.

    Reference and robot use the same measured trajectory, hence C++ initial
    heading alignment is identity. Only the current robot's yaw is removed
    from every future reference orientation. Position values are absolute
    radians in policy order, with no default-angle subtraction or scaling.
    Rotation 6D uses the first two COLUMNS flattened ROW-WISE, unlike Motion's
    separate rotation representation.
    """
    q, dq, roots = (np.asarray(a, dtype=np.float64) for a in (body_q, body_dq, root_wxyz))
    if q.ndim != 2 or q.shape[1] != 29 or dq.shape != q.shape or roots.shape != (len(q), 4):
        raise ValueError("Expected aligned q/dq[N,29] and root_wxyz[N,4]")
    if not np.isfinite(q).all() or not np.isfinite(dq).all():
        raise ValueError("Nonfinite measured reference")
    future = int(current) + FUTURE_OFFSETS
    if int(current) < 0 or future[-1] >= len(q):
        raise ValueError("A full real 0.9-second future is required; no tail padding")
    matrices = matrix_wxyz(roots[future])
    yaw = np.arctan2(matrices[0, 1, 0], matrices[0, 0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    heading_inverse = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    relative = heading_inverse[None] @ matrices
    value = np.zeros((1, 1751), dtype=np.float32)
    # mode is scalar zero + three zero fillers, NOT a one-hot [1,0,0,0].
    value[0, 4:294] = q[future][:, POLICY_TO_BODY].reshape(-1)
    value[0, 294:584] = dq[future][:, POLICY_TO_BODY].reshape(-1)
    value[0, 584:644] = relative[..., :2].reshape(-1)
    if not np.isfinite(value).all():
        raise ValueError("Reference exceeds float32 range")
    return value


def derived_velocity(q, timestamps):
    q, t = np.asarray(q, dtype=np.float64), np.asarray(timestamps, dtype=np.float64)
    if (q.ndim != 2 or len(q) < 2 or t.shape != (len(q),)
            or not np.isfinite(q).all() or not np.isfinite(t).all()
            or not np.allclose(np.diff(t), .02, rtol=0, atol=.0001)):
        raise ValueError("Velocity derivation requires finite chronological native50Hz samples")
    # np.gradient uses unequal-time central differences internally and real
    # one-sided differences at endpoints; no synthetic endpoint observation.
    return np.gradient(q, t, axis=0, edge_order=1)


def selected_source_indices(native_frames):
    valid_native = int(native_frames) - int(FUTURE_OFFSETS[-1])
    if valid_native <= 0:
        raise ValueError("Episode is too short for a real 0.9-second future")
    count = (valid_native * 30 + 49) // 50
    indices = np.arange(count, dtype=np.int64) * 50 // 30
    if indices[-1] + 45 >= native_frames:
        raise AssertionError("Future-window tail selection error")
    return indices


def fixed_split():
    shuffled = np.random.default_rng(20261021).permutation(50).tolist()
    return {"train": sorted(shuffled[:40]), "val": sorted(shuffled[40:45]),
            "test": sorted(shuffled[45:])}


def check_sources(source, sonic, contract):
    info = json.loads((source / "meta/info.json").read_text())
    if (info["fps"], info["total_episodes"], info["total_frames"]) != (50, 50, 57989):
        raise ValueError("This converter is scoped to the fixed official CarryBox pack")
    archive = source.with_suffix(".zip")
    if not archive.is_file() or sha256(archive) != SOURCE_ARCHIVE_SHA:
        raise ValueError("Fixed official CarryBox source archive SHA mismatch")
    extracted_files = {}
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            if member.is_dir():
                continue
            path = Path(member.filename)
            if path.parts[0] != source.name or ".." in path.parts:
                raise ValueError("Unexpected official source archive member")
            relative = Path(*path.parts[1:])
            actual = (source / relative).resolve(strict=True)
            expected = hashlib.sha256(bundle.read(member)).hexdigest()
            if not actual.is_relative_to(source) or sha256(actual) != expected:
                raise ValueError(f"Extracted source differs from verified archive: {relative}")
            extracted_files[str(relative)] = expected
    evidence = json.loads(contract.read_text())
    if (evidence.get("mapping_resolved_from_official_sources") is not True
            or tuple(evidence.get("source_joint_names", ())) != SOURCE_STATE_NAMES
            or tuple(evidence.get("source_action_hand_names", ())) != SIMPLE_HAND_NAMES
            or evidence.get("state_order_resolution") != "official_recording_body29_hands14"):
        raise ValueError("An audited official named state/action contract is required")
    for path, expected in ((sonic / "model_encoder.onnx", ENCODER_SHA),
                           (sonic / "observation_config.yaml", OBSERVATION_SHA),
                           (CPP_DIR / "src/g1_deploy_onnx_ref.cpp", CPP_SHA),
                           (CPP_DIR / "include/policy_parameters.hpp", PARAMETERS_SHA)):
        if sha256(path) != expected:
            raise ValueError(f"Fixed encoder/reference source identity differs: {path}")
    provenance = json.loads((sonic / "PROVENANCE.json").read_text())
    if provenance.get("revision") != SONIC_REVISION:
        raise ValueError("Official encoder source revision differs")
    return info, evidence, extracted_files


def numeric_feature(width, names=None, dtype="float32"):
    return {"dtype": dtype, "shape": [width], "names": list(names) if names else None}


def output_features():
    features = {
        "observation.state": numeric_feature(43, TRAIN_STATE_NAMES),
        "action.body_token_v1_1": numeric_feature(64, [f"token_{i}" for i in range(64)]),
        "action": numeric_feature(14, TRAIN_HAND_NAMES),
        "observation.base_pose": numeric_feature(7, ["pos_x", "pos_y", "pos_z", "quat_w", "quat_x", "quat_y", "quat_z"], "float64"),
        "observation.base_vel": numeric_feature(6, dtype="float64"),
        "observation.body_joint_velocities": numeric_feature(29, BODY_NAMES, "float64"),
        "observation.object_poses": numeric_feature(7, dtype="float64"),
        "task_description": numeric_feature(1, dtype="string"),
        "next.done": numeric_feature(1, dtype="bool"),
        "timestamp": numeric_feature(1),
        "source_timestamp": numeric_feature(1, dtype="float64"),
    }
    for key in ("frame_index", "episode_index", "index", "task_index",
                "source_episode_index", "source_frame_index"):
        features[key] = numeric_feature(1, dtype="int64")
    features[VIDEO_KEY] = {"dtype": "video", "shape": [360, 640, 3],
                           "names": ["height", "width", "channel"],
                           "info": {"video.fps": 30.0, "video.codec": "h264",
                                    "video.pix_fmt": "gbrp", "video.is_depth_map": False,
                                    "has_audio": False}}
    # Existing LeRobot readers and the A100 DINO extractor use different keys
    # for the same video descriptor. Keep identical aliases for both readers.
    features[VIDEO_KEY]["video_info"] = dict(features[VIDEO_KEY]["info"])
    return features


def vector_stats(a):
    a = np.asarray(a, dtype=np.float64)
    return {"min": a.min(0).tolist(), "max": a.max(0).tolist(),
            "mean": a.mean(0).tolist(), "std": a.std(0).tolist(), "count": [len(a)]}


def rewrite_video(source, target, selected, native_timestamps):
    """Decode original RGB and encode selected frames losslessly at 30fps.

    Every output frame is decoded again and its RGB bytes compared to selected
    source RGB. This verifies geometry/colors and temporal identity directly.
    """
    import av
    target.parent.mkdir(parents=True, exist_ok=True)
    wanted = {int(frame): n for n, frame in enumerate(selected)}
    hashes, pts_errors = [], []
    source_frames = 0
    with av.open(str(source)) as original, av.open(str(target), "w") as output:
        video = original.streams.video[0]
        if video.average_rate != 50 or (video.width, video.height) != (640, 360):
            raise ValueError("Source RGB must be original 640x360 at native50fps")
        encoded = output.add_stream("libx264rgb", rate=30)
        encoded.width, encoded.height, encoded.pix_fmt = 640, 360, "rgb24"
        encoded.options = {"crf": "0", "preset": "fast", "bf": "0", "threads": "2"}
        for frame_number, frame in enumerate(original.decode(video)):
            source_frames += 1
            if frame_number >= len(native_timestamps):
                raise ValueError("Source video has more frames than measured parquet")
            if frame.pts is None:
                raise ValueError("Source video frame has no PTS")
            error = abs(float(frame.pts * video.time_base) - float(native_timestamps[frame_number]))
            pts_errors.append(error)
            if error > .0001:
                raise ValueError("Source video PTS does not align with measured state timestamp")
            if frame_number not in wanted:
                continue
            rgb = frame.to_ndarray(format="rgb24")
            hashes.append(hashlib.sha256(rgb.tobytes()).hexdigest())
            fresh = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            fresh.pts, fresh.time_base = wanted[frame_number], Fraction(1, 30)
            for packet in encoded.encode(fresh):
                output.mux(packet)
        for packet in encoded.encode():
            output.mux(packet)
    if source_frames != len(native_timestamps) or len(hashes) != len(selected):
        raise ValueError("Source video/state frame counts or selected video count differ")
    decoded_hashes, output_pts_errors = [], []
    with av.open(str(target)) as output:
        video = output.streams.video[0]
        if video.average_rate != 30:
            raise ValueError("Output video rate is not 30fps")
        for i, frame in enumerate(output.decode(video)):
            decoded_hashes.append(hashlib.sha256(frame.to_ndarray(format="rgb24").tobytes()).hexdigest())
            output_pts_errors.append(abs(float(frame.pts * video.time_base) - i / 30))
    if decoded_hashes != hashes or max(output_pts_errors, default=0) > 1e-7:
        raise ValueError("Output RGB bytes/timestamps differ from selected source frames")
    return {"codec": "libx264rgb", "crf": 0, "source_decoded_frames": source_frames,
            "selected_frames": len(selected), "all_output_rgb_sha256_equal_source": True,
            "max_source_pts_error_s": max(pts_errors),
            "max_output_pts_error_s": max(output_pts_errors),
            "selected_rgb_sequence_sha256": hashlib.sha256("".join(hashes).encode()).hexdigest(),
            "output_sha256": sha256(target)}


def convert_episode(source, source_info, source_id, target, episode_id, row_offset, session):
    import pyarrow as pa
    import pyarrow.parquet as pq
    data_path = source / source_info["data_path"].format(episode_chunk=source_id // 1000, episode_index=source_id)
    original_video = source / source_info["video_path"].format(episode_chunk=source_id // 1000,
                                                             episode_index=source_id, video_key=VIDEO_KEY)
    table = pq.read_table(data_path)
    q = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
    root = np.asarray(table["observation.base_pose"].to_pylist(), dtype=np.float64)
    hands = np.asarray(table["action"].to_pylist(), dtype=np.float64)
    timestamps = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64).reshape(-1)
    n = len(q)
    if q.shape != (n, 43) or hands.shape != (n, 36) or root.shape != (n, 7):
        raise ValueError("Native source state/action/root schema differs")
    if (not np.array_equal(table["frame_index"].to_pylist(), np.arange(n))
            or set(table["episode_index"].to_pylist()) != {source_id}
            or not np.allclose(timestamps, np.arange(n) / 50, rtol=0, atol=.0001)
            or not np.isfinite(q).all() or not np.isfinite(root).all()
            or not np.isfinite(hands).all() or np.any(hands[:, 14:] != 0)):
        raise ValueError("Source frame identities, native timing or hand-only actions differ")
    dq = derived_velocity(q[:, :29], timestamps)
    selected = selected_source_indices(n)
    tokens = np.empty((len(selected), 64), dtype=np.float32)
    start = time.monotonic()
    for i, current in enumerate(selected):
        value = encoder_input(q[:, :29], dq, root[:, 3:7], current)
        token = session.run(["encoded_tokens"], {"obs_dict": value})[0]
        if token.shape != (1, 64) or not np.isfinite(token).all():
            raise ValueError("Official v1.1 encoder returned invalid token")
        tokens[i] = token[0]
    instruction = set(table["task_description"].to_pylist())
    if len(instruction) != 1:
        raise ValueError("One episode must have one task description")
    instruction = instruction.pop()
    length = len(selected)
    values = {
        "observation.state": q[selected][:, STATE_TO_TRAIN].astype(np.float32),
        "action.body_token_v1_1": tokens,
        "action": hands[selected, :14][:, HAND_TO_TRAIN].astype(np.float32),
        "observation.base_pose": root[selected],
        "observation.base_vel": np.asarray(table["observation.base_vel"].to_pylist(), dtype=np.float64)[selected],
        "observation.body_joint_velocities": dq[selected],
        "observation.object_poses": np.asarray(table["observation.object_poses"].to_pylist(), dtype=np.float64)[selected],
        "timestamp": (np.arange(length) / 30).astype(np.float32),
        "source_timestamp": timestamps[selected],
        "frame_index": np.arange(length, dtype=np.int64),
        "episode_index": np.full(length, episode_id, dtype=np.int64),
        "index": np.arange(row_offset, row_offset + length, dtype=np.int64),
        "task_index": np.zeros(length, dtype=np.int64),
        "source_episode_index": np.full(length, source_id, dtype=np.int64),
        "source_frame_index": selected,
        "task_description": [instruction] * length,
        "next.done": np.arange(length) == length - 1,
    }
    for key, value in values.items():
        if key != "task_description" and not np.isfinite(value).all():
            raise ValueError(f"Nonfinite output: {key}")
    video_path = target / f"videos/chunk-000/{VIDEO_KEY}/episode_{episode_id:06d}.mp4"
    video = rewrite_video(original_video, video_path, selected, timestamps)
    arrays = {}
    for key, value in values.items():
        if isinstance(value, np.ndarray) and value.ndim == 2:
            arrays[key] = pa.array(value.tolist(), type=pa.list_(pa.float64() if value.dtype == np.float64 else pa.float32()))
        else:
            arrays[key] = pa.array(value)
    target_data = target / f"data/chunk-000/episode_{episode_id:06d}.parquet"
    target_data.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(arrays), target_data, compression="zstd")
    # Descriptive v2.1 metadata only; normalization remains the A100 run's
    # independent dataset_statistics.json. No local stats_psi0/bounds is made.
    stats = {key: vector_stats(values[key]) for key in
             ("observation.state", "action.body_token_v1_1", "action")}
    entry = {"episode_index": episode_id, "tasks": [instruction], "length": length,
             "source_episode_index": source_id}
    report = {"episode_index": episode_id, "source_episode_index": source_id,
              "source_parquet": str(data_path), "source_parquet_sha256": sha256(data_path),
              "source_video": str(original_video), "source_video_sha256": sha256(original_video),
              "source_frames": n, "output_frames": length, "native_tail_frames_excluded": 45,
              "last_source_frame": int(selected[-1]), "last_reference_frame": int(selected[-1] + 45),
              "output_parquet_sha256": sha256(target_data), "video": video,
              "encoder_cpu_seconds_and_video_conversion": time.monotonic() - start,
              "token_min": float(tokens.min()), "token_max": float(tokens.max()),
              "all_numeric_finite": True}
    return entry, {"episode_index": episode_id, "stats": stats}, report


def prepare(source, sonic, contract, output, *, source_episodes=None, threads=2):
    import av
    import onnxruntime as ort
    import pyarrow
    source, sonic, contract = (p.resolve(strict=True) for p in (source, sonic, contract))
    output = output.absolute()
    if output.exists() or output.is_symlink() or output.resolve().is_relative_to(REPO):
        raise ValueError("Use a fresh output directory outside the repository")
    source_info, evidence, source_files = check_sources(source, sonic, contract)
    if source_episodes is not None and (not source_episodes or len(set(source_episodes)) != len(source_episodes)
                                        or not set(source_episodes) <= set(range(50))):
        raise ValueError("Source episode subset must contain unique IDs in [0,49]")
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = threads, 1
    session = ort.InferenceSession(str(sonic / "model_encoder.onnx"), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if (len(inputs) != 1 or inputs[0].name != "obs_dict" or inputs[0].shape != [1, 1751]
            or inputs[0].type != "tensor(float)" or len(outputs) != 1
            or outputs[0].name != "encoded_tokens" or outputs[0].shape != [1, 64]
            or session.get_providers() != ["CPUExecutionProvider"]):
        raise ValueError("Official encoder CPU contract differs")
    split = fixed_split()
    selected_ids = set(range(50)) if source_episodes is None else set(source_episodes)
    output.mkdir(parents=True)
    provenance = {
        "source_dataset": "USC-PSI-Lab/psi-data", "source_revision": SOURCE_REVISION,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA, "source_root": str(source),
        "source_file_sha256": source_files,
        "encoder_revision": SONIC_REVISION, "encoder_sha256": ENCODER_SHA,
        "encoder_observation_yaml_sha256": OBSERVATION_SHA,
        "cpp_reference_sha256": CPP_SHA, "policy_parameters_sha256": PARAMETERS_SHA,
        "encoder_mode": 0, "encoder_layout": [list(x) for x in ENCODER_LAYOUT],
        "active_encoder_terms": [x[0] for x in ENCODER_LAYOUT[:4]],
        "mode_encoding": "[0,0,0,0] scalar mode zero plus fillers, not one-hot",
        "reference_offsets_native50hz": FUTURE_OFFSETS.tolist(),
        "joint_q": "absolute measured radian body29, reordered to policy order; no default-angle subtraction",
        "joint_dq": "derived np.gradient at measured 50Hz timestamps, central unequal-time differences internally, one-sided real endpoint differences",
        "heading": "inverse yaw(current measured root) * future measured root; reference/robot initial heading alignment identity",
        "rotation6d": "first two matrix columns, row-wise flatten [r00,r01,r10,r11,r20,r21]",
        "inactive_encoder_branches": "all zero per official mode filter",
        "legacy_tokens": "not read as labels, not renamed, not copied to output",
        "resampling": "floor(n*50/30) same source frame for RGB/state/hands/token/root/object",
        "future_tail": "exclude 45 native frames requiring unavailable future; no edge copying",
        "history": "output rows n-[30,24,18,12,6,0] select source indices separated by exactly 10 native frames; first one second has no full history",
        "source_state_names": list(SOURCE_STATE_NAMES), "output_state_names": list(TRAIN_STATE_NAMES),
        "source_action_hand_names": list(SIMPLE_HAND_NAMES), "output_action_hand_names": list(TRAIN_HAND_NAMES),
        "policy_to_body": POLICY_TO_BODY.tolist(),
        "state_action_contract": str(contract), "state_action_contract_sha256": sha256(contract),
        "mapping_evidence": evidence,
        "normalization": "A100 bounds must be reused externally unchanged; descriptive episode stats are not new normalization bounds",
        "split_seed": 20261021, "split_algorithm": "numpy.default_rng PCG64 permutation(50), 40/5/5, sorted IDs within split",
        "source_episode_splits": split, "selected_source_episodes": sorted(selected_ids),
        "code_sha256": sha256(Path(__file__)),
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "onnxruntime": ort.__version__, "pyarrow": pyarrow.__version__, "av": av.__version__,
                        "providers": session.get_providers(), "ort_intra_op_threads": threads},
        "token_reproduction_performed": True,
        "token_reproduction_scope": "new official v1.1 encoding from SIMPLE measured motion; no claim of reproducing pre-existing UnifoLM tokens",
        "gpu_used": False, "dino_cache_generated": False, "physical_rollout_passed": False,
    }
    write_json(output / "CONVERSION_PROVENANCE.json", provenance)
    shutil.copy2(contract, output / "STATE_ACTION_CONTRACT.json")
    all_reports = []
    for split_name, source_ids in split.items():
        source_ids = [i for i in source_ids if i in selected_ids]
        if not source_ids:
            continue
        target = output / split_name
        episodes, episode_stats, reports = [], [], []
        offset = 0
        for episode_id, source_id in enumerate(source_ids):
            entry, stats, report = convert_episode(source, source_info, source_id, target, episode_id, offset, session)
            offset += entry["length"]
            episodes.append(entry); episode_stats.append(stats); reports.append(report)
            print(json.dumps({"split": split_name, "source_episode": source_id,
                              "episode": episode_id, "frames": entry["length"],
                              "finite": True, "rgb_lossless_verified": True}), flush=True)
        features = output_features()
        info = {"codebase_version": "v2.1", "robot_type": "g1_sonic", "total_episodes": len(episodes),
                "total_frames": offset, "total_tasks": 1, "total_videos": len(episodes),
                "total_chunks": 1, "chunks_size": 1000, "fps": 30,
                "splits": {split_name: f"0:{len(episodes)}"},
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
                "features": features}
        write_json(target / "meta/info.json", info)
        for filename, rows in (("episodes.jsonl", episodes), ("episodes_stats.jsonl", episode_stats),
                               ("tasks.jsonl", [{"task_index": 0, "task": episodes[0]["tasks"][0]}])):
            (target / "meta" / filename).write_text("".join(json.dumps(r) + "\n" for r in rows))
        write_json(target / "meta/modality.json", {"state": {"joint_positions": {"start": 0, "end": 43}},
                   "action": {"body_token_v1_1": {"start": 0, "end": 64, "original_key": "action.body_token_v1_1"},
                              "hand_joints": {"start": 0, "end": 14}},
                   "video": {"egocentric": {"original_key": VIDEO_KEY}},
                   "annotation": {"task_description": {}}})
        write_json(target / "meta/conversion_provenance.json", provenance)
        write_json(target / "meta/episode_conversion_reports.json", reports)
        all_reports.extend({"split": split_name, **r} for r in reports)
    ready = {"status": "cpu_conversion_complete", "complete_50_episode_conversion": selected_ids == set(range(50)),
             "episodes": len(all_reports), "output_frames": sum(r["output_frames"] for r in all_reports),
             "split_source_episode_ids": split, "reports": all_reports,
             "encoder_sha256": ENCODER_SHA, "legacy_token_used": False,
             "all_numeric_finite": True, "all_output_rgb_verified_lossless": True,
             "dino_cache_generated": False, "physical_rollout_passed": False}
    write_json(output / "CONVERSION_READY.json", ready)
    return ready


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--sonic", type=Path)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repair-video-metadata", type=Path,
                        help="Only add matching descriptor aliases and a SHA receipt to an existing conversion")
    parser.add_argument("--source-episodes", nargs="+", type=int, help="Smoke test only: convert a subset of original IDs")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.repair_video_metadata is not None:
        print(json.dumps(repair_video_metadata(args.repair_video_metadata), indent=2))
        return
    if any(getattr(args, name) is None for name in ("source", "sonic", "contract", "output")):
        parser.error("conversion requires --source, --sonic, --contract and --output")
    if args.threads < 1:
        parser.error("--threads must be positive")
    report = prepare(args.source, args.sonic, args.contract, args.output,
                     source_episodes=args.source_episodes, threads=args.threads)
    print(json.dumps({k: v for k, v in report.items() if k != "reports"}, indent=2))


if __name__ == "__main__":
    main()
