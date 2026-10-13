"""Source readers. Text/task annotations are intentionally not read."""

import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import pyarrow.compute as pc
from .schema import Episode
from .geometry import matrix_to_quat, rigid_transform


def read_json(path):
    return json.loads(Path(path).read_text())


def selected(uid, config):
    ids = config.get("episode_ids")
    if isinstance(ids, str):
        ids = read_json(ids)
    return ids is None or uid in ids


def discover_egosuite(config):
    root = Path(config["root"])
    infos = [root / "meta/info.json"] if (root / "meta/info.json").exists() else sorted(root.rglob("meta/info.json"))
    if not infos:
        raise FileNotFoundError(f"No LeRobot meta/info.json under {root}")
    emitted = 0
    for info_path in infos:
        dataset_root = info_path.parent.parent
        for episode_file in sorted((dataset_root / "meta/episodes").glob("*/*.parquet")):
            schema = pq.read_schema(episode_file)
            # Only structural metadata and calibration/quality fields; no task text.
            columns = [c for c in schema.names if c not in {"tasks", "task", "annotation", "language_instruction"} and not c.startswith("stats/")]
            for row in pq.read_table(episode_file, columns=columns).to_pylist():
                uid = str(row.get("episode_uuid") or f"{dataset_root.relative_to(root)}:{row['episode_index']}")
                if selected(uid, config):
                    yield dataset_root, row, uid
                    emitted += 1
                    if config.get("limit") and emitted >= int(config["limit"]):
                        return


def column_array(table, key, shape, count, required=False):
    if key not in table.column_names:
        if required:
            raise ValueError(f"Missing LeRobot column: {key}")
        return np.full((count, *shape), np.nan)
    data = table[key].to_pylist()
    try:
        array = np.asarray(data, dtype=float)
        return array.reshape(count, *shape)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Cannot reshape {key} to {(count, *shape)}; only full 22-joint body is accepted") from exc


def read_egosuite(dataset_root, row, uid, config):
    info = read_json(dataset_root / "meta/info.json")
    if not str(info.get("codebase_version", "")).startswith("v3"):
        raise ValueError("EgoSuite reader requires LeRobot v3 metadata")
    template = info.get("data_path", "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet")
    path = dataset_root / template.format(chunk_index=int(row["data/chunk_index"]), file_index=int(row["data/file_index"]))
    keys = ["timestamp", "frame_index", "episode_index", "index"]
    keys += ["observation.state."+name for name in ["body_world", "body_world_rotation", "hand_left_world", "hand_right_world", "head_world", "head_world_rotation"]]
    camera = config.get("camera", "head_left")
    keys += [f"observation.state.{camera}_camera_position", f"observation.state.{camera}_camera_rotation"]
    names = pq.read_schema(path).names
    table = pq.read_table(path, columns=[key for key in keys if key in names])
    table = table.filter(pc.equal(table["episode_index"], int(row["episode_index"])))
    count = len(table)
    if count != int(row["length"]) or count == 0:
        raise ValueError(f"{uid}: episode length metadata mismatch")
    frame_index = np.asarray(table["frame_index"].to_numpy())
    if not np.array_equal(frame_index, np.arange(count)):
        raise ValueError(f"{uid}: unordered/non-contiguous source frame_index")
    timestamps = np.asarray(table["timestamp"].to_numpy(), dtype=float)
    # Vendor timestamps are float32. Recover the nominal clock only if every
    # sample fits that clock within float32 rounding, retaining the error report.
    nominal = timestamps[0]+np.arange(count)/float(info["fps"])
    rounding = np.abs(np.spacing(timestamps.astype(np.float32))).astype(float)*.51+1e-7
    reconstruct = pa_float32(table["timestamp"].type) and np.all(np.abs(timestamps-nominal) <= rounding)
    timestamp_error = float(np.max(np.abs(timestamps-nominal)))
    if reconstruct:
        timestamps = nominal
    positions = {
        "body": column_array(table, "observation.state.body_world", (22, 3), count, True),
        "hand_left": column_array(table, "observation.state.hand_left_world", (21, 3), count, True),
        "hand_right": column_array(table, "observation.state.hand_right_world", (21, 3), count, True),
        "head": column_array(table, "observation.state.head_world", (1, 3), count),
        "camera": column_array(table, f"observation.state.{camera}_camera_position", (1, 3), count)}
    rotations = {"body": column_array(table, "observation.state.body_world_rotation", (22, 4), count, True),
                 "head": column_array(table, "observation.state.head_world_rotation", (1, 4), count),
                 "camera": column_array(table, f"observation.state.{camera}_camera_rotation", (1, 4), count)}
    video_key = f"observation.images.{camera}"
    video = None
    if video_key in info.get("features", {}):
        prefix = f"videos/{video_key}"
        if prefix+"/chunk_index" not in row or prefix+"/from_timestamp" not in row:
            raise ValueError(f"{uid}: missing LeRobot video offsets")
        video_path = dataset_root / info["video_path"].format(video_key=video_key,
                          chunk_index=int(row[prefix+"/chunk_index"]), file_index=int(row[prefix+"/file_index"]))
        video = {"path": str(video_path.resolve()), "from_timestamp": float(row[prefix+"/from_timestamp"]),
                 "to_timestamp": float(row[prefix+"/to_timestamp"]), "timestamp_origin": 0.0,
                 "camera": camera, "preprocessing": config.get("video_preprocessing", {"mode": "preundistorted"})}
    metadata = {"source_root": str(dataset_root.resolve()), "source_data": str(path.resolve()),
                "source_episode_index": int(row["episode_index"]), "source_fps": info["fps"],
                "source_revision": config.get("revision", "unspecified"),
                "body_position_source": "egosuite_mocap", "hand_position_source": "egosuite_mocap",
                "rotation_semantics": "source_global_wxyz", "head_semantics": "egosuite_head_world",
                "camera_pose_basis": "egosuite_native; local axes require vendor calibration",
                "hand_joint_convention": "wrist_then_thumb_index_middle_ring_pinky_4_each",
                "quality": "finite_only; vendor_bad_frame_ratio_not_a_per_joint_mask"}
    metadata["timestamp_reconstruction"] = {"nominal_from_frame_index": bool(reconstruct),
                                             "max_rounding_error_s": timestamp_error,
                                             "source_dtype": str(table["timestamp"].type)}
    for key in ["scene_id", "environment_id", "operator_id", "batch_version", f"camera_intrinsics/{camera}"]:
        if key in row:
            metadata[key] = row[key]
    return Episode(uid, config.get("name", "egosuite"), timestamps, positions, rotations, metadata=metadata, video=video).check()


def pa_float32(dtype):
    import pyarrow as pa
    return dtype == pa.float32()


def discover_hm(config):
    root = Path(config["root"])
    paths = sorted(root.rglob("points_triangulated_world_results_merged.npz"))
    if not paths:
        raise FileNotFoundError(f"No released HM NPZ files under {root}")
    emitted = 0
    for path in paths:
        uid = path.parent.name
        if selected(uid, config):
            yield path, uid
            emitted += 1
            if config.get("limit") and emitted >= int(config["limit"]):
                return


def read_hm(path, uid, config, body_model):
    with np.load(path, allow_pickle=False) as raw:
        required = ["trans", "root_orient", "pose_body", "betas_per_frame", "valid", "joints3d"]
        missing = set(required)-set(raw.files)
        if missing:
            raise ValueError(f"{uid}: missing HM fields {sorted(missing)}")
        def take(key):
            value = raw[key]
            if value.shape[0] != 1:
                raise ValueError(f"{uid}: expected released HM shape [1,T,...] for {key}")
            return value[0]
        trans, root, pose, betas, valid, joints = [take(key) for key in required]
        count = len(trans)
        if joints.shape != (count, 67, 3) or valid.shape != (count,):
            raise ValueError("HM requires 67 OpenPose+hand joints and [1,T] valid")
        pos, rot, mask = body_model.decode(trans, root, pose, betas, valid)
        timestamps = np.arange(count, dtype=float)/30
        positions = {"body": pos, "hand_left": joints[:, 25:46], "hand_right": joints[:, 46:67]}
        rotations = {"body": rot}
        position_valid = {"body": mask, "hand_left": np.broadcast_to(valid.astype(bool)[:, None], (count, 21)),
                          "hand_right": np.broadcast_to(valid.astype(bool)[:, None], (count, 21))}
        rotation_valid = {"body": mask}
        chunks = raw["chunk_ranges"].tolist() if "chunk_ranges" in raw else None
    metadata = {"source_data": str(Path(path).resolve()), "source_fps": 30,
                "source_revision": config.get("revision", "unspecified"),
                "body_position_source": "hm_smplh_fit_fk", "hand_position_source": "hm_openpose67_joints3d",
                "rotation_semantics": "global_smplh_fk_wxyz", "chunk_ranges": chunks,
                "head_semantics": "unavailable", "quality": "HM_valid_is_chunk_finiteness_not_joint_confidence",
                **body_model.metadata}
    video = None
    trajectory_root = config.get("trajectory_root")
    sidecar_path = Path(trajectory_root)/f"{uid}.npz" if trajectory_root else None
    if sidecar_path and sidecar_path.exists():
        with np.load(sidecar_path, allow_pickle=False) as sidecar:
            side_ts = np.asarray(sidecar["timestamp"], float)
            if side_ts.shape != timestamps.shape or not np.allclose(side_ts, timestamps, atol=1e-5, rtol=0):
                raise ValueError("Aria sidecar must already use the aligned take's 30 Hz timestamps")
            if config.get("require_camera", True):
                missing_camera = {"camera_world", "camera_valid"}-set(sidecar.files)
                if missing_camera:
                    raise ValueError(f"{uid}: Aria sidecar missing required camera fields {sorted(missing_camera)}")
            for modality in ["camera", "head"]:
                key = f"{modality}_world"
                if key not in sidecar:
                    continue
                transforms = sidecar[key]
                valid_pose = sidecar[f"{modality}_valid"].astype(bool)
                if transforms.shape != (count, 4, 4) or valid_pose.shape != (count,):
                    raise ValueError("Invalid trajectory sidecar shape")
                for transform in transforms[valid_pose]:
                    rigid_transform(transform)
                positions[modality] = transforms[:, None, :3, 3]
                rotations[modality] = matrix_to_quat(transforms[:, None, :3, :3])
                position_valid[modality] = valid_pose[:, None]
                rotation_valid[modality] = valid_pose[:, None]
        meta_path = sidecar_path.with_suffix(".json")
        if not meta_path.exists():
            raise ValueError("Trajectory sidecar needs matching JSON provenance/video metadata")
        side_meta = read_json(meta_path)
        if side_meta.get("take_name") != uid or side_meta.get("clock") != "aligned_take_seconds":
            raise ValueError("Aria sidecar identity/clock mismatch")
        video = side_meta.get("video")
        if video and not Path(video["path"]).is_absolute():
            video["path"] = str((sidecar_path.parent/video["path"]).resolve())
        metadata["head_semantics"] = side_meta.get("head_semantics", "unspecified")
        metadata["camera_pose_basis"] = side_meta.get("camera_pose_basis", "unspecified")
        metadata["aria_provenance"] = side_meta
    elif config.get("require_camera", True):
        raise ValueError(f"{uid}: Aria sidecar missing. HM cam_R/cam_t are not assumed to be ego. Run prepare-aria or explicitly set require_camera=false.")
    # Optional video override for a previously undistorted, frame-aligned export.
    if config.get("video_root"):
        video = {"path": str((Path(config["video_root"])/config.get("video_template", "{take}.mp4").format(take=uid)).resolve()),
                 "from_timestamp": 0.0, "to_timestamp": (count-1)/30,
                 "timestamp_origin": 0.0, "preprocessing": config.get("video_preprocessing", {"mode": "raw"})}
    return Episode(uid, config.get("name", "egoexo4d_hm"), timestamps, positions, rotations,
                   position_valid, rotation_valid, metadata, video).check()
