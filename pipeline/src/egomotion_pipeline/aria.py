"""Build an aligned Aria trajectory sidecar from original EgoExo4D assets."""

import csv
from decimal import Decimal, InvalidOperation
from pathlib import Path
import re
import numpy as np
from .adapters import read_json, discover_hm
from .storage import atomic_json


def capture_timestamps(path, column, start, end):
    result = []
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if column not in (reader.fieldnames or []):
            raise ValueError(f"Missing sync column {column}")
        for index, row in enumerate(reader):
            if index < start:
                continue
            if index >= end:
                break
            try:
                value = Decimal(row[column])
                stamp = int(value) if value.is_finite() and value == int(value) else -1
            except (InvalidOperation, TypeError, ValueError, OverflowError):
                stamp = -1
            result.append(stamp)
    if len(result) != end-start:
        raise ValueError("timesync slice length mismatch")
    return np.asarray(result, dtype=np.int64)


def prepare_aria(config, output_root):
    from projectaria_tools.core import data_provider, mps
    original = Path(config["egoexo_root"])
    takes = {take["take_name"]: take for take in read_json(original/"takes.json")}
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for hm_path, uid in discover_hm(config):
        take = takes[uid]
        ego = [cam["cam_id"] for cam in take["capture"]["cameras"] if cam["is_ego"] and cam["cam_id"].startswith("aria")]
        if len(ego) != 1:
            raise ValueError(f"{uid}: need exactly one Aria camera")
        take_root = original/"takes"/uid
        vrs_paths = sorted(take_root.glob("*noimagestreams.vrs"))
        if len(vrs_paths) != 1:
            raise ValueError(f"{uid}: expected one noimagestreams.vrs, found {len(vrs_paths)}")
        provider = data_provider.create_vrs_data_provider(str(vrs_paths[0]))
        device = provider.get_device_calibration()
        camera_calib = device.get_camera_calib("camera-rgb")
        device_camera = camera_calib.get_transform_device_camera().to_matrix()
        # CPF is a tracked head reference, distinct from the SMPL anatomical head joint.
        device_head = device.get_transform_device_cpf().to_matrix()
        capture = take.get("capture", {}).get("capture_name") or re.sub(r"_\d+$", "", uid)
        timesync = original/"captures"/capture/"timesync.csv"
        stamps = capture_timestamps(timesync, f"{ego[0]}_214-1_capture_timestamp_ns",
                                    int(take["timesync_start_idx"]), int(take["timesync_end_idx"]))
        with np.load(hm_path, allow_pickle=False) as hm:
            count = hm["trans"].shape[1]
        if len(stamps) != count:
            raise ValueError(f"{uid}: sync frames {len(stamps)} differ from HM frames {count}; check take/release alignment")
        mps_provider = mps.MpsDataProvider(mps.MpsDataPathsProvider(str(take_root)).get_data_paths())
        camera = np.broadcast_to(np.eye(4), (count, 4, 4)).copy()
        head = camera.copy()
        valid = np.zeros(count, bool)
        errors = np.zeros(count, float)
        previous = -1
        tolerance = float(config.get("max_pose_error_s", .02))
        if tolerance <= 0:
            raise ValueError("max_pose_error_s must be positive")
        for index, stamp in enumerate(stamps):
            if stamp < 0 or stamp <= previous:
                continue
            previous = stamp
            pose = mps_provider.get_closed_loop_pose(int(stamp))
            if pose is None:
                continue
            pose_time = pose.tracking_timestamp.total_seconds()
            error = pose_time-int(stamp)/1e9
            errors[index] = error
            if abs(error) > tolerance:
                continue
            world_device = pose.transform_world_device.to_matrix()
            camera[index] = world_device@device_camera
            head[index] = world_device@device_head
            valid[index] = True
        video_paths = sorted((take_root/"frame_aligned_videos").glob(f"{ego[0]}_214-1.mp4"))
        if len(video_paths) != 1:
            raise ValueError(f"{uid}: missing/ambiguous frame-aligned Aria RGB video")
        preprocessing = config.get("video_preprocessing", {"mode": "raw"}).copy()
        if preprocessing.get("mode") == "aria":
            preprocessing["vrs_path"] = str(vrs_paths[0].resolve())
        npz_path = output_root/f"{uid}.npz"
        tmp = output_root/f"{uid}.npz.tmp"
        with tmp.open("wb") as handle:
            np.savez_compressed(handle, timestamp=np.arange(count)/30,
                                camera_world=camera, head_world=head, camera_valid=valid, head_valid=valid,
                                capture_timestamp_ns=stamps, pose_time_error_s=errors)
        tmp.replace(npz_path)
        atomic_json(npz_path.with_suffix(".json"), {"take_name": uid, "clock": "aligned_take_seconds",
                    "head_semantics": "aria_cpf", "pose_sampling": "nearest_MPS_with_time_error_gate",
                    "camera_pose_basis": "aria_rgb_native",
                    "max_pose_error_s": tolerance, "valid_count": int(valid.sum()),
                    "vrs_path": str(vrs_paths[0].resolve()), "timesync_path": str(timesync.resolve()),
                    "take_range": [take["timesync_start_idx"], take["timesync_end_idx"]],
                    "video": {"path": str(video_paths[0].resolve()), "from_timestamp": 0.0,
                              "to_timestamp": (count-1)/30, "timestamp_origin": 0.0,
                              "camera": ego[0], "preprocessing": preprocessing}})
        print(f"Aria {uid}: {int(valid.sum())}/{count}", flush=True)
