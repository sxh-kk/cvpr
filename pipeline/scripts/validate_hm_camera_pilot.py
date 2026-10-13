#!/usr/bin/env python3
"""Read-only HM/Aria pilot; never rewrites an archive or source dataset.

The timestamp prefix below is an explicitly labelled research candidate,
supported by the pinned HM helper's timestamps[:num_frames].  A successful
candidate is not a production adapter acceptance or image ground truth.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

import av
import numpy as np
from PIL import Image, ImageDraw
from projectaria_tools.core import data_provider, mps

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from egomotion_pipeline.aria import capture_timestamps, prepare_aria
from egomotion_pipeline.geometry import quat_to_matrix


DEFAULT_TAKES = ("cmu_bike02_4", "sfu_cooking026_8", "uniandes_dance_020_37")
OFFICIAL_COMMIT = "135d44ee5310c554fde1cbe33e02c7c204717588"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def identity(path):
    p = Path(path).resolve()
    s = p.stat()
    return {"path": str(p), "bytes": s.st_size, "mtime_ns": s.st_mtime_ns,
            "device": s.st_dev, "inode": s.st_ino, "sha256": digest(p)}


def stats(value):
    x = np.asarray(value, float).reshape(-1)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"count": 0}
    return {"count": int(len(x)), "minimum": float(x.min()), "median": float(np.median(x)),
            "p95": float(np.quantile(x, .95)), "maximum": float(x.max())}


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def audit_take(root, take, out, references):
    uid = take["take_name"]
    out = out / uid
    out.mkdir(parents=True, exist_ok=False)
    tr = root / "EgoExo4D/takes" / uid
    ego = [x["cam_id"] for x in take["capture"]["cameras"]
           if x["is_ego"] and x["cam_id"].startswith("aria")]
    if len(ego) != 1:
        raise ValueError(f"{uid}: expected exactly one Aria camera")
    hm_path = root / "EgoExo4D-HM" / uid / "points_triangulated_world_results_merged.npz"
    vrs_paths = list(tr.glob("*noimagestreams.vrs"))
    if len(vrs_paths) != 1:
        raise ValueError(f"{uid}: expected exactly one VRS")
    capture = take["capture"]["capture_name"]
    sync_path = root / "EgoExo4D/captures" / capture / "timesync.csv"
    traj_path = tr / "trajectory/closed_loop_trajectory.csv"
    gopro_path = tr / "trajectory/gopro_calibs.csv"
    video_path = tr / "frame_aligned_videos" / f"{ego[0]}_214-1.mp4"
    source_paths = [hm_path, vrs_paths[0], sync_path, traj_path, gopro_path, video_path]
    before = [identity(p) for p in source_paths]
    with np.load(hm_path, allow_pickle=False) as n:
        raw = {k: n[k] for k in ("trans", "valid", "joints3d", "joints2d", "cam_R", "cam_t", "intrins", "cam_dist", "chunk_ranges")}
    T = raw["trans"].shape[1]
    valid_body = raw["valid"][0].astype(bool)
    joints = raw["joints3d"][0]
    stamps_full = capture_timestamps(sync_path, f"{ego[0]}_214-1_capture_timestamp_ns",
                                    take["timesync_start_idx"], take["timesync_end_idx"])
    if len(stamps_full) < T:
        raise ValueError("HM has more frames than the published timesync slice")
    stamps = stamps_full[:T]
    strict_length_pass = len(stamps_full) == T
    # Exercise the actual adapter instead of inferring its outcome from length.
    strict_output = out / "strict-adapter-attempt"
    try:
        prepare_aria({"root": str(root / "EgoExo4D-HM"), "egoexo_root": str(root / "EgoExo4D"),
                      "episode_ids": [uid], "require_camera": True}, strict_output)
        strict_probe = {"status": "passed", "error": None,
                        "output_files": [str(p) for p in strict_output.glob("*")]}
    except Exception as error:
        strict_probe = {"status": "failed", "error_type": type(error).__name__, "error": str(error),
                        "output_files": [str(p) for p in strict_output.glob("*")]}
    strict_length_pass = bool(strict_length_pass and strict_probe["status"] == "passed")
    monotonic = bool(np.all(stamps > 0) and np.all(np.diff(stamps) > 0))
    vrs = data_provider.create_vrs_data_provider(str(vrs_paths[0]))
    device = vrs.get_device_calibration()
    calib = device.get_camera_calib("camera-rgb")
    device_camera = calib.get_transform_device_camera().to_matrix()
    device_cpf = device.get_transform_device_cpf().to_matrix()
    paths = mps.MpsDataPathsProvider(str(tr)).get_data_paths()
    provider = mps.MpsDataProvider(paths)
    camera = np.repeat(np.eye(4)[None], T, axis=0)
    cpf = camera.copy()
    errors = np.zeros(T, float)
    valid = np.zeros(T, bool)
    graph_uids = set()
    pose_times_ns = np.zeros(T, np.int64)
    previous = -1
    for i, stamp in enumerate(stamps):
        if stamp <= previous or stamp <= 0:
            continue
        previous = int(stamp)
        pose = provider.get_closed_loop_pose(int(stamp))
        if pose is None:
            continue
        ts = int(round(pose.tracking_timestamp.total_seconds() * 1e9))
        pose_times_ns[i] = ts
        errors[i] = (ts - int(stamp)) / 1e9
        if abs(errors[i]) > .02:
            continue
        graph_uids.add(pose.graph_uid)
        wd = pose.transform_world_device.to_matrix()
        camera[i] = wd @ device_camera
        cpf[i] = wd @ device_cpf
        valid[i] = True
    with traj_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    gravity = np.array([[float(row[f"gravity_{axis}_world"]) for axis in "xyz"] for row in rows])
    gravity_error = np.linalg.norm(gravity - [0., 0., -9.81], axis=-1)
    with gopro_path.open(newline="") as f:
        rows = [x for x in csv.DictReader(f) if float(x["quality"]) == 1.]
    if len(rows) < 4:
        raise ValueError("Fewer than four localized static cameras")
    exo = []
    for view, row in enumerate(rows[:4], 1):
        q = np.array([float(row[f"q{axis}_world_cam"]) for axis in "wxyz"])
        q /= np.linalg.norm(q)
        c2w = np.eye(4)
        c2w[:3, :3] = quat_to_matrix(q)
        c2w[:3, 3] = [float(row[f"t{axis}_world_cam"]) for axis in "xyz"]
        w2c = np.linalg.inv(c2w)
        re = np.abs(raw["cam_R"][view] - w2c[:3, :3]).max()
        te = np.abs(raw["cam_t"][view] - w2c[:3, 3]).max()
        exo.append({"view": view, "camera_uid": row["cam_uid"], "graph_uid": row["graph_uid"],
                    "rotation_max_element_error": float(re), "translation_max_element_error_m": float(te),
                    "matches_mps_graph": row["graph_uid"] in graph_uids,
                    "passed": bool(re < 1e-5 and te < 1e-5 and row["graph_uid"] in graph_uids)})
    duplicate = bool(np.array_equal(raw["cam_R"][0], raw["cam_R"][1])
                     and np.array_equal(raw["cam_t"][0], raw["cam_t"][1]))
    reprojection = []
    for view in range(1, 5):
        pc = np.einsum("tij,tkj->tki", raw["cam_R"][view], joints) + raw["cam_t"][view, :, None]
        homogeneous = np.einsum("tij,tkj->tki", raw["intrins"][view], pc)
        pix = homogeneous[..., :2] / homogeneous[..., 2:]
        mask = valid_body[:, None] & np.isfinite(pix).all(-1) & np.isfinite(raw["joints2d"][view-1]).all(-1)
        err = np.linalg.norm(pix - raw["joints2d"][view-1], axis=-1)
        reprojection.append({"view": view, "error_px": stats(err[mask]),
                             "passed": bool(mask.any() and np.max(err[mask]) < .05),
                             "evidence_kind": "released_2d_is_projection_of_released_3d; not_image_ground_truth"})
    sample_indices = sorted(set([0, T // 2, T-1]))
    samples = {}
    pts = []
    with av.open(str(video_path)) as video:
        stream = video.streams.video[0]
        width, height, fps = stream.width, stream.height, float(stream.average_rate)
        metadata_count = stream.frames
        for i, frame in enumerate(video.decode(video=0)):
            pts.append(float(frame.pts * frame.time_base))
            if i in sample_indices:
                samples[i] = frame.to_ndarray(format="rgb24")
    video_length = len(pts)
    pts = np.asarray(pts)
    pts_error = pts - pts[0] - np.arange(video_length)/30
    raw_size = np.asarray(calib.get_image_size())
    native_video_calib = None
    if np.array_equal(raw_size, [2880, 2880]) and (width, height) == (1408, 1408):
        native_video_calib = calib.rescale(np.array([width, height], np.int32), .5, np.array([32., 32.]))
    elif np.array_equal(raw_size, [width, height]):
        native_video_calib = calib
    projection_frames = []
    for idx in sample_indices:
        if idx not in samples or not valid[idx] or native_video_calib is None:
            continue
        image = Image.fromarray(samples[idx])
        draw = ImageDraw.Draw(image)
        wc = np.linalg.inv(camera[idx])
        points = joints[idx] @ wc[:3, :3].T + wc[:3, 3]
        masks = []
        for j, point in enumerate(points):
            projected = native_video_calib.project(point)
            if projected is None:
                masks.append(False)
                continue
            # Official HM camera helper documents stored images as clockwise
            # 90 degrees from calibration-native. Pixel-centre rule uses H-1.
            u, v = height - 1 - float(projected[1]), float(projected[0])
            inside = 0 <= u < width and 0 <= v < height
            masks.append(inside)
            if inside:
                color = "yellow" if j < 25 else "red" if j < 46 else "cyan"
                draw.ellipse((u-3, v-3, u+3, v+3), fill=color)
        draw.rectangle((0, 0, width, 42), fill="black")
        draw.text((8, 8), f"{uid} frame={idx}: HM prediction; yellow body/red L/cyan R", fill="white")
        image.thumbnail((704, 704))
        image.save(out / f"overlay-{idx:06d}.png")
        projection_frames.append({"frame_index": idx, "body_in_image": int(sum(masks[:25])),
                                  "left_hand_in_image": int(sum(masks[25:46])),
                                  "right_hand_in_image": int(sum(masks[46:67]))})
    # This is an observed foot-height proxy, not a stationary-foot ground fit.
    feet = joints[valid_body][:, [19, 20, 21, 22, 23, 24], 2]
    proxy_floor = float(np.quantile(feet, .05)) if feet.size else 0.
    head_proxy = joints[:, 0]  # OpenPose nose; explicitly not SMPL anatomical head.
    head_distance = np.linalg.norm(head_proxy-camera[:, :3, 3], axis=-1)
    rot = camera[valid, :3, :3]
    so3_error = np.abs(np.swapaxes(rot, -1, -2) @ rot - np.eye(3)).max() if len(rot) else 1.
    so3_det = np.linalg.det(rot) if len(rot) else np.zeros(0)
    np.savez_compressed(out / "candidate-camera.npz", timestamp=np.arange(T)/30,
                        camera_world=camera, camera_valid=valid, head_world=cpf,
                        head_valid=valid, capture_timestamp_ns=stamps,
                        pose_time_error_s=errors, pose_timestamp_ns=pose_times_ns)
    candidate_pass = bool(monotonic and valid.all() and np.max(np.abs(errors)) < .02
                          and all(x["passed"] for x in exo) and duplicate
                          and all(x["passed"] for x in reprojection)
                          and video_length >= T and np.max(np.abs(pts_error)) < 1e-4
                          and so3_error < 1e-6 and np.allclose(so3_det, 1, atol=1e-6))
    after = [identity(p) for p in source_paths]
    report = {
        "take_name": uid, "hm_frames": T, "hm_valid_ratio": float(valid_body.mean()),
        "native_camera": ego[0], "camera_pose_basis": "aria_rgb_native",
        "camera_origin": "camera-rgb optical center; T_world_device @ T_device_camera",
        "cpf_is_separate_from_anatomical_head": True,
        "mps_graph_uids": sorted(graph_uids), "gravity_error_m_s2": stats(gravity_error),
        "published_camera_0_is_duplicate_exo": duplicate, "exo_world_calibration_checks": exo,
        "external_published_2d_reprojection": reprojection,
        "production_prepare_aria_length_check_passed": strict_length_pass,
        "production_adapter_acceptance": "passed" if strict_length_pass else "failed",
        "production_prepare_aria_probe": strict_probe,
        "alignment_candidate": {"policy": "first_HM_T_frames_of_published_timesync_and_video",
            "status": "supported_by_official_num_frames_prefix; not_full_release_certification",
            "official_commit": OFFICIAL_COMMIT,
            "reference": "slahmr/util/egoexo4d_utils.py:327 timestamps[:num_frames]",
            "timesync_frames": int(len(stamps_full)), "excess_trailing_frames": int(len(stamps_full)-T),
            "strictly_monotonic_valid_timestamps": monotonic,
            "pose_valid_frames": int(valid.sum()), "pose_time_error_s": stats(np.abs(errors[valid]))},
        "video": {"path": str(video_path), "width": width, "height": height,
            "fps": fps, "decoded_frames": video_length, "metadata_frames": metadata_count,
            "pts_nominal_30hz_error_s": stats(np.abs(pts_error)),
            "calibration_native_size": raw_size.tolist(),
            "crop_scale_candidate": "crop 32px each native edge then 0.5 scale; pixel-centre rescale",
            "rotation_candidate": "native to stored image clockwise90 (official HM helper)",
            "calibration_geometry_supported": native_video_calib is not None,
            "rotation_independently_certified_from_image_annotations": False},
        "camera_numerical_checks": {"so3_orthogonality_max_error": float(so3_error),
            "so3_det": stats(so3_det), "camera_z_m": stats(camera[valid, 2, 3]),
            "camera_z_minus_foot_p05_proxy_m": stats(camera[valid, 2, 3]-proxy_floor),
            "proxy_is_not_stationary_ground_estimate": True,
            "camera_to_openpose_nose_distance_m": stats(head_distance[valid & valid_body])},
        "candidate_hand_body_projection_samples": projection_frames,
        "image_annotation_reprojection_acceptance": "not_available; overlays_require_review",
        "candidate_camera_numerical_checks_passed": candidate_pass,
        "source_files_before": before, "source_files_after": after,
        "source_files_unchanged": before == after,
        "references": references,
        "production_ready": False,
        "passed": bool(candidate_pass and strict_length_pass and before == after),
    }
    save_json(out / "report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--take", action="append")
    args = p.parse_args()
    root = args.source_root.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any((out / uid).exists() for uid in args.take or DEFAULT_TAKES):
        raise ValueError("Refusing to reuse existing per-take reports")
    reference_path = out / "references/manifest.json"
    references = json.loads(reference_path.read_text())
    if references["commit"] != OFFICIAL_COMMIT:
        raise ValueError("Official reference commit mismatch")
    for entry in references["files"]:
        if digest(entry["local_path"]) != entry["sha256"]:
            raise ValueError("Official reference file changed")
    takes_path = root / "EgoExo4D/takes.json"
    takes_before = identity(takes_path)
    takes = {x["take_name"]: x for x in json.loads(takes_path.read_text())}
    reports = [audit_take(root, takes[uid], out, references) for uid in args.take or DEFAULT_TAKES]
    takes_after = identity(takes_path)
    implementation = [identity(Path(__file__).resolve().parents[1] / "src/egomotion_pipeline" / name)
                      for name in ("aria.py", "adapters.py", "schema.py", "geometry.py")]
    summary = {"schema": "egomotion-hm-camera-research-pilot-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(), "python": sys.executable,
        "packages": {name: importlib.metadata.version(name) for name in ["projectaria-tools", "numpy", "pyarrow", "av", "pillow"]},
        "script": identity(__file__), "production_implementation": implementation,
        "source_root": str(root),
        "takes_metadata_before": takes_before, "takes_metadata_after": takes_after,
        "take_names": [x["take_name"] for x in reports],
        "candidate_numerical_checks_passed": all(x["candidate_camera_numerical_checks_passed"] for x in reports),
        "production_prepare_aria_length_checks_passed": all(x["production_prepare_aria_length_check_passed"] for x in reports),
        "source_files_unchanged": takes_before == takes_after and all(x["source_files_unchanged"] for x in reports),
        "passed": takes_before == takes_after and all(x["passed"] for x in reports), "production_ready": False,
        "reports": [str(out / x["take_name"] / "report.json") for x in reports],
        "limitations": ["Official num_frames prefix is a candidate; released HM has no source-frame mapping.",
            "Published joints2d tests calibration consistency, not independent image annotation accuracy.",
            "Crop and CW90 projection overlays need manual/annotation review.",
            "No model asset decoded here; dense22 SMPL-H body pilot is separate.",
            "No archive, source asset, model, DINO feature, split or normalization stat is modified."]}
    save_json(out / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ["passed", "production_ready", "candidate_numerical_checks_passed", "production_prepare_aria_length_checks_passed", "reports"]}, indent=2))


if __name__ == "__main__":
    main()
