"""Read-only EgoStandard SMPL-X body22 fitting pilot; isolated research output.

Uses projected Parquet columns, stable model/code fingerprints, disjoint shape
calibration and pose windows, and matched ablations. This is offline label
fitting, not forecasting validation or certification of native joint axes.
"""

import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_key, "2")

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import torch

from egomotion_pipeline.geometry import quat_to_matrix
from egomotion_pipeline.schema import BODY_NAMES, PARENTS, POSITION_SHAPES, ROTATION_SHAPES
from egomotion_pipeline.storage import atomic_json as _atomic_json
from egomotion_pipeline.windows import prepare_window


DEFAULT_ARCHIVE = Path("/gaozt-test1/datasets/MixEgoMotion")
DEFAULT_MODEL = Path("/gaozt-test1/guanzerong/UniEgoMotion/body_models/smplx/SMPLX_NEUTRAL.npz")
DEFAULT_SELECTION = DEFAULT_ARCHIVE / ".pipeline/axis-research/calibration-24operators-reviewed/manifest.json"
MODES = ("positions_only", "positions_plus_rotation_increment")


def json_safe(value):
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key:json_safe(item) for key,item in value.items()}
    if isinstance(value, (list,tuple)):
        return [json_safe(item) for item in value]
    return value


def atomic_json(path,value):
    _atomic_json(path,json_safe(value))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def array_digest(arrays):
    h = hashlib.sha256()
    for key, value in sorted(arrays.items()):
        value = np.ascontiguousarray(value)
        h.update(key.encode()); h.update(str(value.dtype).encode())
        h.update(str(value.shape).encode()); h.update(value.tobytes())
    return h.hexdigest()


def stat_identity(path):
    st = Path(path).stat()
    return dict(size=st.st_size, mtime_ns=st.st_mtime_ns, ctime_ns=st.st_ctime_ns,
                inode=st.st_ino, device=st.st_dev)


def read_projected(path):
    columns = ["timestamp", "observation.source_timestamp", "observation.source_frame_left",
               "observation.source_frame_right", "observation.environment.floor_height",
               "observation.valid.floor", "observation.valid.head_reference_basis"]
    shapes = {}
    for key, shape in POSITION_SHAPES.items():
        shapes[f"observation.state.{key}_world"] = (shape[0], 3)
        shapes[f"observation.valid.{key}_position"] = (shape[0],)
        shapes[f"observation.interpolated.{key}_position"] = (shape[0],)
    for key, shape in ROTATION_SHAPES.items():
        shapes[f"observation.state.{key}_world_rotation"] = (shape[0], 4)
        shapes[f"observation.valid.{key}_rotation"] = (shape[0],)
        shapes[f"observation.interpolated.{key}_rotation"] = (shape[0],)
    columns.extend(shapes)
    before = stat_identity(path)
    table = pq.read_table(path, columns=columns)
    arrays = {}
    for key in columns:
        value = table[key].combine_chunks()
        raw = value.values.to_numpy(zero_copy_only=False) if key in shapes and shapes[key] != (1,) else value.to_numpy(zero_copy_only=False)
        # Archive flattens states, but even a width-one state is scalar.
        if key in shapes and shapes[key] == (1,) and hasattr(value, "values"):
            raw = value.values.to_numpy(zero_copy_only=False)
        arrays[key] = raw.reshape(len(table), *shapes[key]) if key in shapes else raw.copy()
    if before != stat_identity(path):
        raise RuntimeError("Source Parquet changed during column-projected read")
    return arrays, before


def summarise(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return dict(count=int(values.size), mean=float(values.mean()) if values.size else None,
                p50=float(np.quantile(values, .5)) if values.size else None,
                p95=float(np.quantile(values, .95)) if values.size else None,
                max=float(values.max()) if values.size else None)


def angle_deg(rotations):
    cosine = (np.trace(rotations, axis1=-2, axis2=-1)-1)/2
    return np.degrees(np.arccos(np.clip(cosine, -1, 1)))


def candidate_window(arrays, begin, end, count, stride=2):
    choices = []
    span = (count-1)*stride+1
    for start in range(begin + begin % 2, end-span+1, 30):
        ix = start + np.arange(count)*stride
        mask = arrays["observation.valid.body_position"][ix]
        rmask = arrays["observation.valid.body_rotation"][ix]
        floor = (arrays["observation.valid.floor"][ix]
                 & arrays["observation.valid.head_reference_basis"][ix]
                 & arrays["observation.valid.head_reference_floor_position"][ix,0]
                 & arrays["observation.valid.head_reference_floor_rotation"][ix,0])
        camera = arrays["observation.valid.head_reference_position"][ix, 0] & arrays["observation.valid.head_reference_rotation"][ix, 0]
        if mask.mean() < .92 or rmask.mean() < .92 or not floor.all() or not camera.all() or not mask[:, 0].all():
            continue
        dt = np.diff(arrays["observation.source_timestamp"][ix])
        if not (np.isfinite(dt).all() and np.all(dt > 0) and np.all(dt <= .09)):
            continue
        p = arrays["observation.state.body_world"][ix]
        rel = p[:, [20, 21]] - p[:, :1]
        motion_mask = mask[1:, [20, 21]] & mask[:-1, [20, 21]]
        speed = np.linalg.norm(np.diff(rel, axis=0), axis=-1)/dt[:, None]
        motion = float(np.quantile(speed[motion_mask], .75)) if motion_mask.any() else 0.
        score = float(mask.mean()+.04*np.tanh(motion/.5))
        choices.append((score, start, dict(position_valid_fraction=float(mask.mean()),
                        rotation_valid_fraction=float(rmask.mean()), wrist_relative_speed_p75_m_s=motion)))
    if not choices:
        return None
    return max(choices, key=lambda row: (row[0], -row[1]))


def select_inputs(args, output):
    selection_bytes = args.selection.read_bytes()
    selected_sources = json.loads(selection_bytes)["selected_sources"]
    records, rejected, fingerprints = [], [], []
    for source in selected_sources:
        index = int(source["episode_index"])
        record_path = args.archive/f".pipeline/episodes/{index:08d}.json"
        record_bytes = record_path.read_bytes()
        record = json.loads(record_bytes)
        if record["source"] != "egosuite" or record["source_uid"] != source["uid"]:
            raise ValueError("Selection no longer matches EgoStandard archive records")
        meta = record["metadata"]
        if (not meta.get("coordinate_verified") or meta.get("body_basis_correction") is not None
                or not meta.get("head_reference", {}).get("basis_verified")):
            rejected.append(dict(episode_index=index, reason="incompatible axes/provenance")); continue
        path = args.archive/f"data/chunk-{index//1000:03d}/file-{index%1000:03d}.parquet"
        arrays, identity = read_projected(path)
        middle = len(arrays["timestamp"])//2
        shape = candidate_window(arrays, 0, middle, args.frames)
        evaluation = candidate_window(arrays, middle, len(arrays["timestamp"]), args.frames)
        if shape is None or evaluation is None:
            rejected.append(dict(episode_index=index, reason="no two disjoint high-validity windows")); continue
        if meta.get("operator_id") in {r["operator_id"] for r in records}:
            raise ValueError("Pilot selection repeats an operator")
        label = f"{len(records):02d}-{record['source_uid']}"
        prepared = []
        # The public window reader consumes archive's flattened features;
        # the fitting/selection code above uses structured joint arrays.
        archive_columns={key:value.reshape(len(value),-1) if key.startswith("observation.state.") else value
                         for key,value in arrays.items()}
        for purpose, candidate in (("shape", shape), ("evaluation", evaluation)):
            _, start, quality = candidate
            ix = start+np.arange(args.frames)*2
            values = {key: value[ix].copy() for key, value in arrays.items()}
            window = prepare_window(archive_columns, meta, start_frame=start, num_frames=args.frames,
                                    fps=15, canonicalize=True)
            payload = dict(archive_frame_indices=ix, timestamps=values["observation.source_timestamp"],
                positions=values["observation.state.body_world"],
                source_quaternion_wxyz=values["observation.state.body_world_rotation"],
                source_global_rotations=quat_to_matrix(values["observation.state.body_world_rotation"]),
                position_mask=values["observation.valid.body_position"],
                rotation_mask=values["observation.valid.body_rotation"],
                position_interpolated=values["observation.interpolated.body_position"],
                rotation_interpolated=values["observation.interpolated.body_rotation"],
                source_frame_left=values["observation.source_frame_left"],
                source_frame_right=values["observation.source_frame_right"],
                floor_position=values["observation.state.head_reference_floor_world"][:, 0],
                floor_rotations=quat_to_matrix(values["observation.state.head_reference_floor_world_rotation"][:, 0]),
                floor_height=values["observation.environment.floor_height"],
                floor_valid=values["observation.valid.head_reference_floor_position"][:, 0]
                            & values["observation.valid.head_reference_floor_rotation"][:, 0],
                camera_world=values["observation.state.camera_world"][:, 0],
                camera_world_rotation=values["observation.state.camera_world_rotation"][:, 0],
                head_reference_world=values["observation.state.head_reference_world"][:, 0],
                head_reference_world_rotation=values["observation.state.head_reference_world_rotation"][:, 0],
                trajectory_18d=window.trajectory, trajectory_valid=window.trajectory_valid,
                window_from_archive_world=np.asarray(window.metadata["training_window"]["window_from_archive_world"]),
                hand_wrist_world=np.stack([values[f"observation.state.hand_{side}_world"][:, 0] for side in ("left", "right")], axis=1),
                hand_wrist_valid=np.stack([values[f"observation.valid.hand_{side}_position"][:, 0] for side in ("left", "right")], axis=1))
            source_delta = payload["source_global_rotations"][1:] @ payload["source_global_rotations"][:-1].swapaxes(-1,-2)
            transitions = angle_deg(source_delta) <= 45.
            root_jump = np.linalg.norm(np.diff(payload["positions"][:,0],axis=0),axis=-1) > .5
            transitions[root_jump] = False
            payload["transition_mask"] = transitions
            save_path = output/"inputs"/f"{label}-{purpose}.npz"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(save_path, **payload)
            prepared.append(dict(purpose=purpose, path=str(save_path), sha256=sha(save_path),
                                 start_archive_frame=int(start), frames=args.frames, quality=quality,
                                 interpolated_position_fraction=float(payload["position_interpolated"].mean()),
                                 interpolated_rotation_fraction=float(payload["rotation_interpolated"].mean()),
                                 excluded_jump_transition_count=int((~transitions).sum())))
        records.append(dict(label=label, episode_index=index, source_uid=record["source_uid"],
                source_key=record["source_key"], operator_id=meta.get("operator_id"), task_group=source.get("task"),
                source_native_basis_verified=bool(meta.get("rotation_basis_verified")), windows=prepared))
        fingerprints.append(dict(path=str(path), stat=identity, projected_columns_sha256=array_digest(arrays),
                                 record_path=str(record_path), record_sha256=hashlib.sha256(record_bytes).hexdigest()))
        if len(records) == args.episodes:
            break
    if len(records) != args.episodes:
        raise RuntimeError(f"Only {len(records)} eligible operators, requested {args.episodes}")
    return records, rejected, fingerprints, hashlib.sha256(selection_bytes).hexdigest()


def evaluate_fit(data, result):
    p, g = result.positions, result.global_rotations
    pmask, rmask = data["position_mask"], data["rotation_mask"]
    position_errors = np.linalg.norm(p-data["positions"], axis=-1)
    dt = np.diff(data["timestamps"])
    transitions = (dt > 0) & (dt <= .09)
    src_delta = data["source_global_rotations"][1:] @ data["source_global_rotations"][:-1].swapaxes(-1, -2)
    dst_delta = g[1:] @ g[:-1].swapaxes(-1, -2)
    delta_mask = rmask[1:] & rmask[:-1] & transitions[:, None] & data["transition_mask"]
    # Reject likely source tracking jumps rather than fitting across them.
    delta_mask &= angle_deg(src_delta) <= 45.
    delta_errors = angle_deg(dst_delta @ src_delta.swapaxes(-1, -2))
    floor_r = data["floor_rotations"]
    floor_xyz = np.einsum("tji,tkj->tki", floor_r, p-data["floor_position"][:, None])
    floor_g = floor_r[:, None].swapaxes(-1, -2) @ g
    dense = np.concatenate([floor_g[..., :2, :].reshape(len(p), 22, 6), floor_xyz], axis=-1)
    inverse_xyz = np.einsum("tij,tkj->tki", floor_r, floor_xyz)+data["floor_position"][:, None]
    raw_velocity = np.linalg.norm(np.diff(data["positions"][:, [10, 11]], axis=0), axis=-1)/dt[:, None]
    fit_velocity = np.linalg.norm(np.diff(p[:, [10, 11]], axis=0), axis=-1)/dt[:, None]
    foot_mask = pmask[1:, [10, 11]] & pmask[:-1, [10, 11]] & transitions[:, None]
    # This is an observed stationary-foot proxy, not contact ground truth.
    stationary = foot_mask & (raw_velocity < .15)
    penetration = np.maximum(0, data["floor_height"][:, None]-p[:, [10, 11], 2])
    raw_penetration = np.maximum(0, data["floor_height"][:, None]-data["positions"][:, [10, 11], 2])
    wrist_error = np.linalg.norm(p[:, [20, 21]]-data["hand_wrist_world"], axis=-1)
    raw_wrist_error = np.linalg.norm(data["positions"][:, [20, 21]]-data["hand_wrist_world"], axis=-1)
    wrist_mask = pmask[:, [20, 21]] & data["hand_wrist_valid"]
    return dict(position_error_m=summarise(position_errors[pmask]),
                per_joint_position_error_m={name:summarise(position_errors[pmask[:, j], j]) for j,name in enumerate(BODY_NAMES)},
                rotation_increment_error_deg=summarise(delta_errors[delta_mask]),
                per_joint_rotation_increment_error_deg={name:summarise(delta_errors[delta_mask[:, j], j]) for j,name in enumerate(BODY_NAMES)},
                fitted_foot_penetration_m=summarise(penetration[pmask[:, [10, 11]]]),
                observed_foot_penetration_m=summarise(raw_penetration[pmask[:, [10, 11]]]),
                fitted_stationary_foot_speed_m_s=summarise(fit_velocity[stationary]),
                observed_stationary_foot_speed_m_s=summarise(raw_velocity[stationary]),
                fitted_hand_body_wrist_offset_m=summarise(wrist_error[wrist_mask]),
                observed_hand_body_wrist_offset_m=summarise(raw_wrist_error[wrist_mask]),
                floor_xyz_roundtrip_max_m=float(np.max(np.abs(inverse_xyz-p))),
                so3_orthogonality_max=float(np.max(np.abs(g.swapaxes(-1,-2)@g-np.eye(3)))),
                minimum_rotation_determinant=float(np.min(np.linalg.det(g)))), dense, position_errors, delta_errors, delta_mask


def gauge_diagnostic(model, result):
    """Exhibit two absolute orientations with identical FK XYZ/world deltas."""
    from scipy.spatial.transform import Rotation
    g = result.global_rotations.copy()
    rest = model.rest_joints(torch.as_tensor(result.betas, dtype=model.dtype, device=model.device)).detach().cpu().numpy()
    transformed = g.copy()
    for j in (10, 11, 15):
        transformed[:, j] = g[:, j] @ Rotation.from_rotvec([0, 0, np.pi/2]).as_matrix()
    j, child = 18, 20
    axis = rest[child]-rest[j]; axis /= np.linalg.norm(axis)
    transformed[:, j] = g[:, j] @ Rotation.from_rotvec(axis*np.pi/2).as_matrix()
    local = transformed.copy()
    for k in range(1,22):
        local[:, k] = transformed[:, PARENTS[k]].swapaxes(-1,-2) @ transformed[:, k]
    p2, g2 = model.forward(torch.as_tensor(local, dtype=model.dtype, device=model.device),
                          torch.as_tensor(result.root_world, dtype=model.dtype, device=model.device),
                          torch.as_tensor(result.betas, dtype=model.dtype, device=model.device))
    p2, g2 = p2.detach().cpu().numpy(), g2.detach().cpu().numpy()
    d1 = g[1:] @ g[:-1].swapaxes(-1,-2)
    d2 = g2[1:] @ g2[:-1].swapaxes(-1,-2)
    changed = (10,11,15,18)
    return dict(changed_joints=[BODY_NAMES[j] for j in changed], fixed_twist_deg=90,
        fk_xyz_max_difference_m=float(np.max(np.abs(p2-result.positions))),
        world_rotation_increment_max_difference_deg=float(np.max(angle_deg(d2@d1.swapaxes(-1,-2)))),
        absolute_rotation_difference_deg=summarise(angle_deg(g2[:, changed]@g[:, changed].swapaxes(-1,-2))),
        interpretation="XYZ and world rotation increments do not uniquely authenticate absolute twist; priors may select a gauge")


def plot_summary(output, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = list(dict.fromkeys(r["label"] for r in results))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for mode, color, offset in zip(MODES, ("#6485ad", "#e58b41"), (-.18,.18)):
        rows = [r for r in results if r["mode"] == mode]
        x = np.arange(len(rows))+offset
        metrics = [r["metrics"] for r in rows]
        axes[0,0].bar(x, [m["position_error_m"]["mean"]*100 for m in metrics], .36, color=color, label=mode)
        axes[0,1].bar(x, [m["rotation_increment_error_deg"]["p95"] for m in metrics], .36, color=color)
        axes[1,0].bar(x, [m["fitted_foot_penetration_m"]["p95"]*100 for m in metrics], .36, color=color)
        axes[1,1].bar(x, [r["fit_metadata"]["elapsed_seconds"] for r in rows], .36, color=color)
    for ax, title in zip(axes.flat, ("Mean observed XYZ residual (cm)", "World rotation increment residual P95 (deg)",
                                    "Foot penetration proxy P95 (cm)", "Pose optimization time (seconds)")):
        ax.set_title(title); ax.set_xticks(np.arange(len(labels)), [str(i+1) for i in range(len(labels))])
        ax.set_xlabel("Pilot operator / episode"); ax.grid(axis="y", alpha=.2)
    fig.legend(*axes[0,0].get_legend_handles_labels(), loc="outside lower center", ncol=2)
    fig.suptitle("EgoStandard -> SMPL-X body22: offline fitting pilot (estimated labels)")
    fig.savefig(output/"comparison.png", dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--shape-iterations", type=int, default=150)
    parser.add_argument("--rotation-weight", type=float, default=.02)
    args = parser.parse_args()
    if not 1 <= args.episodes <= 24 or not 2 <= args.frames <= 240 or args.iterations < 1:
        parser.error("Pilot requires 1..24 operators, 2..240 frames, positive iterations")
    if args.output.exists():
        parser.error("Use a new isolated output directory")
    if args.output.resolve() == args.archive.resolve() or args.archive.resolve() in args.output.resolve().parents and ".pipeline" not in args.output.resolve().parts:
        parser.error("Research output must not overwrite archive data or meta")
    args.output.mkdir(parents=True)
    log = (args.output/"process.log").open("a",buffering=1)
    class Tee:
        def __init__(self,stream): self.stream=stream
        def write(self,value): self.stream.write(value); log.write(value)
        def flush(self): self.stream.flush(); log.flush()
    sys.stdout=Tee(sys.stdout)
    sys.stderr=Tee(sys.stderr)
    torch.set_num_threads(2)
    started = time.monotonic()
    status_path = args.output/"status.json"
    atomic_json(status_path, dict(stage="preparing", pid=os.getpid(), started_at=datetime.now(timezone.utc).isoformat()))
    from egomotion_pipeline import smpl_fit
    code = {"runner.py":Path(__file__), "smpl_fit.py":Path(smpl_fit.__file__)}
    for name in ("windows.py","geometry.py","head_reference.py","ground.py","schema.py","storage.py"):
        code[name]=Path(smpl_fit.__file__).parent/name
    code_sha = {}
    for name,path in code.items():
        destination = args.output/"code"/name; destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(path,destination); code_sha[str(path.resolve())] = sha(destination)
    try:
        records, rejected, fingerprints, selection_sha = select_inputs(args,args.output)
        config = smpl_fit.FitConfig(pose_iterations=args.iterations, shape_iterations=args.shape_iterations,
                                   rotation_increment_weight=args.rotation_weight,max_dt=.09)
        model = smpl_fit.load_body22_model(args.model, shape_components=10, device="cpu", dtype=torch.float64)
        manifest = dict(schema="egomotion-smpl-fit-pilot-v1", created_at=datetime.now(timezone.utc).isoformat(),
            archive=str(args.archive), selection=str(args.selection), selection_sha256=selection_sha,
            model=model.metadata, configuration=asdict(config), operators=len(records), episodes=records,
            rejected_sources=rejected, source_fingerprints=fingerprints, code_sha256=code_sha,
            torch_version=torch.__version__, cpu_threads=2, fps=15, body_names=BODY_NAMES,
            sampling="Separate first-half shape and second-half pose windows; grid search for valid moving windows",
            rotation_label_source="smplx_joints_only_temporal_fit", rotation_estimated=True,
            source_native_basis_verified=False, independent_absolute_rotation_reference=False,
            target_axis_convention_defined=True, smplh_hm_validation_performed=False,
            learned_anatomical_pose_prior_used=False, extra_hand_or_head_mount_anchors_used=False,
            source_jump_screening={"max_joint_delta_deg":45,"max_root_step_m":.5,
                "meaning":"conservative edge filter; does not establish a tracking reset"},
            production_archive_modified=False,
            evaluation_semantics="Offline fitting residuals on observed motion; not prediction error or native axis certification")
        atomic_json(args.output/"manifest.json",manifest)
        results=[]; gauges=[]
        all_p={mode:[] for mode in MODES}; all_r={mode:[] for mode in MODES}
        for i,record in enumerate(records):
            with np.load(record["windows"][0]["path"],allow_pickle=False) as archive:
                shape_data = {k:archive[k] for k in archive.files}
            shape = smpl_fit.fit_shape_from_positions(model,shape_data["positions"],shape_data["position_mask"],config=config)
            atomic_json(args.output/"results"/f"{record['label']}-shape.json",shape)
            with np.load(record["windows"][1]["path"],allow_pickle=False) as archive:
                data={k:archive[k] for k in archive.files}
            init=None
            for mode in MODES:
                print(f"[{i+1}/{len(records)}] {record['label']} {mode}",flush=True)
                fit=smpl_fit.fit_body22(model,data["positions"],data["position_mask"],data["timestamps"],
                    source_global_rotations=data["source_global_rotations"],rotation_mask=data["rotation_mask"],
                    mode=mode, config=config, initialization=init, fixed_betas=shape["betas"],
                    transition_mask=data["transition_mask"])
                if init is None: init=fit.initialization
                metrics,dense,perrors,rerrors,rmask=evaluate_fit(data,fit)
                row=dict(label=record["label"],episode_index=record["episode_index"],mode=mode,
                         metrics=metrics,fit_metadata=fit.metadata,shape_metadata=shape["metadata"],history=fit.history)
                results.append(row)
                atomic_json(args.output/"results"/f"{record['label']}-{mode}.json",row)
                np.savez_compressed(args.output/"results"/f"{record['label']}-{mode}.npz",
                    fitted_world_xyz=fit.positions,fitted_world_rotations=fit.global_rotations,
                    local_rotations=fit.local_rotations,root_world=fit.root_world,smpl_trans=fit.smpl_trans,
                    betas=fit.betas,body_floor_rot6d_xyz=dense.astype(np.float32),
                    observed_position_mask=data["position_mask"],observed_rotation_mask=data["rotation_mask"],
                    observed_rotation_increment_mask=rmask,archive_frame_indices=data["archive_frame_indices"],
                    body_floor_position_valid=data["position_mask"] & data["floor_valid"][:,None],
                    body_floor_rotation_estimate_available=np.isfinite(fit.global_rotations).all((-2,-1)) & data["floor_valid"][:,None],
                    rotation_estimated=np.asarray(True),source_native_basis_verified=np.asarray(False),
                    timestamps=data["timestamps"],trajectory_18d=data["trajectory_18d"],trajectory_valid=data["trajectory_valid"],
                    camera_world=data["camera_world"],camera_world_rotation=data["camera_world_rotation"],
                    floor_position=data["floor_position"],floor_rotations=data["floor_rotations"])
                all_p[mode].append(perrors[data["position_mask"]]);all_r[mode].append(rerrors[rmask])
                if mode==MODES[1] and i==0: gauges.append(gauge_diagnostic(model,fit))
                atomic_json(status_path,dict(stage="fitting",pid=os.getpid(),completed_fits=len(results),total_fits=len(records)*2,
                    elapsed_seconds=time.monotonic()-started,last_episode=record["label"],last_mode=mode))
                print(json.dumps(dict(position_mean_cm=metrics["position_error_m"]["mean"]*100,
                    rotation_increment_p95_deg=metrics["rotation_increment_error_deg"]["p95"],
                    elapsed_seconds=fit.metadata["elapsed_seconds"])),flush=True)
        integrity=[]
        for fingerprint in fingerprints:
            arrays,current=read_projected(fingerprint["path"])
            okay=(current==fingerprint["stat"] and array_digest(arrays)==fingerprint["projected_columns_sha256"]
                  and sha(fingerprint["record_path"])==fingerprint["record_sha256"])
            integrity.append(dict(path=fingerprint["path"],unchanged=okay))
        source_code_unchanged=all(sha(path)==expected for path,expected in code_sha.items())
        model_unchanged=sha(args.model)==model.metadata["model_sha256"]
        if not all(row["unchanged"] for row in integrity) or not source_code_unchanged or not model_unchanged:
            raise RuntimeError("Input archive/code changed during pilot; results cannot be accepted")
        aggregates={mode:dict(position_error_m=summarise(np.concatenate(all_p[mode])),
                              rotation_increment_error_deg=summarise(np.concatenate(all_r[mode]))) for mode in MODES}
        report=dict(manifest_path=str(args.output/"manifest.json"),operators=len(records),unique_frames=len(records)*args.frames*2,
                    evaluated_frames_per_mode=len(records)*args.frames,aggregates=aggregates,results=results,
                    gauge_diagnostic=gauges,source_integrity=integrity,source_code_unchanged=source_code_unchanged,
                    model_unchanged=model_unchanged,elapsed_seconds=time.monotonic()-started,
                    native_axes_certified=False,full_22_absolute_rotation_accuracy_established=False,
                    conclusion="Numerical feasibility pilot only; inspect fit/physics residuals before production labels")
        atomic_json(args.output/"report.json",report)
        plot_summary(args.output,results)
        lines=["# EgoStandard → SMPL-X dense22 小规模离线拟合验证", "",
               "仅验证模型 FK 与时序拟合的数值行为；旋转是估计标签，不认证 EgoSuite 原始关节轴。", "",
               f"{len(records)} 位 operator，{len(records)*args.frames*2} 个独立 15Hz 帧；每段前半部估计 shape，后半部拟合 pose。",
               "两组消融使用相同 shape 和相同初始姿态。所有误差在原归档世界系测量，不做 Procrustes 对齐。", "",
               "| 模式 | XYZ 平均残差 cm | XYZ P95 cm | 世界旋转增量平均残差 ° | 增量 P95 ° |",
               "|---|---:|---:|---:|---:|"]
        for mode,m in aggregates.items():
            p,r=m["position_error_m"],m["rotation_increment_error_deg"]
            lines.append(f"| {mode} | {p['mean']*100:.3f} | {p['p95']*100:.3f} | {r['mean']:.3f} | {r['p95']:.3f} |")
        lines += ["", "模型资产：SMPL-X neutral，前 10 个 shape 系数；不是 HM 所需 SMPL-H 模型。",
                  "未使用学习人体姿态先验、额外手掌锚点、相机–头部安装标定或 RGB 重投影约束。",
                  "头/脚等绝对姿态和单骨段 twist 不可由 XYZ 与世界旋转增量唯一确定；见 gauge_diagnostic。",
                  "source_native_basis_verified 保持 false；模型轴定义明确不等于旋转标签是真值。",
                  "相机轨迹、地面、18D 条件及帧序保留；DINO 没有重提取。", "",
                  f"数据完整性：{len(integrity)}/{len(integrity)} 段源运动列与记录保持不变；代码指纹保持不变。",
                  "逐关节残差、脚部代理指标、手/身体腕点偏差、每段参数与 mask 见 report.json 和 results/。",
                  "stationary foot 指标仅使用原始脚速度 <0.15m/s 的代理帧，不是人工接触标注。", "",
                  "![comparison](comparison.png)"]
        (args.output/"report.md").write_text("\n".join(lines)+"\n")
        atomic_json(status_path,dict(stage="completed",pid=os.getpid(),completed_fits=len(results),total_fits=len(results),
                    elapsed_seconds=time.monotonic()-started,report=str(args.output/"report.json")))
        print("Completed: "+str(args.output),flush=True)
    except BaseException as exc:
        atomic_json(status_path,dict(stage="failed",pid=os.getpid(),error=type(exc).__name__+": "+str(exc),
                                    elapsed_seconds=time.monotonic()-started))
        raise


if __name__=="__main__":
    main()
