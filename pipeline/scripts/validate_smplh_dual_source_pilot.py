#!/usr/bin/env python3
"""Read-only SMPL-H retargeting experiment; never publishes training labels.

Center calibration uses five train operators, then freezes its offsets for
five other operators. Held-out poses still fit their observed motion: these
are reconstruction residuals, not prediction accuracy or native-axis truth.
"""

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
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from egomotion_pipeline.schema import BODY_NAMES
from egomotion_pipeline.smpl_fit import initialize_local_rotations, rotation_increment_errors_deg
from validate_smpl_fit_pilot import select_inputs, read_projected


DEFAULT_ARCHIVE = Path("/gaozt-test1/datasets/MixEgoMotion")
DEFAULT_MODEL = DEFAULT_ARCHIVE / "weights/smplh/male/model.npz"
DEFAULT_SELECTION = DEFAULT_ARCHIVE / ".pipeline/axis-research/calibration-24operators-reviewed/manifest.json"
DEFAULT_SPLITS = Path(__file__).resolve().parents[2] / "MotionExpert/exp/dense_k12/splits.json"


def json_safe(value):
    if hasattr(value, "as_dict"):
        return json_safe(value.as_dict())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def summary(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0, "mean": None, "p95": None, "max": None}
    return dict(count=len(values), mean=float(values.mean()),
                p95=float(np.percentile(values, 95)), max=float(values.max()))


def prepare_inputs(args):
    output = args.output
    if (output / "ego-input-manifest.json").exists():
        raise ValueError("Use a new pilot output; input snapshots must not silently be reused")
    records, rejected, fingerprints, selection_sha = select_inputs(args, output)
    splits = json.loads(args.splits.read_text())["splits"]
    memberships = {uid: split for split, uids in splits.items() for uid in uids}
    train = [r for r in records if memberships[r["source_key"]] == "train"]
    if len(train) < args.calibration_operators:
        raise ValueError("Not enough train operators for center calibration")
    calibration_ids = {r["operator_id"] for r in train[:args.calibration_operators]}
    if len(calibration_ids) != args.calibration_operators:
        raise ValueError("Center calibration repeats an operator")
    for record in records:
        record["original_split"] = memberships[record["source_key"]]
        record["pilot_partition"] = ("calibration_operator" if record["operator_id"] in calibration_ids
                                      else "heldout_operator")
        index = int(record["episode_index"])
        arrays, _ = read_projected(args.archive / f"data/chunk-{index//1000:03d}/file-{index%1000:03d}.parquet")
        for window in record["windows"]:
            payload = load_npz(window["path"])
            ix = payload["archive_frame_indices"]
            for side in ("left", "right"):
                payload[f"hand_{side}_world"] = arrays[f"observation.state.hand_{side}_world"][ix]
                payload[f"hand_{side}_valid"] = arrays[f"observation.valid.hand_{side}_position"][ix]
            np.savez_compressed(window["path"], **payload)
            window["sha256"] = sha(window["path"])
    heldout_ids = {r["operator_id"] for r in records if r["pilot_partition"] == "heldout_operator"}
    if calibration_ids & heldout_ids:
        raise AssertionError("Operator leakage")
    manifest = dict(format="smplh-dual-source-ego-inputs-v1", archive=str(args.archive),
        selection=str(args.selection), selection_sha256=selection_sha, source_fingerprints=fingerprints,
        split_manifest_sha256=sha(args.splits), records=records, rejected=rejected,
        calibration_operator_ids=sorted(calibration_ids), heldout_operator_ids=sorted(heldout_ids),
        fps=15, frames_per_window=args.frames, production_archive_modified=False,
        dino_values_read=False, native_axes_certified=False,
        sampling="Fresh projected reads of v5; two non-overlapping high-validity windows per operator",
        created_at=datetime.now(timezone.utc).isoformat())
    write_json(output / "ego-input-manifest.json", manifest)
    return manifest


def input_arguments(data):
    return dict(positions=data["positions"], position_mask=data["position_mask"],
        timestamps=data["timestamps"], source_global_rotations=data["source_global_rotations"],
        rotation_mask=data["rotation_mask"], transition_mask=data["transition_mask"])


def export_result(path, data, result):
    p, g = result.positions, result.global_rotations
    world = np.concatenate((g[..., :2, :].reshape(len(p), 22, 6), p), axis=-1)
    floor_r, floor_p = data["floor_rotations"], data["floor_position"]
    floor_xyz = np.einsum("tji,tkj->tki", floor_r, p-floor_p[:, None])
    floor_rotation = floor_r[:, None].swapaxes(-1, -2) @ g
    floor = np.concatenate((floor_rotation[..., :2, :].reshape(len(p), 22, 6), floor_xyz), axis=-1)
    floor_valid = data["floor_valid"][:, None] & data["position_mask"]
    inverse = np.einsum("tij,tkj->tki", floor_r, floor_xyz)+floor_p[:, None]
    np.savez_compressed(path, canonical_world_rot6d_xyz=world.astype(np.float32),
        canonical_floor_rot6d_xyz=np.where(floor_valid[..., None], floor, 0).astype(np.float32),
        canonical_world_rotations=g, canonical_local_rotations=result.local_rotations,
        source_world_xyz=data["positions"], source_global_rotations=data["source_global_rotations"],
        source_position_valid=data["position_mask"], source_rotation_valid=data["rotation_mask"],
        canonical_floor_reference_valid=data["floor_valid"],
        canonical_rotation_certified=np.zeros_like(data["rotation_mask"]),
        reconstructed_observations=result.reconstructed_observations,
        betas=result.betas, root_world=result.root_world, smpl_trans=result.smpl_trans,
        archive_frame_indices=data["archive_frame_indices"], timestamps=data["timestamps"],
        trajectory_18d=data["trajectory_18d"], trajectory_valid=data["trajectory_valid"],
        hand_left_world=data["hand_left_world"], hand_right_world=data["hand_right_world"],
        hand_left_valid=data["hand_left_valid"], hand_right_valid=data["hand_right_valid"])
    return dict(world_shape=list(world.shape), floor_shape=list(floor.shape),
        floor_inverse_max_abs_m=float(np.max(np.abs(inverse-p))),
        so3_orthogonality_max=float(np.max(np.abs(g.swapaxes(-1,-2) @ g-np.eye(3)))),
        min_determinant=float(np.linalg.det(g).min()), path=str(path), sha256=sha(path))


def metrics(data, result):
    valid = data["position_mask"]
    observed_errors = np.linalg.norm(result.reconstructed_observations-data["positions"], axis=-1)
    center_displacements = np.linalg.norm(result.positions-data["positions"], axis=-1)
    pairs = data["rotation_mask"][1:] & data["rotation_mask"][:-1] & data["transition_mask"]
    pairs &= (np.diff(data["timestamps"]) <= .09)[:, None]
    rerrors = rotation_increment_errors_deg(result.global_rotations, data["source_global_rotations"], pairs)
    wrist_valid = valid[:, [20,21]] & data["hand_wrist_valid"]
    wrist_errors = np.linalg.norm(result.positions[:, [20,21]]-data["hand_wrist_world"], axis=-1)
    penetration = np.maximum(0, data["floor_height"][:, None]-result.positions[:, [10,11],2])
    foot_valid = valid[:, [10,11]] & data["floor_valid"][:, None]
    return dict(observation_reconstruction_error_m=summary(observed_errors[valid]),
        canonical_to_native_center_distance_m=summary(center_displacements[valid]),
        per_joint_observation_error_m={name:summary(observed_errors[valid[:,j],j]) for j,name in enumerate(BODY_NAMES)},
        per_joint_canonical_to_native_center_distance_m={name:summary(center_displacements[valid[:,j],j]) for j,name in enumerate(BODY_NAMES)},
        world_rotation_increment_error_deg=summary(rerrors),
        canonical_body_to_native_hand_wrist_distance_m=summary(wrist_errors[wrist_valid]),
        floor_penetration_proxy_m=summary(penetration[foot_valid])), observed_errors[valid], rerrors


def run(args, manifest):
    from egomotion_pipeline.smplh_retarget import (RetargetConfig, load_smplh_body22_model,
        calibrate_center_offsets, fit_smplh_retarget, pool_center_calibrations,
        fit_safe_limb_betas)
    output = args.output
    model = load_smplh_body22_model(args.model, shape_components=16)
    config = RetargetConfig(pose_iterations=args.iterations, shape_iterations=0, max_dt=.09)
    beta = np.zeros(model.shape_components)
    write_json(output / "ego-experiment-contract.json", dict(model=model.metadata,
        config=asdict(config), fixed_betas=beta, shape_policy="zero_16d_shape_center_isolation_ablation",
        independent_absolute_rotation_reference=False, target_model_axes_defined=True,
        canonical_centers_independently_verified=False, native_axes_certified=False,
        note="Fixed zero shape isolates the center-offset hypothesis; human shape variation remains unresolved"))
    calibrations, calibration_groups = [], []
    for record in manifest["records"]:
        if record["pilot_partition"] != "calibration_operator":
            continue
        data = load_npz(record["windows"][0]["path"])
        print("calibration", record["label"], flush=True)
        result = calibrate_center_offsets(model, **input_arguments(data), fixed_betas=beta, config=config)
        calibrations.append(result.center_calibration)
        calibration_groups.append(record["operator_id"])
        write_json(output / "results" / f"{record['label']}-center-calibration.json",
                   dict(center_calibration=result.center_calibration, metadata=result.metadata))
    pooled = pool_center_calibrations(calibrations, group_ids=calibration_groups)
    write_json(output / "pooled-center-calibration.json", pooled)
    rows, distributions = [], {}
    for record in manifest["records"]:
        data = load_npz(record["windows"][1]["path"])
        initialization = initialize_local_rotations(model, data["positions"], data["position_mask"], beta)
        for mode in ("baseline", "pooled_offset"):
            print("evaluation", record["label"], mode, flush=True)
            result = fit_smplh_retarget(model, **input_arguments(data), fixed_betas=beta,
                initialization=initialization, config=config,
                center_calibration=pooled if mode == "pooled_offset" else None)
            values, perror, rerror = metrics(data, result)
            export = export_result(output / "results" / f"{record['label']}-{mode}.npz", data, result)
            row = dict(label=record["label"], mode=mode, partition=record["pilot_partition"],
                original_split=record["original_split"], operator_id=record["operator_id"],
                source_key=record["source_key"], metrics=values, export=export, metadata=result.metadata)
            rows.append(row)
            key = (record["pilot_partition"], mode)
            distributions.setdefault(key, {"position":[], "rotation":[]})
            distributions[key]["position"].append(perror)
            distributions[key]["rotation"].append(rerror)
            write_json(output / "results" / f"{record['label']}-{mode}.json", row)
            write_json(output / "ego-status.json", dict(state="running", evaluations=len(rows), total=2*len(manifest["records"])))
    aggregates = {f"{partition}/{mode}":dict(
        observation_reconstruction_error_m=summary(np.concatenate(distribution["position"])),
        world_rotation_increment_error_deg=summary(np.concatenate(distribution["rotation"])))
        for (partition,mode),distribution in distributions.items()}
    subject_rows, subject_distributions = [], {}
    if args.limb_shape:
        shape_config = RetargetConfig(pose_iterations=args.iterations, shape_iterations=150, max_dt=.09)
        for record in manifest["records"]:
            calibration_data = load_npz(record["windows"][0]["path"])
            evaluation_data = load_npz(record["windows"][1]["path"])
            print("limb-shape calibration", record["label"], flush=True)
            shape = fit_safe_limb_betas(model, calibration_data["positions"],
                                        calibration_data["position_mask"], config=shape_config)
            subject_beta = shape["betas"]
            calibration_result = calibrate_center_offsets(model, **input_arguments(calibration_data),
                fixed_betas=subject_beta, config=config)
            write_json(output / "results" / f"{record['label']}-subject-calibration.json", dict(
                shape=shape, center_calibration=calibration_result.center_calibration,
                metadata=calibration_result.metadata,
                scope="Each person's calibration window only; not a cross-person shared center mapping"))
            initialization = initialize_local_rotations(model, evaluation_data["positions"],
                                                        evaluation_data["position_mask"], subject_beta)
            for mode in ("limb_shape_baseline", "limb_shape_subject_offset"):
                print("limb-shape evaluation", record["label"], mode, flush=True)
                result = fit_smplh_retarget(model, **input_arguments(evaluation_data),
                    fixed_betas=subject_beta, initialization=initialization, config=config,
                    center_calibration=(calibration_result.center_calibration if mode.endswith("offset") else None))
                values, perror, rerror = metrics(evaluation_data, result)
                export = export_result(output / "results" / f"{record['label']}-{mode}.npz", evaluation_data, result)
                row = dict(label=record["label"], mode=mode, partition=record["pilot_partition"],
                    original_split=record["original_split"], operator_id=record["operator_id"],
                    source_key=record["source_key"], metrics=values, export=export, metadata=result.metadata,
                    calibration_scope="Within-person disjoint first window; pose fitting uses second-window observations")
                subject_rows.append(row)
                subject_distributions.setdefault(mode, {"position":[], "rotation":[]})
                subject_distributions[mode]["position"].append(perror)
                subject_distributions[mode]["rotation"].append(rerror)
                write_json(output / "results" / f"{record['label']}-{mode}.json", row)
    subject_aggregates = {mode:dict(
        observation_reconstruction_error_m=summary(np.concatenate(distribution["position"])),
        world_rotation_increment_error_deg=summary(np.concatenate(distribution["rotation"])))
        for mode,distribution in subject_distributions.items()}
    all_rows = rows + subject_rows
    finite = all(np.isfinite(row["metrics"]["observation_reconstruction_error_m"]["mean"]) for row in all_rows)
    exports_passed = all(row["export"]["floor_inverse_max_abs_m"] < 1e-5 and
        row["export"]["so3_orthogonality_max"] < 1e-8 and row["export"]["min_determinant"] > .999999 for row in all_rows)
    write_json(output / "ego-report.json", dict(format="smplh-dual-source-ego-retarget-pilot-v1",
        execution_passed=finite and exports_passed, operators=len(manifest["records"]),
        calibration_operators=len(calibration_groups), heldout_operators=len(manifest["heldout_operator_ids"]),
        evaluation_frames_per_mode=args.frames*len(manifest["records"]), results=rows,
        aggregates=aggregates, full_production_conversion_ready=False,
        subject_shape_offset_results=subject_rows, subject_shape_offset_aggregates=subject_aggregates,
        subject_shape_offset_semantics="Within-person calibration estimates shape and offsets; not cross-person transfer or independent shape truth",
        independent_absolute_rotation_accuracy_established=False,
        independent_canonical_center_accuracy_established=False,
        shape_variation_resolved=False, production_archive_modified=False,
        native_axes_certified=False, dino_reextracted=False,
        evaluation_semantics="Offline observation reconstruction with frozen train-only center calibration; no pose prediction",
        completed_at=datetime.now(timezone.utc).isoformat()))
    return finite and exports_passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--calibration-operators", type=int, default=5)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--prepared", action="store_true")
    parser.add_argument("--limb-shape", action="store_true",
                        help="Also test per-person limb-only shape and frozen center offsets on disjoint windows")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "inputs").mkdir(exist_ok=True)
    (args.output / "results").mkdir(exist_ok=True)
    torch.set_num_threads(2)
    started = time.monotonic()
    manifest_path = args.output / "ego-input-manifest.json"
    manifest = json.loads(manifest_path.read_text()) if args.prepared else prepare_inputs(args)
    if args.prepared:
        for record in manifest["records"]:
            for window in record["windows"]:
                if sha(window["path"]) != window["sha256"]:
                    raise ValueError("Prepared pilot input changed")
    if args.prepare_only:
        print("prepared", len(manifest["records"]), "operators", flush=True)
        return
    code_dir = args.output / "code"
    code_dir.mkdir(exist_ok=True)
    from egomotion_pipeline import smplh_retarget, smpl_fit
    files = [Path(__file__), Path(smplh_retarget.__file__), Path(smpl_fit.__file__),
             Path(__file__).with_name("validate_smpl_fit_pilot.py")]
    hashes = {}
    for path in files:
        shutil.copyfile(path, code_dir / path.name)
        hashes[str(path.resolve())] = sha(path)
    write_json(args.output / "ego-code.json", hashes)
    try:
        passed = run(args, manifest)
    except Exception as exc:
        write_json(args.output / "ego-status.json", dict(state="failed", error=f"{type(exc).__name__}: {exc}"))
        raise
    for path, digest in hashes.items():
        if sha(path) != digest:
            raise RuntimeError("Research source changed during the experiment")
    write_json(args.output / "ego-status.json", dict(state="complete", execution_passed=passed,
        elapsed_seconds=time.monotonic()-started, full_production_conversion_ready=False))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
