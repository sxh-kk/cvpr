"""Prepare source-informed Handover VAL cold starts without replaying actions.

The source environment_config can contain final object poses. Actual initial
poses therefore come from row zero and the independent initial-object log.
All assets are reused through local symlinks; source data remain read-only.
This does not reproduce the source controller state, warmup or joint velocities.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


TASK = "G1WholebodyHandoverTeleop-v0"
VAL_IDS = (39, 65, 80, 92, 94)
CONTRACT = "cvpr-handover-heldout-initial-v1"
OBJECT_NAMES = [f"{role}.{term}" for role in ("container", "target")
                for term in ("pos_x", "pos_y", "pos_z", "quat_w", "quat_x", "quat_y", "quat_z")]
SCOPE = ("Source-informed VAL cold start, not exact teacher physical replay or an official benchmark. "
         "One initial robot/object state is provided before real camera history fills; "
         "no recorded RGB, teacher tokens, primer, future state or teacher trajectory is executed. "
         "Joint velocities are initialized to zero because complete source joint velocities and "
         "controller integration/history state are not present. Source collection hand/body "
         "controller profile equivalence remains unverified.")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def vector(value, width, label):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (width,) or not np.isfinite(result).all():
        raise ValueError(f"Invalid finite {label}[{width}]")
    return result


def pose(value, label):
    result = vector(value, 7, label)
    if not np.isclose(np.linalg.norm(result[3:]), 1, rtol=0, atol=1e-5):
        raise ValueError(f"Invalid unit quaternion in {label}")
    return result


def remap_paths(value, output, assets):
    if isinstance(value, dict):
        return {key: remap_paths(item, output, assets) for key, item in value.items()}
    if isinstance(value, list):
        return [remap_paths(item, output, assets) for item in value]
    if isinstance(value, str):
        marker = "/SIMPLE/data/"
        relative = (value.split(marker, 1)[1] if marker in value else
                    value[5:] if value.startswith("data/") else None)
        if relative is not None:
            suffix = Path(relative)
            if suffix.is_absolute() or ".." in suffix.parts or not (assets / suffix).exists():
                raise ValueError(f"Missing or unsafe local SIMPLE resource: {relative}")
            return str(output / suffix)
        if value.endswith("/SIMPLE/data"):
            return str(output)
    return value


def _read_first_observation(path):
    import pyarrow.parquet as pq
    columns = ["observation.base_pose", "observation.base_vel", "observation.object_poses"]
    table = next(pq.ParquetFile(path).iter_batches(batch_size=1, columns=columns), None)
    if table is None:
        raise ValueError("Missing heldout first observation")
    return table.to_pylist()[0]


def _read_case(source, row, info, assets, output):
    original_id = row["source_episode_index"]
    local_id = row["episode_index"]
    prefix = source / "validation" / f"source_episode_{original_id:06d}"
    files = {
        "physics": prefix.with_name(prefix.name + "_physics.json"),
        "initial_objects": prefix.with_name(prefix.name + "_initial_objects.json"),
        "reset_joints": prefix.with_name(prefix.name + "_reset_joint_state43.npy"),
        "controller_reset": prefix.with_name(prefix.name + "_controller_reset.json"),
        "data": source / "data/chunk-000" / f"episode_{local_id:06d}.parquet",
    }
    physics = json.loads(files["physics"].read_text())
    initial_objects = json.loads(files["initial_objects"].read_text())
    reset_record = json.loads(files["controller_reset"].read_text())
    names = physics["reset_joint_names"]
    model_xml = assets / "robots/g1_sonic/g1_29dof_with_hand.xml"
    model_names = {element.get("name") for element in ET.parse(model_xml).iter("joint")
                   if element.get("name")}
    if len(names) != 43 or len(set(names)) != 43 or not set(names).issubset(model_names):
        raise ValueError("Source reset must have 43 unique named local model joints")
    if info["features"]["observation.object_poses"].get("names") != OBJECT_NAMES:
        raise ValueError("Recorded object pose order is ambiguous")
    reset_q = vector(np.load(files["reset_joints"], allow_pickle=False), 43, "reset joints")
    if not np.array_equal(reset_q, vector(physics["reset_joint_state43"], 43, "physics reset")):
        raise ValueError("Reset vector differs from independent physics log")
    if (reset_record.get("vector_sha256") != hashlib.sha256(reset_q.astype("<f8").tobytes()).hexdigest()
            or reset_record.get("output_replay_reset_state_key") != "replay.initial_state43"):
        raise ValueError("Reset vector SHA or semantic source differs")
    first = _read_first_observation(files["data"])
    root_pose = pose(first["observation.base_pose"], "root pose")
    root_velocity = vector(first["observation.base_vel"], 6, "root qvel")
    object_poses = vector(first["observation.object_poses"], 14, "initial object poses")
    if initial_objects.get("object_restore_during_control") is not False:
        raise ValueError("Expected source one-time initial object restoration")
    config = json.loads(row["environment_config"])
    source_task_uid = config.get("uid")
    if (source_task_uid not in ("g1_wholebody_handover_teleop", "g1_wholebody_handover_sonic")
            or config["robot_cfg"]["uid"] != "g1_sonic"):
        raise ValueError("Wrong Handover environment identity")
    # The separate collection checkout names the same Handover domain 'sonic'.
    # Record this explicit adapter; it does not certify identical task/runtime
    # code or controller semantics across those two checkouts.
    config["uid"] = "g1_wholebody_handover_teleop"
    if set(config["layout"]["actors"]) != {"target", "container", "table"}:
        raise ValueError("Unexpected heldout actor set")
    objects = {}
    original_final_poses = {}
    for ordinal, role in enumerate(("container", "target")):
        value = pose(object_poses[ordinal * 7:(ordinal + 1) * 7], role)
        documented = initial_objects["objects"][role]
        if (not np.array_equal(value, pose(documented["actual_pose7"], "actual object log"))
                or not np.array_equal(value, pose(documented["recorded_pose7"], "recorded object log"))):
            raise ValueError("Row-zero object poses differ from independent initial-object log")
        body = physics["objects"][role]
        mass = float(body["mass"])
        if not np.isfinite(mass) or mass <= 0 or np.any(vector(body["external_force"], 6, "object force")):
            raise ValueError("Require positive source object mass and zero external force")
        actor = config["layout"]["actors"][role]
        original_final_poses[role] = deepcopy(actor["pose"])
        replacement = {"position": value[:3].tolist(), "quaternion": value[3:].tolist()}
        actor["pose"] = deepcopy(replacement)
        spatial_key = f"container_{actor['uid']}" if role == "container" else str(actor["uid"])
        if spatial_key not in config["dr_state_dict"]["spatial"]:
            raise ValueError("Missing actual SIMPLE spatial reset key")
        config["dr_state_dict"]["spatial"][spatial_key] = deepcopy(replacement)
        objects[role] = {"pose7": value.tolist(), "mass": mass}
    # Keep the original robot spawn transform. Its absolute pelvis state is
    # applied once by the runtime hook; using it as a spawn frame adds the MJCF
    # root height a second time.
    config = remap_paths(config, output, assets)
    prepared_row = deepcopy(row)
    prepared_row["environment_config"] = json.dumps(config, ensure_ascii=False, allow_nan=False)
    prepared_row["evaluation_domain"] = "source_informed_heldout_cold_start"
    prepared_row["exact_source_physics_replay"] = False
    prepared_row["source_task_uid"] = source_task_uid
    initial = {"source_episode_index": original_id, "joint_names": names,
        "joint_qpos": reset_q.tolist(), "root_pose7": root_pose.tolist(),
        "root_qvel6": root_velocity.tolist(), "objects": objects,
        "sources": {str(path): sha256(path) for path in files.values()}, "scope": SCOPE,
        "joint_qvel_policy": "zero; source complete per-joint velocities unavailable",
        "teacher_primer_consumed": False, "future_observation_rows_consumed": False,
        "task_uid_mapping": {"source": source_task_uid, "local": config["uid"],
            "scope": "explicit Handover domain adapter; source and local task/controller implementations are not certified equal"},
        "source_object_layout_poses_replaced": original_final_poses}
    return prepared_row, initial


def prepare(source, assets, output, source_ids=VAL_IDS):
    source = Path(source).resolve(strict=True)
    assets = Path(assets).resolve(strict=True)
    output = Path(output)
    if not output.is_absolute():
        raise ValueError("Output must be absolute")
    output = output.resolve()
    repository = Path(__file__).resolve().parents[2]
    if (output.exists() or output.is_symlink() or output.is_relative_to(repository)
            or output.is_relative_to(source) or output.is_relative_to(assets)):
        raise ValueError("Use a fresh absolute output outside the source, assets and repository")
    selected = list(source_ids)
    if not selected or len(set(selected)) != len(selected) or not set(selected).issubset(VAL_IDS):
        raise ValueError("Only the five original heldout VAL source IDs may be selected")
    manifest_path = source / "conversion_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("status") != "complete" or manifest.get("is_official") is not False
            or manifest.get("accepted_source_episode_indices") != list(VAL_IDS)):
        raise ValueError("Require the existing five-case heldout release")
    info_path, episodes_path = source / "meta/info.json", source / "meta/episodes.jsonl"
    info = json.loads(info_path.read_text())
    if info.get("fps") != 50:
        raise ValueError("Require the original 50 Hz heldout reset pack")
    rows = [json.loads(line) for line in episodes_path.read_text().splitlines() if line]
    if ([row.get("source_episode_index") for row in rows] != list(VAL_IDS)
            or [row["episode_index"] for row in rows] != list(range(5))):
        raise ValueError("Heldout episode ordering or membership differs")
    prepared, cases = [], {}
    for row in rows:
        if row["source_episode_index"] not in selected:
            continue
        adapted, initial = _read_case(source, row, info, assets, output)
        adapted["source_heldout_episode_index"] = adapted["episode_index"]
        adapted["episode_index"] = len(prepared)
        cases[str(len(prepared))] = initial
        prepared.append(adapted)
    # Validate every selected source before creating output or any symlinks.
    output.mkdir(parents=True, exist_ok=False)
    for child in assets.iterdir():
        if child.name != "eval":
            (output / child.name).symlink_to(child, target_is_directory=child.is_dir())
    metadata = output / "eval" / TASK / "dr-level-0/meta"
    metadata.mkdir(parents=True)
    metadata.joinpath("episodes.jsonl").write_text("".join(
        json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in prepared))
    # This directory is solely reset metadata, not an executable teacher dataset.
    write_json(metadata / "info.json", {"fps": 50, "total_episodes": len(prepared),
        "dataset_purpose": "source-informed Handover VAL reset metadata only; no replay labels/video"})
    initialization = {"contract": CONTRACT, "task_id": "simple/" + TASK,
        "episodes": cases, "scope": SCOPE, "compatibility_verified": False}
    write_json(output / "handover_initialization.json", initialization)
    summary = {"contract": CONTRACT, "status": "prepared_cpu_only_not_physically_validated",
        "created_utc": datetime.now(timezone.utc).isoformat(), "source_root": str(source),
        "asset_root": str(assets), "data_root": str(output),
        "source_episode_ids": [row["source_episode_index"] for row in prepared],
        "source_profiles": manifest.get("control_profiles"),
        "teacher_actions_or_primer_consumed": False, "source_videos_consumed": False,
        "control_profile_equivalence_verified": False, "scope": SCOPE,
        "source_sha256": {str(path): sha256(path) for path in (manifest_path, info_path, episodes_path)},
        "generated_sha256": {str(path.relative_to(output)): sha256(path) for path in
            (metadata / "episodes.jsonl", metadata / "info.json", output / "handover_initialization.json")}}
    write_json(output / "PREPARATION.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heldout-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-ids", type=int, nargs="+", default=list(VAL_IDS))
    args = parser.parse_args()
    if not args.output.is_absolute():
        parser.error("--output must be absolute")
    print(json.dumps(prepare(args.heldout_root, args.asset_root, args.output, args.source_ids),
                     ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
