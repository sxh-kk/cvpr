"""Read-only source planning and projected inputs for SMPL-H sequence jobs.

The source v5 archive is immutable. Jobs write motion-only derived products;
camera, floor, hands, RGB and DINO remain associated by the original frame ID.
"""

from collections import defaultdict
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .geometry import quat_to_matrix
from .schema import SCHEMA_VERSION, feature_specs_for_schema
from .storage import Archive, atomic_json, digest


def file_identity(path):
    stat = Path(path).stat()
    return dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                ctime_ns=stat.st_ctime_ns, inode=stat.st_ino, device=stat.st_dev)


def arrays_sha256(arrays):
    h = sha256()
    for key, value in sorted(arrays.items()):
        value = np.ascontiguousarray(value)
        h.update(key.encode())
        h.update(str(value.dtype).encode())
        h.update(json.dumps(value.shape).encode())
        h.update(value.tobytes())
    return h.hexdigest()


def read_source_motion(path):
    """Read physical v5 motion columns, without decoding stored visual features."""
    specs = feature_specs_for_schema(SCHEMA_VERSION)
    specs = {k: v for k, v in specs.items() if k != "observation.features.dino"}
    before = file_identity(path)
    table = pq.read_table(path, columns=list(specs))
    arrays = {}
    for key, spec in specs.items():
        col = table[key].combine_chunks()
        if col.null_count:
            raise ValueError(f"Source nulls in {key}")
        if spec["shape"] == [1]:
            value = col.to_numpy(zero_copy_only=False)
        else:
            if col.values.null_count:
                raise ValueError(f"Source list nulls in {key}")
            value = col.values.to_numpy(zero_copy_only=False).reshape(len(table), *spec["shape"])
        if value.dtype != np.dtype(spec["dtype"]):
            raise ValueError(f"Source dtype mismatch in {key}")
        arrays[key] = value.copy()
    if before != file_identity(path):
        raise ValueError("Source changed during projected read")
    indices = arrays["frame_index"]
    if not np.array_equal(indices, np.arange(len(table))):
        raise ValueError("Source frame identity is not complete and contiguous")
    return arrays, before


def native_contact_proxy(positions, valid, timestamps, height, ground_valid):
    """Conservative source-foot heuristic; never claims contact ground truth."""
    count = len(timestamps)
    contact = np.zeros((count, 2), bool)
    available = np.zeros_like(contact)
    if count < 2:
        return contact, available
    feet, mask = positions[:, [10, 11]], valid[:, [10, 11]]
    dt = np.diff(timestamps)
    edges = mask[1:] & mask[:-1] & (dt > 0)[:, None] & (dt <= .05)[:, None]
    edges &= ground_valid[1:, None] & ground_valid[:-1, None]
    speed = np.linalg.norm(np.diff(feet, axis=0), axis=-1) / np.maximum(dt[:, None], 1e-12)
    distance = feet[1:, :, 2] - height[1:, None]
    positive = edges & (speed < .15) & (distance >= -.03) & (distance <= .08)
    # Isolated positives are insufficient for a foot-position fitting anchor.
    stable = positive.copy()
    if len(positive) > 1:
        stable &= np.vstack((np.zeros((1, 2), bool), positive[:-1])) | np.vstack((positive[1:], np.zeros((1, 2), bool)))
    contact[1:], available[1:] = stable, edges
    return contact, available


def sequence_inputs(arrays):
    count = len(arrays["timestamp"])
    position = arrays["observation.state.body_world"].reshape(count, 22, 3)
    valid = arrays["observation.valid.body_position"]
    rotations = quat_to_matrix(arrays["observation.state.body_world_rotation"].reshape(count, 22, 4))
    rvalid = arrays["observation.valid.body_rotation"]
    timestamps = arrays["observation.source_timestamp"]
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise ValueError("Source timestamps must be strictly increasing")
    hands = np.stack([arrays[f"observation.state.hand_{side}_world"].reshape(count, 21, 3)
                      for side in ("left", "right")], 1)
    hvalid = np.stack([arrays[f"observation.valid.hand_{side}_position"] for side in ("left", "right")], 1)
    dt = np.diff(timestamps)
    root_jump = (np.linalg.norm(np.diff(position[:, 0], axis=0), axis=-1) > .5) & valid[1:, 0] & valid[:-1, 0]
    transition = np.broadcast_to(((dt > 0) & (dt <= .05) & ~root_jump)[:, None], (max(count-1, 0), 22)).copy()
    # Tracking resets in an individual rotation must not be fitted as motion.
    delta = rotations[1:] @ rotations[:-1].swapaxes(-1, -2)
    angle = np.degrees(np.arccos(np.clip((np.trace(delta, axis1=-2, axis2=-1)-1)/2, -1, 1)))
    transition &= ~((rvalid[1:] & rvalid[:-1]) & (angle > 45.))
    contact, cvalid = native_contact_proxy(position, valid, timestamps,
        arrays["observation.environment.floor_height"], arrays["observation.valid.floor"])
    return dict(positions=position, position_mask=valid, timestamps=timestamps,
                source_global_rotations=rotations, rotation_mask=rvalid,
                hand_wrist_world=hands[:, :, 0], hand_wrist_valid=hvalid[:, :, 0],
                hand_world=hands, hand_valid=hvalid, transition_mask=transition,
                source_contact=contact, contact_valid=cvalid)


def source_record_contract(archive, record, arrays, identity):
    index = int(record["episode_index"])
    path = archive.data_path(index)
    return dict(source_archive=str(archive.root.resolve()), source_path=str(path.resolve()),
                source_identity=identity, source_motion_sha256=arrays_sha256(arrays),
                source_projection_columns=sorted(arrays),
                record_path=str(archive.record_path(index).resolve()),
                record_sha256=sha256(archive.record_path(index).read_bytes()).hexdigest(),
                episode_index=index, source_key=record["source_key"], frames=record["length"])


def split_memberships(path, records):
    value = json.loads(Path(path).read_text())
    mapping = {}
    for split in ("train", "val", "test"):
        for key in value["splits"][split]:
            if key in mapping:
                raise ValueError("Duplicate source across splits")
            mapping[key] = split
    operators, recordings = defaultdict(set), defaultdict(set)
    for record in records:
        key = record["source_key"]
        if key not in mapping:
            raise ValueError(f"Missing split membership: {key}")
        meta = record["metadata"]
        operator = meta.get("operator_id")
        if not operator:
            raise ValueError(f"Missing operator ID: {key}")
        operators[operator].add(mapping[key])
        for name in ("source_file_uid", "source_episode_uuid", "source_data"):
            if meta.get(name):
                recordings[(name, meta[name])].add(mapping[key])
    if any(len(x) > 1 for x in list(operators.values()) + list(recordings.values())):
        raise ValueError("Operator or original recording leaks between splits")
    return mapping


def create_plan(archive_root, splits_path, model_path, config, selection, job_dir):
    archive = Archive(archive_root)
    if archive.schema_version != SCHEMA_VERSION:
        raise ValueError("SMPL-H derived processing requires a complete v5 source")
    records = archive.records()
    if any(r["source"] != "egosuite" for r in records):
        raise ValueError("This job processes EgoSuite only; HM uses parameter decoding")
    memberships = split_memberships(splits_path, records)
    groups = defaultdict(list)
    for record in records:
        groups[record["metadata"]["operator_id"]].append(record)
    # Choose a stable, short, well-observed calibration recording per operator.
    groups_plan = {}
    for group_id, members in sorted(groups.items()):
        eligible = sorted(members, key=lambda r: (
            int(r["length"]) < config.min_calibration_frames,
            not 600 <= int(r["length"]) <= 3000,
            -float(r.get("quality", {}).get("body_position", 0)),
            abs(int(r["length"])-1200), int(r["episode_index"])))
        groups_plan[group_id] = dict(group_id=group_id, split=memberships[members[0]["source_key"]],
            calibration_episode_index=int(eligible[0]["episode_index"]),
            episode_indices=[int(r["episode_index"]) for r in members],
            grouping_policy="source_operator_id; per-episode limb consistency is checked before reuse")
    by_index = {r["episode_index"]: r for r in records}
    for purpose in ("calibration_records", "validation_records"):
        selected_indices = set()
        selected_operators = set()
        for row in selection[purpose]:
            record = by_index.get(row["episode_index"])
            if (record is None or record["source_key"] != row["source_key"]
                    or record["metadata"]["operator_id"] != row["operator_id"]
                    or memberships[row["source_key"]] != row["split"]
                    or row["episode_index"] in selected_indices or row["operator_id"] in selected_operators):
                raise ValueError("Selection identity, split or operator uniqueness mismatch")
            selected_indices.add(row["episode_index"])
            selected_operators.add(row["operator_id"])
            if purpose == "calibration_records" and row["split"] != "train":
                raise ValueError("Shared source center calibration must use train only")
    cgroups = {r["operator_id"] for r in selection["calibration_records"]}
    egroups = {r["operator_id"] for r in selection["validation_records"]}
    if cgroups & egroups:
        raise ValueError("Shared calibration and expanded validation operators overlap")
    code = {str(path.resolve()): sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(__file__).parent.glob("smplh*.py"))}
    for name in ("smpl_fit.py", "geometry.py", "schema.py", "body_model.py",
                 "ground.py", "windows.py", "unified_dense22.py", "storage.py",
                 "native_dense22.py", "head_reference.py", "resampling.py"):
        path = Path(__file__).parent/name
        code[str(path.resolve())] = sha256(path.read_bytes()).hexdigest()
    for name in ("process_smplh_dataset.py", "accept_smplh_job.py", "audit_smplh_sidecars.py"):
        script = Path(__file__).parents[2]/"scripts"/name
        code[str(script.resolve())] = sha256(script.read_bytes()).hexdigest()
    contract = dict(format="egosuite-smplh-sequence-job-v1", source_archive=str(archive.root.resolve()),
        source_schema=SCHEMA_VERSION, model_path=str(Path(model_path).resolve()),
        model_sha256=sha256(Path(model_path).read_bytes()).hexdigest(), config=asdict(config),
        config_sha256=digest(asdict(config)), implementation_sha256=code,
        split_manifest_path=str(Path(splits_path).resolve()),
        split_manifest_sha256=sha256(Path(splits_path).read_bytes()).hexdigest(),
        selection_sha256=digest(selection), archive_config_sha256=sha256((archive.root/".pipeline/config.json").read_bytes()).hexdigest(),
        source_frames=sum(r["length"] for r in records), source_episodes=len(records),
        groups=groups_plan, target_joint_basis="smplh_body22_axes_v1", label_kind="model_fitted_estimate",
        source_native_axes_certified=False, archive_fps=30, training_fps=15,
        source_archive_modified=False, camera_ground_policy="reuse_source; source_feet_only_window_fallback",
        visual_policy="reuse_v5_original_frame_join; no_extraction", selection=selection)
    contract["job_signature"] = digest(contract)
    destination = Path(job_dir)/"plan.json"
    if destination.exists():
        if json.loads(destination.read_text()) != contract:
            raise ValueError("Existing job plan differs; use a new job or the frozen code/config")
    else:
        atomic_json(destination, contract)
    return contract


def assert_plan_current(plan):
    unsigned = dict(plan)
    signature = unsigned.pop("job_signature", None)
    if signature != digest(unsigned):
        raise ValueError("Frozen job plan signature mismatch")
    if (plan["config_sha256"] != digest(plan["config"])
            or plan["selection_sha256"] != digest(plan["selection"])):
        raise ValueError("Frozen job config or selection digest mismatch")
    for path, fingerprint in plan["implementation_sha256"].items():
        if sha256(Path(path).read_bytes()).hexdigest() != fingerprint:
            raise ValueError(f"Implementation changed since planning: {path}")
    if sha256(Path(plan["model_path"]).read_bytes()).hexdigest() != plan["model_sha256"]:
        raise ValueError("Model changed since planning")
    if sha256(Path(plan["split_manifest_path"]).read_bytes()).hexdigest() != plan["split_manifest_sha256"]:
        raise ValueError("Split manifest changed since planning")
    if sha256((Path(plan["source_archive"])/".pipeline/config.json").read_bytes()).hexdigest() != plan["archive_config_sha256"]:
        raise ValueError("Source archive config changed since planning")
