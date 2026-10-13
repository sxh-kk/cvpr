"""Audit geometric joint-axis candidates on disjoint EgoSuite operators.

This CPU-only research command writes a separate report and numeric samples.
It never writes the dataset archive and never marks a candidate semantically
verified. Rest-direction alignment cannot authenticate an unknown joint twist.
"""

import os
for _thread_env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_thread_env, "2")

import argparse
from datetime import datetime, timezone
import hashlib
import io
import itertools
import json
from pathlib import Path
import time
import traceback

import numpy as np

from egomotion_pipeline.egosuite_mcap import (
    BAD_TOPICS, LOWER_JOINTS, UPPER_JOINTS, make_proto_reader,
    nearest_indices, timestamp_ns,
)
from egomotion_pipeline.geometry import normalize_quat, quat_to_matrix
from egomotion_pipeline.joint_basis import fit_direction_basis
from egomotion_pipeline.schema import BODY_NAMES, PARENTS
from egomotion_pipeline.storage import atomic_json


DEFAULT_ROOT = Path("/gaozt-test1/datasets/MixEgoMotion/.pipeline/axis-research")
DEFAULT_MODEL = Path("/gaozt-test1/guanzerong/UniEgoMotion/body_models/smplx/SMPLX_NEUTRAL.npz")
SOURCE_MCP = [("index", 5), ("middle", 9), ("ring", 13), ("pinky", 17)]
TARGET_MCP = {"left": [25, 28, 34, 31], "right": [40, 43, 49, 46]}
TARGET_JOINT_NAMES_SOURCE = "https://github.com/vchoutas/smplx/blob/main/smplx/joint_names.py"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def numeric_json(value):
    if isinstance(value, np.ndarray):
        return numeric_json(value.tolist())
    if isinstance(value, (list, tuple)):
        return [numeric_json(v) for v in value]
    if isinstance(value, dict):
        return {str(k): numeric_json(v) for k, v in value.items()}
    if isinstance(value, np.generic):
        return numeric_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def error_summary(values, weights=None):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"count": 0, "mean_deg": None, "rmse_deg": None,
                "p95_deg": None, "max_deg": None}
    weights = np.ones(len(values)) if weights is None else np.asarray(weights, float)
    weights = weights / weights.sum()
    return {"count": len(values), "mean_deg": float(weights @ values),
            "rmse_deg": float(np.sqrt(weights @ values**2)),
            "p95_deg": float(np.quantile(values, .95)), "max_deg": float(values.max())}


def robust_ray(vectors):
    vectors = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(vectors, axis=-1)
    finite = np.isfinite(vectors).all(-1) & (norms > 1e-8)
    vectors, norms = vectors[finite], norms[finite]
    if not len(vectors):
        return None
    unit = vectors / norms[:, None]
    center = np.median(unit, axis=0)
    center /= np.linalg.norm(center)
    angles = np.degrees(np.arccos(np.clip(unit @ center, -1, 1)))
    return {"median_unit": center, "n": len(unit), "stability": error_summary(angles),
            "length_median_m": float(np.median(norms))}


def select_operators(catalog, count, seed):
    """Select one episode per operator, preferring different task groups."""
    eligible = sorted((r for r in catalog if r.get("operator") and r.get("path")),
                      key=lambda r: (r["operator"], r.get("uid", "")))
    if len({r["operator"] for r in eligible}) < count:
        raise ValueError("Catalog contains fewer distinct operators than requested")
    order = np.random.default_rng(seed).permutation(len(eligible))
    chosen, operators, tasks = [], set(), set()
    for require_new_task in (True, False):
        for index in order:
            row = eligible[int(index)]
            if row["operator"] in operators or (require_new_task and row.get("task") in tasks):
                continue
            if row.get("basis_correction") is not None:
                raise ValueError("Raw-MCAP audit requires uncorrected source rotation catalog entries")
            chosen.append(dict(row))
            operators.add(row["operator"])
            tasks.add(row.get("task"))
            if len(chosen) == count:
                # Alternating split distributes the selected task groups across
                # fit/heldout while preserving strict operator disjointness.
                for i, selected in enumerate(chosen):
                    selected["split"] = "fit" if i % 2 == 0 else "heldout"
                return chosen
    raise RuntimeError("Could not complete deterministic operator selection")


def load_target_model(path):
    """Read a licensed user-provided SMPL-X asset without pickle execution."""
    path = Path(path)
    before = source_identity(path.stat())
    snapshot = path.read_bytes()
    snapshot_sha256 = hashlib.sha256(snapshot).hexdigest()
    with np.load(io.BytesIO(snapshot), allow_pickle=False) as asset:
        template = np.asarray(asset["v_template"], dtype=float)
        regressor = np.asarray(asset["J_regressor"], dtype=float)
        tree = np.asarray(asset["kintree_table"])
        if (template.shape != (10475, 3) or regressor.shape != (55, 10475)
                or tree.shape != (2, 55) or not np.isfinite(template).all()
                or not np.isfinite(regressor).all()):
            raise ValueError("Expected a finite full 55-joint SMPL-X neutral asset")
        lookup = {int(child): i for i, child in enumerate(tree[1])}
        parents = np.array([-1] + [lookup[int(v)] for v in tree[0, 1:]])
        if not np.array_equal(parents[:22], PARENTS):
            raise ValueError("SMPL-X first22 parents differ from the dense22 contract")
        joints = regressor @ template
        discrepancy = float(np.max(np.abs(joints - asset["J"]))) if "J" in asset else None
        for side, wrist in (("left", 20), ("right", 21)):
            if not all(parents[k] == wrist for k in TARGET_MCP[side]):
                raise ValueError("Expected SMPL-X MCP roots to be direct children of the wrist")
    if (before != source_identity(path.stat()) or file_sha256(path) != snapshot_sha256):
        raise ValueError("Target model changed while reading its byte snapshot")
    return joints, {"model_path": str(path.resolve()), "model_sha256": snapshot_sha256,
                    "model_byte_snapshot_and_path_hash_match": True,
                    "model_type": "SMPL-X neutral, UniEgoMotion current SMPLXLayer reference",
                    "reference_shape": "zero betas", "vertex_count": len(template),
                    "joint_count": len(joints), "first22_parent_check_passed": True,
                    "first22_parents": parents[:22].tolist(),
                    "mcp_parent_check_passed": True, "J_regressor_template_vs_J_max_m": discrepancy,
                    "not_hm_smplh_decoding_asset": True,
                    "target_local_axes": "SMPL-X joint rest axes: identity at zero local pose"}


def pose_values(transforms, expected):
    if len(transforms) != expected:
        raise ValueError(f"Expected {expected} transforms, found {len(transforms)}")
    positions = np.array([[t.pos.x, t.pos.y, t.pos.z] if t.HasField("pos") else [np.nan]*3
                          for t in transforms], dtype=float)
    quaternions = np.array([[t.quat.w, t.quat.x, t.quat.y, t.quat.z]
                           if t.HasField("quat") else [0.]*4 for t in transforms], dtype=float)
    return positions, quaternions


def matched_flags(rows, body_stamps, tolerance_ns):
    if not rows:
        return np.zeros(len(body_stamps), bool), {"messages": 0, "matched": 0, "bad": 0}
    rows = sorted(rows, key=lambda r: r[0])
    stamps = np.array([r[0] for r in rows], dtype=np.int64)
    if len(stamps) > 1 and np.any(np.diff(stamps) <= 0):
        raise ValueError("Annotation capture timestamps are not unique")
    indices, matched, _ = nearest_indices(stamps, body_stamps, tolerance_ns)
    flags = np.array([r[1] for r in rows], dtype=bool)[indices] & matched
    return flags, {"messages": len(rows), "matched": int(matched.sum()), "bad": int(flags.sum())}


def source_identity(stat):
    return {"size_bytes":stat.st_size,"mtime_ns":stat.st_mtime_ns,
            "ctime_ns":stat.st_ctime_ns,"inode":stat.st_ino,"device":stat.st_dev}


def load_catalog_snapshot(path):
    path = Path(path)
    before = source_identity(path.stat())
    snapshot = path.read_bytes()
    digest = hashlib.sha256(snapshot).hexdigest()
    catalog = json.loads(snapshot)
    if before != source_identity(path.stat()) or file_sha256(path) != digest:
        raise ValueError("Source catalog changed while reading its byte snapshot")
    if not isinstance(catalog,list):
        raise ValueError("Source catalog must be a list")
    return catalog,digest


def read_prefix(path, max_frames, tolerance_ns):
    """Decode pose Protobuf only, then join integer capture timestamps.

    Sorting log times bounds the prefix. Log times are not used for pose joins.
    Continue to one extra body message to collect trailing auxiliary messages;
    auxiliary missing/mismatched poses are masked, and sparse unmatched bad flags
    follow the current pipeline's false-bad convention.
    """
    path = Path(path)
    before = path.stat()
    topics = ["/pose/body", "/pose/left_hand", "/pose/right_hand", *BAD_TOPICS.values()]
    rows = {topic: [] for topic in topics}
    capture_fields = {}
    with path.open("rb") as handle:
        reader = make_proto_reader(handle)
        for schema, channel, message, decoded in reader.iter_decoded_messages(
                topics=topics, log_time_order=True):
            topic = channel.topic
            if channel.message_encoding != "protobuf" or schema.encoding != "protobuf":
                raise ValueError("Pose topics require embedded Protobuf schemas")
            stamp, field = timestamp_ns(decoded, message)
            capture_fields.setdefault(topic, set()).add(field)
            if topic == "/pose/body":
                if len(rows[topic]) == max_frames:
                    break
                p, q = pose_values(decoded.transforms, 22)
                rows[topic].append((stamp, p, q))
            elif topic in ("/pose/left_hand", "/pose/right_hand"):
                p, q = pose_values(decoded.transforms, 21)
                rows[topic].append((stamp, p, q))
            else:
                rows[topic].append((stamp, bool(decoded.is_bad)))
    after = path.stat()
    if source_identity(before) != source_identity(after):
        raise ValueError("Source MCAP changed while reading")
    body = sorted(rows["/pose/body"], key=lambda r: r[0])
    if not body:
        raise ValueError("No body poses in source")
    stamps = np.array([r[0] for r in body], dtype=np.int64)
    if len(stamps) > 1 and np.any(np.diff(stamps) <= 0):
        raise ValueError("Body capture timestamps are not unique")
    p = np.stack([r[1] for r in body])
    q, qvalid = normalize_quat(np.stack([r[2] for r in body]))
    pvalid = np.isfinite(p).all(-1)
    quality = {}
    for key, joints in (("upper_body", UPPER_JOINTS), ("lower_body", LOWER_JOINTS)):
        bad, quality[key] = matched_flags(rows[BAD_TOPICS[key]], stamps, tolerance_ns)
        pvalid[np.ix_(bad, joints)] = False
        qvalid[np.ix_(bad, joints)] = False
    handbad, quality["hand"] = matched_flags(rows[BAD_TOPICS["hand"]], stamps, tolerance_ns)
    hp = np.zeros((len(body), 2, 21, 3), dtype=float)
    hv = np.zeros((len(body), 2, 21), bool)
    hq = np.tile([1.,0.,0.,0.], (len(body),2,1))
    hqvalid = np.zeros((len(body),2), bool)
    handstamps = np.full((len(body),2), -1, dtype=np.int64)
    joins = {}
    for side_i, side in enumerate(("left", "right")):
        hand = sorted(rows[f"/pose/{side}_hand"], key=lambda r:r[0])
        if not hand:
            joins[side] = {"messages": 0, "matched": 0}
            continue
        hs = np.array([r[0] for r in hand], dtype=np.int64)
        if len(hs) > 1 and np.any(np.diff(hs) <= 0):
            raise ValueError("Hand capture timestamps are not unique")
        indices, matched, errors = nearest_indices(hs, stamps, tolerance_ns)
        values = np.stack([r[1] for r in hand])[indices]
        hp[:, side_i] = np.nan_to_num(values)
        hv[:, side_i] = np.isfinite(values).all(-1) & matched[:,None] & ~handbad[:,None]
        quats, valid = normalize_quat(np.stack([r[2][0] for r in hand])[indices])
        hq[:, side_i] = quats
        hqvalid[:, side_i] = valid & matched & ~handbad
        handstamps[:, side_i] = np.where(matched, hs[indices], -1)
        joins[side] = {"messages": len(hand), "matched": int(matched.sum()),
                       "max_matched_error_ns": int(np.abs(errors[matched]).max()) if matched.any() else None}
    data = {"body_capture_time_ns": stamps, "body_xyz": np.nan_to_num(p),
            "body_quaternion_wxyz": q, "body_position_valid": pvalid,
            "body_rotation_valid": qvalid, "hand_xyz": hp, "hand_position_valid": hv,
            "hand_root_quaternion_wxyz": hq, "hand_root_rotation_valid": hqvalid,
            "hand_capture_time_ns": handstamps}
    stat_fingerprint = {"path": str(path.resolve()), **source_identity(before)}
    provenance = {"source_stat": stat_fingerprint,
                  "source_stat_fingerprint_sha256": hashlib.sha256(
                      json.dumps(stat_fingerprint, sort_keys=True).encode()).hexdigest(),
                  "source_full_content_hash_computed": False,
                  "source_unchanged_during_read": True, "body_frames": len(body),
                  "body_capture_start_ns": int(stamps[0]), "body_capture_end_ns": int(stamps[-1]),
                  "capture_timestamp_fields": {k:sorted(v) for k,v in capture_fields.items()},
                  "bad_frame_joins": quality, "hand_joins": joins,
                  "video_decoded": False}
    return data, provenance


def make_anchors(joints):
    anchors = [[] for _ in BODY_NAMES]
    for child in range(1,22):
        parent = int(PARENTS[child])
        anchors[parent].append({"kind":"body_child", "child_index":child,
                                "name":f"{BODY_NAMES[parent]}->{BODY_NAMES[child]}",
                                "target_vector":joints[child]-joints[parent]})
    for side_i, side, wrist in ((0,"left",20),(1,"right",21)):
        for (finger, source_i), target_i in zip(SOURCE_MCP, TARGET_MCP[side]):
            anchors[wrist].append({"kind":"hand_mcp", "side_index":side_i,
                                   "source_hand_index":source_i, "target_joint_index":target_i,
                                   "name":f"{side}_hand_wrist->{finger}_mcp",
                                   "target_vector":joints[target_i]-joints[wrist],
                                   "source_finger_name_verified":False})
    return anchors


def source_rays(data, anchors):
    rotations = quat_to_matrix(data["body_quaternion_wxyz"])
    result = [[] for _ in BODY_NAMES]
    for joint, specs in enumerate(anchors):
        for spec in specs:
            if spec["kind"] == "body_child":
                child = spec["child_index"]
                mask = (data["body_rotation_valid"][:,joint]
                        & data["body_position_valid"][:,joint]
                        & data["body_position_valid"][:,child])
                d = data["body_xyz"][mask,child]-data["body_xyz"][mask,joint]
            else:
                side, point = spec["side_index"], spec["source_hand_index"]
                mask = (data["body_rotation_valid"][:,joint]
                        & data["hand_position_valid"][:,side,0]
                        & data["hand_position_valid"][:,side,point])
                d = data["hand_xyz"][mask,side,point]-data["hand_xyz"][mask,side,0]
            local = np.einsum("tji,tj->ti", rotations[mask,joint],d)
            result[joint].append(robust_ray(local))
    return result


def constraint_arrays(rays, anchors, sample_indices, joint):
    source, target, weights, evidence = [], [], [], []
    for index in sample_indices:
        available = [(spec,ray) for spec,ray in zip(anchors[joint],rays[index][joint]) if ray is not None]
        for spec,ray in available:
            source.append(ray["median_unit"])
            target.append(spec["target_vector"])
            weights.append(1/len(available))
            evidence.append({"sample_index":index, "anchor_name":spec["name"],
                             "valid_frames":ray["n"], "source_stability":ray["stability"]})
    return (np.array(source).reshape(-1,3), np.array(target).reshape(-1,3),
            np.array(weights), evidence)


def errors_for_matrix(source, target, matrix):
    if not len(source):
        return np.empty(0)
    source = source/np.linalg.norm(source,axis=-1,keepdims=True)
    target = target/np.linalg.norm(target,axis=-1,keepdims=True)
    return np.degrees(np.arccos(np.clip(np.sum(source*(target@matrix.T),axis=-1),-1,1)))


def signed_axis_permutations():
    for permutation in itertools.permutations(range(3)):
        base = np.eye(3)[:,permutation]
        for signs in itertools.product((-1.,1.),repeat=3):
            matrix = base@np.diag(signs)
            if np.linalg.det(matrix)>0:
                yield matrix


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_ROOT/"source-catalog.json")
    parser.add_argument("--target-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT/"calibration-24operators")
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--max-frames", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--max-pose-error-ms", type=float, default=2.)
    parser.add_argument("--max-angular-error-deg", type=float, default=5.)
    parser.add_argument("--min-anchor-frames", type=int, default=30)
    args = parser.parse_args(argv)
    if args.episodes<4 or args.episodes%2 or args.max_frames<2 or args.min_anchor_frames<1:
        parser.error("episodes must be even and >=4; max-frames>=2; min-anchor-frames>=1")
    if (not np.isfinite(args.max_pose_error_ms) or args.max_pose_error_ms<0
            or not np.isfinite(args.max_angular_error_deg) or not 0<=args.max_angular_error_deg<=180):
        parser.error("invalid timestamp tolerance or angular threshold")
    output = args.output.resolve()
    # A report output must be a new research directory. Reject archive roots and
    # existing jobs so this command cannot rewrite production metadata or data.
    if output.exists() and any(output.iterdir()):
        parser.error("output directory must be empty; choose a new research output")
    if any((output/name).exists() for name in ("data","meta",".pipeline")):
        parser.error("output must not be a dataset archive")
    output.mkdir(parents=True, exist_ok=True)
    (output/"samples").mkdir()
    started = time.monotonic()
    state = {"pid":os.getpid(), "stage":"starting", "semantic_verified":False,
             "verified_joint_count":0, "production_archive_modified":False}

    def update(**fields):
        state.update(fields, updated_at_utc=datetime.now(timezone.utc).isoformat(),
                     elapsed_seconds=time.monotonic()-started)
        atomic_json(output/"status.json",numeric_json(state))

    def log(message):
        print(message,flush=True)
        with (output/"audit.log").open("a") as handle:
            handle.write(datetime.now(timezone.utc).isoformat()+" "+message+"\n")

    try:
        try:
            from threadpoolctl import threadpool_limits
            threadpool_limits(limits=2)
        except ImportError:
            pass
        joints, target_meta = load_target_model(args.target_model)
        anchors = make_anchors(joints)
        catalog,catalog_sha256 = load_catalog_snapshot(args.catalog)
        selected = select_operators(catalog,args.episodes,args.seed)
        manifest = {"schema":"egosuite-joint-axis-geometric-audit-v1", "seed":args.seed,
                    "catalog":str(args.catalog.resolve()), "catalog_sha256":catalog_sha256,
                    "catalog_byte_snapshot_and_path_hash_match":True,
                    "audit_script":str(Path(__file__).resolve()), "audit_script_sha256":file_sha256(__file__),
                    "pipeline_implementation_sha256":{
                        p.name:file_sha256(p) for p in sorted(Path(__import__("egomotion_pipeline").__file__).parent.glob("*.py"))},
                    "max_frames_per_episode":args.max_frames, "max_pose_error_ns":round(args.max_pose_error_ms*1e6),
                    "max_angular_error_deg":args.max_angular_error_deg,
                    "min_anchor_frames":args.min_anchor_frames, "cpu_threads":2,
                    "fit_operator_count":args.episodes//2,"heldout_operator_count":args.episodes//2,
                    "sampling":"seeded unique operator and task preference; alternating disjoint-operator split",
                    "selected_sources":selected,"target":target_meta,"anchors":anchors,
                    "target_mcp_joint_names_source":TARGET_JOINT_NAMES_SOURCE,
                    "hand_correspondence_assumption":"source thumb/index/middle/ring/pinky, four points each; MCP indices5/9/13/17; source naming not independently authenticated",
                    "reference_source":"source global wxyz; column vectors; C maps target local to source local",
                    "production_archive_modified":False, "semantic_verified":False}
        atomic_json(output/"manifest.json",numeric_json(manifest))
        update(stage="reading_source_prefixes",total_episodes=len(selected),completed_episodes=0)
        all_rays, samples = [], []
        for i,row in enumerate(selected):
            log(f"Reading {i+1}/{len(selected)} operator={row['operator']} split={row['split']} uid={row['uid']}")
            data, provenance = read_prefix(row["path"],args.max_frames,round(args.max_pose_error_ms*1e6))
            sample_file = output/"samples"/f"{i:03d}-{row['uid']}.npz"
            np.savez_compressed(sample_file,**data)
            rays = source_rays(data,anchors)
            for joint_rays in rays:
                for k,ray in enumerate(joint_rays):
                    if ray is not None and ray["n"]<args.min_anchor_frames:
                        joint_rays[k]=None
            all_rays.append(rays)
            sample={"sample_index":i, "operator_id":row["operator"], "source_uid":row["uid"],
                    "split":row["split"], "numeric_sample":str(sample_file),
                    "numeric_sample_sha256":file_sha256(sample_file), "source_provenance":provenance,
                    "source_local_anchors":rays}
            samples.append(sample)
            update(completed_episodes=i+1,frames_read=sum(s["source_provenance"]["body_frames"] for s in samples))
        fit_i=[i for i,row in enumerate(selected) if row["split"]=="fit"]
        held_i=[i for i,row in enumerate(selected) if row["split"]=="heldout"]
        update(stage="fitting_candidates")
        result=[]
        combined_fit, combined_target, combined_weights = [], [], []
        matrices=np.full((22,3,3),np.nan)
        for joint,name in enumerate(BODY_NAMES):
            source,target,weights,evidence=constraint_arrays(all_rays,anchors,fit_i,joint)
            hs,ht,hw,he=constraint_arrays(all_rays,anchors,held_i,joint)
            fit=fit_direction_basis(source,target,weights)
            rank=min(fit["source_rank"],fit["target_rank"])
            candidate=fit["matrix"] if len(source) else None
            if candidate is not None:
                matrices[joint]=candidate
                fe=errors_for_matrix(source,target,candidate)
                held=errors_for_matrix(hs,ht,candidate)
            else:
                fe=held=np.empty(0)
            fstats,hstats=error_summary(fe,weights),error_summary(held,hw)
            fit_ops=len({e["sample_index"] for e in evidence})
            held_ops=len({e["sample_index"] for e in he})
            sufficient=fit_ops>=2 and held_ops>=2
            numerical=bool(sufficient and fstats["max_deg"]<=args.max_angular_error_deg
                           and hstats["max_deg"]<=args.max_angular_error_deg)
            entry={"joint_index":joint,"joint_name":name,"candidate_matrix":candidate,
                   "geometry_rank":rank,"source_rank":fit["source_rank"],"target_rank":fit["target_rank"],
                   "observable_dof":fit["observable_dof"],"unresolved_dof":3-fit["observable_dof"],
                   "semantic_anchor_count":len(anchors[joint]),"anchors":anchors[joint],
                   "fit_operator_count":fit_ops,"heldout_operator_count":held_ops,
                   "fit_error":fstats,"heldout_error":hstats,
                   "fit_operator_anchor_errors_deg":fe,"heldout_operator_anchor_errors_deg":held,
                   "fit_evidence":evidence,"heldout_evidence":he,
                   "source_singular_values":fit["source_singular_values"],
                   "target_singular_values":fit["target_singular_values"],
                   "numerical_direction_agreement_pass":numerical,
                   "full_rotation_observable":rank>=2,
                   "semantic_verified":False,"rotation_basis_verified":False,
                   "reason":("no target anchor / no downstream source anchor" if not len(source) else
                              "bone direction constrained; twist unobservable" if rank==1 else
                              "geometric multi-anchor candidate; official bind/paired orientation reference not authenticated")}
            result.append(entry)
            if len(source):
                # Equal joint weight in the shared hypothesis; within each joint
                # each operator contributes total weight one.
                combined_fit.extend(source)
                combined_target.extend(target)
                combined_weights.extend(weights/weights.sum())
        sf=np.array(combined_fit).reshape(-1,3)
        tf=np.array(combined_target).reshape(-1,3)
        wf=np.array(combined_weights)
        permutations=[]
        for matrix in signed_axis_permutations():
            stats=error_summary(errors_for_matrix(sf,tf,matrix),wf)
            permutations.append({"matrix":matrix,"fit_direction_error":stats})
        permutations.sort(key=lambda r:r["fit_direction_error"]["rmse_deg"])
        shared=permutations[0]
        shared_held_errors,shared_held_weights=[],[]
        for joint in range(22):
            hs,ht,hw,_=constraint_arrays(all_rays,anchors,held_i,joint)
            if len(hs):
                shared_held_errors.extend(errors_for_matrix(hs,ht,shared["matrix"]))
                shared_held_weights.extend(hw/hw.sum())
        shared["heldout_direction_error"]=error_summary(shared_held_errors,shared_held_weights)
        shared.update(hypothesis="one common proper signed-axis permutation for all joints",
                      hypothesis_count=24, semantic_verified=False,
                      limitations="shared basis assumed; native rig and SMPL-X rest geometry differ")
        source_median=np.full((len(selected),22,4,3),np.nan)
        source_valid=np.zeros((len(selected),22,4),bool)
        source_counts=np.zeros((len(selected),22,4),np.int64)
        source_p95=np.full((len(selected),22,4),np.nan)
        target_vectors=np.full((22,4,3),np.nan)
        for joint,specs in enumerate(anchors):
            for k,spec in enumerate(specs):
                target_vectors[joint,k]=spec["target_vector"]
            for i in range(len(selected)):
                for k,ray in enumerate(all_rays[i][joint]):
                    if ray:
                        source_median[i,joint,k]=ray["median_unit"]
                        source_valid[i,joint,k]=True
                        source_counts[i,joint,k]=ray["n"]
                        source_p95[i,joint,k]=ray["stability"]["p95_deg"]
        np.savez_compressed(output/"anchors-and-candidates.npz", source_median=source_median,
                            source_anchor_valid=source_valid,source_anchor_frames=source_counts,
                            source_rigidity_p95_deg=source_p95,target_vectors=target_vectors,
                            candidate_matrices=matrices,fit_samples=np.array([i in fit_i for i in range(len(selected))]),
                            operator_ids=np.array([r["operator"] for r in selected],dtype="U36"),
                            source_uids=np.array([r["uid"] for r in selected],dtype="U36"),
                            observable_dof=np.array([r["observable_dof"] for r in result],np.int64),
                            semantic_verified=np.zeros(22,bool))
        report={"schema":"egosuite-joint-axis-geometric-audit-v1", "manifest":str(output/"manifest.json"),
                "target":target_meta,"samples":samples,"joints":result,
                "shared_signed_permutation_candidate":shared,
                "shared_signed_permutation_all_fit_scores":permutations,
                "fit_operator_ids":[selected[i]["operator"] for i in fit_i],
                "heldout_operator_ids":[selected[i]["operator"] for i in held_i],
                "fit_heldout_operators_disjoint":True,
                "operator_weighting":"equal total direction weight per operator per joint; median unit ray per semantic anchor",
                "semantic_verified_joint_count":0,"all_22_axes_verified":False,
                "full_rotation_observable_joint_count":sum(r["full_rotation_observable"] for r in result),
                "numerical_direction_pass_joint_count":sum(r["numerical_direction_agreement_pass"] for r in result),
                "production_archive_modified":False,"training_modified":False,
                "limitations":["geometric rest-direction alignment is not native-axis authentication",
                               "one child ray leaves arbitrary twist even with multiple poses and operators",
                               "head and both feet have no downstream position anchors in dense22",
                               "wrist source finger identity and target rest geometry correspondence remain an explicit assumption",
                               "SMPL-X neutral morphology differs from source mocap rigs; a shared axis permutation may not align bones",
                               "SMPL-X reference asset must not be substituted for the separate HM SMPL-H decoder",
                               "missing sparse vendor bad-frame flags follow existing pipeline behavior, finite/matched masks still enforced",
                               "raw source prefix audit does not perform a full 6614-episode validation"]}
        atomic_json(output/"report.json",numeric_json(report))
        update(stage="completed",all_22_axes_verified=False,semantic_verified_joint_count=0,
               report=str(output/"report.json"),
               full_rotation_observable_joint_count=report["full_rotation_observable_joint_count"],
               numerical_direction_pass_joint_count=report["numerical_direction_pass_joint_count"])
        log(f"Completed geometric audit: observable={report['full_rotation_observable_joint_count']}/22; semantically verified=0/22")
        return 0
    except BaseException as error:
        update(stage="failed",error=f"{type(error).__name__}: {error}")
        log(traceback.format_exc())
        return 1


if __name__=="__main__":
    raise SystemExit(main())
