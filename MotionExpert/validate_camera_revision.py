"""Independently verify only the new camera masks, continuity and immutable delta.

Camera native metrics/rules are recomputed independently; original final-FK,
hand/world observations and DINO numerical qualification remain inherited.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .dataset.camera_revision_acceptance import (ACCEPTANCE_FORMAT, REVISION_FORMAT,
    AUDIT_FORMAT, INDEPENDENT_AUDIT_FORMAT, ACCEPTANCE_FLAGS, check_camera_revision_acceptance)
from .dataset.smplh_archive import SMPLHParameterArchive, file_sha256
from .dataset.vision_cache import storage_digest
from .dataset.window_cache import atomic_json, digest
from .validate_hm_revision import _owned, _same_file, _assert_values, _arrays, _write_new
from .watch_dino import locked

PROFILE = "vision_profiles/psi0_256x480_v1_30hz"
MANIFEST = ".pipeline/camera-quality-revision.json"
QUALITY = ".pipeline/camera-quality/report.json"


def _now(): return datetime.now(timezone.utc).isoformat()


def independent_rule_check(values, source, policy):
    """Independent scalar-vector policy; does not call producer evaluate_metrics."""
    expected_keys = {"max_pose_error_s", "max_native_gap_s", "max_mps_gap_s", "velocity_margin_m",
        "velocity_multiplier", "angular_margin_rad", "angular_multiplier", "max_camera_speed_m_s", "max_camera_angular_speed_rad_s"}
    if set(policy) != expected_keys or any(not np.isfinite(x) or x<=0 for x in policy.values()):
        raise ValueError("Incomplete independent camera policy")
    point = values["native_valid"] & values["transform_finite"]
    dt = values["dt_s"]
    edge = point & np.isfinite(dt) & (dt>0) & (dt<=policy["max_native_gap_s"])
    edge[0] = point[0]
    if source == "egoexo4d_hm":
        for name in ("closed_score_good", "open_score_good", "closed_time_good", "open_time_good"):
            point &= values[name]
        for prefix in ("closed", "open"):
            q = values[prefix+"_quality"]
            # This is selected raw quality. Dynamic nonfiniteness may further reject,
            # but it must never make a missing or zero confidence valid.
            if np.any(values[prefix+"_score_good"] & ~(np.isfinite(q)&(q>0)&(q<=1))):
                raise ValueError("Camera confidence invalidity was upgraded")
            expected_time = abs(values[prefix+"_time_error_s"]) <= policy["max_pose_error_s"]
            if not np.array_equal(expected_time, values[prefix+"_time_good"]):
                raise ValueError("Camera source timestamp policy differs")
        edge &= values["closed_interval_good"] & values["open_interval_good"]
        velocity_ok = values["camera_distance_m"] <= policy["velocity_margin_m"] + policy["velocity_multiplier"]*values["source_linear_speed_bound_m_s"]*np.maximum(dt,0)
        angular_ok = values["camera_angle_rad"] <= policy["angular_margin_rad"] + policy["angular_multiplier"]*values["source_angular_speed_bound_rad_s"]*np.maximum(dt,0)
        velocity_ok[0] = angular_ok[0] = True
        edge &= velocity_ok & angular_ok
    elif source != "egosuite": raise ValueError("Unknown camera source")
    linear = values["camera_distance_m"] <= policy["max_camera_speed_m_s"]*np.maximum(dt,0)
    angular = values["camera_angle_rad"] <= policy["max_camera_angular_speed_rad_s"]*np.maximum(dt,0)
    linear[0] = angular[0] = True
    edge &= linear & angular & point
    edge[1:] &= point[:-1]
    if not np.array_equal(point, values["quality_valid"]) or not np.array_equal(edge, values["transition_valid"]):
        raise ValueError("Independent camera point/continuity policy disagrees")
    return point, edge


def _raw_mps(path, *, opened):
    """CSV decoding only: all quality/session/time rules are below, independently."""
    import pyarrow as pa
    import pyarrow.csv as csv
    identity = "session_uid" if opened else "graph_uid"
    velocity_frame = "odometry" if opened else "device"
    names = ["tracking_timestamp_us",identity,"quality_score"]
    names += [f"device_linear_velocity_{axis}_{velocity_frame}" for axis in "xyz"]
    names += [f"angular_velocity_{axis}_device" for axis in "xyz"]
    if not opened:
        names += [f"t{axis}_world_device" for axis in "xyz"]+[f"q{axis}_world_device" for axis in "wxyz"]
    types={name:pa.float64() for name in names};types[identity]=pa.string();types["tracking_timestamp_us"]=pa.int64()
    table=csv.read_csv(path,read_options=csv.ReadOptions(use_threads=False),
        convert_options=csv.ConvertOptions(include_columns=names,column_types=types))
    if any(table[name].null_count for name in names):raise ValueError("Independent MPS source contains null required fields")
    return {name:table[name].combine_chunks().to_numpy(zero_copy_only=False) for name in names}


def _independent_source_metrics(record, arrays, values, qc):
    """Re-read raw sources; correlated erroneous producer flags cannot pass."""
    evidence=qc.get("source_evidence",{})
    if record["source"]=="egosuite":
        from google.protobuf.descriptor_pb2 import FileDescriptorSet
        from mcap.reader import make_reader
        source=Path(record["metadata"]["source_data"])
        stat=source.stat();identity={"bytes":stat.st_size,"mtime_ns":stat.st_mtime_ns}
        if evidence.get("source_path")!=str(source) or evidence.get("source_stat")!=identity:
            raise ValueError("EgoSuite original camera source identity differs")
        with source.open("rb") as stream:summary=make_reader(stream).get_summary()
        channels=[channel for channel in summary.channels.values() if channel.topic=="/sensor/camera/head_left/extrinsic"]
        if len(channels)!=1:raise ValueError("EgoSuite original camera schema channel differs")
        schema=summary.schemas[channels[0].schema_id]
        import hashlib
        parsed=FileDescriptorSet();parsed.ParseFromString(schema.data)
        fields={message.name:[field.name for field in message.field]
            for file in parsed.file if file.package=="foxglove"
            for message in file.message_type if message.name in {"FrameTransforms","FrameTransform"}}
        if (schema.name!="foxglove.FrameTransforms" or evidence.get("camera_schema")!=schema.name
            or evidence.get("camera_schema_sha256")!=hashlib.sha256(schema.data).hexdigest()
            or evidence.get("camera_fields")!=fields
            or set(fields.get("FrameTransforms",[]))!={"transforms"}
            or set(fields.get("FrameTransform",[]))!={"timestamp","parent_frame_id","child_frame_id","translation","rotation"}
            or evidence.get("confidence_observed") is not False):
            raise ValueError("EgoSuite missing confidence source policy is not supported by actual schema")
        after=source.stat()
        if {"bytes":after.st_size,"mtime_ns":after.st_mtime_ns}!=identity:raise ValueError("EgoSuite source changed during independent camera audit")
        return
    from scipy.spatial.transform import Rotation
    provenance=record["metadata"]["aria_provenance"]
    closed_path=Path(provenance["trajectory"]["path"])
    paths={"closed":closed_path,"open":closed_path.with_name("open_loop_trajectory.csv")}
    if "source.capture_timestamp_ns" not in arrays:
        raise ValueError("Independent Aria camera audit requires original integer capture_timestamp_ns")
    query=arrays["source.capture_timestamp_ns"]
    policy=qc["policy"]; extracted={}; interval_clock_evidence={}
    for prefix,path in paths.items():
        desc=evidence.get("source_paths",{}).get("closed_loop" if prefix=="closed" else "open_loop",{})
        stat=path.stat();identity={"bytes":stat.st_size,"mtime_ns":stat.st_mtime_ns}
        if desc.get("path")!=str(path) or any(desc.get(k)!=v for k,v in identity.items()) or file_sha256(path)!=desc.get("sha256"):
            raise ValueError("Independent original MPS source identity/SHA differs")
        if prefix=="closed" and desc.get("sha256")!=provenance["trajectory"].get("sha256"):
            raise ValueError("Independent MPS closed source differs from registered archive")
        raw=_raw_mps(path,opened=prefix=="open")
        time=raw["tracking_timestamp_us"]*1e-6
        if not len(time) or not np.isfinite(time).all() or np.any(time[1:]<=time[:-1]):
            raise ValueError("Independent source timestamps are not strictly increasing")
        # MPS1.5.5 stores int64(us)*1000; queryMapByTimestamp compares
        # strictly '<', hence exact integer midpoint ties select AFTER.
        source_ns=raw["tracking_timestamp_us"]*1000
        upper=np.minimum(np.searchsorted(source_ns,query,side="left"),len(time)-1)
        lower=np.maximum(upper-1,0)
        chosen=np.where(abs(query-source_ns[lower]) < abs(source_ns[upper]-query),lower,upper)
        frame="odometry" if prefix=="open" else "device"
        linear=np.sqrt(sum(raw[f"device_linear_velocity_{axis}_{frame}"]**2 for axis in "xyz"))
        angular=np.sqrt(sum(raw[f"angular_velocity_{axis}_device"]**2 for axis in "xyz"))
        quality=raw["quality_score"]
        good=np.isfinite(quality)&(quality>0)&(quality<=1)&np.isfinite(linear)&np.isfinite(angular)
        if prefix=="closed":
            p=np.column_stack([raw[f"t{axis}_world_device"] for axis in "xyz"])
            q=np.column_stack([raw[f"q{axis}_world_device"] for axis in "wxyz"])
            good &= np.isfinite(p).all(1)&np.isfinite(q).all(1)&(np.linalg.norm(q,axis=1)>1e-8)
            extracted["closed_position"]=p[chosen];extracted["closed_quaternion"]=q[chosen]
        ids=raw["session_uid" if prefix=="open" else "graph_uid"]
        source_edge=np.ones(len(time),bool);source_edge[0]=good[0]
        gap_ns=np.diff(source_ns);limit_ns=round(policy["max_mps_gap_s"]*1e9)
        integer_gap_good=gap_ns<=limit_ns
        floating_gap_good=np.diff(time)<=policy["max_mps_gap_s"]
        interval_clock_evidence[prefix]={"source_edges":len(gap_ns),"exact_limit_gap_edges":int((gap_ns==limit_ns).sum()),
            "float_integer_predicate_differences":int((integer_gap_good!=floating_gap_good).sum()),
            "effective_good_sample_predicate_differences":int(((integer_gap_good!=floating_gap_good)&good[1:]&good[:-1]&(ids[1:]==ids[:-1])).sum())}
        source_edge[1:]=good[1:]&good[:-1]&(ids[1:]==ids[:-1])&integer_gap_good
        # Prefix boundary counts, and a separate inclusive point count, qualify
        # every raw source sample, including ones skipped by native30Hz queries.
        bad_edges=np.r_[0,np.cumsum(~source_edge,dtype=np.int64)]
        bad_points=np.r_[0,np.cumsum(~good,dtype=np.int64)]
        continuous=good[chosen].copy()
        low,high=chosen[:-1],chosen[1:]
        continuous[1:] &= (high>=low)&(bad_edges[high+1]==bad_edges[low+1])&(bad_points[high+1]==bad_points[low])
        expected={prefix+"_score_good":good[chosen],prefix+"_time_good":abs(source_ns[chosen]-query)<=round(policy["max_pose_error_s"]*1e9),
            prefix+"_interval_good":continuous,prefix+"_quality":quality[chosen],prefix+"_time_error_s":(source_ns[chosen]-query)*1e-9}
        for name,value in expected.items():_assert_values(values[name],value,"independent original MPS "+name)
        extracted[prefix+"_linear"]=linear[chosen];extracted[prefix+"_angular"]=angular[chosen]
        after=path.stat()
        if {"bytes":after.st_size,"mtime_ns":after.st_mtime_ns}!=identity:raise ValueError("Raw MPS changed during independent camera audit")
    device_camera=np.asarray(provenance["device_camera_transform"],np.float64)
    arm=np.sqrt(np.sum(device_camera[:3,3]**2))
    angular=np.maximum(extracted["closed_angular"],extracted["open_angular"])
    linear=np.maximum(extracted["closed_linear"],extracted["open_linear"])+arm*angular
    for name,value in (("source_linear_speed_bound_m_s",np.maximum(linear,np.r_[linear[:1],linear[:-1]])),
                       ("source_angular_speed_bound_rad_s",np.maximum(angular,np.r_[angular[:1],angular[:-1]]))):
        if not np.allclose(values[name],value,atol=1e-12,rtol=1e-12,equal_nan=True):raise ValueError("Independent raw MPS dynamics bound differs")
    finite=np.isfinite(extracted["closed_quaternion"]).all(1)&(np.linalg.norm(extracted["closed_quaternion"],axis=1)>1e-8)
    checked=values["native_valid"]&values["closed_time_good"]&finite
    if checked.any():
        raw_rotation=Rotation.from_quat(extracted["closed_quaternion"][checked][:,[1,2,3,0]]).as_matrix()
        camera_p=extracted["closed_position"][checked]+raw_rotation@device_camera[:3,3]
        archive_p=arrays["observation.state.camera_world"][checked]
        if np.linalg.norm(camera_p-archive_p,axis=1).max()>.02:raise ValueError("Independent source/registered camera position differs")
        camera_r=raw_rotation@device_camera[:3,:3]
        archive_r=Rotation.from_quat(arrays["observation.state.camera_world_rotation"][checked][:,[1,2,3,0]]).as_matrix()
        if abs(camera_r-archive_r).max()>1e-4:raise ValueError("Independent source/registered camera rotation differs")
    return interval_clock_evidence


def _camera_check(task):
    """All new camera observations only; unrelated full numeric QA is inherited."""
    from scipy.spatial.transform import Rotation
    import pyarrow as pa
    pa.set_cpu_count(1); pa.set_io_thread_count(1)
    root, record, qc, directory = Path(task["parent"]), task["record"], task["qc"], Path(task["directory"])
    index = record["episode_index"]
    from .camera_quality import SOURCE_POLICIES
    if qc.get("source_policy") != SOURCE_POLICIES[record["source"]]:
        raise ValueError("Camera source-specific confidence/world policy differs")
    if (qc.get("record_sha256") != file_sha256(root/f".pipeline/episodes/{index:08d}.json")
            or qc.get("motion_sha256") != record["motion_sha256"] or qc.get("length") != record["length"]):
        raise ValueError("Camera evidence names another parent episode")
    path = _owned(directory, qc["mask_path"])
    if file_sha256(path) != qc.get("mask_sha256"): raise ValueError("Camera sidecar changed")
    with np.load(path, allow_pickle=False) as saved: values = {k:saved[k] for k in saved.files}
    if any(a.shape != (record["length"],) for a in values.values()): raise ValueError("Camera sidecar native population differs")
    point, edge = independent_rule_check(values, record["source"], qc["policy"])
    columns = ["timestamp", "observation.state.camera_world", "observation.state.camera_world_rotation",
        "observation.valid.camera_position", "observation.valid.camera_rotation",
        "observation.valid.head_reference_position", "observation.valid.head_reference_rotation",
        "observation.valid.head_reference_basis"]
    file = pq.ParquetFile(root/record["data_path"])
    if "source.capture_timestamp_ns" in file.schema_arrow.names: columns.append("source.capture_timestamp_ns")
    arrays = _arrays(file.read(columns=columns,use_threads=False))
    native = np.ones(record["length"],bool)
    for name in columns:
        if name.startswith("observation.valid."): native &= arrays[name]
    if not np.array_equal(native, values["native_valid"]): raise ValueError("Native reference availability differs")
    p, q = arrays["observation.state.camera_world"], arrays["observation.state.camera_world_rotation"]
    finite = np.isfinite(p).all(1)&np.isfinite(q).all(1)&(np.linalg.norm(q,axis=1)>1e-8)
    times = arrays["source.capture_timestamp_ns"]*1e-9 if "source.capture_timestamp_ns" in arrays else arrays["timestamp"].astype(np.float64)
    finite &= np.isfinite(times)
    if not np.array_equal(finite, values["transform_finite"]): raise ValueError("Native camera finite mask differs")
    distance = np.r_[0,np.linalg.norm(np.diff(p,axis=0),axis=1)]
    # scipy works in xyzw and normalizes internally; no producer quaternion helper.
    angles=np.zeros(len(q)); validq=finite[1:]&finite[:-1]
    if validq.any():
        ids=np.flatnonzero(validq)
        first=Rotation.from_quat(q[ids][:,[1,2,3,0]])
        second=Rotation.from_quat(q[ids+1][:,[1,2,3,0]])
        angles[ids+1]=(first.inv()*second).magnitude()
    if not np.allclose(values["camera_distance_m"],distance,atol=1e-7,rtol=1e-6): raise ValueError("Camera distance metric differs")
    if not np.allclose(values["camera_angle_rad"][np.r_[True,validq]],angles[np.r_[True,validq]],atol=2e-6,rtol=1e-4):
        raise ValueError("Independent scipy angular metric differs")
    if not np.allclose(values["dt_s"],np.r_[0,np.diff(times)],atol=1e-7,rtol=1e-6): raise ValueError("Camera time metric differs")
    clock_evidence=_independent_source_metrics(record,arrays,values,qc) or {}
    summary=qc["summary"]
    oldedge=native.copy(); oldedge[1:] &= native[:-1]
    expect={"native_valid_frames":int(native.sum()),"quality_valid_frames":int(point.sum()),
        "newly_rejected_frames":int((native&~point).sum()),"newly_rejected_native_edges":int((oldedge&~edge).sum()),
        "invalid_native_frames":int((~native).sum()),"representation_changed_due_native_continuity":bool((~native).any())}
    expect["motion_revision_required"]=bool(expect["newly_rejected_frames"] or expect["newly_rejected_native_edges"] or expect["invalid_native_frames"])
    if any(summary.get(k)!=v for k,v in expect.items()): raise ValueError("Camera revised population/continuity summary differs")
    return {"episode_index":index,"native_frames":len(native),"source":record["source"],"changed":expect["motion_revision_required"],"source_interval_integer_clock":clock_evidence}


def _expected_masks(old, values, source):
    out={}
    for modality in (["camera","head_reference","head"] if source=="egoexo4d_hm" else ["camera","head_reference"]):
        for kind in ("position","rotation"):
            key=f"observation.valid.{modality}_{kind}"
            out[key]=old[key]&values["quality_valid"]
            ikey=f"observation.interpolated.{modality}_{kind}"
            out[ikey]=old[ikey]&out[key]
    out["observation.valid.camera_tracking"]=values["quality_valid"]
    out["observation.valid.camera_transition"]=values["transition_valid"]
    return out


def _snapshot(parent,candidate,manifest,quality):
    records=[]
    for role,archive in (("parent",parent),("candidate",candidate)):
        for record in archive._records:
            index=record["episode_index"]
            for path in (archive.data_path(index), archive.owned_path(f".pipeline/episodes/{index:08d}.json"),
                         archive.owned_path(f"episodes/{index:08d}/receipt.json")):
                s=path.stat(); records.append([role,str(path),s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns])
        records.append([role,archive.binding])
    for path in sorted((candidate.root/".pipeline/camera-quality").glob("episode_*")):
        stat=path.stat();records.append(["camera_evidence",str(path.relative_to(candidate.root)),stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns])
    return digest({"identities":records,"manifest":file_sha256(manifest),"quality":file_sha256(quality),
                   "parent_publication":file_sha256(parent.state_path)})


def audit_revision(parent_root,revision_root,*,workers=8,progress=None):
    if not 1<=workers<=8: raise ValueError("Use1..8 verification workers")
    root=Path(revision_root).resolve(strict=True)
    with locked(root/".revision-build.lock"):
        plan=json.loads(_owned(root,".pipeline/revision-plan.json").read_text())
        parent=SMPLHParameterArchive(parent_root,vision_profile_root=plan["source_vision_root"])
        candidate=SMPLHParameterArchive(root,allow_unpublished=True,vision_profile_root=root/PROFILE)
        manifest_path,quality_path=_owned(root,MANIFEST),_owned(root,QUALITY)
        manifest,quality=json.loads(manifest_path.read_text()),json.loads(quality_path.read_text())
        build=manifest["build_binding"]; build_sha=digest(build)
        if (candidate.state.get("state")!="candidate" or manifest.get("format")!=REVISION_FORMAT
            or manifest.get("scope")!="full" or manifest.get("binding")!=candidate.binding
            or build!=plan.get("binding") or build_sha!=plan.get("binding_sha256")
            or manifest.get("build_binding_sha256")!=build_sha or build.get("scope")!="full"
            or build.get("parent_archive_binding")!=parent.binding or build.get("parent_archive_fingerprint")!=parent.archive_fingerprint
            or build.get("quality_report_sha256")!=file_sha256(quality_path)
            or build.get("policy")!=quality.get("policy")
            or Path(manifest["source_root"]).resolve()!=parent.root or Path(manifest["output_root"]).resolve()!=root):
            raise ValueError("Camera candidate, parent and policy binding differ")
        if (quality.get("format")!=AUDIT_FORMAT or quality.get("state")!="complete" or quality.get("diagnostic") is not False
            or quality.get("production_modified") is not False or quality.get("source_population",{}).get("scope")!="all_archive"
            or quality.get("source_population",{}).get("total")!=len(parent._records)
            or quality.get("binding")!=parent.binding or Path(quality["archive_root"]).resolve()!=parent.root):
            raise ValueError("Camera audit is partial or differs from parent")
        a,b=deepcopy(parent.info),deepcopy(candidate.info)
        old_declaration=a.pop("quality_revision",None); declaration=b.pop("quality_revision",None)
        if (a!=b or declaration.get("format")!=REVISION_FORMAT or declaration.get("parent_revision")!=old_declaration
            or declaration.get("quality_report_sha256")!=file_sha256(quality_path)
            or declaration.get("policy")!=quality.get("policy")):
            raise ValueError("Camera revision changed schema or dropped parent quality qualification")
        if (len(parent._records)!=len(candidate._records) or parent.frames!=candidate.frames
            or parent.model_sha256!=candidate.model_sha256 or parent.split_fingerprint!=candidate.split_fingerprint):
            raise ValueError("Camera revision changes original source/split/model population")
        _same_file(parent.model_path,candidate.model_path,"body model")
        for path in (parent.root/"meta").glob("splits*.json"): _same_file(path,root/"meta"/path.name,"splits")
        prefix=manifest["parent_evidence_prefix"]
        parentpub=_owned(root,prefix+"/.pipeline/dataset-state.json")
        parentaccept=_owned(root,prefix+"/.pipeline/acceptance-report.json")
        _same_file(parent.state_path,parentpub,"parent publication")
        _same_file(parent.root/parent.state["acceptance_report"],parentaccept,"parent acceptance")
        if file_sha256(parentaccept)!=build.get("parent_acceptance_sha256") or file_sha256(parentpub)!=build.get("parent_publication_sha256"):
            raise ValueError("Inherited parent proof identity differs")
        receipts=quality.get("episodes",[]); qc={r["episode_index"]:r for r in receipts}
        required=list(range(len(parent._records)))
        if (sorted(qc)!=required or len(qc)!=len(receipts) or quality.get("episodes_completed")!=len(qc)
            or quality.get("episodes_planned")!=len(qc) or manifest.get("quality_episodes")!=required
            or manifest.get("required_quality_episodes")!=required): raise ValueError("Camera audit population incomplete")
        tasks=[{"parent":str(parent.root),"record":r,"qc":qc[r["episode_index"]],"directory":str(quality_path.parent)} for r in parent._records]
        native_checks=[]
        with ProcessPoolExecutor(workers) as pool:
            for result in pool.map(_camera_check,tasks,chunksize=1):
                native_checks.append(result)
                if progress: progress("native_camera",len(native_checks),len(tasks))
        expected_changed=[r["episode_index"] for r in native_checks if r["changed"]]
        if manifest.get("changed_episode_indices")!=expected_changed or manifest.get("motion_rewritten_episodes")!=expected_changed:
            raise ValueError("Camera changed computation population differs")
        snapshot=_snapshot(parent,candidate,manifest_path,quality_path)
        certificates,comparisons,sources={},{},{}
        for old,new in zip(parent._records,candidate._records):
            index=old["episode_index"]; receipt=qc[index]
            if json.loads(_owned(quality_path.parent,f"episode_{index:08d}.json").read_text())!=receipt or receipt.get("policy")!=quality["policy"]:
                raise ValueError("Camera individual receipt differs from complete audit")
            entry=json.loads(candidate.owned_path(f"episodes/{index:08d}/receipt.json").read_text())["quality_revision"]
            if (entry.get("build_binding_sha256")!=build_sha or entry.get("parent_record_sha256")!=digest(old)
                or entry.get("quality_mask_sha256")!=receipt["mask_sha256"] or entry.get("ground_recomputed") is not False
                or entry.get("motion_rewritten") is not (index in expected_changed)):
                raise ValueError("Camera episode receipt identity/scope differs")
            if index in expected_changed:
                before=_arrays(pq.read_table(parent.data_path(index),use_threads=False))
                after=_arrays(pq.read_table(candidate.data_path(index),use_threads=False))
                with np.load(_owned(quality_path.parent,receipt["mask_path"]),allow_pickle=False) as saved:
                    masks=_expected_masks(before,{name:saved[name] for name in ("quality_valid","transition_valid")},old["source"])
                if set(after)!=set(before)|set(masks): raise ValueError("Camera rewrite changes unauthorized schema")
                for name in after: _assert_values(after[name],masks.get(name,before.get(name)),f"camera episode{index}:{name}")
                if file_sha256(candidate.data_path(index))!=new["motion_sha256"]: raise ValueError("Rewritten camera Parquet digest differs")
                expected=deepcopy(old); expected["motion_sha256"]=new["motion_sha256"]
                meta=new["metadata"].get("camera_quality_revision",{})
                if (meta.get("format")!=REVISION_FORMAT or meta.get("quality_mask_sha256")!=receipt["mask_sha256"]
                    or meta.get("quality_receipt_sha256")!=digest(receipt) or meta.get("quality_report_sha256")!=file_sha256(quality_path)
                    or meta.get("policy")!=quality["policy"] or meta.get("source_policy")!=receipt["source_policy"]):
                    raise ValueError("Rewritten camera metadata differs")
                expected["metadata"]["camera_quality_revision"]=meta
                if new!=expected: raise ValueError("Camera revision changed unauthorized episode metadata")
                comparisons[str(index)]={"all_columns_compared":len(after),"frames":len(after["timestamp"]),"only_camera_mask_columns_changed":True}
            else:
                if old!=new: raise ValueError("Unchanged camera episode record must remain exact")
                _same_file(parent.data_path(index),candidate.data_path(index),"unchanged motion")
                certificates[str(index)]={"source_record_sha256":digest(old),"motion_sha256":old["motion_sha256"],
                    "native_reference_all_valid":True,"native_transition_all_valid":True,
                    "camera_audit_receipt_sha256":digest(receipt),"camera_mask_sha256":receipt["mask_sha256"]}
            for name in ("features.npy","join.npz"):
                _same_file(parent.vision_cache._episodes[index]["artifacts"][name][0],candidate.vision_cache._episodes[index]["artifacts"][name][0],"DINO "+name)
            visual=json.loads(candidate.vision_cache._episodes[index]["receipt_path"].read_text())
            if (visual.get("source_record_sha256")!=storage_digest(new) or visual.get("quality_revision_reuse",{}).get("build_binding_sha256")!=build_sha):
                raise ValueError("Camera revision DINO join receipt binding differs")
            s=sources.setdefault(old["source"],{"episodes":0,"frames":0});s["episodes"]+=1;s["frames"]+=old["length"]
            if progress: progress("column_delta",index+1,len(parent._records))
        # Read real window masks on changed examples; no old full-FK numeric QA.
        from .dataset.smplh_representation import build_smplh_window
        probes=[]
        for index in expected_changed:
            record=candidate._records[index]
            if not record["metadata"].get("training_eligible",True) or not record["metadata"].get("canonical_training_eligible",True) or record["metadata"].get("quarantine_reasons"): continue
            if record["length"]<61: continue
            for start in range(0,min(record["length"]-60,121),30):
                try:
                    arrays,metadata=candidate.read_window(index,start_frame=start,load_dino=False)
                    window=build_smplh_window(arrays,metadata)
                except ValueError as e:
                    if "valid first camera pose" in str(e) or "horizontal projection" in str(e): continue
                    raise
                if not np.isfinite(window.motion).all() or not np.isfinite(window.trajectory).all(): raise ValueError("New camera window is nonfinite")
                invalid=~arrays["observation.valid.head_reference_transition"]; invalid[0]=False
                if window.motion_valid[invalid,108:117].any() or window.trajectory_valid[invalid,9:18].any(): raise ValueError("Window bridges invalid native camera edge")
                probes.append({"episode_index":index,"start_frame":start}); break
            if len(probes)>=8:break
        parent.assert_current(); candidate.assert_current()
        if _snapshot(parent,candidate,manifest_path,quality_path)!=snapshot: raise ValueError("Camera artifacts changed during independent validation")
        population={"episodes":len(parent._records),"frames":parent.frames,"sources":sources}
        if manifest.get("population")!=population: raise ValueError("Camera population summary differs")
        interval_clocks={prefix:{key:sum(result.get("source_interval_integer_clock",{}).get(prefix,{}).get(key,0) for result in native_checks)
            for key in ("source_edges","exact_limit_gap_edges","float_integer_predicate_differences","effective_good_sample_predicate_differences")}
            for prefix in ("closed","open")}
        return {"format":INDEPENDENT_AUDIT_FORMAT,"scope":"full","passed":True,"failures":[],"created_at":_now(),
            "binding":candidate.binding,"population":population,"parent_archive_binding":parent.binding,
            "parent_archive_fingerprint":parent.archive_fingerprint,"validator_sha256":file_sha256(__file__),
            "quality_audit_sha256":file_sha256(quality_path),"revision_manifest_sha256":file_sha256(manifest_path),
            "artifact_snapshot_sha256":snapshot,"changed_episode_indices":expected_changed,
            "unchanged_reference_certificates":certificates,"changed_motion_comparisons":comparisons,
            "camera_native_population":{"episode_indices":required,"native_frames_checked":sum(x["native_frames"] for x in native_checks)},
            "inherited_FK_numerical_QA_repeated":False,"original_source_quality_extraction_independently_repeated":True,
            "source_interval_integer_clock":interval_clocks,
            "camera_mask_policy_independently_recomputed":True,"camera_motion_metrics_independently_recomputed":True,
            "DINO_reextracted":False,"latest_motionexpert_window_probes":probes,**{name:True for name in ACCEPTANCE_FLAGS}}


def publish_revision(parent_root,revision_root,independent_report):
    root=Path(revision_root).resolve(strict=True)
    with locked(root/".revision-build.lock"):
        plan=json.loads(_owned(root,".pipeline/revision-plan.json").read_text())
        parent=SMPLHParameterArchive(parent_root,vision_profile_root=plan["source_vision_root"])
        candidate=SMPLHParameterArchive(root,allow_unpublished=True,vision_profile_root=root/PROFILE)
        path=Path(independent_report).resolve(strict=True)
        if not path.is_relative_to(root): raise ValueError("Independent camera report must be owned")
        report=json.loads(path.read_text()); manifest_path,quality_path=_owned(root,MANIFEST),_owned(root,QUALITY)
        if (candidate.state.get("state")!="candidate" or report.get("format")!=INDEPENDENT_AUDIT_FORMAT
            or report.get("passed") is not True or report.get("scope")!="full" or report.get("failures")!=[]
            or report.get("validator_sha256")!=file_sha256(__file__) or report.get("binding")!=candidate.binding
            or report.get("artifact_snapshot_sha256")!=_snapshot(parent,candidate,manifest_path,quality_path)
            or any(report.get(name) is not True for name in ACCEPTANCE_FLAGS)):
            raise ValueError("Camera independent proof stale, partial or failed")
        manifest=json.loads(manifest_path.read_text()); prefix=manifest["parent_evidence_prefix"]
        proof=lambda p:{"path":str(p.relative_to(root)),"sha256":file_sha256(p)}
        top={"format":ACCEPTANCE_FORMAT,"scope":"full","passed":True,"failures":[],"binding":candidate.binding,
            "population":report["population"],"created_at":_now(),"proofs":{
                "revision_manifest":proof(manifest_path),"quality_audit":proof(quality_path),"independent_revision_audit":proof(path),
                "parent_publication":proof(_owned(root,prefix+"/.pipeline/dataset-state.json")),
                "parent_acceptance":proof(_owned(root,prefix+"/.pipeline/acceptance-report.json"))},
            **{name:True for name in ACCEPTANCE_FLAGS}}
        top_path=root/".pipeline/acceptance-report.json"
        check_camera_revision_acceptance(candidate,top,top_path)
        _write_new(top_path,top)
        atomic_json(candidate.state_path,dict(candidate.state,state="accepted",independent_acceptance_pending=False,
            acceptance_report=str(top_path.relative_to(root)),acceptance_report_sha256=file_sha256(top_path),
            accepted_at=_now(),formal_training_ready=False))
        return top


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--parent-root",required=True)
    p.add_argument("--revision-root",required=True);p.add_argument("--workers",type=int,default=8)
    p.add_argument("--independent-report");p.add_argument("--publish",action="store_true")
    a=p.parse_args();root=Path(a.revision_root)
    report_path=Path(a.independent_report) if a.independent_report else root/".pipeline/camera-quality/independent-revision-audit.json"
    if a.publish: result=publish_revision(a.parent_root,root,report_path)
    else:
        def progress(stage,completed,total):
            state={"stage":stage,"completed":completed,"total":total,"updated_at":_now()}
            atomic_json(root/".pipeline/camera-quality-validation-progress.json",state)
            if completed%100==0 or completed==total:print(json.dumps(state),flush=True)
        result=audit_revision(a.parent_root,root,workers=a.workers,progress=progress)
        _write_new(report_path,result)
    print(json.dumps({"passed":result["passed"],"published":a.publish}),flush=True)


if __name__=="__main__":main()
