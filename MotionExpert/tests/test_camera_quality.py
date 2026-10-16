"""Camera resets cannot survive 30Hz ->5Hz decimation or erase source labels."""
from dataclasses import asdict
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from MotionExpert.camera_quality import (CameraQualityPolicy, camera_metrics, evaluate_metrics,
    interval_valid, selected_transition_valid, SOURCE_POLICIES)
from MotionExpert import revise_camera_archive as revision
from MotionExpert import validate_camera_revision as validate
from MotionExpert.dataset.camera_revision_acceptance import check_camera_revision_acceptance
from MotionExpert.dataset.smplh_archive import SMPLHParameterArchive, file_sha256
from MotionExpert.dataset.smplh_representation import build_smplh_window
from MotionExpert.dataset.window_cache import atomic_json, digest
from MotionExpert.tests.fixtures import owned_archive, parameter_arrays, rewrite_episode
from MotionExpert.tests.test_vision_profile_cache import write_cache


def metrics(count=20,source="egoexo4d_hm"):
    p=np.zeros((count,3));p[:,0]=np.arange(count)*.01
    q=np.tile([1.,0,0,0],(count,1))
    out=camera_metrics(p,q,np.arange(count)/30,np.ones(count,bool))
    if source=="egoexo4d_hm":
        for name in ("closed_score_good","closed_time_good","open_score_good","open_time_good","closed_interval_good","open_interval_good"):
            out[name]=np.ones(count,bool)
        out["source_linear_speed_bound_m_s"]=np.ones(count)
        out["source_angular_speed_bound_rad_s"]=np.ones(count)
        for prefix in ("closed","open"):
            out[prefix+"_quality"]=np.ones(count);out[prefix+"_time_error_s"]=np.zeros(count)
    return out


def test_zero_confidence_and_reset_mask_point_and_adjacent_edges():
    m=metrics();m["closed_score_good"][5]=False;m["closed_quality"][5]=0
    m["open_interval_good"][9]=False
    result=evaluate_metrics(m,source="egoexo4d_hm")
    assert not result["quality_valid"][5]
    assert not result["transition_valid"][[5,6,9]].any()
    assert result["quality_valid"][9]
    validate.independent_rule_check({**m,**result},"egoexo4d_hm",asdict(CameraQualityPolicy()))


def test_half_confidence_is_not_arbitrarily_rejected():
    m=metrics();m["closed_quality"][:]=.5;m["open_quality"][:]=.5
    result=evaluate_metrics(m,source="egoexo4d_hm")
    assert result["quality_valid"].all() and result["transition_valid"].all()


def test_source_interval_checks_hidden_zero_quality_gap_and_session_reset():
    times=np.arange(21)*.001;valid=np.ones(21,bool);valid[5]=False
    ids=np.zeros(21,int)
    assert not interval_valid(np.array([0,10,20]),valid,times,ids,.02)[1]
    valid[:]=True;ids[15:]=1
    assert not interval_valid(np.array([0,10,20]),valid,times,ids,.02)[2]
    times[15:]+=.9;ids[:]=0
    assert not interval_valid(np.array([0,10,20]),valid,times,ids,.02)[2]


def test_5hz_selection_cannot_bridge_hidden_native_invalidity():
    point,edge=np.ones(19,bool),np.ones(19,bool)
    point[3]=False
    valid=selected_transition_valid(point,edge,np.array([0,6,12,18]))
    np.testing.assert_array_equal(valid,[True,False,True,True])
    point[:]=True;edge[8]=False
    np.testing.assert_array_equal(selected_transition_valid(point,edge,np.array([0,6,12,18])),[True,True,False,True])


def test_dynamics_detect_large_finite_closed_loop_correction():
    m=metrics();m["camera_distance_m"][8]=2.57;m["camera_angle_rad"][12]=3.13
    result=evaluate_metrics(m,source="egoexo4d_hm")
    assert result["quality_valid"].all()
    assert not result["transition_valid"][[8,12]].any()
    assert np.isfinite(m["camera_distance_m"]).all()


def test_camera_mask_propagation_preserves_independent_world_labels():
    old=parameter_arrays(episode=0);point=np.ones(91,bool);point[7]=False
    quality={"quality_valid":point,"transition_valid":point.copy()}
    for source in ("egosuite","egoexo4d_hm"):
        new=revision.apply_camera_quality_masks(old,quality,source)
        assert not new["observation.valid.camera_position"][7]
        assert not new["observation.valid.head_reference_rotation"][7]
        for name in ("smplh.trans","smplh.pose_body","smplh.root_orient","smplh.betas","source.feet_valid", "source.feet_world",
                     "observation.valid.hand_left_position","observation.valid.hand_right_position","estimated.valid.body_position",
                     "observation.environment.floor_height","observation.valid.floor","observation.valid.dino"):
            np.testing.assert_array_equal(new[name],old[name])
        assert old["observation.valid.camera_position"][7]


def test_reader_native_span_dependency_and_no_future_anchor(owned_archive):
    def disable(arrays):
        arrays["observation.valid.head_reference_position"][3]=False
    rewrite_episode(owned_archive,0,motion_mutation=disable)
    archive=SMPLHParameterArchive(owned_archive)
    arrays,metadata=archive.read_window(0,load_dino=False)
    assert arrays["observation.valid.head_reference_position"].all()
    assert not arrays["observation.valid.head_reference_transition"][1]
    window=build_smplh_window(arrays,metadata)
    assert not window.motion_valid[1,108:117].any() and not window.trajectory_valid[1,9:18].any()
    arrays["observation.valid.head_reference_position"][0]=False
    with pytest.raises(ValueError,match="valid first camera"):
        build_smplh_window(arrays,metadata)


def write_audit(root,output):
    archive=SMPLHParameterArchive(root);output.mkdir()
    policy=asdict(CameraQualityPolicy());receipts=[]
    for record in archive.records():
        idx=record["episode_index"]
        old=revision.shared._columns(pq.read_table(archive.data_path(idx),use_threads=False))
        m=camera_metrics(old["observation.state.camera_world"],old["observation.state.camera_world_rotation"],
                         old["timestamp"].astype(np.float64),np.ones(91,bool))
        if record["source"]=="egoexo4d_hm":
            for name in ("closed_score_good","closed_time_good","open_score_good","open_time_good","closed_interval_good","open_interval_good"):
                m[name]=np.ones(91,bool)
            for prefix in ("closed","open"):
                m[prefix+"_quality"]=np.ones(91);m[prefix+"_time_error_s"]=np.zeros(91)
            m["source_linear_speed_bound_m_s"]=np.ones(91);m["source_angular_speed_bound_rad_s"]=np.ones(91)
            if idx==1:
                m["closed_score_good"][3]=False;m["closed_quality"][3]=0
        result=evaluate_metrics(m,source=record["source"],policy=policy)
        path=output/f"episode_{idx:08d}.npz";np.savez(path,**m,**result)
        changed=not result["quality_valid"].all() or not result["transition_valid"].all()
        summary={"native_valid_frames":91,"quality_valid_frames":int(result["quality_valid"].sum()),
            "newly_rejected_frames":int((~result["quality_valid"]).sum()),"newly_rejected_native_edges":int((~result["transition_valid"]).sum()),
            "invalid_native_frames":0,"motion_revision_required":changed,"representation_changed_due_native_continuity":False}
        qc={"format":revision.AUDIT_FORMAT,"episode_index":idx,"source":record["source"],"length":91,
            "source_key":record["source_key"],"split":record["split"],"record_sha256":file_sha256(root/f".pipeline/episodes/{idx:08d}.json"),
            "motion_sha256":record["motion_sha256"],"mask_path":path.name,"mask_sha256":file_sha256(path),"policy":policy,
            "source_policy":SOURCE_POLICIES[record["source"]],"summary":summary}
        atomic_json(path.with_suffix(".json"),qc);receipts.append(qc)
    report={"format":revision.AUDIT_FORMAT,"archive_root":str(root),"binding":archive.binding,"state":"complete",
        "diagnostic":False,"production_modified":False,"policy":policy,"source_population":{"scope":"all_archive","total":4,"episodes":4,"sources":{"egosuite":2,"egoexo4d_hm":2}},
        "episodes_completed":4,"episodes_planned":4,"episodes":receipts}
    atomic_json(output/"report.json",report);return output/"report.json"


def qualify_fixture(root):
    archive=SMPLHParameterArchive(root)
    report={"format":"independent-standalone-smplh-parameter-full-audit-v1","scope":"full","failures":[],"passed":True,
            "parameter_and_source_integrity_passed":True,"old_dense_dependency_absent":True,"cleanup_allowed":True,
            "original_egosuite_full_fit_acceptance_passed":True,"population":{"episodes":4,"frames":364},"binding":archive.binding}
    path=root/".pipeline/acceptance-report.json";atomic_json(path,report)
    state=json.loads(archive.state_path.read_text());state.update(acceptance_report=".pipeline/acceptance-report.json",acceptance_report_sha256=file_sha256(path))
    atomic_json(archive.state_path,state)


def test_immutable_camera_revision_and_independent_publish(owned_archive,tmp_path,monkeypatch):
    # This owned-archive fixture deliberately has no original source payload.
    # Independent raw CSV extraction is exercised separately below.
    monkeypatch.setattr(validate,"_independent_source_metrics",lambda *args:None)
    root=owned_archive;qualify_fixture(root)
    vision=write_cache(root,tmp_path/"vision");audit=write_audit(root,tmp_path/"quality")
    output=tmp_path/"v4"
    result=revision.build_revision(root,audit,output,source_vision_root=vision,workers=1)
    assert result["changed_episode_indices"]==[1]
    old=SMPLHParameterArchive(root);new=SMPLHParameterArchive(output,allow_unpublished=True,vision_profile_root=output/revision.PROFILE)
    for idx in (0,2,3):
        assert old._records[idx]==new._records[idx]
        assert os.path.samefile(old.data_path(idx),new.data_path(idx))
    arrays,_=new.read_window(1,load_dino=False)
    assert arrays["observation.valid.camera_position"].all()
    assert not arrays["observation.valid.head_reference_transition"][1]
    report=validate.audit_revision(root,output,workers=1)
    assert report["passed"] and len(report["unchanged_reference_certificates"])==3
    assert report["inherited_FK_numerical_QA_repeated"] is False
    path=output/".pipeline/camera-quality/independent-revision-audit.json";atomic_json(path,report)
    validate.publish_revision(root,output,path)
    accepted=SMPLHParameterArchive(output,vision_profile_root=output/revision.PROFILE)
    assert accepted.state["state"]=="accepted" and accepted.state["formal_training_ready"] is False
    top=json.loads((output/".pipeline/acceptance-report.json").read_text())
    top["proofs"]["quality_audit"]["sha256"]="0"*64
    with pytest.raises(ValueError,match="SHA"):
        check_camera_revision_acceptance(accepted,top,path)


def test_independent_policy_rejects_tampered_transition():
    m=metrics();m["open_interval_good"][6]=False
    result=evaluate_metrics(m,source="egoexo4d_hm");result["transition_valid"][6]=True
    with pytest.raises(ValueError,match="disagrees"):
        validate.independent_rule_check({**m,**result},"egoexo4d_hm",asdict(CameraQualityPolicy()))


def test_index_rejects_future_with_only_invalid_camera_local_labels(owned_archive):
    from MotionExpert.dataset.smplh_motion_dataset import SMPLHMotionDataset
    def disable(arrays):
        for name in ("observation.valid.camera_position", "observation.valid.camera_rotation",
                     "observation.valid.head_reference_position", "observation.valid.head_reference_rotation"):
            arrays[name][36:]=False
    rewrite_episode(owned_archive,0,motion_mutation=disable)
    with pytest.raises(ValueError,match="No usable complete.*no_valid_future_body_labels"):
        SMPLHMotionDataset(archive_root=owned_archive,episode_indices=[0],cond_img_feat=False,do_normalization=False)


def test_independent_raw_source_catches_correlated_interval_flag_error(tmp_path):
    import csv
    from MotionExpert.audit_camera_quality import _source_metrics
    directory=tmp_path/"trajectory";directory.mkdir()
    for opened in (False,True):
        identity="session_uid" if opened else "graph_uid"
        frame="odometry" if opened else "device"
        fields=["tracking_timestamp_us",identity,"quality_score"]
        fields += [f"device_linear_velocity_{axis}_{frame}" for axis in "xyz"]
        fields += [f"angular_velocity_{axis}_device" for axis in "xyz"]
        if not opened:fields += [f"t{axis}_world_device" for axis in "xyz"]+[f"q{axis}_world_device" for axis in "wxyz"]
        path=directory/("open_loop_trajectory.csv" if opened else "closed_loop_trajectory.csv")
        with path.open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            for i in range(31):
                row={name:0 for name in fields};row.update(tracking_timestamp_us=1_000_000+i*1000,quality_score=1.)
                row[identity]="session2" if opened and i>=15 else "session1"
                if not opened:row["qw_world_device"]=1.
                writer.writerow(row)
    record={"source":"egoexo4d_hm","metadata":{"aria_provenance":{
        "trajectory":{"path":str(directory/"closed_loop_trajectory.csv"),"sha256":file_sha256(directory/"closed_loop_trajectory.csv")},
        "device_camera_transform":np.eye(4).tolist()}}}
    a={"timestamp":np.arange(4)/100.,"source.capture_timestamp_ns":np.array([1_000_000_000,1_010_000_000,1_020_000_000,1_030_000_000]),
       "observation.state.camera_world":np.zeros((4,3)),"observation.state.camera_world_rotation":np.tile([1.,0,0,0],(4,1))}
    for key in ("camera_position","camera_rotation","head_reference_position","head_reference_rotation","head_reference_basis"):
        a["observation.valid."+key]=np.ones(4,bool)
    m,evidence=_source_metrics(record,a,CameraQualityPolicy());qc={"source_evidence":evidence,"policy":asdict(CameraQualityPolicy())}
    assert not m["open_interval_good"][2]
    validate._independent_source_metrics(record,a,m,qc)
    m["open_interval_good"][2]=True
    with pytest.raises(ValueError,match="open_interval_good"):
        validate._independent_source_metrics(record,a,m,qc)


def test_integer_nearest_matches_sdk_after_tie_and_one_nanosecond_neighbors():
    from MotionExpert.camera_quality import nearest_indices
    source=np.array([631_635_078_000,631_636_078_000],dtype=np.int64)
    midpoint=631_635_578_000
    query=np.array([source[0]-1,source[0],midpoint-1,midpoint,midpoint+1,source[1],source[1]+1],dtype=np.int64)
    np.testing.assert_array_equal(nearest_indices(source,query),[0,0,0,1,1,1,1])


def test_integer_source_interval_includes_exact_20ms_and_rejects_one_ns_more():
    # Large device timestamps make float-second subtraction unreliable at20ms.
    start=631_635_078_000
    selected=np.array([0,1]);good=np.ones(2,bool);ids=np.zeros(2,int)
    assert interval_valid(selected,good,np.array([start,start+20_000_000],np.int64),ids,20_000_000).all()
    assert not interval_valid(selected,good,np.array([start,start+20_000_001],np.int64),ids,20_000_000)[1]
