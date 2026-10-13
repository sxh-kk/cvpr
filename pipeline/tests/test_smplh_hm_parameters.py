import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import egomotion_pipeline.smplh_hm_parameters as hm
from egomotion_pipeline.geometry import axis_angle_to_quat,quat_to_matrix
from egomotion_pipeline.schema import PARENTS
from egomotion_pipeline.smplh_parameters import validate_parameter_arrays


def fixture(tmp_path,monkeypatch,count=5,uid="unit_take_1"):
    take={"take_name":uid,"take_uid":"unit-take-uid","participant_uid":42,
          "capture_uid":"capture-uid","timesync_start_idx":7,"timesync_end_idx":7+count+1,
          "capture":{"capture_name":"unit_capture","cameras":[{"cam_id":"aria01","is_ego":True}]},
          "frame_aligned_videos":{"aria01":{"rgb":{"relative_path":"frame_aligned_videos/aria01_214-1.mp4"}}}}
    video=tmp_path/"EgoExo4D/takes"/uid/"frame_aligned_videos/aria01_214-1.mp4"
    video.parent.mkdir(parents=True);video.write_bytes(b"unit-video-identity-not-decoded")
    path=tmp_path/"EgoExo4D-HM"/uid/"points_triangulated_world_results_merged.npz"
    path.parent.mkdir(parents=True)
    raw={"trans":np.zeros((1,count,3),np.float32),"root_orient":np.zeros((1,count,3),np.float32),
         "pose_body":np.zeros((1,count,63),np.float32),
         "betas_per_frame":np.arange(count*16,dtype=np.float32).reshape(1,count,16)/100,
         "hand_pose":np.arange(count*90,dtype=np.float32).reshape(1,count,90)/100,
         "valid":np.ones((1,count),np.int8),
         "joints3d":np.arange(count*67*3,dtype=np.float32).reshape(1,count,67,3)/100,
         "chunk_ranges":np.array([[0,2],[2,count]],np.int64),
         "cam_R":np.broadcast_to(np.eye(3,dtype=np.float32),(5,count,3,3)).copy(),
         "cam_t":np.zeros((5,count,3),np.float32)}
    np.savez(path,**raw)
    dino=tmp_path/"DINO/takes"/uid/"aria01_214-1.parquet"
    dino.parent.mkdir(parents=True)
    valid=np.ones(count+1,bool);valid[1]=False
    pq.write_table(pa.table({"frame_index":np.arange(count+1),"timestamp":np.arange(count+1)/30,
        "observation.valid.dino":valid,"observation.video_timestamp":np.arange(count+1)/30+.001,
        "observation.video_time_error":np.full(count+1,.001)}),dino)
    identity=hm.file_identity(video)
    meta={"contract":"egoexo-video-dino5-30hz-v1","state":"complete","length":count+1,
          "parquet_sha256":hm.file_sha256(dino),"signature":"unit-signature",
          "video":{"take_name":uid,"take_uid":take["take_uid"],"length":count+1,
          "fps":30,"clock":"frame_aligned_take_seconds","relative_video_path":f"takes/{uid}/frame_aligned_videos/{video.name}",
          "video_path":str(video),"video_fingerprint":{"size_bytes":identity["bytes"],"mtime_ns":identity["mtime_ns"]},
          "video_info":{"frames":count+1,"average_rate":"30","start_time_s":0.0}}}
    dino.with_suffix(".json").write_text(json.dumps(meta))
    model=tmp_path/"unit-model.npz";model.write_bytes(b"unit-model-placeholder")

    class NumericModelFixture:
        metadata={"model_sha256":hm.MODEL_SHA256,"vertex_count":6890,"joint_count":52,
                  "tree_joint_count":52,"shape_basis_count":16}
        shape_basis=np.zeros((22,3,16))
        def decode(self,trans,root,pose,betas,valid):
            count=len(trans)
            p=np.zeros((count,22,3))
            p[...,2]=1
            p[:,10:12,2]=0
            p+=np.nan_to_num(trans)[:,None]
            local=axis_angle_to_quat(np.concatenate([root[:,None],pose.reshape(count,21,3)],1))
            mask=np.broadcast_to(valid.astype(bool)[:,None],(count,22)).copy()
            mask&=np.isfinite(trans).all(-1)[:,None]&np.isfinite(betas).all(-1)[:,None]
            return p,local,mask
    monkeypatch.setattr(hm,"_verified_model",lambda *args:NumericModelFixture())
    return take,path,raw,dino,meta,model


def camera_fixture(assets,take,count,raw,alignment):
    valid=np.ones(count,bool) if alignment["verified"] else np.zeros(count,bool)
    camera=np.broadcast_to(np.eye(4),(count,4,4)).copy()
    camera[:,2,3]=1.7
    return {"camera_world":camera,"head_world":camera.copy(),"valid":valid,
            "capture_timestamp_ns":np.arange(count,dtype=np.int64)*33333333+1230000000,
            "pose_time_error_s":np.zeros(count),"provenance":{"fixture":True},
            "world":{"verified":bool(valid.any()),"quarantine_reasons":[] if valid.any() else ["unmapped fixture"]}}


def test_release_mapping_has_concrete_generation_chain(tmp_path,monkeypatch):
    take,path,raw,dino,meta,model=fixture(tmp_path,monkeypatch)
    r=hm.alignment_receipt(take,5,raw["chunk_ranges"],meta,
        video_identity=hm.file_identity(Path(meta["video"]["video_path"])),dino_rows=6)
    assert r["verified"] and r["hm_row_to_video_frame"]["offset"]==0
    assert r["hm_row_to_extracted_image"]["offset"]==1
    assert r["last_video_frame_not_in_motion"]==5
    assert len(r["source_evidence"]["files"])==4


@pytest.mark.parametrize("mutate",["length","uid","revision","fingerprint","rows","start_clock","video_path"])
def test_release_mapping_rejects_identity_or_offset_guesses(tmp_path,monkeypatch,mutate):
    take,path,raw,dino,meta,model=fixture(tmp_path,monkeypatch)
    count,revision,rows=5,hm.HM_REVISION,6
    video_identity=hm.file_identity(Path(meta["video"]["video_path"]))
    if mutate=="length":take["timesync_end_idx"]+=1
    if mutate=="uid":meta["video"]["take_uid"]="wrong-take"
    if mutate=="revision":revision="new-unreviewed-release"
    if mutate=="fingerprint":meta["video"]["video_fingerprint"]["size_bytes"]+=1
    if mutate=="rows":rows=5
    if mutate=="start_clock":meta["video"]["video_info"]["start_time_s"]=1/30
    if mutate=="video_path":meta["video"]["video_path"]="/different-take/aria01_214-1.mp4"
    r=hm.alignment_receipt(take,count,raw["chunk_ranges"],meta,revision=revision,
        video_identity=video_identity,dino_rows=rows)
    assert not r["verified"] and r["quarantine_reasons"]


@pytest.mark.parametrize("mutate",[None,"start","gap","partial_chunk","wrong_tail","unregistered_take"])
def test_registered_legacy_mapping_has_strict_chunks_and_tail(tmp_path,monkeypatch,mutate):
    take,path,raw,dino,meta,model=fixture(tmp_path,monkeypatch,count=200,uid="iiith_cooking_110_2")
    take["timesync_end_idx"]=take["timesync_start_idx"]+205
    meta["length"]=meta["video"]["length"]=meta["video"]["video_info"]["frames"]=205
    chunks=np.array([[0,100],[100,200]],np.int64)
    if mutate=="start":chunks[0,0]=1
    if mutate=="gap":chunks[1,0]=101
    if mutate=="partial_chunk":chunks=np.array([[0,90],[90,200]])
    if mutate=="wrong_tail":take["timesync_end_idx"]+=100
    if mutate=="unregistered_take":take["take_name"]="iiith_cooking_not_in_review";meta["video"]["take_name"]=take["take_name"]
    r=hm.alignment_receipt(take,200,chunks,meta,
        video_identity=hm.file_identity(Path(meta["video"]["video_path"])),dino_rows=205)
    assert r["verified"]==(mutate is None)
    if mutate is None:
        assert "viewer_row_contract_legacy" in r["method"]
        assert r["unused_video_frame_range_half_open"]==[200,205]
        assert "cannot be recovered" in r["legacy_generation_flags"]


@pytest.mark.parametrize("chunks",[np.array([[1,5]]),np.array([[0,2],[3,5]]),
    np.array([[0,3],[2,5]]),np.array([[0,4]]),np.array([[0.,5.]])])
def test_chunks_fail_closed_without_trimming(chunks):
    with pytest.raises(ValueError):hm.check_chunks(chunks,5)


def test_native_parameters_and_explicit_hands_preserved(tmp_path,monkeypatch):
    take,path,raw,dino,meta,model=fixture(tmp_path,monkeypatch)
    before=hm.file_sha256(path)
    out=hm.export_hm_parameters(tmp_path,take,model,episode_index=17,index_offset=120,camera_loader=camera_fixture)
    arrays=out.arrays
    assert out.metadata["canonical_training_eligible"]
    for target,source in [("smplh.root_orient","root_orient"),("smplh.pose_body","pose_body"),
            ("smplh.trans","trans"),("smplh.betas","betas_per_frame"),("smplh.pose_hand_pca","hand_pose")]:
        assert np.array_equal(arrays[target],raw[source][0])
    for side,s,e in [("left",25,46),("right",46,67)]:
        assert np.array_equal(arrays[f"observation.state.hand_{side}_world"].reshape(5,21,3),raw["joints3d"][0,:,s:e])
    assert np.array_equal(arrays["frame_index"],np.arange(5))
    assert np.array_equal(arrays["index"],120+np.arange(5))
    assert np.array_equal(arrays["source.video_frame_index"],np.arange(5))
    assert not arrays["observation.valid.dino"][1]
    assert np.all(arrays["episode_index"]==17)
    assert not any("body_world" in key or "floor_world" in key or "hand_left_floor" in key for key in arrays)
    assert not arrays["quality.observation_residual_valid"].any()
    assert "PCA" in out.metadata["hand_pose_encoding"]
    assert hm.file_sha256(path)==before
    validate_parameter_arrays(arrays)


def test_masks_are_independent_and_no_completion_is_fabricated(tmp_path,monkeypatch):
    take,path,raw,dino,meta,model=fixture(tmp_path,monkeypatch)
    raw["trans"][0,1,0]=np.nan
    raw["joints3d"][0,2,30,0]=np.nan
    raw["hand_pose"][0,3,0]=np.nan
    raw["valid"][0,4]=0
    np.savez(path,**raw)
    out=hm.export_hm_parameters(tmp_path,take,model,camera_loader=camera_fixture)
    a=out.arrays
    assert not a["estimated.valid.body_position"][1].any()
    assert a["estimated.valid.body_rotation"][1].all()
    assert not a["observation.valid.hand_left_position"][2,5]
    assert not a["smplh.valid.hand_pose"][3,0] and a["smplh.valid.hand_pose"][3,1]
    assert not a["estimated.valid.body_position"][4].any()
    assert not a["observation.valid.hand_right_position"][4].any()
    assert np.array_equal(a["observation.state.hand_right_world"][4].reshape(21,3),raw["joints3d"][0,4,46:67])
    assert not a["estimated.completed.body_position"].any()
    assert not a["estimated.completed.body_rotation"].any()
    assert np.isfinite(a["smplh.trans"]).all()


def test_unverified_alignment_preserves_motion_but_disables_canonical_camera(tmp_path,monkeypatch):
    take,path,raw,dino,meta,model=fixture(tmp_path,monkeypatch)
    meta["video"]["take_uid"]="another-take"
    dino.with_suffix(".json").write_text(json.dumps(meta))
    out=hm.export_hm_parameters(tmp_path,take,model,camera_loader=camera_fixture)
    assert not out.metadata["canonical_training_eligible"]
    assert out.arrays["estimated.valid.body_position"].all()
    assert not out.arrays["observation.valid.camera_position"].any()
    assert not out.arrays["observation.valid.floor"].any()
    assert np.all(out.arrays["source.video_frame_index"]==-1)
    assert not out.arrays["observation.valid.dino"].any()


def world_fixture(count=3):
    raw={"trans":np.zeros((count,3)),"valid":np.ones(count,bool),
         "cam_R":np.broadcast_to(np.eye(3),(5,count,3,3)).copy(),"cam_t":np.zeros((5,count,3))}
    rows=[{"quality":"1","cam_uid":f"cam{i}","graph_uid":"graph1",
          **{f"q{x}_world_cam":str(float(x=="w")) for x in "wxyz"},
          **{f"t{x}_world_cam":"0" for x in "xyz"}} for i in range(1,5)]
    return raw,rows


def test_world_registration_ignores_duplicate_view_zero_but_requires_all_views():
    raw,rows=world_fixture();raw["cam_t"][0]=999
    assert hm.verify_hm_world(raw,rows,{"graph1"},np.array([[0,0,-9.81]]))["verified"]
    raw["cam_t"][4,1,0]=.01
    assert not hm.verify_hm_world(raw,rows,{"graph1"},np.array([[0,0,-9.81]]))["verified"]


@pytest.mark.parametrize("graphs,gravity",[({"other"},[[0,0,-9.81]]),
    ({"graph1","other"},[[0,0,-9.81]]),({"graph1"},[[0,-9.81,0]]),({"graph1"},[])])
def test_world_registration_rejects_graph_or_vertical_mismatch(graphs,gravity):
    raw,rows=world_fixture()
    assert not hm.verify_hm_world(raw,rows,graphs,np.asarray(gravity))["verified"]


def test_dino_actual_grid_is_checked_not_only_receipt(tmp_path,monkeypatch):
    take,path,raw,dino,meta,model=fixture(tmp_path,monkeypatch)
    table=pq.read_table(dino);table=table.set_column(0,"frame_index",pa.array(np.arange(6)+1))
    pq.write_table(table,dino)
    with pytest.raises(ValueError,match="actual frame index"):
        hm.export_hm_parameters(tmp_path,take,model,camera_loader=camera_fixture)
