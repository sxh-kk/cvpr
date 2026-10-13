"""Persistence, provenance and independent geometry of estimated sidecars."""
from copy import deepcopy
import json

import numpy as np
import pytest

from egomotion_pipeline.body_model import BodyModel
from egomotion_pipeline.geometry import quat_to_matrix
from egomotion_pipeline.schema import PARENTS, SCHEMA_VERSION, feature_specs_for_schema
from egomotion_pipeline.storage import read_arrays, write_arrays
from egomotion_pipeline.unified_dense22 import encode_unified_dense22
from egomotion_pipeline import smplh_sidecar as sidecar
from unified_v4_fixture import v4_archive


@pytest.fixture
def case(tmp_path):
    source = v4_archive(tmp_path/"source",episodes=1,frames=8)
    record = source.records()[0]
    arrays = read_arrays(source.data_path(0),source.specs)
    arrays.update(encode_unified_dense22(arrays,record))
    write_arrays(source.data_path(0),arrays,feature_specs_for_schema(SCHEMA_VERSION))
    # Full 52-joint synthetic SMPL-H structure, with nonzero root pivot and all
    # sixteen bases, makes replacing trans by pelvis or dropping betas observable.
    rest = np.zeros((52,3));rest[0]=[.2,-.1,.3]
    parents=list(PARENTS)
    for wrist,start in ((20,22),(21,37)):
        for finger in range(5):
            j=start+finger*3;parents.extend([wrist,j,j+1])
    parents=np.asarray(parents)
    for j in range(1,52):rest[j]=rest[parents[j]]+[.005*j,.01,.025]
    template=np.zeros((6890,3));template[:52]=rest
    shapedirs=np.zeros((6890,3,16));shapedirs[:52,0,:]=np.arange(52)[:,None]*np.arange(1,17)[None]*.0001
    regressor=np.zeros((52,6890));regressor[:,:52]=np.eye(52)
    model_path=tmp_path/"model.npz"
    np.savez(model_path,v_template=template,shapedirs=shapedirs,J_regressor=regressor,kintree_table=np.array([parents,np.arange(52)]))
    model=BodyModel(model_path)
    T=8;rng=np.random.default_rng(34)
    aa=rng.normal(0,.15,(T,22,3));beta=np.arange(16)*.03
    trans=np.tile([2.5,-1.2,.6],(T,1));trans[:,0]+=np.arange(T)*.01
    pos,q,_=model.decode(trans,aa[:,0],aa[:,1:].reshape(T,63),np.repeat(beta[None],T,axis=0),np.ones(T,bool))
    pv=np.ones((T,22),bool);rv=pv.copy();rv[1,5]=False
    fit={"body_world":pos,"global_rotations":quat_to_matrix(q),"local_axis_angle":aa,
         "root_world":pos[:,0],"smpl_trans":trans,"betas":beta,
         "estimated_position_valid":pv,"estimated_rotation_valid":rv,
         "source_position_valid":arrays["observation.valid.body_position"].copy(),
         "source_rotation_valid":arrays["observation.valid.body_rotation"].copy(),
         "position_confidence":np.full((T,22),.7),"rotation_confidence":np.full((T,22),.4),
         "estimate_interpolated":pv & ~arrays["observation.valid.body_position"],
         "reconstructed_observations":pos.copy()}
    calibration={"source_id":"egosuite","group_id":record["metadata"]["operator_id"],
        "center":{"offsets":np.zeros((22,3)),"anchors":np.arange(22),"offset_valid":np.zeros(22,bool),
                  "betas":beta,"model_sha256":model.metadata["model_sha256"],"input_sha256":"0"*64,"metadata":{}}}
    contract=sidecar.build_sidecar_contract(source.root,source.data_path(0),model_path,
                                           {"policy":"fixed-group-beta"},calibration,arrays)
    return source,record,arrays,fit,contract,tmp_path/"sidecar"


def test_roundtrip_readonly_hands_nativefeet_and_independent_masks(case):
    source,record,arrays,fit,contract,out=case
    before={k:x.copy() for k,x in arrays.items()}
    source_bytes=source.data_path(0).read_bytes()
    report=sidecar.write_episode_sidecar(out,record,arrays,fit,contract)
    assert report["passed"] and report["written"]
    loaded=sidecar._read_table(out/"episodes/00000000/motion.parquet")
    assert not any("dino" in k or "video" in k for k in loaded)
    assert not loaded["source.observed.body_position"][0,3]
    assert loaded["estimated.valid.body_world_position"][0,3]
    assert loaded["estimated.inferred.body_position"][0,3]
    assert not loaded["estimated.valid.body_world_rotation"][1,5]
    assert loaded["quality.rotation_confidence"][1,5]==0
    for side in ("left","right"):
        for frame in ("world","floor"):
            key=f"observation.state.hand_{side}_{frame}"
            np.testing.assert_array_equal(loaded["source."+key],arrays[key])
    np.testing.assert_array_equal(loaded["source.feet_world"],arrays["observation.state.body_world"].reshape(8,22,3)[:,[10,11]].reshape(8,6))
    assert source_bytes==source.data_path(0).read_bytes()
    for k in before:np.testing.assert_array_equal(arrays[k],before[k])
    local=loaded["fitted.body_floor_rot6d_xyz"].reshape(8,22,9)
    floor_R=quat_to_matrix(arrays["observation.state.head_reference_floor_world_rotation"])
    recovered=np.einsum("tij,tkj->tki",floor_R,local[...,6:])+arrays["observation.state.head_reference_floor_world"][:,None]
    np.testing.assert_allclose(recovered,fit["body_world"],atol=8e-7)
    assert sidecar.validate_episode_sidecar(out,record,arrays,contract)["passed"]
    manifest=json.loads((out/"manifest.json").read_text())
    assert manifest["original_archive_root"]==str(source.root)
    assert manifest["join"]==["episode_index","frame_index"]


def test_exact_resume_refuses_new_fit_even_same_model(case):
    _,record,arrays,fit,contract,out=case
    first=sidecar.write_episode_sidecar(out,record,arrays,fit,contract)
    identity=sidecar.file_identity(first["path"])
    assert not sidecar.write_episode_sidecar(out,record,arrays,fit,contract)["written"]
    changed=deepcopy(fit);changed["position_confidence"][0,0]=.2
    with pytest.raises(ValueError,match="fitted estimates differ"):
        sidecar.write_episode_sidecar(out,record,arrays,changed,contract)
    assert sidecar.file_identity(first["path"])==identity


def test_midwrite_failure_commits_neither_file_and_retry_works(case,monkeypatch):
    _,record,arrays,fit,contract,out=case
    original=sidecar._write_table
    def fail(path,data,fingerprint):
        original(path,data,fingerprint)
        raise OSError("simulated disk-full after parquet")
    monkeypatch.setattr(sidecar,"_write_table",fail)
    with pytest.raises(OSError,match="disk-full"):
        sidecar.write_episode_sidecar(out,record,arrays,fit,contract)
    assert not (out/"episodes/00000000").exists()
    assert not list((out/".staging").iterdir())
    monkeypatch.setattr(sidecar,"_write_table",original)
    assert sidecar.write_episode_sidecar(out,record,arrays,fit,contract)["passed"]


def test_corruption_fails_validator_and_window(case):
    _,record,arrays,fit,contract,out=case
    path=sidecar.write_episode_sidecar(out,record,arrays,fit,contract)["path"]
    with open(path,"ab") as f:f.write(b"corruption")
    with pytest.raises(ValueError,match="bytes changed"):
        sidecar.validate_episode_sidecar(out,record,arrays,contract)
    with pytest.raises(ValueError,match="corrupted"):
        sidecar.read_sidecar_window(out,0,num_frames=2)


@pytest.mark.parametrize("field",["config","calibration","source_identity","model_identity"])
def test_mismatched_fingerprints_fail_before_output(case,field):
    _,record,arrays,fit,contract,out=case
    wrong=deepcopy(contract)
    if "identity" in field:wrong[field]["mtime_ns"]+=1
    else:wrong[field]["changed"]=True
    with pytest.raises(ValueError):sidecar.write_episode_sidecar(out,record,arrays,fit,wrong)
    assert not out.exists()


def test_source_changed_on_disk_and_fake_caller_projection_fail(case):
    source,record,arrays,fit,contract,out=case
    bad=deepcopy(arrays);bad["observation.state.hand_left_world"][0,0]+=.01
    with pytest.raises(ValueError,match="projected motion changed"):
        sidecar.write_episode_sidecar(out,record,bad,fit,contract)
    with source.data_path(0).open("ab") as f:f.write(b"unexpected source edit")
    with pytest.raises(ValueError,match="identity changed"):
        sidecar.write_episode_sidecar(out,record,arrays,fit,contract)
    assert not out.exists()


def test_native_ground_missing_stays_invalid_without_losing_world(case):
    source,record,arrays,fit,contract,out=case
    arrays=deepcopy(arrays)
    for part in ("position","rotation"):
        arrays[f"observation.valid.head_reference_floor_{part}"][:]=False
    arrays["observation.valid.floor"][:]=False
    arrays["observation.valid.head_reference_height"][:]=False
    arrays["observation.state.head_reference_floor_world"][:]=0
    arrays["observation.state.head_reference_floor_world_rotation"][:]=[1,0,0,0]
    record=deepcopy(record);record["metadata"]["ground"]={"height_m":0.,"valid":False,"reason":"no_stationary_feet"}
    write_arrays(source.data_path(0),arrays,feature_specs_for_schema(SCHEMA_VERSION))
    contract=sidecar.build_sidecar_contract(source.root,source.data_path(0),contract["model_identity"]["path"],contract["config"],contract["calibration"],arrays)
    sidecar.write_episode_sidecar(out,record,arrays,fit,contract)
    x=sidecar._read_table(out/"episodes/00000000/motion.parquet")
    assert not x["estimated.valid.body_floor_position"].any()
    assert not x["estimated.valid.body_floor_rotation"].any()
    assert not x["fitted.body_floor_rot6d_xyz"].any()
    assert x["estimated.valid.body_world_position"].all()
    assert not x["source.observation.valid.floor"].any()
    np.testing.assert_array_equal(x["source.feet_valid"],arrays["observation.valid.body_position"][:,[10,11]])


def test_15hz_window_selects_exact_even_frames_and_preserves_dtype(case):
    _,record,arrays,fit,contract,out=case
    sidecar.write_episode_sidecar(out,record,arrays,fit,contract)
    window=sidecar.read_sidecar_window(out,0,start_frame=0,num_frames=4,fps=15,
                                      columns=["frame_index","fitted.body_world_rot6d_xyz","source.feet_valid"])
    np.testing.assert_array_equal(window["frame_index"],[0,2,4,6])
    assert window["fitted.body_world_rot6d_xyz"].dtype==np.float32
    assert window["source.feet_valid"].dtype==np.bool_
    with pytest.raises(ValueError,match="exceeds"):sidecar.read_sidecar_window(out,0,num_frames=5)


@pytest.mark.parametrize("mutate",["frame","fps","episode","shape","root_pivot","rotation","source_mask","confidence","inference_mask"])
def test_invalid_time_model_or_mask_contract_fails_closed(case,mutate):
    _,record,arrays,fit,contract,out=case
    arrays,fit,record=deepcopy(arrays),deepcopy(fit),deepcopy(record)
    if mutate=="frame":arrays["frame_index"][3]+=1
    elif mutate=="fps":arrays["timestamp"][3]+=.01
    elif mutate=="episode":record["episode_index"]=1
    elif mutate=="shape":fit["betas"]=fit["betas"][:10]
    elif mutate=="root_pivot":fit["root_world"][:]=fit["smpl_trans"]
    elif mutate=="rotation":fit["global_rotations"][0,3]=np.eye(3)
    elif mutate=="source_mask":fit["source_position_valid"][0,3]=True
    elif mutate=="confidence":fit["rotation_confidence"][0,3]=1.1
    elif mutate=="inference_mask":fit["estimated_position_valid"][0,3]=False
    with pytest.raises(ValueError):sidecar.write_episode_sidecar(out,record,arrays,fit,contract)
    assert not out.exists()


def test_structured_source_arrays_and_poison_dino_are_supported(case):
    _,record,arrays,fit,contract,out=case
    class Poison:
        def __array__(self,*a,**kw):raise AssertionError("DINO was decoded")
    arrays=dict(arrays);arrays["observation.features.dino"]=Poison()
    arrays["observation.state.body_world"]=arrays["observation.state.body_world"].reshape(8,22,3)
    arrays["observation.state.body_world_rotation"]=arrays["observation.state.body_world_rotation"].reshape(8,22,4)
    assert sidecar.write_episode_sidecar(out,record,arrays,fit,contract)["passed"]


def test_output_cannot_overwrite_source_or_nonempty_unknown_directory(case):
    source,record,arrays,fit,contract,out=case
    with pytest.raises(ValueError,match="overwrite archive"):
        sidecar.write_episode_sidecar(source.root,record,arrays,fit,contract)
    out.mkdir();(out/"unrelated.txt").write_text("keep")
    with pytest.raises(ValueError,match="nonempty unrecognized"):
        sidecar.write_episode_sidecar(out,record,arrays,fit,contract)
    assert (out/"unrelated.txt").read_text()=="keep"


@pytest.mark.parametrize("field", ["manifest", "receipt_paths", "receipt_specs", "receipt_join"])
def test_semantic_receipt_and_manifest_mismatch_fail_closed(case, field):
    _,record,arrays,fit,contract,out=case
    sidecar.write_episode_sidecar(out,record,arrays,fit,contract)
    path=out/("manifest.json" if field=="manifest" else "episodes/00000000/receipt.json")
    payload=json.loads(path.read_text())
    if field=="manifest":payload["join"]=["frame_index"]
    elif field=="receipt_paths":payload["paths"]["parquet"]="../foreign.parquet"
    elif field=="receipt_specs":payload["specs"]["fitted.betas"]["shape"]=[10]
    elif field=="receipt_join":payload["declaration"]["episode_index"]=1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):sidecar.validate_episode_sidecar(out,record,arrays,contract)
    with pytest.raises(ValueError):sidecar.read_sidecar_window(out,0,num_frames=2)


def test_nonzero_center_offsets_are_independently_reconstructed(case):
    source,record,arrays,fit,contract,out=case
    calibration=deepcopy(contract["calibration"])
    calibration["center"]["offsets"][9]=[.02,-.03,.01]
    calibration["center"]["anchors"][9]=6
    calibration["center"]["offset_valid"][9]=True
    offsets=np.asarray(calibration["center"]["offsets"])
    anchors=np.asarray(calibration["center"]["anchors"])
    fit=deepcopy(fit)
    fit["reconstructed_observations"]=fit["body_world"]+(fit["global_rotations"][:,anchors]@offsets[...,None])[...,0]
    contract=sidecar.build_sidecar_contract(source.root,source.data_path(0),contract["model_identity"]["path"],contract["config"],calibration,arrays)
    assert sidecar.write_episode_sidecar(out,record,arrays,fit,contract)["passed"]
    loaded=sidecar._read_table(out/"episodes/00000000/motion.parquet")
    reconstructed=loaded["fitted.reconstructed_source_landmarks_world"].reshape(8,22,3)
    np.testing.assert_allclose(reconstructed,fit["reconstructed_observations"],atol=3e-7)
    residual=np.linalg.norm(reconstructed-arrays["observation.state.body_world"].reshape(8,22,3),axis=-1)
    np.testing.assert_allclose(loaded["quality.observation_residual_m"][:,9],residual[:,9],atol=4e-7)
    bad=deepcopy(fit);bad["reconstructed_observations"]=bad["body_world"]
    with pytest.raises(ValueError,match="frozen center offsets"):
        sidecar.write_episode_sidecar(out,record,arrays,bad,contract)


@pytest.mark.parametrize("valid", [True,False])
def test_recording_calibration_binds_exact_operator_and_source_key(case,valid):
    source,record,arrays,fit,contract,out=case
    cal=deepcopy(contract["calibration"])
    cal["group_id"]=str(record["metadata"]["operator_id"])+":record:"+(record["source_key"] if valid else "another_record")
    contract=sidecar.build_sidecar_contract(source.root,source.data_path(0),contract["model_identity"]["path"],contract["config"],cal,arrays)
    if valid:
        assert sidecar.write_episode_sidecar(out,record,arrays,fit,contract)["passed"]
    else:
        with pytest.raises(ValueError,match="operator/recording"):
            sidecar.write_episode_sidecar(out,record,arrays,fit,contract)
        assert not out.exists()


def test_full_model_hand_topology_is_required(case):
    source,_,arrays,_,contract,_=case
    path=contract["model_identity"]["path"]
    with np.load(path,allow_pickle=False) as model:
        payload={key:model[key].copy() for key in model.files}
    payload["kintree_table"][0,23]=20  # a finger middle joint must follow its base
    np.savez(path,**payload)
    with pytest.raises(ValueError,match="full52 hand parent"):
        sidecar.build_sidecar_contract(source.root,source.data_path(0),path,contract["config"],contract["calibration"],arrays)


def test_window_rejects_changed_source_and_noninteger_or_unknown_columns(case):
    source,record,arrays,fit,contract,out=case
    sidecar.write_episode_sidecar(out,record,arrays,fit,contract)
    for kwargs in ({"start_frame":.5},{"episode_index":True},{"columns":["missing"]},{"columns":[]}):
        params={"episode_index":0,"num_frames":2,**kwargs}
        with pytest.raises(ValueError):sidecar.read_sidecar_window(out,**params)
    with source.data_path(0).open("ab") as f:f.write(b"modified")
    with pytest.raises(ValueError,match="identity changed"):
        sidecar.read_sidecar_window(out,0,num_frames=2)


def test_actual_sequence_execution_receipt_preserved_and_resume_guarded(case):
    _,record,arrays,fit,contract,out=case
    fit=deepcopy(fit)
    fit["metadata"]={"label_kind":"model_fitted_estimate","target_joint_basis":"smplh_body22_axes_v1",
        "native_axis_certified":False,"model":{"model_sha256":contract["model_sha256"]},
        "config":contract["config"],"calibration_input_sha256":contract["calibration"]["center"]["input_sha256"],
        "source_id":"egosuite","group_id":contract["calibration"]["group_id"],
        "segments":[[0,8]],"blocks":[{"start":0,"end":8}],"input_sha256":"a"*64}
    fit["quality"]={"seams":[],"proxy_rotation_accuracy_established":False}
    assert sidecar.write_episode_sidecar(out,record,arrays,fit,contract)["passed"]
    path=out/"episodes/00000000/receipt.json"
    receipt=json.loads(path.read_text())
    assert receipt["execution"]["metadata"]==fit["metadata"]
    assert receipt["execution"]["quality"]==fit["quality"]
    changed=deepcopy(fit);changed["metadata"]["segments"]=[[0,4],[4,8]]
    with pytest.raises(ValueError,match="execution metadata differs"):
        sidecar.write_episode_sidecar(out,record,arrays,changed,contract)
    receipt["execution"]["metadata"]["segments"]=[[0,4],[4,8]]
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match="receipt contract"):
        sidecar.validate_episode_sidecar(out,record,arrays,contract)
