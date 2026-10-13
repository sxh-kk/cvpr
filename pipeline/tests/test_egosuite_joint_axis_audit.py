"""Audit safety, capture-time joins, and independently named hand correspondences."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from egomotion_pipeline.schema import PARENTS
from mcap_fixtures import BASE_NS, make_mcap


MODULE_PATH = Path(__file__).parents[1]/"scripts"/"audit_egosuite_joint_axes.py"
spec = importlib.util.spec_from_file_location("axis_audit_script", MODULE_PATH)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def neutral_smplx_fixture(path):
    # Finger order is from the upstream 55-joint named contract. Source order
    # puts ring before pinky; the SMPL-X order puts pinky before ring.
    parents = np.full(55, -1, dtype=np.int64)
    parents[:22] = PARENTS
    parents[22:25] = 15
    names = ["body_"+str(i) for i in range(25)]
    joints = np.zeros((55,3), dtype=float)
    for i in range(22):
        joints[i] = [i*.01, i*.02, (i%3)*.015]
    finger_offsets = {"index":[.06,.01,.025], "middle":[.07,0.,.01],
                      "pinky":[.04,-.02,-.035], "ring":[.065,-.01,-.02],
                      "thumb":[.035,.02,.04]}
    index = 25
    for side,wrist in (("left",20),("right",21)):
        for finger in ("index","middle","pinky","ring","thumb"):
            offset = np.array(finger_offsets[finger])*([1,1,1] if side=="left" else [-1,1,1])
            for k in range(3):
                parents[index] = wrist if k==0 else index-1
                names.append(f"{side}_{finger}{k+1}")
                joints[index] = joints[wrist]+offset*(1+k*.3)
                index+=1
    template = np.zeros((10475,3),dtype=float)
    template[:55] = joints
    regressor = np.zeros((55,10475),dtype=float)
    regressor[np.arange(55),np.arange(55)] = 1.
    np.savez_compressed(path, v_template=template,J_regressor=regressor,J=joints,
                        kintree_table=np.stack([parents,np.arange(55)]))
    return joints, names


def test_mcp_correspondences_follow_named_ring_and_pinky_contract(tmp_path):
    model = tmp_path/"neutral.npz"
    expected, names = neutral_smplx_fixture(model)
    joints, metadata = audit.load_target_model(model)
    anchors = audit.make_anchors(joints)
    assert metadata["first22_parent_check_passed"]
    assert metadata["mcp_parent_check_passed"]
    assert metadata["not_hm_smplh_decoding_asset"]
    for side,wrist in (("left",20),("right",21)):
        by_name={s["name"].split("->")[1]:s for s in anchors[wrist]}
        for finger,source_index in (("index",5),("middle",9),("ring",13),("pinky",17)):
            spec=by_name[finger+"_mcp"]
            # Name lookup in the independent full joint fixture catches a
            # ring/pinky swap even when both mistaken roots have wrist parent.
            target_index=names.index(f"{side}_{finger}1")
            assert spec["target_joint_index"]==target_index
            assert spec["source_hand_index"]==source_index
            assert np.array_equal(spec["target_vector"],expected[target_index]-expected[wrist])
            assert spec["source_finger_name_verified"] is False


def test_neutral_model_rejects_mcp_with_wrong_parent(tmp_path):
    path=tmp_path/"bad.npz"
    neutral_smplx_fixture(path)
    with np.load(path,allow_pickle=False) as asset:
        values={k:asset[k] for k in asset.files}
    values["kintree_table"][0,34]=18
    np.savez_compressed(path,**values)
    with pytest.raises(ValueError,match="MCP roots"):
        audit.load_target_model(path)


def test_model_and_catalog_hashes_cover_the_loaded_byte_snapshot(tmp_path,monkeypatch):
    model=tmp_path/"neutral.npz"
    neutral_smplx_fixture(model)
    catalog=tmp_path/"catalog.json"
    catalog.write_text('[{"operator":"p","path":"/source.mcap"}]')
    loaded,digest=audit.load_catalog_snapshot(catalog)
    assert loaded[0]["operator"]=="p" and digest==audit.file_sha256(catalog)
    original=audit.file_sha256

    def replace_between_snapshot_and_hash(path):
        path=Path(path)
        raw=path.read_bytes()
        if path==model:
            # Alter an unused ZIP tail byte after the already loaded model:
            # merely hashing the filename would silently bind data to a new hash.
            path.write_bytes(raw[:-1]+bytes([raw[-1]^1]))
        elif path==catalog:
            path.write_text('[{"operator":"different","path":"/source.mcap"}]')
        return original(path)

    monkeypatch.setattr(audit,"file_sha256",replace_between_snapshot_and_hash)
    with pytest.raises(ValueError,match="Target model changed"):
        audit.load_target_model(model)
    with pytest.raises(ValueError,match="Source catalog changed"):
        audit.load_catalog_snapshot(catalog)


def test_operator_split_is_disjoint_deterministic_and_task_diverse():
    catalog=[{"uid":f"e{i:02d}","operator":f"p{i//2:02d}","task":f"t{i%7}",
              "path":f"/{i}.mcap","basis_correction":None} for i in range(24)]
    selected=audit.select_operators(catalog,12,123)
    assert selected==audit.select_operators(list(reversed(catalog)),12,123)
    fit={r["operator"] for r in selected if r["split"]=="fit"}
    held={r["operator"] for r in selected if r["split"]=="heldout"}
    assert len(fit)==len(held)==6 and fit.isdisjoint(held)
    assert len({r["task"] for r in selected})==7
    with pytest.raises(ValueError,match="fewer distinct"):
        audit.select_operators(catalog,14,123)


def test_prefix_preserves_integer_capture_time_bad_masks_and_missing_hands(tmp_path):
    path=make_mcap(tmp_path/"pose.mcap",count=5,reverse_body=True,missing_hand=[1],
                   bad={"upper_body":[1],"lower_body":[2],"hand":[0]})
    before=audit.file_sha256(path)
    data,provenance=audit.read_prefix(path,3,2_000_000)
    assert len(data["body_capture_time_ns"])==3
    assert np.array_equal(data["body_capture_time_ns"],
                          [BASE_NS+round(i*1e9/30) for i in range(3)])
    assert data["body_capture_time_ns"].dtype==np.int64
    assert not data["body_position_valid"][1,audit.UPPER_JOINTS].any()
    assert data["body_position_valid"][1,audit.LOWER_JOINTS].all()
    assert not data["body_rotation_valid"][2,audit.LOWER_JOINTS].any()
    assert not data["hand_position_valid"][0].any()
    assert not data["hand_position_valid"][1].any()
    assert data["hand_position_valid"][2].all()
    assert provenance["video_decoded"] is False
    assert provenance["capture_timestamp_fields"]["/pose/body"]==["header.ts"]
    assert set(("inode","device","ctime_ns"))<=provenance["source_stat"].keys()
    assert audit.file_sha256(path)==before


def test_shared_axis_candidates_are_exactly_24_proper_rotations():
    candidates=list(audit.signed_axis_permutations())
    assert len(candidates)==24
    assert len({tuple(c.ravel()) for c in candidates})==24
    for candidate in candidates:
        assert np.array_equal(candidate.T@candidate,np.eye(3))
        assert np.linalg.det(candidate)==pytest.approx(1)


def test_end_to_end_report_keeps_partial_and_leaf_axes_unverified(tmp_path):
    model=tmp_path/"neutral.npz"
    neutral_smplx_fixture(model)
    rows=[]
    originals={}
    for i in range(4):
        path=make_mcap(tmp_path/f"source-{i}.mcap",count=5)
        originals[path]=audit.file_sha256(path)
        rows.append({"uid":f"source-{i}","operator":f"operator-{i}","task":f"task-{i}",
                     "path":str(path),"basis_correction":None})
    catalog=tmp_path/"catalog.json"
    catalog.write_text(json.dumps(rows))
    output=tmp_path/"research-output"
    result=audit.main(["--catalog",str(catalog),"--target-model",str(model),"--output",str(output),
                       "--episodes","4","--max-frames","5","--min-anchor-frames","1"])
    assert result==0
    report=json.loads((output/"report.json").read_text())
    assert len(report["joints"])==22
    assert report["semantic_verified_joint_count"]==0
    assert report["all_22_axes_verified"] is False
    assert report["production_archive_modified"] is False
    assert set(report["fit_operator_ids"]).isdisjoint(report["heldout_operator_ids"])
    for joint in report["joints"]:
        assert joint["semantic_verified"] is False
        if joint["joint_name"] in ("head","left_foot","right_foot"):
            assert joint["candidate_matrix"] is None and joint["observable_dof"]==0
        elif joint["semantic_anchor_count"]==1:
            assert joint["geometry_rank"]==1 and joint["observable_dof"]==2
    for path in output.rglob("*.npz"):
        with np.load(path,allow_pickle=False) as arrays:
            assert all(arrays[name].dtype.kind!="O" for name in arrays.files)
    assert all(audit.file_sha256(path)==digest for path,digest in originals.items())
    with pytest.raises(SystemExit):
        audit.main(["--output",str(output)])
