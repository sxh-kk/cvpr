from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest


PATH=Path(__file__).resolve().parents[1]/"scripts/smplh_parameter_inventory.py"
SPEC=importlib.util.spec_from_file_location("parameter_inventory_tested",PATH)
inventory=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(inventory)


def ego_fixture():
    rows=[];offset=0
    for index,(length,split) in enumerate([(12,"train"),(4,"val"),(4,"test")]):
        rows.append({"source":"egosuite","episode_index":index,"source_episode_index":index,
            "source_key":f"egosuite:{index}","length":length,"split":split,
            "dataset_from_index":offset,"dataset_to_index":offset+length,
            "source_record":{"original":index}})
        offset+=length
    value={"source":"egosuite","episodes":len(rows),"frames":offset,"rows":rows}
    value["inventory_sha256"]=inventory.digest(value)
    return value


def hm_fixture(people_captures):
    items=[];takes={}
    for index,(person,capture,length) in enumerate(people_captures):
        name=f"unit_{index:03d}";uid=f"take-uid-{index}"
        groups=[f"hm:participant:{person}" if person is not None else "hm:participant:unknown",
                f"hm:capture:{capture}"]
        item={"take_name":name,"take_uid":uid,"hm_rows":length,"split_group_ids":groups,
            "hm_path":f"/source/{name}/hm.npz","assets":{"dino":f"/source/DINO/{name}.parquet"},
            "alignment":{"hm_rows":length,"verified":True},"preflight_eligible":True,
            "source_split":None,"quarantine_reasons":[]}
        items.append(item);takes[name]={"take_name":name,"take_uid":uid,"participant_uid":person,
                                        "capture_uid":capture,"original_extra":{"preserved":True}}
    value={"hm_revision":"unit-revision","take_count":len(items),"hm_frames":sum(i["hm_rows"] for i in items),"items":items}
    return value,takes


def test_transitive_person_capture_links_never_cross_split():
    hm,takes=hm_fixture([(1,"a",50),(1,"b",20),(2,"b",25),(2,"c",10),(3,"d",15),
                         (None,"x",11),(None,"y",8)])
    result,combined=inventory.build_hm_inventory(ego_fixture(),hm,takes)
    rows={r["take_name"]:r for r in result["rows"]}
    assert len({rows[f"unit_{i:03d}"]["split"] for i in range(4)})==1
    assert rows["unit_005"]["split"]==rows["unit_006"]["split"]
    assert len({rows[f"unit_{i:03d}"]["split_component_id"] for i in range(4)})==1
    assert result["split_policy"]["components"]==3
    assert result["split_policy"]["leakage_check"]["cross_split_groups"]==0


def test_disconnected_components_are_frame_balanced_and_deterministic():
    hm,takes=hm_fixture([(i,str(i),10) for i in range(100)])
    value,combined=inventory.build_hm_inventory(ego_fixture(),hm,takes)
    assert value["split_policy"]["frames"]=={"train":800,"val":100,"test":100}
    shuffled=deepcopy(hm);shuffled["items"]=list(reversed(shuffled["items"]))
    another,_=inventory.build_hm_inventory(ego_fixture(),shuffled,takes)
    assert value["rows"]==another["rows"]
    assert value["inventory_sha256"]==another["inventory_sha256"]


def test_global_indices_and_ego_splits_are_preserved():
    ego=ego_fixture();original=deepcopy(ego)
    hm,takes=hm_fixture([(1,"a",7),(2,"b",9)])
    value,combined=inventory.build_hm_inventory(ego,hm,takes)
    assert ego==original and combined["rows"][:3]==original["rows"]
    assert value["rows"][0]["episode_index"]==3
    assert value["rows"][0]["dataset_from_index"]==20
    assert value["rows"][1]["dataset_to_index"]==36
    assert combined["episodes"]==5 and combined["frames"]==36
    assert value["rows"][0]["take_metadata"]==takes["unit_000"]
    assert value["rows"][0]["source_split"] is None


def test_leakage_checker_rejects_cross_split_component():
    rows=[{"split":"train","split_group_ids":["hm:participant:1","hm:capture:a"]},
          {"split":"test","split_group_ids":["hm:participant:1","hm:capture:b"]}]
    with pytest.raises(ValueError,match="crosses splits"):
        inventory.assert_no_split_leakage(rows)


@pytest.mark.parametrize("corruption",["ego_digest","ego_offset","ego_total","hm_uid","hm_total","hm_duplicate","group_identity","alignment"])
def test_input_corruption_fails_before_output(corruption):
    ego=ego_fixture();hm,takes=hm_fixture([(1,"a",7),(2,"b",9)])
    if corruption=="ego_digest":ego["rows"][0]["split"]="test"
    if corruption=="ego_offset":
        ego["rows"][1]["dataset_from_index"]+=1
        ego["inventory_sha256"]=inventory.digest({k:v for k,v in ego.items() if k!="inventory_sha256"})
    if corruption=="ego_total":
        ego["frames"]+=1;ego["inventory_sha256"]=inventory.digest({k:v for k,v in ego.items() if k!="inventory_sha256"})
    if corruption=="hm_uid":hm["items"][0]["take_uid"]="wrong"
    if corruption=="hm_total":hm["hm_frames"]+=1
    if corruption=="hm_duplicate":hm["items"][1]=deepcopy(hm["items"][0])
    if corruption=="group_identity":hm["items"][0]["split_group_ids"][0]="hm:participant:another"
    if corruption=="alignment":hm["items"][0]["alignment"]["hm_rows"]+=1
    with pytest.raises(ValueError):inventory.build_hm_inventory(ego,hm,takes)


def test_unknown_person_token_cannot_be_replaced_with_unique_fabricated_subject():
    hm,takes=hm_fixture([(None,"a",7),(None,"b",9)])
    hm["items"][0]["split_group_ids"][0]="hm:participant:fake-person0"
    with pytest.raises(ValueError,match="group IDs"):
        inventory.build_hm_inventory(ego_fixture(),hm,takes)


def test_quarantine_is_preserved_not_dropped():
    hm,takes=hm_fixture([(1,"a",7),(2,"b",9)])
    hm["items"][0].update(preflight_eligible=False,quarantine_reasons=["missing-camera"])
    value,combined=inventory.build_hm_inventory(ego_fixture(),hm,takes)
    assert value["episodes"]==2 and value["frames"]==16
    assert value["preflight_eligible"]==1
    assert value["rows"][0]["quarantine_reasons"]==["missing-camera"]


def grouped_ego_fixture():
    ego=ego_fixture()
    for i,row in enumerate(ego["rows"]):
        row["source_record"]={"source":"egosuite","source_key":row["source_key"],
            "metadata":{"operator_id":f"operator-{i}","source_file_uid":f"file-{i}",
                        "source_episode_uuid":f"record-{i}","source_data":f"/raw/source-{i}.mcap"}}
    ego["inventory_sha256"]=inventory.digest({k:v for k,v in ego.items() if k!="inventory_sha256"})
    manifest={"format":"motionexpert-archive-splits-v1","group_by":["operator_id","original_recording"],
        "archive_fingerprint":"original-v5-fingerprint", "splits":{
            split:[row["source_key"] for row in ego["rows"] if row["split"]==split]
            for split in ["train","val","test"]}}
    return ego,manifest


def test_ego_group_normalization_uses_actual_original_fields_and_keeps_splits():
    ego,manifest=grouped_ego_fixture();out=inventory.normalize_ego_groups(ego,manifest)
    row=out["rows"][0]
    assert row["split_group_ids"]==["operator:operator-0","recording:egosuite:source_file_uid:file-0",
        "recording:egosuite:source_episode_uuid:record-0","recording:egosuite:source_data:/raw/source-0.mcap",
        "source_key:egosuite:0"]
    assert [row["split"] for row in out["rows"]]==[row["split"] for row in ego["rows"]]
    assert inventory.normalize_ego_groups(out,manifest)==out
    assert "split_group_ids" not in ego["rows"][0]


@pytest.mark.parametrize("problem",["operator_missing","group_leakage","membership","policy"])
def test_ego_group_normalization_rejects_missing_or_conflicting_identity(problem):
    ego,manifest=grouped_ego_fixture()
    if problem=="operator_missing":ego["rows"][0]["source_record"]["metadata"].pop("operator_id")
    if problem=="group_leakage":ego["rows"][1]["source_record"]["metadata"]["operator_id"]="operator-0"
    if problem=="membership":manifest["splits"]["train"]=[]
    if problem=="policy":manifest["group_by"]=["guessed_subject"]
    ego["inventory_sha256"]=inventory.digest({k:v for k,v in ego.items() if k!="inventory_sha256"})
    with pytest.raises(ValueError):inventory.normalize_ego_groups(ego,manifest)
