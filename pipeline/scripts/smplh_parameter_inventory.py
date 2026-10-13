#!/usr/bin/env python3
"""Join frozen source inventories and allocate leak-free HM group splits."""

import argparse
from collections import Counter,defaultdict
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import uuid


EXPECTED={"egosuite_episodes":6614,"egosuite_frames":26841893,
          "hm_episodes":2649,"hm_frames":11295991,
          "combined_episodes":9263,"combined_frames":38137884}
SPLIT_FRACTIONS={"train":.8,"val":.1,"test":.1}
SPLIT_SEED="MixEgoMotion-HM-parameters-v1-person-capture"
DEFAULT_EGO_SPLITS=Path("/root/workspace/wbc/guanzerong/cvpr/MotionExpert/exp/dense_k12/splits.json")
EGO_GROUPING_SOURCE=Path("/root/workspace/wbc/guanzerong/cvpr/MotionExpert/dataset/split_manifest.py")


def json_bytes(value):
    return (json.dumps(value,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)+"\n").encode()


def digest(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def file_sha(path):
    result=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for data in iter(lambda:handle.read(8<<20),b""):
            result.update(data)
    return result.hexdigest()


def identity(path):
    path=Path(path).resolve();stat=path.stat()
    return {"path":str(path),"bytes":stat.st_size,"mtime_ns":stat.st_mtime_ns,
            "device":stat.st_dev,"inode":stat.st_ino}


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+".tmp-"+uuid.uuid4().hex)
    try:
        with temporary.open("wb") as handle:
            handle.write(json_bytes(value));handle.flush();os.fsync(handle.fileno())
        os.replace(temporary,path)
        fd=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
    finally:
        temporary.unlink(missing_ok=True)


def _positive_integer(value,label):
    if isinstance(value,bool) or not isinstance(value,int) or value<1:
        raise ValueError(label+" must be a positive integer")
    return value


def validate_ego_inventory(ego):
    if ego.get("source")!="egosuite" or not isinstance(ego.get("rows"),list) or not ego["rows"]:
        raise ValueError("Missing EgoSuite inventory rows")
    unsigned={k:v for k,v in ego.items() if k!="inventory_sha256"}
    if ego.get("inventory_sha256")!=digest(unsigned):
        raise ValueError("EgoSuite inventory canonical SHA changed")
    if ego.get("episodes")!=len(ego["rows"]):
        raise ValueError("EgoSuite episode count differs from rows")
    offset=0;keys=set()
    for index,row in enumerate(ego["rows"]):
        length=_positive_integer(row.get("length"),"EgoSuite length")
        if (row.get("episode_index")!=index or row.get("source_episode_index")!=index
                or row.get("source")!="egosuite" or row.get("split") not in SPLIT_FRACTIONS
                or row.get("dataset_from_index")!=offset or row.get("dataset_to_index")!=offset+length):
            raise ValueError("EgoSuite original index/split contract changed")
        if not isinstance(row.get("source_key"),str) or row["source_key"] in keys:
            raise ValueError("Duplicate/missing EgoSuite source key")
        keys.add(row["source_key"]);offset+=length
    if ego.get("frames")!=offset:
        raise ValueError("EgoSuite frame total differs from original contiguous indices")


def ego_recording_keys(record):
    """Exact original split_manifest.recording_keys policy, from actual metadata."""
    meta=record.get("metadata",{})
    if not meta.get("operator_id"):
        raise ValueError("Original EgoSuite split requires an actual operator_id")
    keys=[f"operator:{meta['operator_id']}"]
    for name in ("source_file_uid","source_episode_uuid","source_data"):
        if meta.get(name):keys.append(f"recording:{record['source']}:{name}:{meta[name]}")
    keys.append(f"source_key:{record['source_key']}")
    return keys


def normalize_ego_groups(ego,manifest,*,provenance=None):
    validate_ego_inventory(ego)
    if (manifest.get("format")!="motionexpert-archive-splits-v1"
            or manifest.get("group_by")!=["operator_id","original_recording"]):
        raise ValueError("Unsupported original EgoSuite grouped split policy")
    membership={}
    for split in SPLIT_FRACTIONS:
        for key in manifest.get("splits",{}).get(split,[]):
            if key in membership:raise ValueError("Original split manifest repeats a sequence")
            membership[key]=split
    if set(membership)!={row["source_key"] for row in ego["rows"]}:
        raise ValueError("Original EgoSuite split membership differs from inventory population")
    result=deepcopy(ego)
    for row in result["rows"]:
        record=row.get("source_record",{})
        if (record.get("source")!="egosuite" or record.get("source_key")!=row["source_key"]
                or row["split"]!=membership[row["source_key"]]):
            raise ValueError("EgoSuite original record/split identity changed")
        groups=ego_recording_keys(record)
        if "split_group_ids" in row and row["split_group_ids"]!=groups:
            raise ValueError("EgoSuite inventory group IDs differ from the original grouping policy")
        row["split_group_ids"]=groups
    leakage=assert_no_split_leakage(result["rows"])
    result["split_group_provenance"]={"group_by":manifest["group_by"],
        "original_archive_fingerprint":manifest.get("archive_fingerprint"),
        "original_split_assignment_preserved":True,"leakage_check":leakage,
        "grouping_policy":"exact original recording_keys: operator_id + actual source_file_uid/source_episode_uuid/source_data + source_key",
        **(provenance or {})}
    result.pop("inventory_sha256",None);result["inventory_sha256"]=digest(result)
    return result


def connected_components(items):
    """Episodes sharing any person/capture token belong to one component."""
    parents=list(range(len(items)))
    def find(index):
        while parents[index]!=index:
            parents[index]=parents[parents[index]];index=parents[index]
        return index
    def union(left,right):
        left,right=find(left),find(right)
        if left!=right:parents[max(left,right)]=min(left,right)
    first={}
    for index,item in enumerate(items):
        groups=item.get("split_group_ids")
        if (not isinstance(groups,list) or len(groups)<2 or any(not isinstance(g,str) or not g for g in groups)
                or not any(g.startswith("hm:participant:") for g in groups)
                or not any(g.startswith("hm:capture:") for g in groups)):
            raise ValueError("HM episode requires explicit person and capture grouping; unknown is a shared token")
        for group in groups:
            if group in first:union(index,first[group])
            else:first[group]=index
    partitions=defaultdict(list)
    for index in range(len(items)):partitions[find(index)].append(index)
    components=[]
    for indices in partitions.values():
        keys=sorted(items[i]["take_name"] for i in indices)
        groups=sorted({group for i in indices for group in items[i]["split_group_ids"]})
        component_id=hashlib.sha256(json_bytes({"groups":groups,"takes":keys})).hexdigest()
        components.append({"component_id":component_id,"take_names":keys,"split_group_ids":groups,
            "frames":sum(_positive_integer(items[i]["hm_rows"],"HM length") for i in indices),"indices":indices})
    return components


def assign_component_splits(items):
    components=connected_components(items)
    total=sum(c["frames"] for c in components)
    targets={split:total*weight for split,weight in SPLIT_FRACTIONS.items()}
    frames={split:0 for split in SPLIT_FRACTIONS}
    names=list(SPLIT_FRACTIONS)
    components.sort(key=lambda c:(-c["frames"],hashlib.sha256((SPLIT_SEED+c["component_id"]).encode()).hexdigest()))
    for component in components:
        split=max(names,key=lambda s:(targets[s]-frames[s],-names.index(s)))
        component["split"]=split;frames[split]+=component["frames"]
    # Move whole components only if this strictly improves frame balance.
    while True:
        best=None;improvement=0.
        for index,component in enumerate(components):
            origin=component["split"];size=component["frames"]
            for destination in names:
                if origin==destination:continue
                change=2*size*(frames[destination]-targets[destination]-frames[origin]+targets[origin])+2*size*size
                if change<improvement:
                    improvement=change;best=(index,origin,destination,size)
        if best is None:break
        index,origin,destination,size=best
        components[index]["split"]=destination;frames[origin]-=size;frames[destination]+=size
    membership={name:c["split"] for c in components for name in c["take_names"]}
    component_for={name:c["component_id"] for c in components for name in c["take_names"]}
    summary={"seed":SPLIT_SEED,"algorithm":"largest_component_first_remaining_frame_deficit_then_whole_component_squared_error_improvement",
        "targets":targets,"frames":frames,"frame_fractions":{s:frames[s]/total for s in names},
        "components":len(components),"component_records":[{k:v for k,v in c.items() if k!="indices"} for c in components],
        "unknown_person_policy":"all absent participant_uid share hm:participant:unknown; never treated as distinct subjects",
        "cross_source_identity":"unavailable; EgoSuite published groups/splits retained unchanged; no numeric subject ID conflation"}
    return membership,component_for,summary


def assert_no_split_leakage(rows):
    group_splits=defaultdict(set)
    for row in rows:
        for group in row["split_group_ids"]:group_splits[group].add(row["split"])
    leaking={group:sorted(splits) for group,splits in group_splits.items() if len(splits)!=1}
    if leaking:raise ValueError("HM person/capture group crosses splits: "+json.dumps(leaking,sort_keys=True))
    return {"passed":True,"group_count":len(group_splits),"cross_split_groups":0}


def build_hm_inventory(ego,hm,takes,*,source_files=None,expected=None):
    validate_ego_inventory(ego)
    if not isinstance(hm.get("items"),list) or not hm["items"] or hm.get("take_count")!=len(hm["items"]):
        raise ValueError("HM inventory counts/items mismatch")
    items=sorted(hm["items"],key=lambda x:x["take_name"])
    if len({i["take_name"] for i in items})!=len(items):raise ValueError("Duplicate HM take identity")
    if hm.get("hm_frames")!=sum(_positive_integer(i.get("hm_rows"),"HM length") for i in items):
        raise ValueError("HM inventory frame sum mismatch")
    for item in items:
        name=item["take_name"]
        if name not in takes or item.get("take_uid")!=takes[name].get("take_uid"):
            raise ValueError("HM take UID differs from original takes.json")
        take=takes[name]
        participant=take.get("participant_uid");capture=take.get("capture_uid")
        expected_groups=[f"hm:participant:{participant}" if participant is not None else "hm:participant:unknown",
                         f"hm:capture:{capture}" if capture else "hm:capture:unknown"]
        if sorted(item.get("split_group_ids",[]))!=sorted(expected_groups):
            raise ValueError("HM group IDs do not match original person/capture identity")
        if not item.get("hm_path") or not item.get("assets",{}).get("dino"):
            raise ValueError("HM source/vision path missing from inventory")
        if item.get("alignment",{}).get("hm_rows")!=item["hm_rows"]:
            raise ValueError("HM alignment row count differs from parameters")
    membership,component_for,split_summary=assign_component_splits(items)
    offset=ego["frames"];rows=[]
    for source_index,item in enumerate(items):
        take=takes[item["take_name"]];length=item["hm_rows"]
        rows.append({"episode_index":ego["episodes"]+source_index,"source_episode_index":source_index,
            "source":"egoexo4d_hm","source_key":"egoexo4d_hm:"+item["take_name"],
            "source_uid":take["take_uid"],"take_name":item["take_name"],"length":length,
            "dataset_from_index":offset,"dataset_to_index":offset+length,
            "source_path":item["hm_path"],"vision_source_path":item["assets"]["dino"],
            "take_metadata":deepcopy(take),"split":membership[item["take_name"]],
            "source_split":item.get("source_split"),"split_group_ids":deepcopy(item["split_group_ids"]),
            "split_component_id":component_for[item["take_name"]],
            "preflight_eligible":item.get("preflight_eligible",False),
            "frame_alignment":deepcopy(item["alignment"]),"quarantine_reasons":deepcopy(item.get("quarantine_reasons",[])),
            "source_identity":deepcopy(item.get("hm_identity")),"video_identity":deepcopy(item.get("video_identity")),
            "vision_identity":deepcopy(item.get("dino_identity")),"vision_source_sha256":item.get("dino_sha256")})
        offset+=length
    leakage=assert_no_split_leakage(rows)
    if expected is not None:
        actual={"egosuite_episodes":ego["episodes"],"egosuite_frames":ego["frames"],
            "hm_episodes":len(rows),"hm_frames":hm["hm_frames"],"combined_episodes":ego["episodes"]+len(rows),"combined_frames":offset}
        if actual!=expected:raise ValueError("Production population differs from frozen expected counts: "+json.dumps(actual))
    split_summary["leakage_check"]=leakage
    value={"source":"egoexo4d_hm","episodes":len(rows),"frames":hm["hm_frames"],"rows":rows,
        "source_inventory_identities":source_files or {},"source_revision":hm["hm_revision"],
        "split_policy":split_summary,"official_split_policy":"source_split is null when original metadata provides no official partition; mixed splits are separate",
        "row_mapping":{"episode_index":"EgoSuite episode count + sorted HM take index",
            "frame_index":"zero-based released HM row", "index":"dataset_from_index + frame_index",
            "source.video_frame_index":"explicit certified per-take released row/video mapping; undefined remains -1",
            "join":["episode_index","frame_index"]},
        "preflight_eligible":sum(r["preflight_eligible"] for r in rows),
        "preflight_semantics":"assets/frame mapping only; world and numerical acceptance are checked during export/audit"}
    value["inventory_sha256"]=digest(value)
    combined={"source":"mixed_egosuite_egoexo4d_hm","episodes":ego["episodes"]+len(rows),"frames":offset,
        "source_counts":{"egosuite":{"episodes":ego["episodes"],"frames":ego["frames"]},
            "egoexo4d_hm":{"episodes":len(rows),"frames":hm["hm_frames"]}},
        "source_inventory_identities":source_files or {},"source_inventory_canonical_sha256":{
            "egosuite":ego["inventory_sha256"],"egoexo4d_hm":value["inventory_sha256"]},
        "rows":deepcopy(ego["rows"])+deepcopy(rows),"split_policy":{"egosuite":"original fixed split copied verbatim","egoexo4d_hm":split_summary},
        "created_at":datetime.now(timezone.utc).isoformat()}
    combined["inventory_sha256"]=digest(combined)
    return value,combined


def run(job):
    job=Path(job);ego_path=job/"inventory/egosuite.json";hm_path=job/"inventory/hm.json"
    before={"egosuite":identity(ego_path),"hm":identity(hm_path)}
    ego=json.loads(ego_path.read_text());hm=json.loads(hm_path.read_text())
    split_before=identity(DEFAULT_EGO_SPLITS)
    grouping_before=identity(EGO_GROUPING_SOURCE)
    split_manifest=json.loads(DEFAULT_EGO_SPLITS.read_text())
    normalized=normalize_ego_groups(ego,split_manifest,provenance={
        "split_manifest_identity":{**split_before,"sha256":file_sha(DEFAULT_EGO_SPLITS)},
        "grouping_source_identity":{**grouping_before,"sha256":file_sha(EGO_GROUPING_SOURCE)}})
    for key,stat in before.items():
        if identity(stat["path"])!=stat:raise ValueError("Source inventory changed during group normalization: "+key)
    if identity(DEFAULT_EGO_SPLITS)!=split_before or identity(EGO_GROUPING_SOURCE)!=grouping_before:
        raise ValueError("Original EgoSuite split/grouping source changed during normalization")
    if normalized!=ego:atomic_json(ego_path,normalized)
    ego=normalized;before["egosuite"]=identity(ego_path)
    takes_path=Path(hm["dataset_root"])/"EgoExo4D/takes.json"
    before["takes_metadata"]=identity(takes_path)
    takes_raw=json.loads(takes_path.read_text());takes={t["take_name"]:t for t in takes_raw}
    if len(takes)!=len(takes_raw):raise ValueError("Duplicate original takes metadata")
    files={key:{**stat,"sha256":file_sha(stat["path"])} for key,stat in before.items()}
    files["egosuite_split_manifest"]={**split_before,"sha256":file_sha(DEFAULT_EGO_SPLITS)}
    expected_take=hm.get("metadata_identities",{}).get("EgoExo4D/takes.json")
    if expected_take is None or expected_take!=files["takes_metadata"]:
        raise ValueError("Original takes metadata differs from the source HM inventory identity")
    value,combined=build_hm_inventory(ego,hm,takes,source_files=files,expected=EXPECTED)
    for key,stat in before.items():
        if identity(stat["path"])!=stat:raise ValueError("Source inventory changed during assembly: "+key)
    atomic_json(job/"inventory/egoexo4d_hm.json",value)
    atomic_json(job/"inventory/combined.json",combined)
    return {"hm_episodes":value["episodes"],"hm_frames":value["frames"],
        "combined_episodes":combined["episodes"],"combined_frames":combined["frames"],
        "hm_inventory_sha256":value["inventory_sha256"],"combined_inventory_sha256":combined["inventory_sha256"],
        "split_frames":value["split_policy"]["frames"],"split_frame_fractions":value["split_policy"]["frame_fractions"],
        "components":value["split_policy"]["components"],"split_leakage":value["split_policy"]["leakage_check"]}


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--job",required=True)
    arguments=parser.parse_args();print(json.dumps(run(arguments.job),indent=2,allow_nan=False))
