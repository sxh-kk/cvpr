"""Native HM SMPL-H parameters and an explicit released-row/video join.

This is independent of the legacy dense archive. No pose fitting, trimming,
world registration, or hand landmark correction is performed here. The known
release drops the final video frame because its generator uses [0, N-1).
Unexpected releases/assets remain exportable with invalid camera/floor masks
and a quarantine receipt; they cannot silently become canonical training data.
"""

from collections import Counter
import csv
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import zipfile

import numpy as np
import pyarrow.parquet as pq

from .body_model import BodyModel, file_sha256
from .geometry import matrix_to_quat, quat_to_matrix
from .schema import Episode, PARENTS


HM_REVISION = "6e1cd8627a0c5751b02493a87172dd3709c0c652"
OFFICIAL_COMMIT = "135d44ee5310c554fde1cbe33e02c7c204717588"
MODEL_SHA256 = "be42bff1ff0589cf0467893bcfbbeb3f9b533ba151031c8cdfb64ca1dcb34947"
LEGACY_FULL_CHUNK_TAKES = frozenset({"iiith_cooking_110_2","iiith_cooking_120_4",
    "iiith_cooking_23_1","iiith_cooking_24_1","iiith_cooking_43_1","iiith_cooking_62_2","sfu_cooking020_6"})
OFFICIAL_SOURCES = {
    "scripts/run_pipeline.py": "371461de4ea40f4f4a3ccc45ec90988dfa7590657a14999bfba55db3c3eb6f55",
    "slahmr/data/dataset.py": "bf2295037540f1034027489b5657881f610e473130190aa705e94e8260b97a37",
    "scripts/run_mesh_vis_hands_egoexo.py": "b0dde65e6c82a0c7d3aa46b51261ac04e0bda6ea71dc0aafe0e691b42b109b1f",
    "scripts/undistort_egoexo.py": "024b4cb51147e9c09b6577c5324744554eee822249acf1db9e0165dfc009b10d",
}
PARAMETER_FIELDS = {"trans": (3,), "root_orient": (3,), "pose_body": (63,),
                    "betas_per_frame": (16,), "hand_pose": (90,), "valid": (),
                    "joints3d": (67, 3)}


@lru_cache(maxsize=4096)
def _stable_sha(path, byte_count, mtime_ns, device, inode):
    return file_sha256(path)


def file_identity(path, *, sha256=False):
    path = Path(path).resolve()
    stat = path.stat()
    value = {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
             "device": stat.st_dev, "inode": stat.st_ino}
    if sha256:
        value["sha256"] = _stable_sha(str(path),stat.st_size,stat.st_mtime_ns,stat.st_dev,stat.st_ino)
    return value


@lru_cache(maxsize=2)
def _verified_model(path, identity_token):
    model=BodyModel(path)
    if (model.metadata["model_sha256"]!=MODEL_SHA256 or model.metadata["vertex_count"]!=6890
            or model.metadata["joint_count"]!=52 or model.shape_basis.shape[-1]!=16):
        raise ValueError("HM export requires the exact released male SMPL-H 52/16 model")
    return model


def source_evidence():
    return {"repository": "https://github.com/Abhiram824/egoexo4d_human_meshes",
            "commit": OFFICIAL_COMMIT,
            "files": [{"path": path, "sha256": sha,
                "url": f"https://raw.githubusercontent.com/Abhiram824/egoexo4d_human_meshes/{OFFICIAL_COMMIT}/{path}"}
                for path, sha in OFFICIAL_SOURCES.items()],
            "chain": [
                "run_pipeline.py:307-311 default start=0, end=video_len-1",
                "run_pipeline.py:223-248 optimize half-open chunks [start,end)",
                "data/dataset.py:94-98 images[start_idx:end_idx]",
                "run_mesh_vis_hands_egoexo.py:139-144 HM row i uses image (i+1):06d.jpg",
                "undistort_egoexo.py:104 ffmpeg %06d.jpg starts at image 1"],
            "row_mapping": "HM row i -> one-based extracted image i+1 -> zero-based video/DINO row i",
            "last_video_frame": "intentionally excluded by published generator default",
            "certification": "release_generation_and_viewer_contract; not_image_annotation_ground_truth"}


def check_chunks(chunks, count):
    raw = np.asarray(chunks)
    if raw.ndim != 2 or raw.shape[1] != 2 or not len(raw) or raw.dtype.kind not in "iu":
        raise ValueError("HM chunk_ranges must be a nonempty integer [C,2]")
    if (raw[0, 0] != 0 or raw[-1, 1] != count or np.any(raw[:, 1] <= raw[:, 0])
            or np.any(raw[1:, 0] != raw[:-1, 1])):
        raise ValueError("HM chunks must cover all released rows exactly once, starting at zero")
    return raw.astype(np.int64, copy=True)


def read_parameter_source(path):
    """Read only native parameters/landmarks; forbid pickles and shape guesses."""
    with np.load(path, allow_pickle=False) as data:
        missing = set(PARAMETER_FIELDS) - set(data.files)
        if missing or "chunk_ranges" not in data.files:
            raise ValueError(f"Missing released HM fields: {sorted(missing | ({'chunk_ranges'}-set(data.files)))}")
        count = data["trans"].shape[1] if data["trans"].ndim == 3 else -1
        if count < 1:
            raise ValueError("Empty/malformed HM trans")
        out = {}
        for key, tail in PARAMETER_FIELDS.items():
            value = data[key]
            if value.shape != (1, count, *tail) or value.dtype.kind not in "fiu":
                raise ValueError(f"HM {key} must have released shape {(1,count,*tail)}")
            out[key] = value[0].copy()
        if not np.isin(out["valid"], [0, 1]).all():
            raise ValueError("HM valid is not a binary chunk-finiteness flag")
        out["chunk_ranges"] = check_chunks(data["chunk_ranges"], count)
        for key in ("cam_R", "cam_t"):
            if key in data.files:
                out[key] = data[key].copy()
    return out


def npz_headers(path):
    """Cheap full-take inventory: NPY headers plus the tiny chunk table."""
    headers = {}
    with zipfile.ZipFile(path) as archive:
        for key in (*PARAMETER_FIELDS, "cam_R", "cam_t"):
            with archive.open(key+".npy") as stream:
                version = np.lib.format.read_magic(stream)
                shape, _, dtype = np.lib.format._read_array_header(stream, version)
                headers[key] = {"shape": list(shape), "dtype": str(dtype)}
        with archive.open("chunk_ranges.npy") as stream:
            chunks = np.lib.format.read_array(stream, allow_pickle=False)
    return headers, chunks


def alignment_receipt(take, count, chunks, dino_metadata, *, revision=HM_REVISION,
                      video_identity=None, dino_rows=None):
    """Certify the documented mapping only if all independent identities agree."""
    reasons = []
    valid_chunks = False
    try:
        validated_chunks=check_chunks(chunks, count)
        valid_chunks=True
    except ValueError as error:
        reasons.append(str(error))
    video_count = int(take["timesync_end_idx"] - take["timesync_start_idx"])
    if revision != HM_REVISION:
        reasons.append("HM revision differs from pinned released parameters")
    legacy = bool(valid_chunks and take["take_name"] in LEGACY_FULL_CHUNK_TAKES
        and count == (video_count//100)*100 and 0 < video_count-count < 100
        and np.all(validated_chunks[:,1]-validated_chunks[:,0]==100))
    if count != video_count-1 and not legacy:
        reasons.append("HM length does not match the default or explicitly registered legacy full-chunk contract")
    video = dino_metadata.get("video", {})
    if (dino_metadata.get("contract") != "egoexo-video-dino5-30hz-v1"
            or dino_metadata.get("state") != "complete"
            or video.get("take_name") != take["take_name"]
            or video.get("take_uid") != take["take_uid"]
            or video.get("length") != video_count or dino_metadata.get("length") != video_count
            or video.get("fps") != 30 or video.get("clock") != "frame_aligned_take_seconds"
            or video.get("video_info", {}).get("frames") != video_count
            or video.get("video_info", {}).get("average_rate") != "30"
            or video.get("video_info", {}).get("start_time_s") != 0.0):
        reasons.append("DINO/video/take clock or identity mismatch")
    expected_relative = f"takes/{take['take_name']}/frame_aligned_videos/{Path(video.get('video_path','')).name}"
    if video.get("relative_video_path") != expected_relative:
        reasons.append("Video does not belong to this take")
    if video_identity is None or video.get("video_fingerprint") != {
            "size_bytes": video_identity.get("bytes"), "mtime_ns": video_identity.get("mtime_ns")}:
        reasons.append("Source video fingerprint changed or unavailable")
    if video_identity is None or str(Path(video.get("video_path","")).resolve()) != video_identity.get("path"):
        reasons.append("DINO source video path does not match the selected take asset")
    if dino_rows != video_count:
        reasons.append("DINO actual Parquet row count differs from timesync/video count")
    method="verified_by_pinned_release_viewer_row_contract_legacy_full100_tail" if legacy else "verified_by_pinned_release_generation_contract"
    return {"verified": not reasons, "method": method if not reasons else "unverified_release_frame_join",
            "hm_rows": int(count), "video_rows": video_count,
            "hm_row_to_video_frame": {"offset": 0, "step": 1, "length": int(count)},
            "hm_row_to_extracted_image": {"offset": 1, "step": 1, "length": int(count)},
            "take_uid": take["take_uid"], "hm_revision": revision,
            "source_evidence": source_evidence(), "quarantine_reasons": reasons,
            "last_video_frame_not_in_motion": video_count-1 if count == video_count-1 else None,
            "unused_video_frame_range_half_open": [int(count),video_count] if count<=video_count else None,
            "legacy_generation_flags": "historical flags cannot be recovered; consistent with legacy complete100 chunk rule; row mapping follows explicit pinned merged viewer" if legacy else None,
            "legacy_rule_evidence": "run_pipeline.py:310 historical commented floor(video_len/100)*100; this alone is not certification" if legacy else None}


def _ego_assets(root, take):
    root = Path(root)
    take_root = root/"EgoExo4D/takes"/take["take_name"]
    cameras = [c for c in take["capture"]["cameras"] if c.get("is_ego")]
    arias = [c for c in cameras if c["cam_id"].startswith("aria")]
    candidates = arias if len(arias) == 1 else cameras
    assets = {"take_root": take_root, "ego_camera": None, "is_aria": False}
    if len(candidates) != 1:
        return assets
    camera = candidates[0]["cam_id"]
    streams = take["frame_aligned_videos"].get(camera, {})
    stream = streams.get("rgb") if camera.startswith("aria") else streams.get("0")
    if not stream or not stream.get("relative_path"):
        return assets
    video = take_root/stream["relative_path"]
    assets.update(ego_camera=camera, is_aria=camera.startswith("aria"), video=video,
        dino=root/"DINO/takes"/take["take_name"]/(video.stem+".parquet"),
        timesync=root/"EgoExo4D/captures"/take["capture"]["capture_name"]/"timesync.csv",
        vrs=list(take_root.glob("*noimagestreams.vrs")),
        trajectory=take_root/"trajectory/closed_loop_trajectory.csv",
        gopro_calibrations=take_root/"trajectory/gopro_calibs.csv")
    return assets


def inventory_hm(root):
    """Inventory every planned take without reading DINO tensors or body arrays."""
    root = Path(root)
    plan = json.loads((root/"download_plan.json").read_text())
    takes = {t["take_name"]: t for t in json.loads((root/"EgoExo4D/takes.json").read_text())}
    items = []
    for entry in plan["hm_files"]:
        path = root/"EgoExo4D-HM"/entry["relative_path"]
        uid = path.parent.name
        item = {"take_name": uid, "hm_path": str(path), "quarantine_reasons": []}
        try:
            if not path.is_file() or path.stat().st_size != entry["size"]:
                raise ValueError("HM source missing/incomplete")
            take = takes[uid]
            headers, chunks = npz_headers(path)
            count = headers["trans"]["shape"][1]
            for key, tail in PARAMETER_FIELDS.items():
                if headers[key]["shape"] != [1,count,*tail]:
                    raise ValueError(f"Released parameter header mismatch: {key}")
            check_chunks(chunks, count)
            assets = _ego_assets(root, take)
            participant=take.get("participant_uid")
            capture=take.get("capture_uid")
            groups=[f"hm:participant:{participant}" if participant is not None else "hm:participant:unknown"]
            groups.append(f"hm:capture:{capture}" if capture else "hm:capture:unknown")
            item.update(take_uid=take["take_uid"], participant_uid=participant,capture_uid=capture,
                split_group_ids=groups,source_split=take.get("split",take.get("partition")),
                hm_rows=count, video_rows=int(take["timesync_end_idx"]-take["timesync_start_idx"]),
                chunk_ranges=chunks.tolist(), hm_identity=file_identity(path), parameter_headers=headers,
                assets={key: [str(p) for p in value] if isinstance(value,list) else str(value) if isinstance(value,Path) else value
                        for key,value in assets.items()})
            if not assets.get("dino") or not assets["dino"].with_suffix(".json").is_file():
                raise ValueError("Selected ego DINO sidecar missing")
            meta = json.loads(assets["dino"].with_suffix(".json").read_text())
            actual_rows = pq.ParquetFile(assets["dino"]).metadata.num_rows
            alignment = alignment_receipt(take,count,chunks,meta,revision=plan["hm_revision"],
                video_identity=file_identity(assets["video"]), dino_rows=actual_rows)
            item.update(alignment=alignment, video_identity=file_identity(assets["video"]),dino_identity=file_identity(assets["dino"]),
                dino_sha256=meta["parquet_sha256"], dino_signature=meta["signature"])
            item["quarantine_reasons"].extend(alignment["quarantine_reasons"])
            if not assets["is_aria"]:
                item["quarantine_reasons"].append("No uniquely selected Aria camera; ego GoPro pose not fabricated")
            for key in ("timesync","trajectory","gopro_calibrations"):
                if not assets.get(key) or not assets[key].is_file():
                    item["quarantine_reasons"].append("Missing camera asset: "+key)
            if len(assets.get("vrs",[])) != 1:
                item["quarantine_reasons"].append("Missing/ambiguous Aria VRS calibration")
        except (ValueError,KeyError,IndexError,OSError,zipfile.BadZipFile) as error:
            item["quarantine_reasons"].append(str(error))
        item["preflight_eligible"] = not item["quarantine_reasons"]
        items.append(item)
    counts = Counter(reason for item in items for reason in item["quarantine_reasons"])
    return {"contract":"egoexo-hm-native-smplh-parameter-inventory-v1", "dataset_root":str(root.resolve()),
            "hm_revision":plan["hm_revision"], "take_count":len(items),
            "hm_frames":sum(i.get("hm_rows",0) for i in items),
            "metadata_identities":{name:file_identity(root/name,sha256=True) for name in ("download_plan.json","EgoExo4D/takes.json")},
            "preflight_eligible":sum(i["preflight_eligible"] for i in items),
            "quarantine_count":sum(not i["preflight_eligible"] for i in items),
            "reasons":dict(counts), "source_evidence":source_evidence(), "items":items,
            "world_registration": "requires per-take camera/calibration verification at export; inventory is not final acceptance"}


def verify_hm_world(raw, gopro_rows, graph_uids, gravity, *, tolerance=1e-5):
    """Prove HM w2c and Aria MPS use the same graph/world; view zero is ignored."""
    errors, reasons = [], []
    rows = [row for row in gopro_rows if float(row["quality"]) == 1.0]
    count = len(raw["trans"])
    if raw.get("cam_R",np.empty(0)).shape != (5,count,3,3) or raw.get("cam_t",np.empty(0)).shape != (5,count,3):
        return {"verified":False,"quarantine_reasons":["HM external camera tensors missing/malformed"]}
    if len(rows) < 4 or not graph_uids:
        reasons.append("Fewer than four localized external cameras or missing MPS graph")
    frame_valid = raw["valid"].astype(bool)
    if not frame_valid.any():
        reasons.append("No valid HM frame to compare world calibration")
    for view,row in enumerate(rows[:4],1):
        q = np.array([float(row[f"q{a}_world_cam"]) for a in "wxyz"])
        if not np.isfinite(q).all() or np.linalg.norm(q)<1e-8:
            reasons.append("Invalid external camera quaternion")
            continue
        R = quat_to_matrix(q/np.linalg.norm(q))
        p = np.array([float(row[f"t{a}_world_cam"]) for a in "xyz"])
        w2c_R, w2c_t = R.T,-R.T@p
        re = float(np.max(abs(raw["cam_R"][view,frame_valid]-w2c_R))) if frame_valid.any() else None
        te = float(np.max(abs(raw["cam_t"][view,frame_valid]-w2c_t))) if frame_valid.any() else None
        passed = re is not None and np.isfinite([re,te]).all() and max(re,te)<tolerance and {str(row["graph_uid"])} == graph_uids
        errors.append({"view":view,"camera_uid":row["cam_uid"],"graph_uid":row["graph_uid"],
                       "rotation_max_element_error":re,"translation_max_element_error_m":te,"passed":bool(passed)})
        if not passed:
            reasons.append(f"HM external view {view} does not match Aria MPS world graph/calibration")
    gravity = np.asarray(gravity,float)
    vertical = bool(gravity.ndim==2 and gravity.shape[1]==3 and len(gravity)
                    and np.isfinite(gravity).all() and np.max(np.linalg.norm(gravity-[0,0,-9.81],axis=-1))<1e-4)
    if not vertical:
        reasons.append("MPS gravity does not establish Z-up meters")
    return {"verified":not reasons,"vertical_axis_verified":vertical,"external_views":errors,
            "mps_graph_uids":sorted(graph_uids),"world_from_source":np.eye(4).tolist(),
            "position_scale":1.0,"quarantine_reasons":reasons,
            "semantics":"same_graph_and_external_calibration_numeric_evidence; not_annotation_accuracy"}


def load_aria_camera(assets, take, count, raw, alignment, *, max_pose_error_s=.02):
    from .aria import capture_timestamps
    identity_camera = np.broadcast_to(np.eye(4),(count,4,4)).copy()
    result = {"camera_world":identity_camera,"head_world":identity_camera.copy(),
              "valid":np.zeros(count,bool),"capture_timestamp_ns":np.full(count,-1,np.int64),
              "pose_time_error_s":np.zeros(count),"provenance":{},
              "world":{"verified":False,"quarantine_reasons":["Aria mapping/assets unavailable"]}}
    if not alignment["verified"] or not assets.get("is_aria") or len(assets.get("vrs",[]))!=1:
        return result
    if not np.isfinite(max_pose_error_s) or max_pose_error_s<=0:
        raise ValueError("Camera time-error gate must be finite and positive")
    from projectaria_tools.core import data_provider,mps
    sources = {key:assets[key] for key in ("timesync","trajectory","gopro_calibrations")}
    sources["vrs"] = assets["vrs"][0]
    result["provenance"] = {key:file_identity(path,sha256=True) for key,path in sources.items()}
    column = f"{assets['ego_camera']}_214-1_capture_timestamp_ns"
    stamps = capture_timestamps(assets["timesync"],column,int(take["timesync_start_idx"]),int(take["timesync_end_idx"]))
    if len(stamps)!=alignment["video_rows"]:
        raise ValueError("Actual timesync rows differ from the verified video/DINO clock")
    alignment["actual_timesync_rows"]=int(len(stamps))
    # This slice is the documented i->video_i mapping, after its length/identity gate.
    result["capture_timestamp_ns"] = stamps[np.arange(count)]
    provider=data_provider.create_vrs_data_provider(str(assets["vrs"][0]))
    device=provider.get_device_calibration()
    rgb=device.get_camera_calib("camera-rgb")
    dc=rgb.get_transform_device_camera().to_matrix()
    dh=device.get_transform_device_cpf().to_matrix()
    mps_provider=mps.MpsDataProvider(mps.MpsDataPathsProvider(str(assets["take_root"])).get_data_paths())
    graphs=set()
    for index,stamp in enumerate(result["capture_timestamp_ns"]):
        if stamp<=0 or (index and stamp<=result["capture_timestamp_ns"][index-1]):
            continue
        pose=mps_provider.get_closed_loop_pose(int(stamp))
        if pose is None:
            continue
        error=pose.tracking_timestamp.total_seconds()-int(stamp)/1e9
        result["pose_time_error_s"][index]=error
        if abs(error)>max_pose_error_s:
            continue
        wd=pose.transform_world_device.to_matrix()
        result["camera_world"][index]=wd@dc
        result["head_world"][index]=wd@dh
        result["valid"][index]=True
        graphs.add(str(pose.graph_uid))
    with Path(assets["trajectory"]).open(newline="") as handle:
        gravity=np.array([[float(row[f"gravity_{axis}_world"]) for axis in "xyz"] for row in csv.DictReader(handle)])
    with Path(assets["gopro_calibrations"]).open(newline="") as handle:
        rows=list(csv.DictReader(handle))
    result["world"]=verify_hm_world(raw,rows,graphs,gravity)
    if result["valid"].any():
        matrices=result["camera_world"][result["valid"],:3,:3]
        if (not np.isfinite(result["camera_world"][result["valid"]]).all()
                or np.max(abs(np.swapaxes(matrices,-1,-2)@matrices-np.eye(3)))>1e-5
                or np.max(abs(np.linalg.det(matrices)-1))>1e-5):
            raise ValueError("Mapped Aria camera matrices are not finite proper rigid transforms")
    result["provenance"].update(camera_pose_basis="aria_rgb_native",
        device_camera_transform=dc.tolist(),device_cpf_transform=dh.tolist(),
        capture_timestamp_column=column,take_range=[take["timesync_start_idx"],take["timesync_end_idx"]],
        pose_sampling="nearest_MPS_with_individual_time_error_gate",max_pose_error_s=max_pose_error_s,
        camera_position_semantics="selected_Aria_RGB_optical_center",head_position_semantics="aria_CPF_distinct_from_camera",
        video_geometry="source_raw_no_undistortion_no_rotation",
        valid_pose_count=int(result["valid"].sum()),invalid_pose_count=int((~result["valid"]).sum()),
        valid_pose_max_abs_error_s=float(np.max(abs(result["pose_time_error_s"][result["valid"]]))) if result["valid"].any() else None)
    for key,path in sources.items():
        if file_identity(path)!= {k:v for k,v in result["provenance"][key].items() if k!="sha256"}:
            raise ValueError("Aria input changed while exporting: "+key)
    return result


@dataclass
class HMParameterEpisode:
    arrays: dict
    metadata: dict
    alignment: dict

    def as_mapping(self):
        return self.arrays


def export_hm_parameters(root, take, model_path, *, episode_index=0, index_offset=0,
                         revision=HM_REVISION, ground_config=None, camera_loader=load_aria_camera):
    """Return standalone motion columns; the caller owns/copies the DINO sidecar."""
    root=Path(root);uid=take["take_name"]
    source=root/"EgoExo4D-HM"/uid/"points_triangulated_world_results_merged.npz"
    before=file_identity(source,sha256=True)
    raw=read_parameter_source(source);count=len(raw["trans"])
    model_identity=file_identity(model_path)
    model=_verified_model(str(Path(model_path).resolve()),tuple(model_identity.values()))
    assets=_ego_assets(root,take)
    dino_meta={};dino_rows=None;video_id=None
    if assets.get("dino") and assets["dino"].is_file() and assets["dino"].with_suffix(".json").is_file():
        dino_meta=json.loads(assets["dino"].with_suffix(".json").read_text())
        dino_rows=pq.ParquetFile(assets["dino"]).metadata.num_rows
    if assets.get("video") and assets["video"].is_file():
        video_id=file_identity(assets["video"])
    alignment=alignment_receipt(take,count,raw["chunk_ranges"],dino_meta,revision=revision,
                               video_identity=video_id,dino_rows=dino_rows)
    quarantine=list(alignment["quarantine_reasons"])
    try:
        camera=camera_loader(assets,take,count,raw,alignment)
    except (ValueError,OSError,RuntimeError,ImportError) as error:
        camera={"camera_world":np.broadcast_to(np.eye(4),(count,4,4)).copy(),
                "head_world":np.broadcast_to(np.eye(4),(count,4,4)).copy(),"valid":np.zeros(count,bool),
                "capture_timestamp_ns":np.full(count,-1,np.int64),"pose_time_error_s":np.zeros(count),
                "provenance":{},"world":{"verified":False,"quarantine_reasons":[str(error)]}}
    quarantine.extend(camera["world"]["quarantine_reasons"])
    if not camera["valid"].any():
        quarantine.append("No valid mapped ego-camera poses")
    positions,rotations,pmask=model.decode(raw["trans"],raw["root_orient"],raw["pose_body"],raw["betas_per_frame"],raw["valid"])
    aa=np.concatenate([raw["root_orient"][:,None],raw["pose_body"].reshape(count,21,3)],axis=1)
    rmask=np.isfinite(aa).all(-1)&raw["valid"].astype(bool)[:,None]
    for joint,parent in enumerate(PARENTS[1:],1):
        rmask[:,joint]&=rmask[:,parent]
    hvalid={side:raw["valid"].astype(bool)[:,None]&np.isfinite(raw["joints3d"][:,s:e]).all(-1)
            for side,s,e in (("left",25,46),("right",46,67))}
    metadata={"source_data":str(source),"source_identity":before,"source_revision":revision,"source_fps":30,
        "coordinate_verified":camera["world"]["verified"],"world_registration":camera["world"],
        "camera_pose_basis":"aria_rgb_native","aria_provenance":camera["provenance"],
        "rotation_basis_verified":True,"target_joint_basis":"smplh_body22_axes_v1",
        "rotation_semantics":"global_smplh_fk_wxyz","body_position_source":"hm_optimizer_native_smplh_parameters",
        "label_kind":"model_fitted_estimate","source_observed_semantics":"released_HM_optimizer_numeric_availability_not_capture_ground_truth",
        "absolute_rotation_accuracy_established":False,"quality_confidence_semantics":"binary_numeric_availability_proxy; not_calibrated_accuracy",
        "chunk_ranges":raw["chunk_ranges"].tolist(),"beta_policy":"native_per_frame16; changes_at_chunk_boundaries_preserved",
        "parameter_translation_semantics":"SMPL-H trans; root_world=J0(beta)+trans; no_EgoSuite_center_offsets",
        "invalid_nonfinite_parameter_storage":"zero_with_separate_invalid_mask; original_source_identified_by_SHA256",
        "hand_position_source":"released_joints3d_openpose67_left25:46_right46:67",
        "hand_storage_policy":"retain_all_native_finite_XYZ_even_if_chunk_mask_invalid; nonfinite_cells_zero_with_independent_joint_mask",
        "hand_pose_encoding":"left45_right45_PCA_coefficients; use_pca=True; flat_hand_mean=False; borrowed_SMPLX_basis_required",
        "hand_params_used_for_explicit_landmarks":False,"head_semantics":"aria_cpf",
        "frame_alignment":alignment,"quarantine_reasons":quarantine,"canonical_training_eligible":not quarantine,
        "dino_source_identity":file_identity(assets["dino"]) if assets.get("dino") and assets["dino"].exists() else None,
        "video_identity":video_id,"dino_source_sha256":dino_meta.get("parquet_sha256"),
        "native_source_feet_semantics":"released_HM_model_FK_feet; not_additional_capture_ground_truth",
        **model.metadata}
    episode=Episode(uid,"egoexo4d_hm",np.arange(count)/30,
        {"body":positions,"hand_left":raw["joints3d"][:,25:46],"hand_right":raw["joints3d"][:,46:67],
         "camera":camera["camera_world"][:,None,:3,3],"head":camera["head_world"][:,None,:3,3]},
        {"body":rotations,"camera":matrix_to_quat(camera["camera_world"][:,None,:3,:3]),
         "head":matrix_to_quat(camera["head_world"][:,None,:3,:3])},
        {"body":pmask,"hand_left":hvalid["left"],"hand_right":hvalid["right"],"camera":camera["valid"][:,None],"head":camera["valid"][:,None]},
        {"body":rmask,"camera":camera["valid"][:,None],"head":camera["valid"][:,None]},metadata).check()
    from .ground import attach_ground
    from .head_reference import attach_head_reference
    from .resampling import resample
    attach_ground(episode,ground_config)
    attach_head_reference(episode)
    arrays=resample(episode)
    # Preserve even finite values in rejected optimization chunks. An invalid
    # source mask never becomes a training label merely because XYZ is finite.
    for side,s,e in (("left",25,46),("right",46,67)):
        arrays[f"observation.state.hand_{side}_world"]=np.nan_to_num(raw["joints3d"][:,s:e],nan=0,posinf=0,neginf=0).reshape(count,63).copy()
    arrays={key:value for key,value in arrays.items() if not key.startswith("observation.state.body_")
        and not key.startswith("observation.valid.body_") and not key.startswith("observation.interpolated.body_")
        and key!="observation.features.dino"}
    arrays.update({"frame_index":np.arange(count,dtype=np.int64),"episode_index":np.full(count,episode_index,np.int64),
        "index":index_offset+np.arange(count,dtype=np.int64),"task_index":np.zeros(count,np.int64),
        "source.capture_timestamp_ns":camera["capture_timestamp_ns"],
        "source.video_frame_index":np.arange(count,dtype=np.int64) if alignment["verified"] else np.full(count,-1,np.int64),
        "smplh.root_orient":np.nan_to_num(raw["root_orient"],nan=0,posinf=0,neginf=0),
        "smplh.pose_body":np.nan_to_num(raw["pose_body"],nan=0,posinf=0,neginf=0),
        "smplh.trans":np.nan_to_num(raw["trans"],nan=0,posinf=0,neginf=0),
        "smplh.betas":np.nan_to_num(raw["betas_per_frame"],nan=0,posinf=0,neginf=0),
        "smplh.pose_hand_pca":np.nan_to_num(raw["hand_pose"],nan=0,posinf=0,neginf=0),
        "smplh.valid.hand_pose":np.isfinite(raw["hand_pose"].reshape(count,2,45)).all(-1)&raw["valid"].astype(bool)[:,None],
        "source.feet_world":episode.positions["body"][:,[10,11]].reshape(count,6).astype(np.float32),
        "source.feet_valid":pmask[:,[10,11]],"source.feet_interpolated":np.zeros((count,2),bool),
        "quality.observation_residual_m":np.zeros((count,22),np.float32),
        "quality.observation_residual_valid":np.zeros((count,22),bool)})
    for component,mask in (("position",pmask),("rotation",rmask)):
        arrays[f"estimated.valid.body_{component}"]=mask.copy()
        arrays[f"source.observed.body_{component}"]=mask.copy()
        arrays[f"source.interpolated.body_{component}"]=np.zeros((count,22),bool)
        arrays[f"estimated.completed.body_{component}"]=np.zeros((count,22),bool)
        arrays[f"quality.{component}_confidence"]=mask.astype(np.float32)
    if alignment["verified"]:
        names=["frame_index","timestamp","observation.valid.dino","observation.video_timestamp","observation.video_time_error"]
        table=pq.read_table(assets["dino"],columns=names)
        joined={name:table[name].to_numpy() for name in names}
        if (not np.array_equal(joined["frame_index"],np.arange(alignment["video_rows"]))
                or not np.array_equal(joined["timestamp"],np.arange(alignment["video_rows"])/30)):
            raise ValueError("DINO actual frame index/clock changed from documented row contract")
        for name in names[2:]:
            arrays[name]=joined[name][:count].copy()
    from .smplh_parameters import parameter_specs,OPTIONAL_SPECS,validate_parameter_arrays
    columns=set(parameter_specs())|set(OPTIONAL_SPECS)
    arrays={key:value for key,value in arrays.items() if key in columns}
    arrays=validate_parameter_arrays(arrays)
    if file_identity(source)!= {k:v for k,v in before.items() if k!="sha256"}:
        raise ValueError("Native HM source changed while exporting")
    if file_identity(model_path)!=model_identity:
        raise ValueError("SMPL-H asset changed while exporting")
    return HMParameterEpisode(arrays,episode.metadata,alignment)
