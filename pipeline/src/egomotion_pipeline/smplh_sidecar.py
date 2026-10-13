"""Immutable, motion-only SMPL-H estimates joined to an unchanged v5 archive.

Episode directories are the transaction boundary: Parquet and its receipt are
staged together and committed by one rename. Camera, native feet and hand labels
remain source observations. This module does not create contact labels, repair
ground, authenticate source joint axes, or read image/DINO numeric columns.
"""
from collections.abc import Mapping
from contextlib import contextmanager
from functools import lru_cache
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .body_model import BodyModel
from .geometry import axis_angle_to_quat, quat_to_matrix
from .schema import SCHEMA_VERSION, feature_specs_for_schema


SIDECAR_VERSION = "egomotion-smplh-body22-estimates-sidecar-v1"
TARGET_JOINT_BASIS = "smplh_body22_axes_v1"
_GROUND_POLICY = "reuse_archived_ground; window_fallback_uses_original_native_feet_and_masks"
_CONTACT_POLICY = "not_stored; recompute_from_fitted_feet_after_15Hz_selection"
_SPECS = feature_specs_for_schema(SCHEMA_VERSION)
_JOIN = ("timestamp", "frame_index", "episode_index", "index", "observation.source_timestamp",
         "observation.source_frame_left", "observation.source_frame_right", "observation.source_interpolation_weight")
_SOURCE = [*_JOIN, "observation.state.body_world", "observation.state.body_world_rotation"]
for _modality in ("body", "camera", "head_reference", "head_reference_floor"):
    if _modality != "body":
        _SOURCE += [f"observation.state.{_modality}_world", f"observation.state.{_modality}_world_rotation"]
    for _kind in ("valid", "interpolated"):
        _SOURCE += [f"observation.{_kind}.{_modality}_{_part}" for _part in ("position", "rotation")]
for _side in ("left", "right"):
    _SOURCE += [f"observation.state.hand_{_side}_world", f"observation.state.hand_{_side}_floor"]
    for _kind in ("valid", "interpolated"):
        _SOURCE += [f"observation.{_kind}.hand_{_side}_position", f"observation.{_kind}.hand_{_side}_floor_position"]
_SOURCE += ["observation.environment.floor_height", "observation.environment.floor_normal",
            "observation.valid.floor", "observation.valid.floor_verified", "observation.valid.head_reference_basis",
            "observation.state.head_reference_height", "observation.valid.head_reference_height"]
SOURCE_COLUMNS = tuple(_SOURCE)


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "as_dict"):
        return value.as_dict()
    raise TypeError(f"Cannot fingerprint {type(value).__name__}")


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json, allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_bytes(value)).hexdigest()


def file_identity(path):
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    if not path.is_file():
        raise ValueError("Fingerprint path must be a regular file")
    return {"path": str(path), "device": stat.st_dev, "inode": stat.st_ino,
            "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _file_sha(path):
    before = file_identity(path)
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    if file_identity(path) != before:
        raise ValueError("File changed during fingerprinting")
    return h.hexdigest()


def _fit_code_sha():
    return {name: _file_sha(Path(__file__).with_name(name))
            for name in ("smplh_sequence.py", "smplh_retarget.py", "body_model.py")}


def _array_digest(arrays):
    h = hashlib.sha256()
    for key in sorted(arrays):
        x = np.ascontiguousarray(arrays[key])
        h.update(_bytes([key, x.dtype.str, list(x.shape)]))
        h.update(x.tobytes())
    return h.hexdigest()


def _source(arrays):
    missing = set(SOURCE_COLUMNS) - arrays.keys()
    if missing:
        raise ValueError(f"Missing source projection columns: {sorted(missing)}")
    T = len(arrays["timestamp"])
    out = {}
    for key in SOURCE_COLUMNS:
        spec = _SPECS[key]
        shape = (T,) if spec["shape"] == [1] else (T, spec["shape"][0])
        try:
            out[key] = np.asarray(arrays[key], dtype=spec["dtype"]).reshape(shape).copy()
        except (ValueError, TypeError) as error:
            raise ValueError(f"Bad source shape/dtype: {key}") from error
    if not T or not np.array_equal(out["frame_index"], np.arange(T)):
        raise ValueError("Sidecars require complete contiguous 30Hz episodes")
    if not np.array_equal(out["timestamp"], (np.arange(T)/30).astype(np.float32)):
        raise ValueError("Source frame clock is not the exact archived 30Hz grid")
    if T > 1 and not np.allclose(np.diff(out["observation.source_timestamp"]), 1/30, atol=2e-6, rtol=0):
        raise ValueError("Source capture clock is inconsistent with 30Hz archive")
    if not all(np.isfinite(x).all() for x in out.values()):
        raise ValueError("Source projection has nonfinite values")
    for key, values in out.items():
        if key.startswith("observation.interpolated."):
            valid_key = key.replace("observation.interpolated.", "observation.valid.")
            if np.any(values & ~out[valid_key]):
                raise ValueError(f"Source interpolation flag exceeds observation validity: {key}")
    return out


def _read_source(path):
    table = pq.read_table(path, columns=list(SOURCE_COLUMNS))
    arrays = {key: np.asarray(table[key].to_pylist(), dtype=_SPECS[key]["dtype"]) for key in SOURCE_COLUMNS}
    return _source(arrays)


@lru_cache(maxsize=8)
def _model(path, identity_json):
    model = BodyModel(path)
    if (model.metadata["vertex_count"] != 6890 or model.metadata["joint_count"] != 52
            or model.metadata["tree_joint_count"] != 52 or Path(path).suffix != ".npz"):
        raise ValueError("Production sidecar requires a full 6890-vertex 52-joint SMPL-H NPZ model")
    with np.load(path,allow_pickle=False) as asset:
        tree=np.asarray(asset["kintree_table"])
    lookup={int(child):i for i,child in enumerate(tree[1])}
    parents=np.array([-1]+[lookup[int(p)] for p in tree[0,1:]])
    from .schema import PARENTS
    expected=[*PARENTS]
    for wrist,start in ((20,22),(21,37)):
        for finger in range(5):
            first=start+finger*3
            expected.extend([wrist,first,first+1])
    if not np.array_equal(parents,np.asarray(expected)):
        raise ValueError("SMPL-H full52 hand parent hierarchy mismatch")
    if file_identity(path) != json.loads(identity_json):
        raise ValueError("SMPL-H model changed while loading")
    return model


def build_sidecar_contract(dataset_root, source_path, model_path, config, calibration, source_arrays,
                           *, target_joint_basis=TARGET_JOINT_BASIS, source_full_sha256=None):
    """Freeze motion projection + stat identity; optional full SHA is evidence only.

    Full source Parquet bytes are not hashed here, so DINO values are not decoded
    or copied. Model asset SHA is cached by its immutable file identity.
    """
    root, source = Path(dataset_root).resolve(strict=True), Path(source_path).resolve(strict=True)
    if not source.is_relative_to(root / "data"):
        raise ValueError("Source Parquet must belong to the declared archive data root")
    identity = file_identity(model_path)
    model = _model(identity["path"], _bytes(identity).decode())
    arrays = _source(source_arrays)
    contract = {"schema": SIDECAR_VERSION, "source_schema": SCHEMA_VERSION, "archive_fps": 30,
                "dataset_root": str(root), "source_identity": file_identity(source),
                "source_motion_sha256": _array_digest(arrays), "source_projection_columns": list(SOURCE_COLUMNS),
                "model_identity": identity, "model_sha256": model.metadata["model_sha256"],
                "shape_components": model.shape_basis.shape[-1],
                "config": json.loads(_bytes(config)), "config_sha256": _digest(config),
                "calibration": json.loads(_bytes(calibration)), "calibration_sha256": _digest(calibration),
                "target_joint_basis": target_joint_basis, "label_kind": "model_fitted_estimate",
                "native_axis_certified": False, "target_model_axes_defined": True,
                "sidecar_implementation_sha256": _file_sha(__file__),
                "fitter_implementation_sha256": _fit_code_sha(),
                "source_full_sha256": source_full_sha256,
                "ground_policy": _GROUND_POLICY,
                "contact_policy": _CONTACT_POLICY}
    _check_contract(contract, arrays)
    return contract


def _check_contract(contract, arrays):
    if (contract.get("schema") != SIDECAR_VERSION or contract.get("source_schema") != SCHEMA_VERSION
            or contract.get("archive_fps") != 30 or contract.get("label_kind") != "model_fitted_estimate"
            or contract.get("native_axis_certified") is not False or contract.get("target_model_axes_defined") is not True
            or contract.get("target_joint_basis") != TARGET_JOINT_BASIS
            or contract.get("ground_policy") != _GROUND_POLICY or contract.get("contact_policy") != _CONTACT_POLICY):
        raise ValueError("Sidecar declaration must describe 30Hz SMPL-H estimates, not certified native axes")
    full_sha = contract.get("source_full_sha256")
    if full_sha is not None and (not isinstance(full_sha, str) or len(full_sha) != 64
                                 or any(c not in "0123456789abcdef" for c in full_sha)):
        raise ValueError("Optional source full SHA must be a SHA256 hex digest")
    if contract["sidecar_implementation_sha256"] != _file_sha(__file__):
        raise ValueError("Sidecar implementation fingerprint changed")
    if contract.get("fitter_implementation_sha256") != _fit_code_sha():
        raise ValueError("Fitter/decoder implementation fingerprint changed")
    if contract["config_sha256"] != _digest(contract["config"]) or contract["calibration_sha256"] != _digest(contract["calibration"]):
        raise ValueError("Configuration or calibration fingerprint mismatch")
    for key in ("source_identity", "model_identity"):
        if file_identity(contract[key]["path"]) != contract[key]:
            raise ValueError(f"{key}: file identity changed")
    if contract["source_motion_sha256"] != _array_digest(arrays):
        raise ValueError("Source projected motion changed")
    if contract["source_projection_columns"] != list(SOURCE_COLUMNS):
        raise ValueError("Source projection contract changed")
    if not Path(contract["source_identity"]["path"]).is_relative_to(Path(contract["dataset_root"]) / "data"):
        raise ValueError("Source does not belong to archive")
    actual = _read_source(contract["source_identity"]["path"])
    if _array_digest(actual) != contract["source_motion_sha256"]:
        raise ValueError("Caller projection differs from source Parquet")
    model = _model(contract["model_identity"]["path"], _bytes(contract["model_identity"]).decode())
    if model.metadata["model_sha256"] != contract["model_sha256"] or model.shape_basis.shape[-1] != contract["shape_components"]:
        raise ValueError("Wrong model or shape basis")
    return model


def _fit_mapping(result):
    if hasattr(result, "as_mapping"):
        return result.as_mapping()
    if isinstance(result, Mapping):
        return result
    raise ValueError("fit_result must expose the SequenceResult.as_mapping contract")


def _encode(source, result, contract, model):
    T = len(source["timestamp"])
    fit = _fit_mapping(result)
    def array(key, shape, dtype=np.float64):
        x = np.asarray(fit[key])
        if x.shape != shape or (dtype == bool and x.dtype != np.bool_):
            raise ValueError(f"Fitted {key}: wrong shape/dtype")
        x = x.astype(dtype, copy=True)
        if not np.isfinite(x).all():
            raise ValueError(f"Fitted {key}: nonfinite values")
        return x
    positions = array("body_world", (T, 22, 3))
    global_R = array("global_rotations", (T, 22, 3, 3))
    aa = array("local_axis_angle", (T, 22, 3))
    root = array("root_world", (T, 3))
    trans = array("smpl_trans", (T, 3))
    beta = np.asarray(fit["betas"], float)
    if beta.shape != (contract["shape_components"],) or not np.isfinite(beta).all():
        raise ValueError("Shape must be a finite fixed model-matched vector for the group")
    center=contract["calibration"].get("center")
    if not isinstance(center,Mapping):
        raise ValueError("Production sidecar requires a frozen GroupCalibration.center")
    offsets=np.asarray(center["offsets"],float)
    anchors=np.asarray(center["anchors"])
    available=np.asarray(center["offset_valid"])
    if (offsets.shape!=(22,3) or not np.isfinite(offsets).all() or anchors.shape!=(22,) or anchors.dtype.kind not in "iu"
            or np.any((anchors<0)|(anchors>=22)) or available.shape!=(22,) or available.dtype!=np.bool_
            or available[0] or np.any(offsets[~available]!=0)
            or center["model_sha256"]!=contract["model_sha256"]
            or not np.array_equal(np.asarray(center["betas"],float),beta)):
        raise ValueError("Fixed center calibration/model/betas mismatch")
    pv, rv = (array(f"estimated_{part}_valid", (T, 22), bool) for part in ("position", "rotation"))
    source_pv, source_rv = (source[f"observation.valid.body_{part}"] for part in ("position", "rotation"))
    for part, original in (("position", source_pv), ("rotation", source_rv)):
        if f"source_{part}_valid" in fit and not np.array_equal(fit[f"source_{part}_valid"], original):
            raise ValueError("Fitter changed original observation masks")
    confidence = {}
    for part in ("position", "rotation"):
        key = f"{part}_confidence" if f"{part}_confidence" in fit else "fit_quality"
        confidence[part] = array(key, (T, 22))
        if np.any((confidence[part] < 0) | (confidence[part] > 1)):
            raise ValueError("Confidence must be a finite [0,1] proxy")
    inferred = np.asarray(fit.get("estimate_interpolated", pv & ~source_pv))
    if inferred.dtype != np.bool_ or inferred.shape != (T,22) or not np.array_equal(inferred, pv & ~source_pv):
        raise ValueError("Estimated completion flag must identify exactly available unobserved positions")
    # Independent model decoder checks centers, root pivot and rotation chains.
    expected_p, expected_q, _ = model.decode(trans, aa[:,0], aa[:,1:].reshape(T,63),
                                            np.repeat(beta[None],T,axis=0), np.ones(T,bool))
    expected_R = quat_to_matrix(expected_q)
    if ((pv.any() and np.max(abs(expected_p[pv]-positions[pv])) > 2e-5)
            or (rv.any() and np.max(abs(expected_R[rv]-global_R[rv])) > 2e-5)
            or np.max(abs(root-expected_p[:,0])) > 2e-5):
        raise ValueError("Fitted labels/parameters disagree with model FK or root pivot")
    if not np.allclose(np.swapaxes(global_R,-1,-2) @ global_R, np.eye(3), atol=2e-5) or not np.allclose(np.linalg.det(global_R),1,atol=2e-5):
        raise ValueError("Fitted global rotations are not proper SO(3)")
    fq = source["observation.state.head_reference_floor_world_rotation"].reshape(T,4)
    floor_R = quat_to_matrix(fq)
    origin = source["observation.state.head_reference_floor_world"]
    floor_pv = source["observation.valid.head_reference_floor_position"].reshape(T)
    floor_rv = source["observation.valid.head_reference_floor_rotation"].reshape(T)
    basis = source["observation.valid.head_reference_basis"].reshape(T)
    if np.any((floor_pv | floor_rv) & ~basis):
        raise ValueError("Valid source floor reference has unverified axes")
    local_p = np.einsum("tji,tkj->tki",floor_R,positions-origin[:,None])
    local_R = np.swapaxes(floor_R,-1,-2)[:,None] @ global_R
    out = {k: source[k].copy() for k in _JOIN}
    for key, value in source.items():
        if key not in _JOIN:
            out["source."+key] = value.copy()
    for frame, R, p, p_mask, r_mask in (("world",global_R,positions,pv,rv),
            ("floor",local_R,local_p,pv & floor_pv[:,None] & floor_rv[:,None],rv & floor_rv[:,None])):
        r6 = np.where(r_mask[...,None],R[...,:2,:].reshape(T,22,6),0)
        xyz = np.where(p_mask[...,None],p,0)
        out[f"fitted.body_{frame}_rot6d_xyz"] = np.concatenate([r6,xyz],-1).reshape(T,198).astype(np.float32)
        out[f"estimated.valid.body_{frame}_position"] = p_mask
        out[f"estimated.valid.body_{frame}_rotation"] = r_mask
    out.update({"fitted.body_world_rotation": np.where(rv[...,None],expected_q,np.array([1,0,0,0])).reshape(T,88).astype(np.float32),
        "fitted.local_axis_angle": aa.reshape(T,66).astype(np.float32),
        "fitted.local_rotation": quat_to_matrix(axis_angle_to_quat(aa)).reshape(T,198).astype(np.float32),
        "fitted.root_world":root.astype(np.float32),"fitted.smpl_trans":trans.astype(np.float32),
        "fitted.betas":np.repeat(beta[None],T,axis=0).astype(np.float32),
        "source.observed.body_position":source_pv.copy(),"source.observed.body_rotation":source_rv.copy(),
        "estimated.inferred.body_position": inferred,
        "source.feet_world":source["observation.state.body_world"].reshape(T,22,3)[:,[10,11]].reshape(T,6).copy(),
        "source.feet_valid":source_pv[:,[10,11]].copy(),
        "quality.position_confidence":np.where(pv,confidence["position"],0).astype(np.float32),
        "quality.rotation_confidence":np.where(rv,confidence["rotation"],0).astype(np.float32)})
    predicted_source=positions+(global_R[:,anchors]@offsets[...,None])[...,0]
    reconstructed = np.asarray(fit.get("reconstructed_observations",predicted_source),float)
    if reconstructed.shape != positions.shape or not np.isfinite(reconstructed).all():
        raise ValueError("Bad reconstructed source landmark positions")
    if pv.any() and np.max(abs(reconstructed[pv]-predicted_source[pv]))>2e-5:
        raise ValueError("Source landmark reconstruction disagrees with frozen center offsets")
    out["fitted.reconstructed_source_landmarks_world"] = reconstructed.reshape(T,66).astype(np.float32)
    residual = np.linalg.norm(reconstructed-source["observation.state.body_world"].reshape(T,22,3),axis=-1)
    out["quality.observation_residual_m"] = np.where(source_pv & pv,residual,0).astype(np.float32)
    out["quality.observation_residual_valid"] = source_pv & pv
    if not all(np.isfinite(x).all() for x in out.values()):
        raise ValueError("Encoded fitted sidecar overflowed storage dtype")
    return out


def _write_table(path, arrays, fingerprint):
    columns = {}
    for key,x in arrays.items():
        if x.ndim == 1:
            columns[key] = pa.array(x)
        else:
            columns[key] = pa.FixedSizeListArray.from_arrays(pa.array(x.reshape(-1)),x.shape[1])
    table = pa.table(columns).replace_schema_metadata({b"sidecar_version":SIDECAR_VERSION.encode(),b"contract_sha256":fingerprint.encode()})
    pq.write_table(table,path,compression="zstd",row_group_size=1024)
    with Path(path).open("rb") as f:
        os.fsync(f.fileno())


def _read_table(path):
    return _table_arrays(pq.read_table(path))


def _table_arrays(table):
    arrays = {}
    for key in table.column_names:
        typ = table[key].type
        dtype = np.bool_ if pa.types.is_boolean(typ) or (pa.types.is_fixed_size_list(typ) and pa.types.is_boolean(typ.value_type)) else None
        arrays[key] = np.asarray(table[key].to_pylist(),dtype=dtype)
        value_type = typ.value_type if pa.types.is_fixed_size_list(typ) else typ
        arrays[key] = arrays[key].astype(value_type.to_pandas_dtype())
    return arrays


@contextmanager
def _locked(root):
    root.mkdir(parents=True,exist_ok=True)
    with (root/".writer.lock").open("a") as f:
        # Independent worker processes serialize their episode commits here.
        # Model fitting and encoding happen before acquiring the root lock.
        fcntl.flock(f,fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f,fcntl.LOCK_UN)


def _location(output_dir,episode,contract):
    root = Path(output_dir).resolve()
    source_root = Path(contract["dataset_root"]).resolve()
    if root == source_root or source_root.is_relative_to(root) or root.is_relative_to(source_root/"data") or root.is_relative_to(source_root/"meta"):
        raise ValueError("Sidecar output must not overwrite archive or its data/metadata")
    idx = episode["episode_index"]
    if isinstance(idx,bool) or not isinstance(idx,(int,np.integer)) or idx<0:
        raise ValueError("Invalid episode index")
    return root, root/"episodes"/f"{idx:08d}"


def _declaration(episode,contract):
    cal=contract["calibration"]
    if cal.get("source_id")!=episode.get("source"):
        raise ValueError("Calibration source differs from episode")
    if "operator_id" in episode.get("metadata",{}):
        operator = str(episode["metadata"]["operator_id"])
        recording_group = operator + ":record:" + episode["source_key"]
        if str(cal.get("group_id")) not in (operator, recording_group):
            raise ValueError("Calibration group differs from original operator/recording")
        if "operator_id" in cal and str(cal["operator_id"]) != operator:
            raise ValueError("Calibration operator identity mismatch")
    if "source_key" in cal and cal["source_key"] != episode["source_key"]:
        raise ValueError("Calibration recording identity mismatch")
    return {"contract":contract,"source_record_sha256":_digest(episode),
            "episode_index":int(episode["episode_index"]),"source_key":episode["source_key"],
            "frames":int(episode["length"]),"source_metadata":episode.get("metadata",{}),
            "fitted_axes_defined_by_model":True,"source_axes_certified":False,
            "confidence_semantics":"numerical_fit_proxy; not_absolute_orientation_accuracy",
            "hand_policy":"original_world_and_floor_21_labels_and_masks_copied_exactly",
            "window_floor_policy":"use_source.feet_world/source.feet_valid for ground fallback; fore prefix only"}


def _manifest(contract):
    common = {k: contract[k] for k in ("schema", "source_schema", "archive_fps", "dataset_root", "model_identity",
              "model_sha256", "shape_components", "config", "config_sha256", "target_joint_basis", "label_kind",
              "native_axis_certified", "target_model_axes_defined", "sidecar_implementation_sha256", "ground_policy", "contact_policy")}
    common["fitter_implementation_sha256"] = contract["fitter_implementation_sha256"]
    return {"format": SIDECAR_VERSION, "contract": common, "join": ["episode_index", "frame_index"],
            "data_path": "episodes/{episode_index:08d}/motion.parquet", "receipt_path": "episodes/{episode_index:08d}/receipt.json",
            "original_archive_root": contract["dataset_root"], "source_archive_modified": False,
            "images_dino_videos": "join original v5; never copied", "native_ground_fallback": "source.feet_world/source.feet_valid"}


def _array_specs(arrays):
    return {key: {"dtype": str(x.dtype), "shape": list(x.shape[1:]) or [1]} for key, x in arrays.items()}


def _receipt_header(root, folder, receipt):
    declaration = receipt.get("declaration", {})
    contract = declaration.get("contract", {})
    if (receipt.get("state") != "complete" or receipt.get("contract_sha256") != _digest(declaration)
            or receipt.get("execution_sha256") != _digest(receipt.get("execution"))
            or json.loads((root/"manifest.json").read_text()) != _manifest(contract)
            or receipt.get("paths") != {"parquet": str(folder.relative_to(root)/"motion.parquet"),
                                        "receipt": str(folder.relative_to(root)/"receipt.json")}):
        raise ValueError("Sidecar manifest/receipt contract mismatch")
    metadata = pq.ParquetFile(folder/"motion.parquet").schema_arrow.metadata or {}
    if (metadata.get(b"sidecar_version") != SIDECAR_VERSION.encode()
            or metadata.get(b"contract_sha256") != receipt["contract_sha256"].encode()):
        raise ValueError("Sidecar Parquet schema provenance mismatch")
    schema = pq.ParquetFile(folder/"motion.parquet").schema_arrow
    actual_specs = {}
    for field in schema:
        typ = field.type
        value = typ.value_type if pa.types.is_fixed_size_list(typ) else typ
        actual_specs[field.name] = {"dtype": str(np.dtype(value.to_pandas_dtype())),
                                    "shape": [typ.list_size] if pa.types.is_fixed_size_list(typ) else [1]}
    if receipt.get("specs") != actual_specs:
        raise ValueError("Sidecar declared shapes/dtypes changed")


def validate_episode_sidecar(output_dir,episode,source_arrays,contract):
    """Validate immutable receipt, original source projection, FK and floor math."""
    source = _source(source_arrays)
    model = _check_contract(contract,source)
    root, folder = _location(output_dir,episode,contract)
    declaration = _declaration(episode,contract)
    receipt = json.loads((folder/"receipt.json").read_text())
    if receipt.get("state")!="complete" or receipt.get("declaration") != declaration or receipt.get("contract_sha256") != _digest(declaration):
        raise ValueError("Sidecar provenance/join contract mismatch")
    path = folder/"motion.parquet"
    if receipt.get("parquet_sha256") != _file_sha(path):
        raise ValueError("Sidecar Parquet bytes changed")
    _receipt_header(root, folder, receipt)
    arrays = _read_table(path)
    if receipt.get("numeric_sha256") != _array_digest(arrays):
        raise ValueError("Sidecar numeric payload changed")
    if receipt.get("specs") != _array_specs(arrays):
        raise ValueError("Sidecar declared shapes/dtypes changed")
    T = len(source["timestamp"])
    if int(episode["length"])!=T or not np.all(source["episode_index"]==episode["episode_index"]):
        raise ValueError("Source episode index/length mismatch")
    for key,value in source.items():
        stored = key if key in _JOIN else "source."+key
        if stored not in arrays or arrays[stored].dtype != value.dtype or not np.array_equal(arrays[stored],value):
            raise ValueError(f"Source value changed in sidecar: {key}")
    fit = {"body_world":arrays["fitted.body_world_rot6d_xyz"].reshape(T,22,9)[...,6:],
        "global_rotations":quat_to_matrix(arrays["fitted.body_world_rotation"].reshape(T,22,4)),
        "local_axis_angle":arrays["fitted.local_axis_angle"].reshape(T,22,3),
        "root_world":arrays["fitted.root_world"],"smpl_trans":arrays["fitted.smpl_trans"],
        "betas":np.asarray(contract["calibration"]["center"]["betas"],float),
        "estimated_position_valid":arrays["estimated.valid.body_world_position"],
        "estimated_rotation_valid":arrays["estimated.valid.body_world_rotation"],
        "position_confidence":arrays["quality.position_confidence"],
        "rotation_confidence":arrays["quality.rotation_confidence"],
        "estimate_interpolated":arrays["estimated.inferred.body_position"],
        "reconstructed_observations":arrays["fitted.reconstructed_source_landmarks_world"].reshape(T,22,3)}
    # Reconstruct unavailable rotations from finite model parameters, avoiding
    # treating the identity sentinel as a fitted FK orientation.
    _,q,_ = model.decode(fit["smpl_trans"],fit["local_axis_angle"][:,0],fit["local_axis_angle"][:,1:].reshape(T,63),
                          arrays["fitted.betas"],np.ones(T,bool))
    fit["global_rotations"] = quat_to_matrix(q)
    expected = _encode(source,fit,contract,model)
    if set(arrays)!=set(expected):
        raise ValueError("Unexpected/missing sidecar columns")
    for key in expected:
        if arrays[key].shape != expected[key].shape or arrays[key].dtype != expected[key].dtype:
            raise ValueError(f"Incorrect sidecar shape/dtype: {key}")
        if arrays[key].dtype.kind in "biu":
            valid=np.array_equal(arrays[key],expected[key])
        else:
            valid=np.allclose(arrays[key],expected[key],atol=2e-5,rtol=0)
        if not valid:
            raise ValueError(f"Incorrect sidecar encoding: {key}")
    mask=arrays["quality.observation_residual_valid"]
    if np.any(mask & ~(arrays["source.observed.body_position"] & arrays["estimated.valid.body_world_position"])) or np.any(arrays["quality.observation_residual_m"]<0) or np.any(arrays["quality.observation_residual_m"][~mask]!=0):
        raise ValueError("Invalid observation quality mask/value")
    _check_contract(contract,source)
    return {"passed":True,"frames":T,"episode_index":int(episode["episode_index"]),
            "numeric_sha256":receipt["numeric_sha256"],"source_inputs_unchanged":True,
            "contact_stored":False,"dino_copied":False,"native_axes_certified":False}


def write_episode_sidecar(output_dir,episode,source_arrays,fit_result,contract):
    """Commit one complete episode; repeat only if all inputs and labels agree."""
    source=_source(source_arrays)
    model=_check_contract(contract,source)
    T=len(source["timestamp"])
    if episode.get("source")!="egosuite" or episode["length"]!=T or not np.all(source["episode_index"]==episode["episode_index"]):
        raise ValueError("Only matching complete EgoSuite episodes can be retargeted")
    arrays=_encode(source,fit_result,contract,model)
    fit=_fit_mapping(fit_result)
    execution=json.loads(_bytes({"metadata":fit.get("metadata",{}),"quality":fit.get("quality",{})}))
    metadata=execution["metadata"]
    if metadata:
        if (metadata.get("label_kind") != "model_fitted_estimate" or metadata.get("target_joint_basis") != TARGET_JOINT_BASIS
                or metadata.get("native_axis_certified") is not False
                or metadata.get("model",{}).get("model_sha256") != contract["model_sha256"]
                or metadata.get("config") != contract["config"]
                or metadata.get("calibration_input_sha256") != contract["calibration"]["center"]["input_sha256"]
                or metadata.get("source_id") != contract["calibration"]["source_id"]
                or metadata.get("group_id") != contract["calibration"]["group_id"]):
            raise ValueError("Sequence execution metadata disagrees with frozen model/config/calibration")
    root,folder=_location(output_dir,episode,contract)
    declaration=_declaration(episode,contract)
    fingerprint=_digest(declaration)
    manifest = _manifest(contract)
    with _locked(root):
        manifest_path=root/"manifest.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text())!=manifest:
                raise ValueError("Output contains an incompatible sidecar configuration/model")
        else:
            if any(p.name != ".writer.lock" for p in root.iterdir()):
                raise ValueError("Refusing to initialize nonempty unrecognized sidecar output")
            tmp=root/(".manifest-"+uuid.uuid4().hex)
            tmp.write_bytes(_bytes(manifest))
            with tmp.open("rb") as f:os.fsync(f.fileno())
            os.replace(tmp,manifest_path)
        if folder.exists():
            report=validate_episode_sidecar(root,episode,source,contract)
            receipt=json.loads((folder/"receipt.json").read_text())
            if receipt["numeric_sha256"]!=_array_digest(arrays):
                raise ValueError("Resume refused: fitted estimates differ")
            if receipt["execution"]!=execution:
                raise ValueError("Resume refused: fitter execution metadata differs")
            return {**report,"written":False,"path":str(folder/"motion.parquet")}
        stage=root/".staging"/uuid.uuid4().hex
        stage.mkdir(parents=True)
        try:
            path=stage/"motion.parquet"
            _write_table(path,arrays,fingerprint)
            if _array_digest(_read_table(path))!=_array_digest(arrays):
                raise ValueError("Parquet staging numeric roundtrip failed")
            receipt={"declaration":declaration,"contract_sha256":fingerprint,"parquet_sha256":_file_sha(path),
                     "numeric_sha256":_array_digest(arrays),"state":"complete",
                     "execution":execution,"execution_sha256":_digest(execution),
                     "paths":{"parquet":str(folder.relative_to(root)/"motion.parquet"),
                              "receipt":str(folder.relative_to(root)/"receipt.json")},
                     "specs":_array_specs(arrays)}
            (stage/"receipt.json").write_bytes(_bytes(receipt))
            with (stage/"receipt.json").open("rb") as f:os.fsync(f.fileno())
            fd=os.open(stage,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(fd)
            finally:os.close(fd)
            _check_contract(contract,source)
            folder.parent.mkdir(parents=True,exist_ok=True)
            os.rename(stage,folder)
            fd=os.open(folder.parent,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(fd)
            finally:os.close(fd)
        finally:
            if stage.exists():shutil.rmtree(stage)
    report=validate_episode_sidecar(root,episode,source,contract)
    return {**report,"written":True,"path":str(folder/"motion.parquet")}


def read_sidecar_window(output_dir,episode_index,*,start_frame=0,num_frames=120,fps=15,columns=None):
    """Read selected rows with exact frame joins; ground fallback remains source-only.

    Returns stored motion labels, not training conditions or contacts. Consumers
    must resolve a missing ground using original native feet (fore prefix only)
    and rebuild fitted floor caches after any such resolution/canonicalization.
    """
    if (any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) for x in (episode_index, start_frame, num_frames, fps))
            or episode_index < 0 or fps not in (15,30) or start_frame<0 or num_frames<1):
        raise ValueError("Invalid sidecar window")
    root=Path(output_dir).resolve()
    folder=root/"episodes"/f"{episode_index:08d}"
    receipt=json.loads((folder/"receipt.json").read_text())
    path=folder/"motion.parquet"
    if receipt["parquet_sha256"]!=_file_sha(path):
        raise ValueError("Window refuses a corrupted sidecar")
    _receipt_header(root, folder, receipt)
    declaration = receipt["declaration"]
    if declaration["episode_index"] != episode_index:
        raise ValueError("Window episode join mismatch")
    for key in ("source_identity", "model_identity"):
        identity = declaration["contract"][key]
        if file_identity(identity["path"]) != identity:
            raise ValueError("Window source/model identity changed")
    if columns is not None and (not isinstance(columns, (tuple, list)) or not columns
                               or len(set(columns)) != len(columns) or not set(columns).issubset(receipt["specs"])):
        raise ValueError("Window requires declared, unique projection columns")
    selected=start_frame+np.arange(num_frames)*(30//fps)
    if selected[-1]>=receipt["declaration"]["frames"]:
        raise ValueError("Window exceeds episode")
    file=pq.ParquetFile(path)
    groups=[];first=0
    for i in range(file.metadata.num_row_groups):
        last=first+file.metadata.row_group(i).num_rows
        if np.any((selected>=first)&(selected<last)):
            groups.append((i,first))
        first=last
    out={}
    for i,first in groups:
        table=file.read_row_group(i,columns=columns)
        ids=selected[(selected>=first)&(selected<first+len(table))]-first
        for key,values in _table_arrays(table).items():
            x=values[ids]
            out.setdefault(key,[]).append(x)
    return {key:np.concatenate(parts) for key,parts in out.items()}
