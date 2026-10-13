"""Standalone SMPL-H body parameters, with owned hands, camera and vision.

The archive owns every file read by this module. Body FK is a transient window
representation, not a stored dense22 dependency. Parameter finiteness never
creates an observed label or upgrades an estimated rotation to ground truth.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .body_model import BodyModel
from .geometry import axis_angle_to_quat, matrix_to_quat, quat_multiply, quat_to_matrix
from .ground import resolve_window_ground
from .head_reference import derive_ground_features
from .schema import PARENTS, POSITION_SHAPES, ROTATION_SHAPES, SCHEMA_VERSION, feature_specs_for_schema
from .unified_dense22 import encode_unified_dense22, unified_metadata
from .windows import TrainingWindow, head_trajectory


PARAMETER_SCHEMA_VERSION = "egomotion-smplh16-bodyparams-hands21-camera-floor-dino5-v1"
MODEL_SHA256 = "be42bff1ff0589cf0467893bcfbbeb3f9b533ba151031c8cdfb64ca1dcb34947"
DINO_COLUMN = "observation.features.dino"
OPTIONAL_SPECS = {
    "smplh.pose_hand_pca": {"dtype": "float32", "shape": [90], "names": None},
    "smplh.valid.hand_pose": {"dtype": "bool", "shape": [2], "names": ["left", "right"]},
    "source.capture_timestamp_ns": {"dtype": "int64", "shape": [1], "names": None},
    "source.video_frame_index": {"dtype": "int64", "shape": [1], "names": None},
}


def _independent_columns():
    return {key: deepcopy(spec) for key, spec in feature_specs_for_schema(SCHEMA_VERSION).items()
            if ".body_" not in key and ".hand_left_floor" not in key
            and ".hand_right_floor" not in key and ".head_reference_floor" not in key
            and ".head_reference_height" not in key}


def parameter_specs(*, include_dino=False):
    """Exact mandatory physical columns; optional hand pose is never inferred."""
    specs = _independent_columns()
    # HM DINO stores the original PTS error as float64. Promote EgoSuite's
    # float32 values exactly, rather than truncating HM's source precision.
    specs["observation.video_time_error"]["dtype"] = "float64"
    if not include_dino:
        specs.pop(DINO_COLUMN)
    for name, width in (("smplh.root_orient", 3), ("smplh.pose_body", 63),
                        ("smplh.trans", 3), ("smplh.betas", 16), ("source.feet_world", 6)):
        specs[name] = {"dtype": "float32", "shape": [width], "names": None}
    for name in ("estimated.valid.body_position", "estimated.valid.body_rotation",
                 "source.observed.body_position", "source.observed.body_rotation",
                 "source.interpolated.body_position", "source.interpolated.body_rotation",
                 "estimated.completed.body_position", "estimated.completed.body_rotation",
                 "quality.observation_residual_valid"):
        specs[name] = {"dtype": "bool", "shape": [22], "names": None}
    for name in ("source.feet_valid", "source.feet_interpolated"):
        specs[name] = {"dtype": "bool", "shape": [2], "names": ["left", "right"]}
    for name in ("quality.position_confidence", "quality.rotation_confidence",
                 "quality.observation_residual_m"):
        specs[name] = {"dtype": "float32", "shape": [22], "names": None}
    return specs


def _cast_feature(value, name, spec, count):
    raw = np.asarray(value)
    width = spec["shape"][0]
    shape = (count,) if width == 1 else (count, width)
    if raw.size != count * width:
        raise ValueError(f"{name}: expected shape {shape}")
    if spec["dtype"] == "bool" and not np.isin(raw, [False, True]).all():
        raise ValueError(f"{name}: mask must contain booleans")
    if spec["dtype"] == "int64" and (not np.isfinite(raw).all() or np.any(raw != np.floor(raw))):
        raise ValueError(f"{name}: indices must be finite integers")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(value, dtype=spec["dtype"]).reshape(shape).copy()
    if not np.isfinite(result).all():
        raise ValueError(f"{name}: nonfinite/overflowing stored values")
    return result


def validate_parameter_arrays(arrays, *, require_contiguous=True):
    """Validate numeric availability separately from source observation masks."""
    if "timestamp" not in arrays or not len(arrays["timestamp"]):
        raise ValueError("Parameters require a nonempty episode")
    count = len(arrays["timestamp"])
    specs = parameter_specs(include_dino=DINO_COLUMN in arrays)
    specs.update({key: spec for key, spec in OPTIONAL_SPECS.items() if key in arrays})
    missing, unknown = set(specs) - set(arrays), set(arrays) - set(specs)
    if missing or unknown:
        raise ValueError(f"Parameter columns mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}")
    out = {key: _cast_feature(arrays[key], key, spec, count) for key, spec in specs.items()}
    frames = out["frame_index"]
    if frames[0] < 0 or np.any(np.diff(frames) <= 0):
        raise ValueError("Frame indices must be nonnegative and strictly increasing")
    if require_contiguous and not np.array_equal(frames, np.arange(count)):
        raise ValueError("Parameter episodes require contiguous zero-based 30Hz frames")
    if not np.array_equal(out["timestamp"], (frames / 30).astype(np.float32)):
        raise ValueError("Parameter timestamps must equal the exact 30Hz frame grid")
    if not np.all(out["episode_index"] == out["episode_index"][0]):
        raise ValueError("A parameter episode cannot mix episode identities")
    for component in ("position", "rotation"):
        valid = out[f"estimated.valid.body_{component}"]
        source = out[f"source.observed.body_{component}"]
        completion = out[f"estimated.completed.body_{component}"]
        if not np.array_equal(completion, valid & ~source):
            raise ValueError(f"{component}: completion must retain its source observation boundary")
        if np.any(out[f"source.interpolated.body_{component}"] & ~source):
            raise ValueError("Source interpolation exceeds source availability")
        confidence = out[f"quality.{component}_confidence"]
        if np.any((confidence < 0) | (confidence > 1)) or np.any(confidence[~valid] != 0):
            raise ValueError(f"{component}: confidence must be in [0,1] and zero for unavailable estimates")
    if np.any(out["quality.observation_residual_m"] < 0):
        raise ValueError("Observation residuals cannot be negative")
    if np.any(out["quality.observation_residual_valid"] & ~(
            out["source.observed.body_position"] & out["estimated.valid.body_position"])):
        raise ValueError("Residual validity exceeds observed and estimated availability")
    if np.any(out["source.feet_interpolated"] & ~out["source.feet_valid"]):
        raise ValueError("Native foot interpolation exceeds its original validity")
    if (not np.array_equal(out["source.feet_valid"], out["source.observed.body_position"][:, [10, 11]])
            or not np.array_equal(out["source.feet_interpolated"], out["source.interpolated.body_position"][:, [10, 11]])):
        raise ValueError("Native foot masks differ from retained source body availability")
    for key in out:
        if key.startswith("observation.interpolated."):
            valid_key = key.replace("observation.interpolated.", "observation.valid.")
            if valid_key in out and np.any(out[key] & ~out[valid_key]):
                raise ValueError(f"Independent modality interpolation exceeds validity: {key}")
    if ("smplh.pose_hand_pca" in out) != ("smplh.valid.hand_pose" in out):
        raise ValueError("Optional hand pose requires a separate availability mask")
    return out


def parameters_from_fitted(source_arrays, fit_mapping):
    """Project an existing v5 source plus explicit fitted parameters, not XYZ caches."""
    if hasattr(fit_mapping, "as_mapping"):
        fit_mapping = fit_mapping.as_mapping()
    count = len(source_arrays["timestamp"])
    def fit(name, *aliases):
        for key in (name, *aliases):
            if key in fit_mapping:
                return np.asarray(fit_mapping[key])
        raise ValueError(f"Missing explicit fitted field: {name}")
    aa = fit("local_axis_angle", "fitted.local_axis_angle").reshape(count, 22, 3)
    beta = fit("betas", "fitted.betas")
    if beta.shape == (16,):
        beta = np.broadcast_to(beta, (count, 16))
    out = {key: np.asarray(source_arrays[key]).copy() for key in _independent_columns()
           if key != DINO_COLUMN or key in source_arrays}
    out.update({"smplh.root_orient": aa[:, 0], "smplh.pose_body": aa[:, 1:].reshape(count, 63),
                "smplh.trans": fit("smpl_trans", "fitted.smpl_trans"), "smplh.betas": beta})
    for component in ("position", "rotation"):
        source = np.asarray(source_arrays[f"observation.valid.body_{component}"], bool).reshape(count, 22)
        available = fit(f"estimated_{component}_valid", f"estimated.valid.body_world_{component}").reshape(count, 22)
        out[f"source.observed.body_{component}"] = source
        out[f"source.interpolated.body_{component}"] = source_arrays[f"observation.interpolated.body_{component}"]
        out[f"estimated.valid.body_{component}"] = available
        out[f"estimated.completed.body_{component}"] = available & ~source
        confidence = fit(f"{component}_confidence", f"quality.{component}_confidence").reshape(count, 22)
        out[f"quality.{component}_confidence"] = np.where(available, confidence, 0)
    out["source.feet_world"] = np.asarray(source_arrays["observation.state.body_world"]).reshape(count, 22, 3)[:, [10, 11]].reshape(count, 6)
    out["source.feet_valid"] = out["source.observed.body_position"][:, [10, 11]]
    out["source.feet_interpolated"] = out["source.interpolated.body_position"][:, [10, 11]]
    if "quality.observation_residual_m" in fit_mapping:
        out["quality.observation_residual_m"] = fit_mapping["quality.observation_residual_m"]
        out["quality.observation_residual_valid"] = fit("quality.observation_residual_valid")
    elif "reconstructed_observations" in fit_mapping:
        predicted = np.asarray(fit_mapping["reconstructed_observations"]).reshape(count, 22, 3)
        residual = np.linalg.norm(predicted - np.asarray(source_arrays["observation.state.body_world"]).reshape(count, 22, 3), axis=-1)
        valid = out["source.observed.body_position"] & out["estimated.valid.body_position"]
        out["quality.observation_residual_m"], out["quality.observation_residual_valid"] = np.where(valid, residual, 0), valid
    else:
        out["quality.observation_residual_m"] = np.zeros((count, 22))
        out["quality.observation_residual_valid"] = np.zeros((count, 22), bool)
    return validate_parameter_arrays(out)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_parameter_model(path, *, expected_sha256=MODEL_SHA256, allow_test_fixture=False):
    """Verify the numeric full52 asset and its real sixteen shape components."""
    path = Path(path)
    actual = _sha256(path)
    if actual != expected_sha256 or (not allow_test_fixture and actual != MODEL_SHA256):
        raise ValueError("Parameter model SHA mismatch; the shared official SMPL-H asset is required")
    with np.load(path, allow_pickle=False) as asset:
        template, shape, reg, tree = [np.asarray(asset[key]) for key in
                                      ("v_template", "shapedirs", "J_regressor", "kintree_table")]
        if (template.shape != (6890, 3) or shape.shape != (6890, 3, 16)
                or reg.shape != (52, 6890) or tree.shape != (2, 52)):
            raise ValueError("Parameters require the full6890/52 SMPL-H model with sixteen actual shape bases")
        lookup = {int(child): idx for idx, child in enumerate(tree[1])}
        parents = np.array([-1] + [lookup[int(parent)] for parent in tree[0, 1:]])
        expected = list(PARENTS)
        for wrist, start in ((20, 22), (21, 37)):
            for finger in range(5):
                first = start + finger * 3
                expected.extend((wrist, first, first + 1))
        if not np.array_equal(parents, expected):
            raise ValueError("Parameter model full52 hand hierarchy mismatch")
    model = BodyModel(path)
    if _sha256(path) != actual:
        raise ValueError("Parameter model changed during loading")
    model.metadata["parameter_test_fixture_allowed"] = bool(allow_test_fixture)
    return model


def _selected(start_frame, num_frames, fps, length):
    for name, value in (("start_frame", start_frame), ("num_frames", num_frames), ("fps", fps)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
    if fps not in (15, 30) or start_frame < 0 or num_frames < 2:
        raise ValueError("Select at least two frames at 15Hz or 30Hz")
    if fps == 15 and start_frame % 2:
        raise ValueError("15Hz windows require the zero-based even archive phase")
    selected = start_frame + np.arange(num_frames) * (30 // fps)
    if selected[-1] >= length:
        raise ValueError("Parameter window exceeds episode; no padding/extrapolation")
    return selected


def _window_from_selected(arrays, metadata, model, *, fps, canonicalize, task, forecast_prefix):
    count = len(arrays["timestamp"])
    if canonicalize and (metadata.get("training_eligible") is False
                         or metadata.get("canonical_training_eligible") is False
                         or bool(metadata.get("quarantine_reasons", []))):
        raise ValueError("Quarantined or ineligible parameter episodes cannot be canonicalized")
    if task not in ("recon", "fore", "gen"):
        raise ValueError("Unknown motion task")
    if task == "fore" and (isinstance(forecast_prefix, bool) or not isinstance(forecast_prefix, (int, np.integer))
                           or not 1 <= forecast_prefix < count):
        raise ValueError("Forecast prefix must lie inside the selected training window")
    if (model.metadata["shape_basis_count"] != 16 or
            (model.metadata["model_sha256"] != MODEL_SHA256 and not model.metadata.get("parameter_test_fixture_allowed", False))):
        raise ValueError("Window FK requires sixteen real model shape bases")
    p, q, numeric = model.decode(arrays["smplh.trans"], arrays["smplh.root_orient"],
                                arrays["smplh.pose_body"], arrays["smplh.betas"], np.ones(count, bool))
    pv, rv = arrays["estimated.valid.body_position"], arrays["estimated.valid.body_rotation"]
    if np.any((pv | rv) & ~numeric) or not np.isfinite(p).all():
        raise ValueError("Explicit estimate masks refer to unavailable model FK")
    result = {key: value.copy() for key, value in arrays.items()}
    meta = deepcopy(metadata)
    native = np.zeros((count, 22, 3)); native_valid = np.zeros((count, 22), bool)
    native[:, [10, 11]] = arrays["source.feet_world"].reshape(count, 2, 3)
    native_valid[:, [10, 11]] = arrays["source.feet_valid"]
    prefix = forecast_prefix if task == "fore" else count
    ground = resolve_window_ground(native[:prefix], native_valid[:prefix], meta["ground"], meta.get("coordinate_verified", False))
    ground.update(fallback_body_source="native_source_feet", fallback_selected_frames=int(prefix))
    ground["native_window_fallback_policy"] = deepcopy(ground.get("window_fallback", {}))
    ground["window_fallback"] = {"mode": "disabled_after_native_resolution"}
    result["observation.state.body_world"] = np.where(pv[..., None], p, 0).reshape(count, 66).astype(np.float32)
    result["observation.state.body_world_rotation"] = np.where(rv[..., None], q, [1, 0, 0, 0]).reshape(count, 88).astype(np.float32)
    for component, valid in (("position", pv), ("rotation", rv)):
        result[f"observation.valid.body_{component}"] = valid.copy()
        result[f"observation.source_observed.body_{component}"] = arrays[f"source.observed.body_{component}"].copy()
        result[f"observation.source_interpolated.body_{component}"] = arrays[f"source.interpolated.body_{component}"].copy()
        result[f"observation.estimated.completed.body_{component}"] = arrays[f"estimated.completed.body_{component}"].copy()
        result[f"observation.interpolated.body_{component}"] = (arrays[f"source.interpolated.body_{component}"] | arrays[f"estimated.completed.body_{component}"]) & valid
        result[f"observation.quality.body_{component}"] = arrays[f"quality.{component}_confidence"].copy()
    # Derived reference placeholders are replaced by derive_ground_features.
    for modality in ("head_reference_floor",):
        result[f"observation.state.{modality}_world"] = np.zeros((count, 3), np.float32)
        result[f"observation.state.{modality}_world_rotation"] = np.tile([1., 0, 0, 0], (count, 1)).astype(np.float32)
        for kind in ("valid", "interpolated"):
            for component in ("position", "rotation"):
                result[f"observation.{kind}.{modality}_{component}"] = np.zeros(count, bool)
    transform = np.eye(4)
    if canonicalize:
        reference_valid = result["observation.valid.head_reference_position"].reshape(count) & result["observation.valid.head_reference_rotation"].reshape(count)
        if not (reference_valid[0] and meta.get("head_reference", {}).get("basis_verified", False)
                and meta.get("coordinate_verified", False)):
            raise ValueError("Canonical parameter windows require verified world/camera axes and a valid first reference")
        heading = quat_to_matrix(result["observation.state.head_reference_world_rotation"])[0, :2, 0]
        length = np.linalg.norm(heading)
        if length <= 1e-6:
            raise ValueError("First camera reference axis has degenerate horizontal projection")
        x, y = heading / length
        A = np.array([[x, y, 0.], [-y, x, 0.], [0., 0., 1.]])
        origin = result["observation.state.head_reference_world"][0].copy(); origin[2] = 0
        t = -A @ origin
        transform[:3, :3], transform[:3, 3] = A, t
        aq = matrix_to_quat(A)
        for modality, shape in POSITION_SHAPES.items():
            key = f"observation.state.{modality}_world"
            value = result[key].reshape(count, *shape)
            valid = result[f"observation.valid.{modality}_position"].reshape(count, shape[0])
            result[key] = np.where(valid[..., None], value @ A.T + t, 0).reshape(count, -1).astype(np.float32)
        for modality, shape in ROTATION_SHAPES.items():
            key = f"observation.state.{modality}_world_rotation"
            value = quat_multiply(aq, result[key].reshape(count, *shape))
            valid = result[f"observation.valid.{modality}_rotation"].reshape(count, shape[0])
            result[key] = np.where(valid[..., None], value, [1, 0, 0, 0]).reshape(count, -1).astype(np.float32)
        # Keep returned parameter and native-foot coordinates in the same world.
        root_q = quat_multiply(aq, axis_angle_to_quat(arrays["smplh.root_orient"]))
        from scipy.spatial.transform import Rotation
        result["smplh.root_orient"] = Rotation.from_matrix(quat_to_matrix(root_q)).as_rotvec().astype(np.float32)
        j0 = model.rest[0] + np.einsum("tb,cb->tc", arrays["smplh.betas"], model.shape_basis[0])
        result["smplh.trans"] = ((p[:, 0] @ A.T + t) - j0).astype(np.float32)
        feet = arrays["source.feet_world"].reshape(count, 2, 3)
        result["source.feet_world"] = np.where(arrays["source.feet_valid"][..., None], feet @ A.T + t, 0).reshape(count, 6).astype(np.float32)
    result["timestamp"] = (np.arange(count) / fps).astype(np.float32)
    meta.update(ground=ground, target_joint_basis="smplh_body22_axes_v1", label_kind="model_fitted_estimate",
                body_position_source="model_fitted_estimate", rotation_semantics="model_fitted_global_wxyz",
                target_model_axes_defined=True, rotation_basis_verified=False,
                source_native_axes_certified=False, absolute_rotation_accuracy_established=False)
    derive_ground_features(result, ground, bool(meta.get("head_reference", {}).get("basis_verified", False)))
    meta["unified_dense22"] = unified_metadata(meta)
    meta["unified_dense22"].update(label_kind="model_fitted_estimate", target_model_axes_defined=True,
                                   absolute_rotation_accuracy_established=False, native_axis_certified=False,
                                   world_fields="transient_model_FK_estimate_from_owned_parameters")
    result.update(encode_unified_dense22(result, meta))
    meta["training_window"] = {"archive_fps": 30, "fps": int(fps), "source_frame_indices": arrays["frame_index"].tolist(),
                               "canonicalized": bool(canonicalize), "window_from_archive_world": transform.tolist(),
                               "task": task, "forecast_prefix": int(forecast_prefix) if task == "fore" else None,
                               "ground_fallback_body": "native_source_feet", "floor_resolution_stage": "after_training_fps_and_window_selection",
                               "derived_body": "transient_smplh_body22_fk", "requires_old_dense_archive": False}
    trajectory, trajectory_valid = head_trajectory(result)
    return TrainingWindow(result, meta, trajectory, trajectory_valid)


def prepare_parameter_window(arrays, metadata, model, *, start_frame, num_frames, fps=15,
                             canonicalize=True, task="recon", forecast_prefix=30):
    checked = validate_parameter_arrays(arrays)
    selected = _selected(start_frame, num_frames, fps, len(checked["timestamp"]))
    return _window_from_selected({key: value[selected] for key, value in checked.items()}, metadata, model,
                                 fps=fps, canonicalize=canonicalize, task=task, forecast_prefix=forecast_prefix)


def write_parameter_table(path, arrays, *, row_group_size=512, compression="zstd"):
    checked = validate_parameter_arrays(arrays)
    specs = parameter_specs(include_dino=DINO_COLUMN in checked)
    specs.update({key: value for key, value in OPTIONAL_SPECS.items() if key in checked})
    columns = {}
    for key, values in checked.items():
        columns[key] = (pa.array(values) if specs[key]["shape"] == [1] else
                        pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1)), specs[key]["shape"][0]))
    table = pa.table(columns).replace_schema_metadata({b"parameter_schema_version": PARAMETER_SCHEMA_VERSION.encode()})
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, row_group_size=row_group_size, compression=compression)
    return {"length": len(checked["timestamp"]), "columns": list(checked), "sha256": _sha256(path)}


def _read_rows(path, columns, specs, selected=None):
    parquet = pq.ParquetFile(path)
    blocks, begin = [], 0
    for index in range(parquet.metadata.num_row_groups):
        end = begin + parquet.metadata.row_group(index).num_rows
        if selected is None or np.any((selected >= begin) & (selected < end)):
            table = parquet.read_row_group(index, columns=columns)
            take = None if selected is None else selected[(selected >= begin) & (selected < end)] - begin
            block = {}
            for key in columns:
                column = table[key].combine_chunks()
                value = (column.to_numpy(zero_copy_only=False) if specs[key]["shape"] == [1] else
                         column.values.to_numpy(zero_copy_only=False).reshape(len(table), specs[key]["shape"][0]))
                block[key] = value.copy() if take is None else value[take].copy()
            blocks.append(block)
        begin = end
    if not blocks:
        raise ValueError("Selected parameter rows are unavailable")
    return {key: np.concatenate([block[key] for block in blocks]) for key in columns}


class SMPLHParameterArchive:
    """Read only owned archive-relative motion, model and DINO files."""
    def __init__(self, root, *, allow_test_fixture=False):
        self.root = Path(root).resolve()
        self.info = json.loads((self.root / "meta/info.json").read_text())
        if self.info.get("schema_version") != PARAMETER_SCHEMA_VERSION or self.info.get("fps") != 30:
            raise ValueError("Not a native30Hz standalone SMPL-H parameter archive")
        identity = self.info["model"]
        self.model = load_parameter_model(self._owned(identity["path"]), expected_sha256=identity["model_sha256"],
                                          allow_test_fixture=allow_test_fixture)
        self.hand_basis = self.info.get("hand_basis")
        if self.hand_basis is not None:
            hand_path = self._owned(self.hand_basis["path"])
            if _sha256(hand_path) != self.hand_basis["sha256"]:
                raise ValueError("Owned hand PCA basis SHA mismatch")
            with np.load(hand_path, allow_pickle=False) as hand_asset:
                if not hand_asset.files or any(not np.isfinite(hand_asset[key]).all() for key in hand_asset.files):
                    raise ValueError("Owned hand PCA basis must contain finite numeric arrays")
        self._records = [json.loads(path.read_text()) for path in sorted((self.root / ".pipeline/episodes").glob("*.json"))]
        if any(record.get("episode_index") != i for i, record in enumerate(self._records)):
            raise ValueError("Standalone episode records must have contiguous identities")

    def _owned(self, value):
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Standalone dependencies must use owned relative paths")
        resolved = (self.root / path).resolve()
        if not resolved.is_relative_to(self.root) or not resolved.is_file():
            raise ValueError("Standalone dependency is absent or points outside this archive")
        return resolved

    def records(self):
        return deepcopy(self._records)

    def data_path(self, episode_index):
        record = self._records[episode_index]
        return self._owned(record.get("data_path", f"data/chunk-{episode_index//1000:03d}/file-{episode_index%1000:03d}.parquet"))

    def read_window(self, episode_index, *, start_frame, num_frames, fps=15, canonicalize=True,
                    task="recon", forecast_prefix=30, load_dino=True):
        if isinstance(episode_index, bool) or not isinstance(episode_index, (int, np.integer)) or not 0 <= episode_index < len(self._records):
            raise ValueError("Episode index is outside this parameter archive")
        record = self._records[episode_index]
        selected = _selected(start_frame, num_frames, fps, record["length"])
        path = self.data_path(episode_index)
        parquet = pq.ParquetFile(path)
        available = set(parquet.schema_arrow.names)
        if parquet.metadata.num_rows != record["length"]:
            raise ValueError("Standalone motion length differs from its episode record")
        allowed = set(parameter_specs(include_dino=True)) | set(OPTIONAL_SPECS)
        if available - allowed:
            raise ValueError("Standalone parameter motion contains unknown or stored dense body columns")
        specs = parameter_specs(include_dino=load_dino and DINO_COLUMN in available)
        specs.update({key: value for key, value in OPTIONAL_SPECS.items() if key in available})
        if not set(specs) <= available:
            raise ValueError("Standalone motion file is missing mandatory parameter columns")
        arrays = _read_rows(path, list(specs), specs, selected)
        arrays = validate_parameter_arrays(arrays, require_contiguous=False)
        if arrays.get("smplh.valid.hand_pose", np.zeros(0, bool)).any() and self.hand_basis is None:
            raise ValueError("Available hand PCA parameters require an owned hand_basis identity")
        if not np.array_equal(arrays["frame_index"], selected) or not np.all(arrays["episode_index"] == episode_index):
            raise ValueError("Standalone motion frame join differs from episode identity")
        if load_dino and DINO_COLUMN not in arrays:
            if not record.get("vision_path"):
                raise ValueError("Standalone DINO requires its own relative vision_path")
            vision_path = self._owned(record["vision_path"])
            vspecs = feature_specs_for_schema(SCHEMA_VERSION)
            columns = ["frame_index", "episode_index", DINO_COLUMN]
            vision_parquet = pq.ParquetFile(vision_path)
            if vision_parquet.metadata.num_rows != record["length"]:
                raise ValueError("Standalone DINO length differs from its episode record")
            vision_available = set(vision_parquet.schema_arrow.names)
            vspecs.update(OPTIONAL_SPECS)
            columns += [key for key in ("timestamp", "observation.source_timestamp", "source.video_frame_index",
                                       "observation.valid.dino", "observation.video_timestamp", "observation.video_time_error")
                        if key in vision_available]
            vision = _read_rows(vision_path, columns, vspecs, selected)
            if not np.array_equal(vision["frame_index"], selected) or not np.all(vision["episode_index"] == episode_index):
                raise ValueError("Standalone DINO frame join differs from parameter rows")
            if any(not np.array_equal(vision[key], arrays[key]) for key in columns
                   if key != DINO_COLUMN and key in arrays):
                raise ValueError("Standalone DINO time/validity/source-frame join differs from parameter rows")
            arrays[DINO_COLUMN] = _cast_feature(vision[DINO_COLUMN], DINO_COLUMN, vspecs[DINO_COLUMN], num_frames)
        return _window_from_selected(arrays, record["metadata"], self.model, fps=fps, canonicalize=canonicalize,
                                     task=task, forecast_prefix=forecast_prefix)
