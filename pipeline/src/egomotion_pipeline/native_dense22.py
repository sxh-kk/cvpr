"""Lossless-source dense22 sidecars using EgoSuite's native joint axes.

This module changes the rotation encoding and the external reference frame.
It does not calibrate anatomical axes, fit a body model, change joint centers,
estimate a floor, or consume image features.  Original projected source
columns remain in the sidecar so every derived value can be checked.
"""

from collections.abc import Mapping
import copy
import json

import numpy as np

from .geometry import quat_to_matrix
from .schema import BODY_NAMES, PARENTS, feature_specs


SCHEMA_VERSION = "egosuite-native-dense22-rot6d-xyz-v1"
MOTION_COLUMNS = (
    "timestamp", "index", "frame_index", "episode_index", "task_index",
    "observation.source_timestamp", "observation.source_frame_left",
    "observation.source_frame_right", "observation.source_interpolation_weight",
    "observation.state.body_world", "observation.state.body_world_rotation",
    "observation.valid.body_position", "observation.valid.body_rotation",
    "observation.interpolated.body_position", "observation.interpolated.body_rotation",
    "observation.state.head_reference_floor_world",
    "observation.state.head_reference_floor_world_rotation",
    "observation.valid.head_reference_floor_position",
    "observation.valid.head_reference_floor_rotation",
    "observation.interpolated.head_reference_floor_position",
    "observation.interpolated.head_reference_floor_rotation",
)
STATE_COLUMNS = tuple(f"observation.state.body_{frame}_rot6d_xyz" for frame in ("world", "floor"))
MASK_COLUMNS = tuple(
    f"observation.{kind}.body_{frame}_rot6d_xyz_{component}"
    for frame in ("world", "floor") for kind in ("valid", "interpolated")
    for component in ("rotation", "position")
)


def native_dense22_specs():
    """Return flat Parquet specifications, including unchanged source columns."""
    source_specs = feature_specs()
    specs = {key: copy.deepcopy(source_specs[key]) for key in MOTION_COLUMNS}
    components = ["r00", "r01", "r02", "r10", "r11", "r12", "x", "y", "z"]
    for key in STATE_COLUMNS:
        specs[key] = {"dtype": "float32", "shape": [198],
                      "names": [f"{joint}.{component}" for joint in BODY_NAMES for component in components]}
    for key in MASK_COLUMNS:
        specs[key] = {"dtype": "bool", "shape": [22], "names": list(BODY_NAMES)}
    return specs


def _source_metadata(metadata):
    if not isinstance(metadata, Mapping):
        raise ValueError("source metadata must be a mapping")
    nested = metadata.get("metadata")
    source_meta = nested if isinstance(nested, Mapping) else metadata
    source = metadata.get("source", source_meta.get("source"))
    if source != "egosuite":
        raise ValueError("native dense22 export requires source='egosuite'")
    if source_meta.get("rotation_semantics") != "source_global_wxyz":
        raise ValueError("native export requires source_global_wxyz global quaternions")
    if "body_basis_correction" not in source_meta or source_meta["body_basis_correction"] is not None:
        raise ValueError("native export requires explicit null body_basis_correction")
    if source_meta.get("coordinate_verified") is not True:
        raise ValueError("native export requires explicitly verified output-world coordinates")
    if source_meta.get("rotation_basis_verified", False) is not False:
        raise ValueError("native export cannot relabel an already calibrated/certified rotation basis")
    return source_meta


def _reshape_feature(arrays, key, count, shape):
    if key not in arrays:
        raise ValueError(f"missing source column: {key}")
    value = np.asarray(arrays[key])
    flat_shape = (count, int(np.prod(shape)))
    shaped = (count, *shape)
    scalar = shape == (1,) and value.shape == (count,)
    if value.shape not in {shaped, flat_shape} and not scalar:
        raise ValueError(f"{key}: shape {value.shape}, expected {shaped} or {flat_shape}")
    return value.reshape(shaped)


def _mask(arrays, key, count, width):
    value = _reshape_feature(arrays, key, count, (width,))
    if value.dtype != np.bool_:
        raise ValueError(f"{key}: validity/interpolation masks must be boolean")
    return value


def _proper_so3(rotation, atol=1e-8):
    gram = np.swapaxes(rotation, -1, -2) @ rotation
    return (np.isfinite(rotation).all(axis=(-2, -1))
            & (np.max(np.abs(gram-np.eye(3)), axis=(-2, -1)) <= atol)
            & (np.abs(np.linalg.det(rotation)-1.) <= atol))


def _quaternions(raw):
    """Preserve raw validity before quat_to_matrix's safe identity fallback."""
    raw = np.asarray(raw, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        norm = np.linalg.norm(raw, axis=-1)
    valid = np.isfinite(raw).all(-1) & np.isfinite(norm) & (norm > 1e-8)
    safe = np.zeros_like(raw)
    safe[..., 0] = 1.
    np.divide(raw, norm[..., None], out=safe, where=valid[..., None])
    rotation = quat_to_matrix(safe)
    return safe, rotation, valid & _proper_so3(rotation)


def _inputs(arrays):
    if not isinstance(arrays, Mapping) or "timestamp" not in arrays:
        raise ValueError("source arrays must contain timestamp")
    count = len(np.asarray(arrays["timestamp"]))
    if not count:
        raise ValueError("cannot encode an empty episode")
    for key in MOTION_COLUMNS:
        if key not in arrays:
            raise ValueError(f"missing source column: {key}")
    timestamp = _reshape_feature(arrays, "timestamp", count, (1,)).reshape(count)
    if not np.isfinite(timestamp).all() or np.any(np.diff(timestamp) <= 0):
        raise ValueError("source timestamp must be finite and strictly increasing")
    for key in ("index", "frame_index", "episode_index", "task_index",
                "observation.source_frame_left", "observation.source_frame_right"):
        value = _reshape_feature(arrays, key, count, (1,))
        if not np.issubdtype(value.dtype, np.integer):
            raise ValueError(f"{key}: source index must have integer dtype")
    source_ts = _reshape_feature(arrays, "observation.source_timestamp", count, (1,))
    if not np.isfinite(source_ts).all():
        raise ValueError("source capture timestamp must be finite")
    weight = _reshape_feature(arrays, "observation.source_interpolation_weight", count, (1,))
    if not np.isfinite(weight).all() or np.any((weight < 0) | (weight > 1)):
        raise ValueError("source interpolation weight must be finite in [0, 1]")
    position = _reshape_feature(arrays, "observation.state.body_world", count, (22, 3)).astype(np.float64)
    raw_quat = _reshape_feature(arrays, "observation.state.body_world_rotation", count, (22, 4))
    floor_position = _reshape_feature(arrays, "observation.state.head_reference_floor_world", count, (3,)).astype(np.float64)
    floor_raw = _reshape_feature(arrays, "observation.state.head_reference_floor_world_rotation", count, (4,))
    unit_q, rotation, quaternion_valid = _quaternions(raw_quat)
    floor_unit_q, floor_rotation, floor_quaternion_valid = _quaternions(floor_raw)
    position_valid = _mask(arrays, "observation.valid.body_position", count, 22) & np.isfinite(position).all(-1)
    rotation_valid = _mask(arrays, "observation.valid.body_rotation", count, 22) & quaternion_valid
    floor_position_valid = (_mask(arrays, "observation.valid.head_reference_floor_position", count, 1).reshape(count)
                            & np.isfinite(floor_position).all(-1))
    floor_rotation_valid = (_mask(arrays, "observation.valid.head_reference_floor_rotation", count, 1).reshape(count)
                            & floor_quaternion_valid)
    interpolated = {
        "body_position": _mask(arrays, "observation.interpolated.body_position", count, 22),
        "body_rotation": _mask(arrays, "observation.interpolated.body_rotation", count, 22),
        "floor_position": _mask(arrays, "observation.interpolated.head_reference_floor_position", count, 1).reshape(count),
        "floor_rotation": _mask(arrays, "observation.interpolated.head_reference_floor_rotation", count, 1).reshape(count),
    }
    return {"count": count, "position": position, "rotation": rotation, "unit_q": unit_q,
            "floor_position": floor_position, "floor_rotation": floor_rotation, "floor_unit_q": floor_unit_q,
            "position_valid": position_valid, "rotation_valid": rotation_valid,
            "floor_position_valid": floor_position_valid, "floor_rotation_valid": floor_rotation_valid,
            "interpolated": interpolated}


def _output_masks(source):
    pv, rv = source["position_valid"], source["rotation_valid"]
    fpv, frv = source["floor_position_valid"][:, None], source["floor_rotation_valid"][:, None]
    interp = source["interpolated"]
    masks = {("world", "valid", "position"): pv,
             ("world", "valid", "rotation"): rv,
             ("floor", "valid", "position"): pv & fpv & frv,
             ("floor", "valid", "rotation"): rv & frv}
    masks[("world", "interpolated", "position")] = interp["body_position"] & pv
    masks[("world", "interpolated", "rotation")] = interp["body_rotation"] & rv
    masks[("floor", "interpolated", "position")] = (
        interp["body_position"] | interp["floor_position"][:, None] | interp["floor_rotation"][:, None]
    ) & masks[("floor", "valid", "position")]
    masks[("floor", "interpolated", "rotation")] = (
        interp["body_rotation"] | interp["floor_rotation"][:, None]
    ) & masks[("floor", "valid", "rotation")]
    return masks


def _mask_key(frame, kind, component):
    return f"observation.{kind}.body_{frame}_rot6d_xyz_{component}"


def _check_source_join(arrays, metadata, count):
    frame = np.asarray(arrays["frame_index"]).reshape(count)
    if not np.array_equal(frame, np.arange(count)):
        raise ValueError("source frame_index must be contiguous from zero")
    index = np.asarray(arrays["index"]).reshape(count)
    if not np.array_equal(index, np.arange(int(index[0]), int(index[0])+count)):
        raise ValueError("source global index must be contiguous")
    episode = np.asarray(arrays["episode_index"]).reshape(count)
    if not np.all(episode == episode[0]):
        raise ValueError("source episode_index must identify one episode")
    task = np.asarray(arrays["task_index"]).reshape(count)
    if not np.all(task == task[0]):
        raise ValueError("source task_index must identify one task per episode")
    expected = {"episode_index": int(episode[0]), "length": count,
                "dataset_from_index": int(index[0]), "dataset_to_index": int(index[0])+count}
    for key, value in expected.items():
        if key in metadata and (type(metadata[key]) is not int or metadata[key] != value):
            raise ValueError(f"source record and frame identity disagree: {key}")


def _check_floor_basis(arrays, source_meta):
    # An unavailable floor cannot affect the independently valid world output.
    # If the source claims either floor component is available, its documented
    # camera-reference basis must be explicit; coordinate verification alone
    # does not establish the camera axis convention.
    claimed = (np.asarray(arrays["observation.valid.head_reference_floor_position"]).any()
               or np.asarray(arrays["observation.valid.head_reference_floor_rotation"]).any())
    reference = source_meta.get("head_reference")
    verified = isinstance(reference, Mapping) and reference.get("basis_verified") is True
    if claimed and not verified:
        raise ValueError("available camera-floor reference requires head_reference.basis_verified=true")


def _episode_metadata(metadata, source_meta, count, masks):
    return {
        "schema_version": SCHEMA_VERSION, "source": "egosuite", "length": count,
        "shape": [22, 9], "flat_shape": [198], "storage_dtype": "float32",
        "joint_order": list(BODY_NAMES), "parents": PARENTS.tolist(),
        "flatten_order": "joint_major", "component_order": ["r00", "r01", "r02", "r10", "r11", "r12", "x", "y", "z"],
        "rotation_6d": "first_two_matrix_rows_flattened", "matrix_convention": "column_vectors; local_to_reference",
        "position_units": "meters", "position_semantics": "egosuite_native_dense22_joint_centers",
        "rotation_semantics": "source_native_global_rotation", "source_quaternion_semantics": "source_global_wxyz",
        "target_joint_basis": "egosuite_native_v1", "body_basis_correction": None,
        "rotation_basis_verified": False, "axis_calibration_performed": False,
        "axis_certification": "none; numerical_availability_is_not_anatomical_axis_verification",
        "coordinate_verified": True,
        "world_position": "unchanged_valid_archived_body_world_xyz",
        "world_rotation": "quaternion_to_matrix(archived_body_world_rotation); no_right_correction",
        "floor_position": "R_world_floor.T @ (p_world - p_world_floor)",
        "floor_rotation": "R_world_floor.T @ R_native_world_joint",
        "floor_policy": "stored_camera_ground_projection; no_reestimation_or_window_fallback",
        "mask_semantics": "independent_xyz_and_rotation; invalid_subblocks_zero; interpolation_requires_validity",
        "source_inputs_modified": False, "source_quaternions_preserved": True,
        "source_columns": list(MOTION_COLUMNS),
        "source_identity": {key: copy.deepcopy(metadata[key]) for key in ("episode_index", "source_uid", "source_key") if key in metadata},
        "source_coordinate_provenance": {key: copy.deepcopy(source_meta[key]) for key in ("world_from_source", "position_scale", "head_reference", "ground") if key in source_meta},
        "valid_counts": {f"{frame}_{component}": int(mask.sum())
                         for (frame, kind, component), mask in masks.items() if kind == "valid"},
    }


def encode_native_dense22(arrays, metadata):
    """Return copied source columns, native world/floor 22x9, and provenance.

    ``metadata`` is preferably a complete archive episode record. A flat
    metadata mapping is accepted when it includes explicit ``source``. Floor
    orientation needs a valid reference rotation; floor positions additionally
    need a valid reference origin. Missing floor data never invalidates world
    outputs. Rotation availability is not anatomical-axis certification.
    """
    source_meta = _source_metadata(metadata)
    source = _inputs(arrays)
    count = source["count"]
    _check_source_join(arrays, metadata, count)
    _check_floor_basis(arrays, source_meta)
    out = {key: np.array(arrays[key], copy=True) for key in MOTION_COLUMNS}
    masks = _output_masks(source)
    # Zero unavailable operands before arithmetic, including masked NaNs.
    world_position = np.where(source["position_valid"][..., None], source["position"], 0.)
    floor_position = np.where(source["floor_position_valid"][:, None], source["floor_position"], 0.)
    floor_local_position = np.einsum("tji,tkj->tki", source["floor_rotation"], world_position-floor_position[:, None])
    floor_local_rotation = np.swapaxes(source["floor_rotation"], -1, -2)[:, None] @ source["rotation"]
    for frame, rotations, positions in (
        ("world", source["rotation"], world_position),
        ("floor", floor_local_rotation, floor_local_position),
    ):
        block = np.zeros((count, 22, 9), dtype=np.float32)
        rmask, pmask = masks[(frame, "valid", "rotation")], masks[(frame, "valid", "position")]
        with np.errstate(over="ignore", invalid="ignore"):
            encoded_rotation = rotations[..., :2, :].reshape(count, 22, 6).astype(np.float32)
            encoded_position = positions.astype(np.float32)
        if not np.isfinite(encoded_rotation[rmask]).all() or not np.isfinite(encoded_position[pmask]).all():
            raise ValueError(f"{frame}: a valid source value cannot be represented as float32")
        block[..., :6] = np.where(rmask[..., None], encoded_rotation, 0.)
        block[..., 6:] = np.where(pmask[..., None], encoded_position, 0.)
        out[f"observation.state.body_{frame}_rot6d_xyz"] = block.reshape(count, 198)
    for (frame, kind, component), mask in masks.items():
        out[_mask_key(frame, kind, component)] = mask.copy()
    return out, _episode_metadata(metadata, source_meta, count, masks)


def _independent_quaternion_matrix(unit_q):
    """Quaternion sandwich expressed through outer products, for auditing."""
    w, v = unit_q[..., 0], unit_q[..., 1:]
    cross = np.zeros((*v.shape[:-1], 3, 3), dtype=np.float64)
    cross[..., 0, 1], cross[..., 0, 2] = -v[..., 2], v[..., 1]
    cross[..., 1, 0], cross[..., 1, 2] = v[..., 2], -v[..., 0]
    cross[..., 2, 0], cross[..., 2, 1] = -v[..., 1], v[..., 0]
    return ((w*w-(v*v).sum(-1))[..., None, None]*np.eye(3)
            + 2*v[..., :, None]*v[..., None, :] + 2*w[..., None, None]*cross)


def _decode_row6d(value):
    first, second = value[..., :3], value[..., 3:6]
    nfirst = np.linalg.norm(first, axis=-1, keepdims=True)
    if np.any(nfirst <= 1e-8):
        raise ValueError("valid rotation contains a degenerate first 6D row")
    first = first/nfirst
    second = second-(first*second).sum(-1, keepdims=True)*first
    nsecond = np.linalg.norm(second, axis=-1, keepdims=True)
    if np.any(nsecond <= 1e-8):
        raise ValueError("valid rotation contains collinear 6D rows")
    second = second/nsecond
    return np.stack((first, second, np.cross(first, second)), axis=-2)


def _maximum(value):
    return float(np.max(np.abs(value))) if np.size(value) else 0.


def validate_native_dense22(source_arrays, output_arrays, metadata, episode_metadata=None, *, atol=2e-5):
    """Audit every row and source column; raise ValueError on any mismatch.

    Verification independently computes quaternion rotations via sandwich
    algebra and decodes row-6D back to SO(3). It checks each component mask,
    zero padding, XYZ preservation, floor inverse transforms, and unchanged
    capture/frame identities. This proves the export, not anatomical accuracy.
    """
    source_meta = _source_metadata(metadata)
    if not np.isfinite(atol) or atol <= 0:
        raise ValueError("validation atol must be finite and positive")
    source = _inputs(source_arrays)
    count = source["count"]
    _check_source_join(source_arrays, metadata, count)
    _check_floor_basis(source_arrays, source_meta)
    expected_keys = set(native_dense22_specs())
    if set(output_arrays) != expected_keys:
        raise ValueError("native dense22 output columns differ from the schema")
    for key in MOTION_COLUMNS:
        before, after = np.asarray(source_arrays[key]), np.asarray(output_arrays[key])
        if before.dtype != after.dtype or before.shape != after.shape or not np.array_equal(before, after, equal_nan=True):
            raise ValueError(f"source column changed during export: {key}")
    masks = _output_masks(source)
    for (frame, kind, component), expected in masks.items():
        key = _mask_key(frame, kind, component)
        actual = np.asarray(output_arrays[key])
        if actual.dtype != np.bool_ or actual.shape != (count, 22) or not np.array_equal(actual, expected):
            raise ValueError(f"incorrect output validity/interpolation mask: {key}")
    world_rotation = _independent_quaternion_matrix(source["unit_q"])
    floor_rotation = _independent_quaternion_matrix(source["floor_unit_q"])
    local_rotation = np.einsum("tji,tkjl->tkil", floor_rotation, world_rotation)
    safe_position = np.where(source["position_valid"][..., None], source["position"], 0.)
    safe_origin = np.where(source["floor_position_valid"][:, None], source["floor_position"], 0.)
    local_position = np.einsum("tji,tkj->tki", floor_rotation, safe_position-safe_origin[:, None])
    report = {"schema_version": SCHEMA_VERSION, "rows": count, "source_columns_checked": len(MOTION_COLUMNS),
              "source_columns_preserved": True, "all_masks_verified": True,
              "axis_certification": False, "numerical_validation_only": True, "atol": float(atol)}
    for frame, expected_rotation, expected_position in (
        ("world", world_rotation, safe_position), ("floor", local_rotation, local_position),
    ):
        key = f"observation.state.body_{frame}_rot6d_xyz"
        value = np.asarray(output_arrays[key])
        if value.dtype != np.float32 or value.shape != (count, 198) or not np.isfinite(value).all():
            raise ValueError(f"{key}: derived state must be finite float32 [T,198]")
        block = value.reshape(count, 22, 9).astype(np.float64)
        rmask, pmask = masks[(frame, "valid", "rotation")], masks[(frame, "valid", "position")]
        if np.any(block[..., :6][~rmask] != 0) or np.any(block[..., 6:][~pmask] != 0):
            raise ValueError(f"{key}: invalid rotation/position subblocks must be zero")
        expected_6d = expected_rotation[..., :2, :].reshape(count, 22, 6)
        rotation_encoding_error = _maximum(block[..., :6][rmask]-expected_6d[rmask])
        position_error = _maximum(block[..., 6:][pmask]-expected_position[pmask])
        decoded = _decode_row6d(block[..., :6][rmask])
        rotation_roundtrip_error = _maximum(decoded-expected_rotation[rmask])
        if max(rotation_encoding_error, position_error, rotation_roundtrip_error) > atol:
            raise ValueError(f"{key}: source rotation mapping or XYZ preservation failed")
        if not _proper_so3(decoded, atol=max(atol, 1e-8)).all():
            raise ValueError(f"{key}: decoded 6D rotation is not proper SO(3)")
        if frame == "world":
            # Cast equivalence is stricter than a tolerance: source valid XYZ
            # must be preserved exactly in the float32 derived output.
            if not np.array_equal(value.reshape(count, 22, 9)[..., 6:][pmask], source["position"].astype(np.float32)[pmask]):
                raise ValueError("valid world XYZ were changed")
            position_inverse_error = position_error
        else:
            inverse_position = np.einsum("tij,tkj->tki", floor_rotation, block[..., 6:])+safe_origin[:, None]
            position_inverse_error = _maximum(inverse_position[pmask]-source["position"][pmask])
            if position_inverse_error > 3*atol:
                raise ValueError("floor XYZ inverse transform does not recover source world XYZ")
        report[frame] = {"valid_rotation_count": int(rmask.sum()), "valid_position_count": int(pmask.sum()),
                         "rotation_encoding_max_abs": rotation_encoding_error,
                         "rotation_roundtrip_max_abs": rotation_roundtrip_error,
                         "position_encoding_max_abs_m": position_error,
                         "position_inverse_max_abs_m": position_inverse_error}
    if episode_metadata is not None:
        expected = _episode_metadata(metadata, source_meta, count, masks)
        try:
            actual_json = json.dumps(episode_metadata, sort_keys=True, allow_nan=False)
            expected_json = json.dumps(expected, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("native export provenance must be finite JSON with declared types") from exc
        if actual_json != expected_json:
            raise ValueError("incorrect native export provenance: full declaration differs from source")
    report["passed"] = True
    return report
