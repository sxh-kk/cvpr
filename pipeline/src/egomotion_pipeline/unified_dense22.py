"""Physical v5 dense body/hand encoding and independently checked caches.

World body XYZ and world quaternions are the source of truth. Floor fields
are caches of the currently selected camera ground projection, so callers
must rebuild them after changing a window's world frame or ground estimate.
Encoding availability does not certify a source's anatomical joint axes.
"""

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from .schema import BODY_NAMES, HAND_NAMES, PARENTS, SCHEMA_VERSION
from .native_dense22 import (_reshape_feature, _mask, _quaternions, _output_masks,
                             _independent_quaternion_matrix, _decode_row6d, _proper_so3)


BODY_STATE_COLUMNS = tuple(f"observation.state.body_{frame}_rot6d_xyz" for frame in ("world", "floor"))
HAND_STATE_COLUMNS = tuple(f"observation.state.hand_{side}_floor" for side in ("left", "right"))
STATE_COLUMNS = BODY_STATE_COLUMNS + HAND_STATE_COLUMNS
MASK_COLUMNS = tuple(
    f"observation.{kind}.body_{frame}_rot6d_xyz_{component}"
    for frame in ("world", "floor") for kind in ("valid", "interpolated")
    for component in ("rotation", "position")
) + tuple(f"observation.{kind}.hand_{side}_floor_position"
          for side in ("left", "right") for kind in ("valid", "interpolated"))
ENCODING_POLICY = {
    "schema_version": SCHEMA_VERSION,
    "body_order": list(BODY_NAMES), "parents": PARENTS.tolist(), "hand_order": list(HAND_NAMES),
    "body_shape": [22, 9], "hand_shape": [21, 3], "flatten_order": "joint_major",
    "component_order": ["r00", "r01", "r02", "r10", "r11", "r12", "x", "y", "z"],
    "rotation_6d": "first_two_matrix_rows_flattened",
    "world_rotation": "normalized_archived_global_wxyz; no_additional_right_axis_correction",
    "matrix_convention": "column_vectors; joint_local_to_reference",
    "floor_rotation": "R_world_floor.T @ R_world_joint",
    "floor_position": "R_world_floor.T @ (p_world - p_world_floor)",
    "position_units": "meters", "world_fields": "authoritative_source_encoding",
    "floor_fields": "derived_cache; rebuild_after_world_or_ground_change",
    "ground_policy": "stored_camera_ground_projection; no_encoder_reestimation",
    "mask_policy": "independent_xyz_rotation; invalid_subblocks_zero; interpolated_remains_valid",
    "axis_policy": "retain_source_axes_and_verification; no_axis_calibration_performed",
}


def unified_specs():
    """The sixteen new v5 columns, with no image or original source columns."""
    specs = {}
    for key in BODY_STATE_COLUMNS:
        specs[key] = {"dtype": "float32", "shape": [198], "names": [
            f"{joint}.{component}" for joint in BODY_NAMES for component in ENCODING_POLICY["component_order"]]}
    for key in HAND_STATE_COLUMNS:
        specs[key] = {"dtype": "float32", "shape": [63],
                      "names": [f"{joint}.{axis}" for joint in HAND_NAMES for axis in "xyz"]}
    for key in MASK_COLUMNS:
        joints = BODY_NAMES if ".body_" in key else HAND_NAMES
        specs[key] = {"dtype": "bool", "shape": [len(joints)], "names": list(joints)}
    return specs


def _source_metadata(metadata):
    if not isinstance(metadata, Mapping):
        raise ValueError("source metadata must be a mapping")
    nested = metadata.get("metadata")
    return nested if isinstance(nested, Mapping) else metadata


def unified_metadata(metadata):
    """Describe numerical encoding while retaining the actual axis policy."""
    source = _source_metadata(metadata)
    verified = source.get("rotation_basis_verified", False) is True
    basis = source.get("target_joint_basis")
    if basis is None:
        source_name = metadata.get("source", source.get("source"))
        basis = "egosuite_native_v1" if source_name == "egosuite" and not verified else "source_native_unspecified"
    result = deepcopy(ENCODING_POLICY)
    result.update({"target_joint_basis": basis, "rotation_basis_verified": verified,
                   "axis_calibration_performed": False,
                   "source_rotation_semantics": source.get("rotation_semantics"),
                   "source_body_basis_correction": deepcopy(source.get("body_basis_correction")),
                   "coordinate_verified": source.get("coordinate_verified", False) is True,
                   "source_inputs_modified": False})
    return result


def _inputs(arrays, metadata):
    if not isinstance(arrays, Mapping) or "timestamp" not in arrays:
        raise ValueError("source arrays must contain timestamp")
    count = len(np.asarray(arrays["timestamp"]))
    if not count:
        raise ValueError("cannot encode an empty batch")
    # Batches can start at any frame and windows can use sparse frame indices.
    position = _reshape_feature(arrays, "observation.state.body_world", count, (22, 3)).astype(np.float64)
    raw = _reshape_feature(arrays, "observation.state.body_world_rotation", count, (22, 4))
    q, rotation, qvalid = _quaternions(raw)
    origin = _reshape_feature(arrays, "observation.state.head_reference_floor_world", count, (3,)).astype(np.float64)
    fq, floor_rotation, fqvalid = _quaternions(
        _reshape_feature(arrays, "observation.state.head_reference_floor_world_rotation", count, (4,)))
    source = {"count": count, "position": position, "rotation": rotation, "unit_q": q,
              "floor_position": origin, "floor_rotation": floor_rotation, "floor_unit_q": fq,
              "position_valid": _mask(arrays, "observation.valid.body_position", count, 22) & np.isfinite(position).all(-1),
              "rotation_valid": _mask(arrays, "observation.valid.body_rotation", count, 22) & qvalid,
              "floor_position_valid": _mask(arrays, "observation.valid.head_reference_floor_position", count, 1).reshape(count) & np.isfinite(origin).all(-1),
              "floor_rotation_valid": _mask(arrays, "observation.valid.head_reference_floor_rotation", count, 1).reshape(count) & fqvalid,
              "interpolated": {"body_position": _mask(arrays, "observation.interpolated.body_position", count, 22),
                               "body_rotation": _mask(arrays, "observation.interpolated.body_rotation", count, 22),
                               "floor_position": _mask(arrays, "observation.interpolated.head_reference_floor_position", count, 1).reshape(count),
                               "floor_rotation": _mask(arrays, "observation.interpolated.head_reference_floor_rotation", count, 1).reshape(count)}}
    reference = _source_metadata(metadata).get("head_reference", {})
    if (source["floor_position_valid"].any() or source["floor_rotation_valid"].any()) and (
            not isinstance(reference, Mapping) or reference.get("basis_verified") is not True):
        raise ValueError("available camera-floor reference requires head_reference.basis_verified=true")
    source["hands"] = {}
    for side in ("left", "right"):
        p = _reshape_feature(arrays, f"observation.state.hand_{side}_world", count, (21, 3)).astype(np.float64)
        valid = _mask(arrays, f"observation.valid.hand_{side}_position", count, 21) & np.isfinite(p).all(-1)
        interp = _mask(arrays, f"observation.interpolated.hand_{side}_position", count, 21)
        source["hands"][side] = {"position": p, "valid": valid, "interpolated": interp}
    return source


def _float32(value, mask, name):
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(value).astype(np.float32)
    if not np.isfinite(result[mask]).all():
        raise ValueError(f"{name}: valid source values cannot be represented as float32")
    return np.where(mask[..., None], result, 0.).astype(np.float32)


def encode_unified_dense22(arrays, metadata):
    """Return only v5's new columns, without touching inputs or image data.

    Accepts an episode, a row-group batch, or a selected/canonicalized window.
    The caller owns episode/frame identity checks and any floor resolution.
    """
    source = _inputs(arrays, metadata)
    count = source["count"]
    masks = _output_masks(source)
    safe_position = np.where(source["position_valid"][..., None], source["position"], 0.)
    safe_origin = np.where(source["floor_position_valid"][:, None], source["floor_position"], 0.)
    floor_rotation = source["floor_rotation"]
    local_position = np.einsum("tji,tkj->tki", floor_rotation, safe_position-safe_origin[:, None])
    local_rotation = np.swapaxes(floor_rotation, -1, -2)[:, None] @ source["rotation"]
    out = {}
    for frame, rotation, position in (("world", source["rotation"], safe_position),
                                      ("floor", local_rotation, local_position)):
        rvalid, pvalid = masks[(frame, "valid", "rotation")], masks[(frame, "valid", "position")]
        r6 = _float32(rotation[..., :2, :].reshape(count, 22, 6), rvalid, frame+" rotation")
        xyz = _float32(position, pvalid, frame+" position")
        out[f"observation.state.body_{frame}_rot6d_xyz"] = np.concatenate((r6, xyz), -1).reshape(count, 198)
    for (frame, kind, component), mask in masks.items():
        out[f"observation.{kind}.body_{frame}_rot6d_xyz_{component}"] = mask.copy()
    for side, hand in source["hands"].items():
        valid = hand["valid"] & source["floor_position_valid"][:, None] & source["floor_rotation_valid"][:, None]
        safe_hand = np.where(hand["valid"][..., None], hand["position"], 0.)
        local_hand = np.einsum("tji,tkj->tki", floor_rotation, safe_hand-safe_origin[:, None])
        out[f"observation.state.hand_{side}_floor"] = _float32(local_hand, valid, side+" hand").reshape(count, 63)
        out[f"observation.valid.hand_{side}_floor_position"] = valid
        out[f"observation.interpolated.hand_{side}_floor_position"] = (
            hand["interpolated"] | source["interpolated"]["floor_position"][:, None]
            | source["interpolated"]["floor_rotation"][:, None]) & valid
    return out


def _maximum(value):
    return float(np.max(np.abs(value))) if np.size(value) else 0.


def validate_unified_dense22(source_arrays, output_arrays, metadata, *, atol=2e-5):
    """Audit masks, analytic quaternion/6D round trips, and body/hand XYZ.

    ``output_arrays`` may contain the full archive or only the new fields.
    This checks the encoding, not anatomical axis calibration or wrist fit.
    """
    if not np.isfinite(atol) or atol <= 0:
        raise ValueError("validation atol must be finite and positive")
    source = _inputs(source_arrays, metadata)
    count = source["count"]
    expected = encode_unified_dense22(source_arrays, metadata)
    for key, spec in unified_specs().items():
        if key not in output_arrays:
            raise ValueError(f"missing unified dense22 column: {key}")
        actual = np.asarray(output_arrays[key])
        if actual.dtype != np.dtype(spec["dtype"]) or actual.shape != (count, spec["shape"][0]):
            raise ValueError(f"{key}: incorrect dtype/shape")
        if spec["dtype"] == "bool":
            if not np.array_equal(actual, expected[key]):
                raise ValueError(f"{key}: incorrect validity/interpolation mask")
        elif not np.isfinite(actual).all() or _maximum(actual-expected[key]) > atol:
            raise ValueError(f"{key}: incorrect derived state")
    world_rotation = _independent_quaternion_matrix(source["unit_q"])
    floor_rotation = _independent_quaternion_matrix(source["floor_unit_q"])
    report = {"schema_version": SCHEMA_VERSION, "rows": count, "all_masks_verified": True,
              "numerical_validation_only": True, "axis_certification": False, "atol": float(atol)}
    for frame, rotation in (("world", world_rotation),
                             ("floor", np.swapaxes(floor_rotation, -1, -2)[:, None] @ world_rotation)):
        block = np.asarray(output_arrays[f"observation.state.body_{frame}_rot6d_xyz"]).reshape(count, 22, 9)
        rvalid = expected[f"observation.valid.body_{frame}_rot6d_xyz_rotation"]
        pvalid = expected[f"observation.valid.body_{frame}_rot6d_xyz_position"]
        if np.any(block[..., :6][~rvalid] != 0) or np.any(block[..., 6:][~pvalid] != 0):
            raise ValueError(f"{frame}: invalid subblocks must be exactly zero")
        decoded = _decode_row6d(block[..., :6][rvalid].astype(np.float64))
        error = _maximum(decoded-rotation[rvalid])
        if error > atol or not _proper_so3(decoded, atol=max(atol, 1e-8)).all():
            raise ValueError(f"{frame}: quaternion/row6D mapping failed")
        xyz = block[..., 6:].astype(np.float64)
        if frame == "floor":
            xyz = np.einsum("tij,tkj->tki", floor_rotation, xyz)+source["floor_position"][:, None]
        inverse_error = _maximum(xyz[pvalid]-source["position"][pvalid])
        if inverse_error > 3*atol:
            raise ValueError(f"{frame}: inverse body XYZ mapping failed")
        if frame == "world" and not np.array_equal(block[..., 6:][pvalid], source["position"].astype(np.float32)[pvalid]):
            raise ValueError("valid world XYZ must be preserved exactly")
        report[frame] = {"rotation_roundtrip_max_abs": error, "position_inverse_max_abs_m": inverse_error,
                         "valid_rotation_count": int(rvalid.sum()), "valid_position_count": int(pvalid.sum())}
    for side, hand in source["hands"].items():
        valid = expected[f"observation.valid.hand_{side}_floor_position"]
        block = np.asarray(output_arrays[f"observation.state.hand_{side}_floor"]).reshape(count, 21, 3)
        if np.any(block[~valid] != 0):
            raise ValueError(f"{side}: invalid hand XYZ must be exactly zero")
        world = np.einsum("tij,tkj->tki", floor_rotation, block.astype(np.float64))+source["floor_position"][:, None]
        error = _maximum(world[valid]-hand["position"][valid])
        if error > 3*atol:
            raise ValueError(f"{side}: inverse hand XYZ mapping failed")
        report[f"hand_{side}"] = {"position_inverse_max_abs_m": error, "valid_position_count": int(valid.sum())}
    report["passed"] = True
    return report


def producer_signature():
    """Stable encoding signature; I/O-only changes do not alter the contract."""
    names = ("schema.py", "geometry.py", "native_dense22.py", "unified_dense22.py", "numeric_fingerprint.py")
    parent = Path(__file__).parent
    implementation = {name: hashlib.sha256((parent/name).read_bytes()).hexdigest() for name in names}
    payload = {"implementation": implementation, "encoding_policy": ENCODING_POLICY, "features": unified_specs()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def representation_provenance(arrays):
    """Per-episode semantic digest independent of later DINO extraction."""
    from .numeric_fingerprint import numeric_hashes, motion_digest_from_hashes, NON_MOTION_COLUMNS
    from .schema import feature_specs_for_schema
    # Resampling uses [T,1] masks; scalar Parquet fields use [T]. Hash the
    # physical layout so conversion and read-back produce the same digest.
    count = len(arrays["timestamp"])
    motion = {}
    for key, spec in feature_specs_for_schema(SCHEMA_VERSION).items():
        if key in NON_MOTION_COLUMNS:
            continue
        value = np.asarray(arrays[key], dtype=spec["dtype"])
        shape = (count,) if spec["shape"] == [1] else (count, *spec["shape"])
        motion[key] = value.reshape(shape)
    return {"schema_version": SCHEMA_VERSION, "producer_signature": producer_signature(),
            "motion_digest": motion_digest_from_hashes(numeric_hashes(motion))}
