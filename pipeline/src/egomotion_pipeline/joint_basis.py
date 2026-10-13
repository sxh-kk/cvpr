"""Estimate right-multiplied joint bases without claiming reference provenance.

All rotations use column vectors. A correction C maps target-local coordinates
to source-local coordinates, so the unified orientation is R_source @ C. These
functions establish geometric observability and numerical agreement only. A
trusted, synchronized reference and an independent provenance review are still
required before a dataset may mark its joint basis as verified.
"""

import hashlib
import json
from pathlib import Path
import re

import numpy as np

from .schema import BODY_NAMES


def validate_so3(rotations, *, atol=1e-5, name="rotation"):
    """Return float64 matrices, rejecting non-finite, scaled or reflected frames."""
    matrices = np.asarray(rotations, dtype=np.float64)
    if matrices.ndim < 2 or matrices.shape[-2:] != (3, 3):
        raise ValueError(f"{name} must have shape (..., 3, 3)")
    if not np.isfinite(atol) or atol <= 0:
        raise ValueError("atol must be finite and positive")
    if not np.isfinite(matrices).all():
        raise ValueError(f"{name} contains non-finite matrices")
    gram = np.swapaxes(matrices, -1, -2) @ matrices
    if (not np.allclose(gram, np.eye(3), atol=atol, rtol=0)
            or not np.allclose(np.linalg.det(matrices), 1., atol=atol, rtol=0)):
        raise ValueError(f"{name} must contain proper SO(3) rotations")
    return matrices


def rotation_geodesic_deg(source, target):
    """Geodesic angle in degrees for broadcastable proper rotation matrices."""
    source = validate_so3(source, name="source rotation")
    target = validate_so3(target, name="target rotation")
    relative = np.swapaxes(source, -1, -2) @ target
    cosine = (np.trace(relative, axis1=-2, axis2=-1) - 1.) / 2.
    return np.rad2deg(np.arccos(np.clip(cosine, -1., 1.)))


def _project_so3(matrix):
    left, singular_values, right = np.linalg.svd(matrix)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(left @ right)
    return left @ correction @ right, singular_values


def _direction_rank(vectors, weights, tolerance):
    singular_values = np.linalg.svd(vectors * np.sqrt(weights[:, None]),
                                    compute_uv=False)
    if not len(singular_values) or singular_values[0] == 0:
        return 0, np.zeros(3)
    rank = int(np.count_nonzero(singular_values > tolerance * singular_values[0]))
    return rank, np.pad(singular_values, (0, 3 - len(singular_values)))


def fit_direction_basis(source_vectors, target_vectors, weights=None,
                        rank_tolerance=.02):
    """Fit C with C @ target_direction ~= source_direction using weighted Wahba.

    Directions are normalized; weights express confidence rather than bone
    length. Zero-weight rows are excluded. Rank one determines two rotational
    degrees of freedom and leaves twist unobservable. Two non-collinear
    constraints determine all three degrees, but neither case authenticates
    the reference. Empty constraints return an unverified identity placeholder.
    """
    source = np.asarray(source_vectors, dtype=np.float64)
    target = np.asarray(target_vectors, dtype=np.float64)
    if source.ndim != 2 or source.shape[1:] != (3,) or target.shape != source.shape:
        raise ValueError("source_vectors and target_vectors must have equal (N, 3) shape")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("direction constraints must be finite")
    if not np.isfinite(rank_tolerance) or not 0 < rank_tolerance < 1:
        raise ValueError("rank_tolerance must be finite and between zero and one")
    weights = np.ones(len(source)) if weights is None else np.asarray(weights, dtype=np.float64)
    if weights.shape != (len(source),) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("weights must be finite, non-negative and have shape (N,)")
    active = weights > 0
    with np.errstate(over="ignore", invalid="ignore"):
        source_norm = np.linalg.norm(source[active], axis=-1)
        target_norm = np.linalg.norm(target[active], axis=-1)
    if (not np.isfinite(source_norm).all() or not np.isfinite(target_norm).all()
            or np.any(source_norm <= 1e-12) or np.any(target_norm <= 1e-12)):
        raise ValueError("positive-weight directions must have finite non-zero length")
    count = int(active.sum())
    errors = np.full(len(source), np.nan)
    if count:
        source_unit = source[active] / source_norm[:, None]
        target_unit = target[active] / target_norm[:, None]
        with np.errstate(over="ignore", invalid="ignore"):
            weight_sum = weights[active].sum()
        if not np.isfinite(weight_sum) or weight_sum < np.finfo(np.float64).tiny:
            raise ValueError("positive weights must have a finite, non-subnormal sum")
        active_weights = weights[active] / weight_sum
        if not np.isfinite(active_weights).all():
            raise ValueError("normalized weights must be finite")
        matrix, singular_values = _project_so3(
            source_unit.T @ (active_weights[:, None] * target_unit))
        source_rank, source_singular = _direction_rank(source_unit, active_weights, rank_tolerance)
        target_rank, target_singular = _direction_rank(target_unit, active_weights, rank_tolerance)
        predicted = target_unit @ matrix.T
        errors[active] = np.rad2deg(np.arccos(np.clip(
            np.sum(predicted * source_unit, axis=-1), -1., 1.)))
        rmse = float(np.sqrt(np.sum(active_weights * errors[active] ** 2)))
        maximum = float(errors[active].max())
    else:
        matrix, singular_values = np.eye(3), np.zeros(3)
        source_rank = target_rank = 0
        source_singular = target_singular = np.zeros(3)
        rmse = maximum = None
    rank = min(source_rank, target_rank)
    return {"matrix": matrix, "source_rank": source_rank, "target_rank": target_rank,
            "observable_dof": 3 if rank >= 2 else 2 if rank == 1 else 0,
            "singular_values": singular_values,
            "source_singular_values": source_singular,
            "target_singular_values": target_singular,
            "angular_errors_deg": errors, "weighted_rmse_deg": rmse,
            "max_error_deg": maximum, "constraint_count": count,
            "constraint_mask": active, "verified": False}


def _positive_sample_count(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _error_stats(errors):
    if not len(errors):
        return {"mean_deg": None, "rmse_deg": None, "p95_deg": None, "max_deg": None}
    return {"mean_deg": float(errors.mean()),
            "rmse_deg": float(np.sqrt(np.mean(errors ** 2))),
            "p95_deg": float(np.percentile(errors, 95)),
            "max_deg": float(errors.max())}


def fit_paired_rotation_basis(source_rotations, target_rotations, fit_mask,
                              validation_mask, *, max_error_deg=5.,
                              min_fit_samples=8, min_validation_samples=8):
    """Fit 22 constant bases from paired, synchronized same-world orientations.

    Both masks must be boolean arrays of shape (T, 22), with no overlap. Masked
    poses must be proper SO(3); unselected poses may be unavailable. The fit is
    the chordal SO(3) mean of R_source.T @ R_target. Numerical acceptance requires
    enough fit AND held-out samples and every selected sample within the error
    limit. ``verified`` remains false even when ``numerical_pass`` is true:
    numerical self-consistency cannot establish that a reference is authentic.
    """
    source = np.asarray(source_rotations, dtype=np.float64)
    target = np.asarray(target_rotations, dtype=np.float64)
    if (source.ndim != 4 or source.shape[1:] != (len(BODY_NAMES), 3, 3)
            or target.shape != source.shape):
        raise ValueError("paired rotations must have equal (T, 22, 3, 3) shape")
    fit = np.asarray(fit_mask)
    validation = np.asarray(validation_mask)
    expected = source.shape[:2]
    if (fit.shape != expected or validation.shape != expected
            or fit.dtype != np.bool_ or validation.dtype != np.bool_):
        raise ValueError("fit_mask and validation_mask must be boolean arrays of shape (T, 22)")
    if np.any(fit & validation):
        raise ValueError("fit and validation masks must not overlap")
    if not np.isfinite(max_error_deg) or not 0 <= max_error_deg <= 180:
        raise ValueError("max_error_deg must be finite and between zero and 180")
    min_fit_samples = _positive_sample_count(min_fit_samples, "min_fit_samples")
    min_validation_samples = _positive_sample_count(min_validation_samples, "min_validation_samples")
    selected = fit | validation
    validate_so3(source[selected], name="selected source rotations")
    validate_so3(target[selected], name="selected target rotations")
    matrices = np.repeat(np.eye(3)[None], len(BODY_NAMES), axis=0)
    numerical_pass = np.zeros(len(BODY_NAMES), dtype=bool)
    joints = []
    for joint, name in enumerate(BODY_NAMES):
        fit_count = int(fit[:, joint].sum())
        validation_count = int(validation[:, joint].sum())
        if fit_count:
            source_fit = source[fit[:, joint], joint]
            target_fit = target[fit[:, joint], joint]
            relative = np.swapaxes(source_fit, -1, -2) @ target_fit
            matrices[joint], singular_values = _project_so3(relative.mean(axis=0))
            fit_errors = rotation_geodesic_deg(source_fit @ matrices[joint], target_fit)
            source_val = source[validation[:, joint], joint]
            target_val = target[validation[:, joint], joint]
            validation_errors = rotation_geodesic_deg(source_val @ matrices[joint], target_val)
        else:
            singular_values = np.zeros(3)
            fit_errors = validation_errors = np.empty(0)
        fit_stats, validation_stats = _error_stats(fit_errors), _error_stats(validation_errors)
        numerical_pass[joint] = (
            fit_count >= min_fit_samples and validation_count >= min_validation_samples
            and fit_stats["max_deg"] <= max_error_deg
            and validation_stats["max_deg"] <= max_error_deg)
        joints.append({"joint_index": joint, "joint_name": name,
                       "fit_count": fit_count, "validation_count": validation_count,
                       "candidate_available": bool(fit_count),
                       "singular_values": singular_values,
                       "fit_errors_deg": fit_errors, "validation_errors_deg": validation_errors,
                       "fit_error": fit_stats, "validation_error": validation_stats,
                       "numerical_pass": bool(numerical_pass[joint]), "verified": False})
    return {"joint_names": list(BODY_NAMES), "matrices": matrices, "joints": joints,
            "numerical_pass": numerical_pass, "verified": np.zeros(len(BODY_NAMES), dtype=bool),
            "reference_verified": False, "max_error_deg": float(max_error_deg),
            "min_fit_samples": min_fit_samples, "min_validation_samples": min_validation_samples,
            "convention": "target_world_rotation = source_world_rotation @ correction"}


def _strict_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value):
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def _all_verified_flags(value, name):
    if (not isinstance(value, list) or len(value) != len(BODY_NAMES)
            or not all(type(flag) is bool for flag in value)
            or not all(value)):
        raise ValueError(f"{name} must contain exactly 22 boolean true values")


def _hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_verified_joint_basis(path, *, target_basis="smpl_body22_axes_v1"):
    """Load a previously reviewed, fully verified calibration and its provenance.

    This is a strict artifact consumer, not a supplier/reference authenticator.
    It rejects candidates and missing or changed reference evidence. JSON is
    parsed without pickle, duplicate keys and non-standard non-finite constants
    are rejected, and evidence files are hashed without deserializing them.
    Relative evidence paths are resolved against the artifact directory.
    """
    artifact_path = Path(path).expanduser().resolve()
    raw = artifact_path.read_bytes()
    try:
        artifact = json.loads(raw, object_pairs_hook=_strict_json_object,
                              parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("joint basis artifact must be valid UTF JSON") from error
    if not isinstance(artifact, dict):
        raise ValueError("joint basis artifact must be a JSON object")
    schema = "egomotion-joint-basis-calibration-v1"
    if artifact.get("schema") != schema:
        raise ValueError(f"joint basis artifact schema must be {schema}")
    if artifact.get("joint_names") != BODY_NAMES:
        raise ValueError("joint_names must exactly match the ordered dense22 BODY_NAMES")
    if not isinstance(target_basis, str) or not target_basis or artifact.get("target_basis") != target_basis:
        raise ValueError("joint basis artifact target_basis does not match the requested basis")
    convention = "target_world_rotation = source_world_rotation @ correction"
    if artifact.get("convention") != convention:
        raise ValueError("joint basis artifact convention must specify right-multiplied correction")
    _all_verified_flags(artifact.get("verified"), "verified")
    _all_verified_flags(artifact.get("numerical_pass"), "numerical_pass")
    if artifact.get("reference_verified") is not True:
        raise ValueError("reference_verified must be boolean true following independent review")
    reference_kind = artifact.get("reference_kind")
    if (not isinstance(reference_kind, str)
            or reference_kind not in {"supplier_bind_pose", "independent_paired_orientations"}):
        raise ValueError("reference_kind must describe supplier bind pose or independent paired orientations")
    try:
        matrices = np.asarray(artifact.get("correction_matrices"), dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("correction_matrices must be numeric (22, 3, 3)") from error
    if matrices.shape != (len(BODY_NAMES), 3, 3):
        raise ValueError("correction_matrices must have shape (22, 3, 3)")
    validate_so3(matrices, name="correction_matrices")
    references = artifact.get("reference_files")
    if not isinstance(references, list) or not references:
        raise ValueError("reference_files must contain non-empty reference evidence")
    provenance_files = []
    seen = set()
    for reference in references:
        if not isinstance(reference, dict):
            raise ValueError("each reference_files entry must be an object with path and sha256")
        reference_path = reference.get("path")
        expected_hash = reference.get("sha256")
        if not isinstance(reference_path, str) or not reference_path.strip():
            raise ValueError("each reference file path must be a non-empty string")
        if not isinstance(expected_hash, str) or re.fullmatch(r"[0-9a-fA-F]{64}", expected_hash) is None:
            raise ValueError("each reference file sha256 must be a 64-character hexadecimal digest")
        evidence = Path(reference_path).expanduser()
        if not evidence.is_absolute():
            evidence = artifact_path.parent / evidence
        evidence = evidence.resolve()
        if evidence in seen:
            raise ValueError("reference_files must not repeat the same evidence path")
        seen.add(evidence)
        if not evidence.is_file():
            raise ValueError(f"reference evidence file is missing: {evidence}")
        actual_hash = _hash_file(evidence)
        if actual_hash != expected_hash.lower():
            raise ValueError(f"reference evidence sha256 mismatch: {evidence}")
        provenance_files.append({"path": str(evidence), "sha256": actual_hash})
    return {"matrices": matrices,
            "provenance": {"artifact_path": str(artifact_path),
                           "artifact_sha256": hashlib.sha256(raw).hexdigest(),
                           "schema": schema, "joint_names": list(BODY_NAMES),
                           "target_basis": target_basis, "convention": convention,
                           "reference_kind": reference_kind,
                           "reference_verified": True, "reference_files": provenance_files}}
