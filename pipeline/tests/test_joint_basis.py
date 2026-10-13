import hashlib
import json

import numpy as np
import pytest

from egomotion_pipeline.geometry import axis_angle_to_quat, quat_to_matrix
from egomotion_pipeline.joint_basis import (
    fit_direction_basis, fit_paired_rotation_basis, load_verified_joint_basis, rotation_geodesic_deg,
    validate_so3,
)
from egomotion_pipeline.schema import BODY_NAMES


def rotations(axis_angles):
    return quat_to_matrix(axis_angle_to_quat(np.asarray(axis_angles, dtype=float)))


def paired_fixture(count=40):
    generator = np.random.default_rng(721)
    source = rotations(generator.normal(size=(count, 22, 3)))
    corrections = rotations(generator.normal(size=(22, 3)) * .7)
    target = source @ corrections
    fit = np.zeros((count, 22), bool)
    fit[:count // 2] = True
    return source, target, corrections, fit, ~fit


def test_recovers_22_different_right_multiplied_corrections():
    source, target, corrections, fit, holdout = paired_fixture()
    result = fit_paired_rotation_basis(source, target, fit, holdout)
    np.testing.assert_allclose(result["matrices"], corrections, atol=1e-12)
    assert result["numerical_pass"].all()
    assert result["joint_names"] == BODY_NAMES
    assert all(j["fit_count"] == j["validation_count"] == 20 for j in result["joints"])
    assert all(j["validation_error"]["max_deg"] < 1e-4 for j in result["joints"])
    assert not result["verified"].any()
    assert result["reference_verified"] is False


def test_common_world_transform_cancels_per_frame():
    source, target, corrections, fit, holdout = paired_fixture()
    world = rotations(np.random.default_rng(73).normal(size=(len(source), 3)))[:, None]
    result = fit_paired_rotation_basis(world @ source, world @ target, fit, holdout)
    np.testing.assert_allclose(result["matrices"], corrections, atol=1e-12)
    assert result["numerical_pass"].all()


def test_left_multiplication_is_not_a_constant_joint_basis():
    source, _, corrections, fit, holdout = paired_fixture()
    left_multiplied_target = corrections @ source
    result = fit_paired_rotation_basis(source, left_multiplied_target, fit, holdout,
                                       max_error_deg=1.)
    assert not result["numerical_pass"].any()
    assert all(j["validation_error"]["max_deg"] > 1. for j in result["joints"])


def test_single_bone_does_not_determine_twist():
    target = np.tile([1., 0., 0.], (10, 1))
    known = rotations([.5, -.7, .2])
    source = target @ known.T
    result = fit_direction_basis(source, target)
    assert result["source_rank"] == result["target_rank"] == 1
    assert result["observable_dof"] == 2
    assert result["verified"] is False
    assert result["max_error_deg"] < 1e-4
    arbitrary_twist = rotations([1.1, 0., 0.])
    np.testing.assert_allclose(target @ (known @ arbitrary_twist).T, source, atol=1e-12)
    assert rotation_geodesic_deg(known, known @ arbitrary_twist) > 60.


def test_non_collinear_directions_determine_full_basis():
    target = np.array([[1., 0., 0.], [0., 1., 0.], [-1., 0., 0.]])
    known = rotations([.9, -.2, .7])
    source = target @ known.T
    result = fit_direction_basis(source, target, weights=[1., 3., 1.])
    assert result["source_rank"] == result["target_rank"] == 2
    assert result["observable_dof"] == 3
    np.testing.assert_allclose(result["matrix"], known, atol=1e-12)
    assert result["verified"] is False


def test_rank_tolerance_rejects_nearly_collinear_constraints():
    directions = np.array([[1., 0., 0.], [1., .001, 0.]])
    result = fit_direction_basis(directions, directions)
    assert result["source_rank"] == result["target_rank"] == 1
    assert result["observable_dof"] == 2


@pytest.mark.parametrize("empty, weights", [(np.empty((0, 3)), None), (np.zeros((2, 3)), [0., 0.])])
def test_empty_direction_constraints_remain_unverified(empty, weights):
    result = fit_direction_basis(empty, empty, weights)
    np.testing.assert_array_equal(result["matrix"], np.eye(3))
    assert result["observable_dof"] == result["constraint_count"] == 0
    assert result["weighted_rmse_deg"] is None
    assert result["verified"] is False


@pytest.mark.parametrize("matrix", [np.diag([1., 1., -1.]), np.eye(3) * 2.,
                                     np.full((3, 3), np.nan), np.full((3, 3), np.inf)])
def test_invalid_rotation_frames_are_rejected(matrix):
    with pytest.raises(ValueError):
        validate_so3(matrix)
    with pytest.raises(ValueError):
        rotation_geodesic_deg(matrix, np.eye(3))
    source, target, _, fit, holdout = paired_fixture()
    source[0, 0] = matrix
    with pytest.raises(ValueError):
        fit_paired_rotation_basis(source, target, fit, holdout)


def test_geodesic_rotation_angle_and_broadcast():
    target = rotations([[0., 0., 0.], [0., 0., np.pi / 2], [np.pi, 0., 0.]])
    np.testing.assert_allclose(rotation_geodesic_deg(np.eye(3), target), [0., 90., 180.], atol=1e-7)


def test_independent_holdout_errors_reject_fit_only_success():
    source, target, _, fit, holdout = paired_fixture()
    target[holdout[:, 5], 5] = target[holdout[:, 5], 5] @ rotations([0., 0., .2])
    result = fit_paired_rotation_basis(source, target, fit, holdout, max_error_deg=2.)
    assert not result["numerical_pass"][5]
    assert result["numerical_pass"].sum() == 21
    assert result["joints"][5]["fit_error"]["max_deg"] < 1e-4
    assert result["joints"][5]["validation_error"]["max_deg"] > 10.


def test_partial_and_missing_joints_never_gain_verified_mask():
    source, target, _, fit, holdout = paired_fixture()
    fit[:, 0] = holdout[:, 0] = False
    source[:, 0] = target[:, 0] = np.nan
    fit[1:, 1] = False
    holdout[:, 2] = False
    result = fit_paired_rotation_basis(source, target, fit, holdout)
    assert result["numerical_pass"].sum() == 19
    assert not result["verified"].any()
    assert result["joints"][0]["candidate_available"] is False
    assert result["joints"][0]["fit_error"]["max_deg"] is None
    assert result["joints"][1]["fit_count"] == 1
    assert result["joints"][2]["validation_count"] == 0


def test_no_paired_constraints_returns_no_verified_joints():
    unavailable = np.full((3, 22, 3, 3), np.nan)
    empty_mask = np.zeros((3, 22), bool)
    result = fit_paired_rotation_basis(unavailable, unavailable, empty_mask, empty_mask)
    assert not result["numerical_pass"].any()
    assert not result["verified"].any()
    assert not any(j["candidate_available"] for j in result["joints"])


def test_fit_and_holdout_must_not_overlap():
    source, target, _, fit, holdout = paired_fixture()
    holdout[0, 8] = True
    with pytest.raises(ValueError, match="must not overlap"):
        fit_paired_rotation_basis(source, target, fit, holdout)


@pytest.mark.parametrize("case", ["flat", "float", "wrong_joint_count"])
def test_masks_are_strictly_per_frame_per_joint_boolean(case):
    source, target, _, fit, holdout = paired_fixture()
    if case == "flat":
        fit = fit[:, 0]
    elif case == "float":
        fit = fit.astype(float)
    else:
        fit = fit[:, :-1]
    with pytest.raises(ValueError, match="boolean arrays"):
        fit_paired_rotation_basis(source, target, fit, holdout)


@pytest.mark.parametrize("option, value", [("min_fit_samples", 0), ("min_fit_samples", True),
                                          ("min_validation_samples", 1.5),
                                          ("max_error_deg", np.nan)])
def test_acceptance_parameters_are_validated(option, value):
    source, target, _, fit, holdout = paired_fixture()
    with pytest.raises(ValueError):
        fit_paired_rotation_basis(source, target, fit, holdout, **{option: value})


@pytest.mark.parametrize("source, target, weights", [
    ([[0., 0., 0.]], [[1., 0., 0.]], None),
    ([[np.nan, 0., 0.]], [[1., 0., 0.]], None),
    ([[1., 0., 0.]], [[1., 0., 0.]], [-1.]),
    ([[1., 0., 0.]], [[1., 0., 0.]], [np.inf]),
    ([[1., 0., 0.]], [[1., 0., 0.]], [[1.]]),
])
def test_invalid_direction_constraints_are_rejected(source, target, weights):
    with pytest.raises(ValueError):
        fit_direction_basis(source, target, weights)


@pytest.mark.parametrize("weights", [[1e308, 1e308], [1e-320, 1e-320]])
def test_non_normalizable_weight_sums_are_rejected(weights):
    directions = np.array([[1., 0., 0.], [0., 1., 0.]])
    with pytest.raises(ValueError, match="finite, non-subnormal sum"):
        fit_direction_basis(directions, directions, weights)


def test_direction_norm_overflow_is_rejected():
    with pytest.raises(ValueError, match="finite non-zero length"):
        fit_direction_basis([[1e308, 0., 0.]], [[1., 0., 0.]])


@pytest.fixture
def synthetic_calibration(tmp_path):
    # Flags only simulate an artifact already reviewed by its owner. This
    # fixture provides no scientific evidence about actual EgoSuite axes.
    evidence = tmp_path / "synthetic_reference.json"
    evidence.write_text('{"purpose": "synthetic loader test, not an EgoSuite calibration"}')
    _, _, matrices, _, _ = paired_fixture()
    artifact = {"schema": "egomotion-joint-basis-calibration-v1",
                "joint_names": list(BODY_NAMES),
                "target_basis": "smpl_body22_axes_v1",
                "convention": "target_world_rotation = source_world_rotation @ correction",
                "correction_matrices": matrices.tolist(),
                "verified": [True] * 22, "numerical_pass": [True] * 22,
                "reference_verified": True,
                "reference_kind": "independent_paired_orientations",
                "reference_files": [{"path": evidence.name,
                                     "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest()}],
                "synthetic": True}
    path = tmp_path / "synthetic_calibration.json"
    path.write_text(json.dumps(artifact))
    return path, artifact, evidence, matrices


def test_loader_accepts_synthetic_reviewed_artifact_and_records_provenance(synthetic_calibration):
    path, artifact, evidence, expected = synthetic_calibration
    result = load_verified_joint_basis(path)
    np.testing.assert_allclose(result["matrices"], expected, atol=0)
    assert result["provenance"]["artifact_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result["provenance"]["reference_files"][0]["path"] == str(evidence.resolve())
    assert result["provenance"]["reference_kind"] == artifact["reference_kind"]
    assert result["provenance"]["convention"] == artifact["convention"]


@pytest.mark.parametrize("field, value", [
    ("verified", [False] * 22),
    ("verified", [True] * 21),
    ("verified", [1] * 22),
    ("numerical_pass", [False] * 22),
    ("numerical_pass", None),
    ("reference_verified", False),
    ("reference_verified", 1),
    ("reference_kind", "estimated"),
    ("reference_kind", "position_only"),
    ("reference_kind", ["independent_paired_orientations"]),
    ("target_basis", "another_basis"),
    ("convention", "target_world_rotation = correction @ source_world_rotation"),
    ("schema", "candidate_calibration"),
    ("joint_names", list(reversed(BODY_NAMES))),
    ("reference_files", []),
    ("reference_files", [{"path": "missing", "sha256": "0" * 64}]),
    ("reference_files", [{"path": "synthetic_reference.json", "sha256": "0" * 64}]),
    ("reference_files", [{"path": "synthetic_reference.json", "sha256": "invalid"}]),
])
def test_loader_rejects_candidates_and_incomplete_or_mismatched_evidence(synthetic_calibration, field, value):
    path, artifact, _, _ = synthetic_calibration
    artifact[field] = value
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError):
        load_verified_joint_basis(path)


@pytest.mark.parametrize("matrix", [np.diag([1., 1., -1.]), np.eye(3) * 2., np.full((3, 3), np.nan)])
def test_loader_rejects_improper_or_nonfinite_corrections(synthetic_calibration, matrix):
    path, artifact, _, _ = synthetic_calibration
    artifact["correction_matrices"][8] = matrix.tolist()
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError):
        load_verified_joint_basis(path)


def test_loader_rejects_changed_evidence_without_loading_it(synthetic_calibration):
    path, _, evidence, _ = synthetic_calibration
    evidence.write_bytes(b"this could be an unsafe serialized object, but is only hashed")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        load_verified_joint_basis(path)


def test_loader_rejects_deleted_validation_field(synthetic_calibration):
    path, artifact, _, _ = synthetic_calibration
    del artifact["numerical_pass"]
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="numerical_pass"):
        load_verified_joint_basis(path)


def test_loader_rejects_missing_multiplication_convention(synthetic_calibration):
    path, artifact, _, _ = synthetic_calibration
    del artifact["convention"]
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="right-multiplied"):
        load_verified_joint_basis(path)


def test_loader_rejects_wrong_requested_target(synthetic_calibration):
    path, _, _, _ = synthetic_calibration
    with pytest.raises(ValueError, match="target_basis"):
        load_verified_joint_basis(path, target_basis="unexpected")


def test_loader_rejects_duplicate_json_keys(synthetic_calibration):
    path, artifact, _, _ = synthetic_calibration
    raw = json.dumps(artifact)
    path.write_text(raw[:-1] + ', "reference_verified": false}')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_verified_joint_basis(path)


def test_loader_accepts_supplier_bind_pose_kind(synthetic_calibration):
    path, artifact, _, _ = synthetic_calibration
    artifact["reference_kind"] = "supplier_bind_pose"
    path.write_text(json.dumps(artifact))
    assert load_verified_joint_basis(path)["provenance"]["reference_kind"] == "supplier_bind_pose"


def test_loader_rejects_duplicate_evidence_paths(synthetic_calibration):
    path, artifact, _, _ = synthetic_calibration
    artifact["reference_files"] *= 2
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="repeat"):
        load_verified_joint_basis(path)
