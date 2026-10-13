"""Independent offset observation, isolation, and SMPL-H contract checks."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from egomotion_pipeline.schema import PARENTS
from egomotion_pipeline.smpl_fit import FitConfig, load_body22_model
from egomotion_pipeline.smplh_retarget import (
    CenterCalibration, RetargetConfig, calibrate_center_offsets,
    fit_safe_limb_betas, fit_smplh_retarget, load_smplh_body22_model,
    pool_center_calibrations, reconstruct_source_landmarks,
)


def _rotation_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def _numpy_fk(rest, local, root):
    p, g = np.empty((len(local), 22, 3)), np.empty_like(local)
    p[:, 0], g[:, 0] = root, local[:, 0]
    for joint in range(1, 22):
        parent = PARENTS[joint]
        g[:, joint] = g[:, parent] @ local[:, joint]
        p[:, joint] = p[:, parent] + np.einsum("tij,j->ti", g[:, parent], rest[joint] - rest[parent])
    return p, g


@pytest.fixture
def model(model_path):
    return load_smplh_body22_model(model_path, shape_components=2, allow_test_fixture=True)


def _scene(model, *, phase=0., offsets=None):
    count, betas = 14, np.asarray([.25, -.15])
    local = np.broadcast_to(np.eye(3), (count, 22, 3, 3)).copy()
    for frame in range(count):
        local[frame, 0] = _rotation_z(phase + .06 * frame)
        local[frame, 12] = _rotation_z(.03 * frame)
    root = np.stack([np.linspace(.1, .2, count), np.linspace(-.2, -.1, count), np.ones(count)], axis=-1)
    rest = model.rest_joints(torch.tensor(betas)).detach().numpy()
    p, g = _numpy_fk(rest, local, root)
    anchors = np.arange(22)
    anchors[[12, 13, 14, 15]] = [9, 9, 9, 12]
    if offsets is None:
        offsets = np.zeros((22, 3))
        offsets[12] = [.025, -.01, .04]
        offsets[13] = [-.03, .015, .02]
        offsets[14] = [.03, -.015, .02]
        offsets[15] = [.01, .02, -.045]
    # Explicit per-frame NumPy mapping is independent of the implementation.
    observed = p.copy()
    for frame in range(count):
        for joint in range(22):
            observed[frame, joint] += g[frame, anchors[joint]] @ offsets[joint]
    # Native joint axes are intentionally different from the target rig.
    corrections = np.stack([_rotation_z(.013 * joint) for joint in range(22)])
    native = g @ corrections
    return dict(positions=observed, position_mask=np.ones((count, 22), bool),
                timestamps=np.arange(count) / 15., source_global_rotations=native,
                rotation_mask=np.ones((count, 22), bool), fixed_betas=betas,
                initialization={"local_rotations": local, "root_world": root,
                                "betas": betas.copy(), "metadata": {"method": "synthetic_known_pose"}}), p, g, offsets, anchors


def _config(**changes):
    return RetargetConfig(pose_iterations=0, offset_regularization=0., **changes)


def test_fixture_requires_explicit_flag_and_is_never_labelled_real_smplh(model_path):
    with pytest.raises(ValueError, match="Required SMPL-H"):
        load_smplh_body22_model(model_path, shape_components=2)
    model = load_smplh_body22_model(model_path, shape_components=2, allow_test_fixture=True)
    assert model.metadata["test_fixture"] is True
    assert model.metadata["required_layout_verified"] is False
    assert model.metadata["model_kind"] == "body22_fixture"
    assert model.metadata["target_joint_basis"] == "synthetic_body22_fixture"


def test_generic_model_cannot_bypass_required_smplh_contract(model_path):
    model = load_body22_model(model_path, shape_components=2)
    data, *_ = _scene(model)
    with pytest.raises(ValueError, match="explicit layout contract"):
        calibrate_center_offsets(model, **data, config=_config())


def _full_asset(path, *, bad_last_parent=False, tree_count=52):
    template = np.zeros((6890, 3))
    template[:22, 1] = np.arange(22) * .05
    regressor = np.zeros((52, 6890))
    regressor[np.arange(52), np.arange(52)] = 1.
    parents = PARENTS.tolist()
    for wrist in (20, 21):
        for _ in range(5):
            first = len(parents)
            parents += [wrist, first, first + 1]
    if bad_last_parent:
        parents[-1] = 0
    np.savez(path, v_template=template, shapedirs=np.zeros((6890, 3, 16)),
             J_regressor=regressor,
             kintree_table=np.array([parents[:tree_count], np.arange(tree_count)]))


def test_full_layout_checks_all_52_parents_and_16_actual_shape_components(tmp_path):
    path = tmp_path / "synthetic_full_layout.npz"
    _full_asset(path)
    model = load_smplh_body22_model(path)
    assert model.shape_components == 16
    assert model.metadata["asset_joint_count"] == 52
    assert model.metadata["implicit_shape_padding"] is False
    assert model.metadata["asset_authenticity_verified"] is False
    _full_asset(path, bad_last_parent=True)
    with pytest.raises(ValueError, match="full kinematic hierarchy"):
        load_smplh_body22_model(path)
    _full_asset(path, tree_count=51)
    with pytest.raises(ValueError, match="52 regressed/tree"):
        load_smplh_body22_model(path)


def test_model_layout_validation_rejects_asset_replacement(model_path, monkeypatch):
    import egomotion_pipeline.smplh_retarget as module
    original = module.load_body22_model

    def mismatched(*args, **kwargs):
        model = original(*args, **kwargs)
        model.metadata["model_sha256"] = "a" * 64
        return model

    monkeypatch.setattr(module, "load_body22_model", mismatched)
    with pytest.raises(ValueError, match="changed"):
        module.load_smplh_body22_model(model_path, shape_components=2, allow_test_fixture=True)


def test_parent_offsets_reconstruct_rotating_landmarks_without_changing_fk_centers(model):
    data, p, g, offsets, anchors = _scene(model)
    fit = calibrate_center_offsets(model, **data, config=_config())
    np.testing.assert_allclose(fit.center_calibration.offsets, offsets, atol=1e-14)
    np.testing.assert_array_equal(fit.center_calibration.anchors, anchors)
    np.testing.assert_allclose(fit.positions, p, atol=1e-14)
    np.testing.assert_allclose(fit.global_rotations, g, atol=1e-14)
    np.testing.assert_allclose(fit.reconstructed_observations, data["positions"], atol=1e-14)
    np.testing.assert_allclose(reconstruct_source_landmarks(p, g, offsets, anchors), data["positions"], atol=1e-14)
    assert fit.metadata["model_center_to_source_landmark_distance_m"]["mean"] > .005
    assert fit.metadata["observation_error_m"]["max"] < 1e-14
    assert fit.metadata["rotation_increment_error_deg"]["max"] < 5e-5
    assert fit.metadata["native_axis_certified"] is False
    assert fit.metadata["absolute_rotation_accuracy_established"] is False
    assert fit.metadata["target_model_axes_defined"] is True
    assert fit.metadata["label_kind"] == "model_fitted_estimate"


def test_heldout_freezes_calibration_beta_offsets_and_uses_fair_baseline_initialization(model):
    calibration_data, *_ = _scene(model)
    calibration = calibrate_center_offsets(model, **calibration_data, config=_config()).center_calibration
    data, p, _, offsets, _ = _scene(model, phase=.5)
    original_offsets, original_betas = calibration.offsets.copy(), calibration.betas.copy()
    fitted = fit_smplh_retarget(model, **data, center_calibration=calibration, config=_config())
    baseline = fit_smplh_retarget(model, **data, config=_config())
    assert fitted.metadata["initialization_sha256"] == baseline.metadata["initialization_sha256"]
    assert fitted.metadata["center_offsets_frozen"] is True
    assert fitted.metadata["center_offsets_optimized"] is False
    np.testing.assert_array_equal(calibration.offsets, original_offsets)
    np.testing.assert_array_equal(calibration.betas, original_betas)
    np.testing.assert_allclose(fitted.positions, p, atol=1e-14)
    assert fitted.metadata["observation_error_m"]["mean"] < 1e-14
    assert baseline.metadata["observation_error_m"]["mean"] > .005
    data["positions"][:, 12] += [1., -2., 3.]
    contaminated = fit_smplh_retarget(model, **data, center_calibration=calibration, config=_config())
    np.testing.assert_array_equal(contaminated.center_calibration.offsets, original_offsets)
    with pytest.raises(ValueError):
        calibration.offsets[12, 0] = 99.
    with pytest.raises(ValueError, match="betas must match"):
        fit_smplh_retarget(model, **{**data, "fixed_betas": [0., 0.]}, center_calibration=calibration, config=_config())


def test_offset_bounds_apply_at_initialization_and_are_reported(model):
    data, _, _, offsets, _ = _scene(model)
    offsets[12] *= 10
    data, *_ = _scene(model, offsets=offsets)
    fit = calibrate_center_offsets(model, **data, config=_config(offset_bound_m=.05))
    assert np.linalg.norm(fit.center_calibration.offsets[12]) == pytest.approx(.05)
    assert fit.metadata["offset_diagnostics"]["offset_bound_saturated"][12] is True
    assert fit.metadata["offset_diagnostics"]["offset_bound_saturated_count"] >= 1
    assert fit.metadata["offset_diagnostics"]["offset_regularization"] == 0.


def test_no_unavailable_or_pelvis_offsets_are_silently_calibrated(model):
    data, *_ = _scene(model)
    data["position_mask"][:, 15] = False
    data["positions"][:, 15] = np.nan
    fit = calibrate_center_offsets(model, **data, config=_config())
    assert fit.center_calibration.offset_valid[15] == np.bool_(False)
    np.testing.assert_array_equal(fit.center_calibration.offsets[15], 0.)
    data["position_mask"][:, [12, 13, 14]] = False
    with pytest.raises(ValueError, match="enough paired"):
        calibrate_center_offsets(model, **data, config=_config())
    with pytest.raises(ValueError, match="Pelvis offsets"):
        replace(_config(), offset_joints=(0,), offset_anchors=(0,)).validate()


def test_pooling_has_equal_person_weights_dispersion_and_distinct_group_contract(model):
    data, _, _, offsets, _ = _scene(model)
    first = calibrate_center_offsets(model, **data, config=_config()).center_calibration
    changed = offsets.copy()
    changed[12] += [.008, 0., 0.]
    data, *_ = _scene(model, offsets=changed)
    second = calibrate_center_offsets(model, **data, config=_config()).center_calibration
    pooled = pool_center_calibrations([first, second], ["operator-a", "operator-b"])
    np.testing.assert_allclose(pooled.offsets, (offsets + changed) / 2, atol=1e-14)
    assert pooled.metadata["per_joint_offset_scatter_rms_m"][12] == pytest.approx(.004)
    assert pooled.metadata["per_joint_calibration_groups"][12] == 2
    with pytest.raises(ValueError, match="distinct"):
        pool_center_calibrations([first, second], ["a", "a"])
    incompatible = CenterCalibration(second.offsets, second.anchors, second.offset_valid,
                                     second.betas + .01, second.model_sha256, second.input_sha256)
    with pytest.raises(ValueError, match="betas"):
        pool_center_calibrations([first, incompatible], ["a", "b"])


def test_invalid_values_gaps_and_resets_never_create_rotation_or_temporal_edges(model):
    data, *_ = _scene(model)
    data["timestamps"][7:] += 1.
    data["position_mask"][3] = False
    data["rotation_mask"][3] = False
    data["positions"][3] = np.nan
    data["source_global_rotations"][3] = np.nan
    reset_mask = np.ones(len(data["timestamps"]) - 1, dtype=bool)
    reset_mask[9] = False
    fit = fit_smplh_retarget(model, **data, transition_mask=reset_mask, config=_config())
    expected = np.ones_like(fit.rotation_increment_valid)
    expected[[2, 3, 6, 9]] = False
    np.testing.assert_array_equal(fit.rotation_increment_valid, expected)
    assert fit.metadata["rotation_increment_constraints"] == int(expected.sum())
    assert fit.metadata["position_neighbor_constraints"] == int(expected.sum())
    assert not fit.estimate_available[3].any()
    data["positions"][3] = np.inf
    data["source_global_rotations"][3] = -np.inf
    refit = fit_smplh_retarget(model, **data, transition_mask=reset_mask, config=_config())
    np.testing.assert_array_equal(refit.positions, fit.positions)
    assert refit.metadata["input_sha256"] == fit.metadata["input_sha256"]


def test_isolated_selected_bad_rotation_and_nonmonotonic_time_are_rejected(model):
    data, *_ = _scene(model)
    data["rotation_mask"][:] = False
    data["rotation_mask"][4, 12] = True
    data["source_global_rotations"][4, 12] = np.diag([-1., 1., 1.])
    with pytest.raises(ValueError, match="proper SO"):
        fit_smplh_retarget(model, **data, config=_config())
    data["rotation_mask"][:] = False
    data["timestamps"][3] = data["timestamps"][2]
    with pytest.raises(ValueError, match="strictly increasing"):
        fit_smplh_retarget(model, **data, config=_config())


def test_joint_center_objective_optimizes_without_changing_fixed_beta(model):
    data, *_ = _scene(model)
    data["initialization"]["root_world"] += [.015, -.02, .01]
    config = RetargetConfig(pose_iterations=75, offset_regularization=.001,
                            pose_lr=.005, root_lr=.004, offset_lr=.003)
    fit = calibrate_center_offsets(model, **data, config=config)
    assert fit.metadata["final"]["loss"] < fit.history[0]["loss"] * .2
    assert fit.metadata["observation_error_m"]["mean"] < .003
    np.testing.assert_array_equal(fit.betas, data["fixed_betas"])
    assert fit.metadata["shape_optimized"] is False
    assert np.max(np.linalg.norm(fit.center_calibration.offsets, axis=-1)) <= config.offset_bound_m + 1e-12


def test_absolute_leaf_twist_remains_unobservable_even_with_parent_landmark_offsets(model):
    data, *_ = _scene(model)
    first = calibrate_center_offsets(model, **data, config=_config())
    data["initialization"]["local_rotations"][:, 15] @= _rotation_z(np.pi / 2)
    second = calibrate_center_offsets(model, **data, config=_config())
    np.testing.assert_allclose(first.positions, second.positions, atol=1e-14)
    np.testing.assert_allclose(first.reconstructed_observations, second.reconstructed_observations, atol=1e-14)
    assert first.metadata["final"]["position_loss"] == pytest.approx(second.metadata["final"]["position_loss"], abs=1e-20)
    assert first.metadata["final"]["rotation_increment_loss"] == pytest.approx(second.metadata["final"]["rotation_increment_loss"], abs=1e-20)
    relative = np.swapaxes(first.global_rotations[:, 15], -1, -2) @ second.global_rotations[:, 15]
    angles = np.rad2deg(np.arccos(np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))
    np.testing.assert_allclose(angles, 90., atol=1e-10)
    assert second.metadata["absolute_rotation_accuracy_established"] is False


def test_safe_limb_shape_excludes_center_mismatched_torso_bones(model):
    data, *_ = _scene(model)
    result = fit_safe_limb_betas(model, data["positions"], data["position_mask"], config=FitConfig(shape_iterations=0))
    assert result["metadata"]["bone_constraints"] == len(data["positions"]) * 10
    assert result["metadata"]["torso_head_centers_used"] is False
    distorted = data["positions"].copy()
    distorted[:, [0, 3, 6, 9, 12, 13, 14, 15]] += 100.
    refit = fit_safe_limb_betas(model, distorted, data["position_mask"], config=FitConfig(shape_iterations=0))
    np.testing.assert_array_equal(result["betas"], refit["betas"])
    assert result["metadata"]["final"]["bone_loss"] == refit["metadata"]["final"]["bone_loss"]


@pytest.mark.skipif(not Path("/gaozt-test1/datasets/MixEgoMotion/weights/smplh/male/model.npz").is_file(),
                    reason="licensed user-owned SMPL-H asset is not installed")
def test_real_smplh_16beta_differentiable_fk_matches_production_decoder():
    from egomotion_pipeline.body_model import BodyModel
    from egomotion_pipeline.geometry import axis_angle_to_quat, quat_to_matrix

    path = Path("/gaozt-test1/datasets/MixEgoMotion/weights/smplh/male/model.npz")
    model = load_smplh_body22_model(path)
    decoder = BodyModel(path)
    rng = np.random.default_rng(49381)
    count = 5
    local_aa = rng.normal(0, .15, (count, 22, 3))
    local = quat_to_matrix(axis_angle_to_quat(local_aa))
    betas = rng.normal(0, .2, 16)
    transl = rng.normal(0, .3, (count, 3))
    positions, quaternions, valid = decoder.decode(transl, local_aa[:, 0], local_aa[:, 1:].reshape(count, 63),
                                                 np.broadcast_to(betas, (count, 16)), np.ones(count, bool))
    root = transl + model.rest_joints(torch.tensor(betas)).detach().numpy()[0]
    actual_p, actual_g = model(torch.tensor(local), torch.tensor(root), torch.tensor(betas))
    np.testing.assert_allclose(actual_p.detach(), positions, atol=1e-12)
    np.testing.assert_allclose(actual_g.detach(), quat_to_matrix(quaternions), atol=1e-12)
    assert valid.all()
    assert model.shape_components == 16
    assert model.metadata["asset_authenticity_verified"] is False
