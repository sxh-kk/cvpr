"""Independent checks for the experimental body22 model-fitting pilot.

These tests check kinematics and optimization inputs, not the authenticity of
EgoSuite's original local joint axes. A successful fit remains an estimate.
"""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from egomotion_pipeline.schema import PARENTS
from egomotion_pipeline.smpl_fit import (
    FitConfig,
    build_transition_mask,
    fit_body22,
    fit_shape_from_positions,
    initialize_local_rotations,
    load_body22_model,
    rotation_increment_errors_deg,
    so3_exp,
    world_rotation_increments,
)


def _numpy_exp(vectors):
    """Rodrigues with NumPy, independent of the differentiable implementation."""
    vectors = np.asarray(vectors, dtype=np.float64)
    matrices = np.empty((*vectors.shape[:-1], 3, 3))
    for index in np.ndindex(vectors.shape[:-1]):
        vector = vectors[index]
        angle = np.linalg.norm(vector)
        if angle < 1e-12:
            matrices[index] = np.eye(3)
            continue
        x, y, z = vector / angle
        skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        matrices[index] = (np.eye(3) + np.sin(angle) * skew
                           + (1 - np.cos(angle)) * (skew @ skew))
    return matrices


def _numpy_fk(rest, local, root_world):
    """Column-vector FK: a parent orientation acts on each child rest offset."""
    count = len(local)
    rest = np.broadcast_to(rest, (count, 22, 3))
    positions = np.empty((count, 22, 3))
    rotations = np.empty((count, 22, 3, 3))
    positions[:, 0] = root_world
    rotations[:, 0] = local[:, 0]
    for joint in range(1, 22):
        parent = PARENTS[joint]
        rotations[:, joint] = rotations[:, parent] @ local[:, joint]
        positions[:, joint] = positions[:, parent] + np.einsum(
            "tij,tj->ti", rotations[:, parent], rest[:, joint] - rest[:, parent])
    return positions, rotations


def _asset_rest(path, betas):
    with np.load(path, allow_pickle=False) as asset:
        regressor = asset["J_regressor"][:22]
        rest = regressor @ asset["v_template"]
        basis = np.einsum("jv,vck->jck", regressor,
                          asset["shapedirs"][..., :len(betas)])
    return rest + np.einsum("k,jck->jc", betas, basis)


def test_so3_exp_matches_independent_rodrigues_and_has_finite_zero_gradient():
    vectors = np.array([[0., 0., 0.], [1e-9, -2e-9, 3e-9],
                        [.3, -.4, .2], [np.pi - 1e-7, 0., 0.]])
    value = torch.tensor(vectors, dtype=torch.float64, requires_grad=True)
    result = so3_exp(value)
    np.testing.assert_allclose(result.detach().numpy(), _numpy_exp(vectors),
                               atol=1e-12, rtol=0)
    result.sum().backward()
    assert torch.isfinite(value.grad).all()
    at_rest = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    so3_exp(at_rest)[2, 1].backward()
    np.testing.assert_allclose(at_rest.grad, [1., 0., 0.], atol=1e-12)


def test_shaped_fk_matches_numpy_and_root_is_world_pelvis(model_path):
    model = load_body22_model(model_path, shape_components=2)
    betas = np.array([.4, -.2])
    rest = _asset_rest(model_path, betas)
    np.testing.assert_allclose(model.rest_joints(torch.tensor(betas)).detach(), rest,
                               atol=1e-12)
    rng = np.random.default_rng(4291)
    local = _numpy_exp(rng.normal(0, .35, (4, 22, 3)))
    # The fixture has a non-zero rest root; do not silently treat this as trans.
    root_world = np.array([[1., 2., .9], [1.1, 2.2, .85],
                           [-.2, .1, .95], [.3, -.4, 1.2]])
    actual_positions, actual_rotations = model.forward(
        torch.tensor(local), torch.tensor(root_world), torch.tensor(betas))
    expected_positions, expected_rotations = _numpy_fk(rest, local, root_world)
    np.testing.assert_allclose(actual_positions.detach(), expected_positions,
                               atol=1e-12, rtol=0)
    np.testing.assert_allclose(actual_rotations.detach(), expected_rotations,
                               atol=1e-12, rtol=0)
    np.testing.assert_allclose(actual_positions.detach()[:, 0], root_world,
                               atol=1e-12, rtol=0)


def test_per_frame_shape_changes_follow_independent_fk(model_path):
    model = load_body22_model(model_path, shape_components=2)
    betas = np.array([[.2, -.1], [-.3, .4], [.5, .1]])
    rest = np.stack([_asset_rest(model_path, beta) for beta in betas])
    local = _numpy_exp(np.arange(3 * 22 * 3).reshape(3, 22, 3) / 1000.)
    root_world = np.array([[.1, .2, .9], [.1, .3, 1.], [.2, .3, 1.]])
    positions, rotations = model.forward(
        torch.tensor(local), torch.tensor(root_world), torch.tensor(betas))
    expected_positions, expected_rotations = _numpy_fk(rest, local, root_world)
    np.testing.assert_allclose(positions.detach(), expected_positions, atol=1e-12)
    np.testing.assert_allclose(rotations.detach(), expected_rotations, atol=1e-12)


@pytest.mark.skipif(not Path(
    "/gaozt-test1/guanzerong/UniEgoMotion/body_models/smplx/SMPLX_NEUTRAL.npz"
).is_file(), reason="licensed user-owned SMPL-X asset is not installed")
def test_real_smplx_body22_fk_matches_independent_joint_regression():
    path = Path("/gaozt-test1/guanzerong/UniEgoMotion/body_models/smplx/SMPLX_NEUTRAL.npz")
    model = load_body22_model(path, shape_components=3)
    betas = np.array([.2, -.1, .3])
    rest = _asset_rest(path, betas)
    rng = np.random.default_rng(618)
    local = _numpy_exp(rng.normal(0., .2, (2, 22, 3)))
    root_world = np.array([[0., 0., 1.], [.1, .2, 1.]])
    positions, rotations = model.forward(
        torch.tensor(local), torch.tensor(root_world), torch.tensor(betas))
    expected_positions, expected_rotations = _numpy_fk(rest, local, root_world)
    np.testing.assert_allclose(positions.detach(), expected_positions, atol=1e-10)
    np.testing.assert_allclose(rotations.detach(), expected_rotations, atol=1e-12)


def test_unknown_constant_right_basis_preserves_world_but_not_local_increments():
    rng = np.random.default_rng(5932)
    source = _numpy_exp(rng.normal(0., .5, (9, 22, 3)))
    correction = _numpy_exp(rng.normal(0., .8, (22, 3)))
    target = source @ correction
    world_source = source[1:] @ source[:-1].swapaxes(-1, -2)
    world_target = target[1:] @ target[:-1].swapaxes(-1, -2)
    np.testing.assert_allclose(world_source, world_target, atol=1e-14, rtol=0)
    np.testing.assert_allclose(world_rotation_increments(source), world_source,
                               atol=1e-14, rtol=0)
    np.testing.assert_allclose(world_rotation_increments(torch.tensor(source)),
                               world_source, atol=1e-14, rtol=0)
    local_source = source[:-1].swapaxes(-1, -2) @ source[1:]
    local_target = target[:-1].swapaxes(-1, -2) @ target[1:]
    assert np.max(np.abs(local_source - local_target)) > .1
    assert np.max(np.abs(source - target)) > .1


def test_transition_mask_excludes_invalid_endpoints_gaps_and_explicit_resets():
    timestamps = np.array([0., .05, .10, .40, .45, .50])
    valid = np.ones((6, 22), dtype=bool)
    valid[2, 7] = False
    explicit = np.ones((5, 22), dtype=bool)
    explicit[3, 11] = False
    result = build_transition_mask(timestamps, valid,
                                   transition_mask=explicit, max_dt=.1)
    expected = valid[1:] & valid[:-1]
    expected[2] = False
    expected &= explicit
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("timestamps", [
    [0., .1, .1], [0., .2, .1], [0., np.nan, .2], [0., np.inf, .2],
])
def test_transition_mask_rejects_nonincreasing_or_nonfinite_times(timestamps):
    with pytest.raises(ValueError):
        build_transition_mask(np.array(timestamps), np.ones((3, 22), dtype=bool))


def test_fk_matches_official_smplx_rigid_transform(model_path):
    lbs = pytest.importorskip("smplx.lbs")
    model = load_body22_model(model_path, shape_components=2)
    betas = torch.tensor([.2, -.4], dtype=torch.float64)
    rest = model.rest_joints(betas).expand(3, -1, -1)
    local = torch.tensor(_numpy_exp(
        np.random.default_rng(714).normal(0., .3, (3, 22, 3))))
    root_world = torch.tensor([[1., 2., 3.], [.1, .2, .3], [-1., -.2, .8]])
    positions, rotations = model.forward(local, root_world, betas)
    official_positions, official_transforms = lbs.batch_rigid_transform(
        local, rest, torch.tensor(PARENTS), dtype=torch.float64)
    official_positions += (root_world - rest[:, 0])[:, None]
    np.testing.assert_allclose(positions.detach(), official_positions.detach(),
                               atol=1e-12, rtol=0)
    np.testing.assert_allclose(rotations.detach(),
                               official_transforms.detach()[..., :3, :3],
                               atol=1e-12, rtol=0)


def _synthetic_observations(model_path, frames=6):
    betas = np.array([.25, -.15])
    rest = _asset_rest(model_path, betas)
    phase = np.linspace(0., .5, frames)
    vectors = np.zeros((frames, 22, 3))
    vectors[:, 0, 2] = .15 + phase * .1
    vectors[:, 1, 0] = .15 * np.sin(phase)
    vectors[:, 2, 0] = -.12 * np.sin(phase)
    vectors[:, 16, 2] = .2 + phase * .15
    vectors[:, 18, 1] = -.3 + phase * .2
    local = _numpy_exp(vectors)
    root_world = np.stack([phase * .08, phase * .02, .9 + phase * .01], axis=1)
    positions, global_rotations = _numpy_fk(rest, local, root_world)
    return {
        "positions": positions,
        "global_rotations": global_rotations,
        "local_rotations": local,
        "root_world": root_world,
        "betas": betas,
        "timestamps": np.arange(frames) / 15.,
    }


def _short_fit_config(iterations=30):
    return FitConfig(shape_iterations=0, pose_iterations=iterations,
                     pose_lr=.015, root_lr=.01,
                     temporal_regularization=1e-7,
                     local_rest_regularization=1e-8,
                     pose_increment_regularization=1e-8,
                     history_every=10)


def test_synthetic_positions_fit_reduces_known_world_error_without_certification(model_path):
    model = load_body22_model(model_path, shape_components=2)
    observed = _synthetic_observations(model_path)
    initialization = {
        "local_rotations": observed["local_rotations"] @ _numpy_exp(
            np.random.default_rng(801).normal(0., .025, (6, 22, 3))),
        "root_world": observed["root_world"] + [.015, -.01, .02],
        "betas": observed["betas"].copy(),
    }
    initial_positions, _ = _numpy_fk(
        _asset_rest(model_path, observed["betas"]),
        initialization["local_rotations"], initialization["root_world"])
    initial_error = np.linalg.norm(initial_positions - observed["positions"], axis=-1).mean()
    result = fit_body22(
        model, observed["positions"], np.ones((6, 22), dtype=bool),
        observed["timestamps"], mode="positions_only", initialization=initialization,
        fixed_betas=observed["betas"], config=_short_fit_config(150))
    final_error = np.linalg.norm(result.positions - observed["positions"], axis=-1).mean()
    assert final_error < initial_error / 4
    assert final_error < .005
    assert np.isfinite(result.positions).all()
    np.testing.assert_allclose(result.positions[:, 0], result.root_world, atol=1e-12)
    rest = _asset_rest(model_path, result.betas)
    np.testing.assert_allclose(result.smpl_trans, result.root_world - rest[0], atol=1e-12)
    assert result.metadata["label_kind"] == "model_fitted_estimate"
    assert result.metadata["target_model_axes_defined"] is True
    assert result.metadata["rotation_basis_verified"] is False
    assert result.metadata["native_axis_certified"] is False
    assert result.metadata["absolute_twist_observed"] is False


def test_fit_world_increment_objective_is_invariant_to_unknown_right_basis(model_path):
    model = load_body22_model(model_path, shape_components=2)
    observed = _synthetic_observations(model_path)
    initialization = {key: observed[key].copy()
                      for key in ("local_rotations", "root_world", "betas")}
    # An imperfect, time-varying start ensures the objective has to optimize;
    # comparing two already-exact minima would be a weaker invariance test.
    initialization["local_rotations"] = initialization["local_rotations"] @ _numpy_exp(
        np.random.default_rng(312).normal(0., .01, (6, 22, 3)))
    correction = _numpy_exp(np.random.default_rng(183).normal(0., .7, (22, 3)))
    kwargs = dict(
        model=model, positions=observed["positions"],
        position_mask=np.ones((6, 22), dtype=bool),
        timestamps=observed["timestamps"], rotation_mask=np.ones((6, 22), dtype=bool),
        mode="positions_plus_rotation_increment", initialization=initialization,
        fixed_betas=observed["betas"], config=_short_fit_config(12))
    a = fit_body22(source_global_rotations=observed["global_rotations"], **kwargs)
    b = fit_body22(source_global_rotations=observed["global_rotations"] @ correction, **kwargs)
    np.testing.assert_allclose(a.positions, b.positions, atol=1e-7, rtol=0)
    np.testing.assert_allclose(a.global_rotations, b.global_rotations, atol=1e-7, rtol=0)


def test_explicit_reset_prevents_piecewise_basis_change_from_becoming_motion(model_path):
    model = load_body22_model(model_path, shape_components=2)
    observed = _synthetic_observations(model_path)
    source = observed["global_rotations"].copy()
    correction = _numpy_exp(np.random.default_rng(394).normal(0., .5, (22, 3)))
    source[3:] = source[3:] @ correction
    reset = np.ones(5, dtype=bool)
    reset[2] = False
    valid = np.ones((6, 22), dtype=bool)
    pair_mask = build_transition_mask(observed["timestamps"], valid, reset)
    errors = rotation_increment_errors_deg(observed["global_rotations"], source, pair_mask)
    assert errors.max() < 2e-6
    initialization = {key: observed[key].copy()
                      for key in ("local_rotations", "root_world", "betas")}
    initialization["local_rotations"] = initialization["local_rotations"] @ _numpy_exp(
        np.random.default_rng(643).normal(0., .01, (6, 22, 3)))
    kwargs = dict(model=model, positions=observed["positions"], position_mask=valid,
                  timestamps=observed["timestamps"], rotation_mask=valid,
                  mode="positions_plus_rotation_increment", transition_mask=reset,
                  initialization=initialization, fixed_betas=observed["betas"],
                  config=_short_fit_config(12))
    a = fit_body22(source_global_rotations=observed["global_rotations"], **kwargs)
    b = fit_body22(source_global_rotations=source, **kwargs)
    np.testing.assert_allclose(a.positions, b.positions, atol=1e-7, rtol=0)
    np.testing.assert_allclose(a.global_rotations, b.global_rotations, atol=1e-7, rtol=0)
    assert b.metadata["rotation_increment_constraints"] == 4 * 22


def test_shape_fit_uses_only_calibration_frames_and_recovers_observable_bone_lengths(model_path):
    model = load_body22_model(model_path, shape_components=2)
    observed = _synthetic_observations(model_path)
    positions = observed["positions"]
    calibration = np.array([True, True, True, False, False, False])
    contaminated = positions.copy()
    # The evaluation frames are valid observations but must not leak into shape.
    contaminated[~calibration, :, 0] += np.arange(22)[None] * .2
    config = FitConfig(shape_iterations=180, shape_lr=.03, shape_regularization=0.,
                       pose_iterations=0)
    kwargs = dict(model=model, position_mask=np.ones((6, 22), dtype=bool),
                  frame_mask=calibration, config=config)
    a = fit_shape_from_positions(positions=positions, **kwargs)
    b = fit_shape_from_positions(positions=contaminated, **kwargs)
    np.testing.assert_allclose(a["betas"], b["betas"], atol=1e-12, rtol=0)
    fitted_rest = _asset_rest(model_path, a["betas"])
    target_rest = _asset_rest(model_path, observed["betas"])
    fitted_lengths = np.linalg.norm(fitted_rest[1:] - fitted_rest[PARENTS[1:]], axis=-1)
    target_lengths = np.linalg.norm(target_rest[1:] - target_rest[PARENTS[1:]], axis=-1)
    assert np.max(np.abs(fitted_lengths - target_lengths)) < 1e-5
    assert a["metadata"]["calibration_frames"] == 3
    assert a["metadata"]["surface_shape_observed"] is False


def test_single_bone_twist_gauge_has_equal_observations_and_loss_but_different_absolute_rotation(model_path):
    model = load_body22_model(model_path, shape_components=2)
    observed = _synthetic_observations(model_path)
    rest = _asset_rest(model_path, observed["betas"])
    elbow, wrist = 18, 20
    offset = rest[wrist] - rest[elbow]
    twist = _numpy_exp(.7 * offset / np.linalg.norm(offset))
    alternate_local = observed["local_rotations"].copy()
    alternate_local[:, elbow] = alternate_local[:, elbow] @ twist
    alternate_local[:, wrist] = twist.T @ alternate_local[:, wrist]
    alternate_positions, alternate_global = _numpy_fk(
        rest, alternate_local, observed["root_world"])
    np.testing.assert_allclose(alternate_positions, observed["positions"], atol=1e-12)
    np.testing.assert_allclose(world_rotation_increments(alternate_global),
                               world_rotation_increments(observed["global_rotations"]),
                               atol=1e-12)
    assert np.max(np.abs(alternate_global[:, elbow]
                         - observed["global_rotations"][:, elbow])) > .1
    config = FitConfig(shape_iterations=0, pose_iterations=0,
                       local_rest_regularization=0., temporal_regularization=0.,
                       pose_increment_regularization=0.)
    kwargs = dict(model=model, positions=observed["positions"],
                  position_mask=np.ones((6, 22), dtype=bool), timestamps=observed["timestamps"],
                  source_global_rotations=observed["global_rotations"],
                  rotation_mask=np.ones((6, 22), dtype=bool), fixed_betas=observed["betas"],
                  config=config)
    a = fit_body22(initialization={"local_rotations": observed["local_rotations"],
                                  "root_world": observed["root_world"]}, **kwargs)
    b = fit_body22(initialization={"local_rotations": alternate_local,
                                  "root_world": observed["root_world"]}, **kwargs)
    assert abs(a.metadata["final"]["loss"] - b.metadata["final"]["loss"]) < 1e-12
    assert a.metadata["absolute_twist_observed"] is False
    assert b.metadata["absolute_twist_observed"] is False


def test_initializer_anchors_missing_initial_hips_before_full_root_geometry_returns(model_path):
    model = load_body22_model(model_path, shape_components=2)
    betas = np.array([.2, -.1])
    local_vectors = np.zeros((7, 22, 3))
    local_vectors[:, 0, 2] = 1.4
    local = _numpy_exp(local_vectors)
    root_world = np.stack([np.arange(7) * .01, np.zeros(7), np.ones(7)], axis=1)
    positions, _ = _numpy_fk(_asset_rest(model_path, betas), local, root_world)
    mask = np.ones((7, 22), dtype=bool)
    lower_body = [1, 2, 4, 5, 7, 8, 10, 11]
    mask[:3, lower_body] = False
    positions[~mask] = np.nan
    initialization = initialize_local_rotations(model, positions, mask, betas)
    root_rotations = initialization["local_rotations"][:, 0]
    assert initialization["metadata"]["root_orientation_anchor_frame"] == 3
    assert initialization["metadata"]["offline_future_geometry_seed"] is True
    assert initialization["metadata"]["source_rotations_used"] is False
    assert initialization["metadata"]["native_axis_certified"] is False
    # A later multi-direction pelvis observation fixes the otherwise arbitrary
    # initial yaw. A constant real pose must not jump when hips become available.
    np.testing.assert_allclose(root_rotations, local[:, 0], atol=1e-12, rtol=0)
    changes = world_rotation_increments(root_rotations[:, None])
    np.testing.assert_allclose(changes, np.broadcast_to(np.eye(3), changes.shape),
                               atol=1e-12, rtol=0)


def test_masked_nan_observations_do_not_contaminate_position_or_increment_loss(model_path):
    model = load_body22_model(model_path, shape_components=2)
    observed = _synthetic_observations(model_path)
    position_mask = np.ones((6, 22), dtype=bool)
    position_mask[2:4, 5] = False
    rotation_mask = np.ones((6, 22), dtype=bool)
    rotation_mask[2:4, 18] = False
    positions_nan = observed["positions"].copy()
    positions_finite = observed["positions"].copy()
    positions_nan[~position_mask] = np.nan
    positions_finite[~position_mask] = 1e7
    rotations_nan = observed["global_rotations"].copy()
    rotations_finite = observed["global_rotations"].copy()
    rotations_nan[~rotation_mask] = np.nan
    rotations_finite[~rotation_mask] = 1234.  # Proper SO(3) is required only where valid.
    initialization = {key: observed[key].copy()
                      for key in ("local_rotations", "root_world", "betas")}
    kwargs = dict(model=model, position_mask=position_mask, timestamps=observed["timestamps"],
                  rotation_mask=rotation_mask, mode="positions_plus_rotation_increment",
                  initialization=initialization, fixed_betas=observed["betas"],
                  config=_short_fit_config(12))
    a = fit_body22(positions=positions_nan, source_global_rotations=rotations_nan, **kwargs)
    b = fit_body22(positions=positions_finite, source_global_rotations=rotations_finite, **kwargs)
    assert np.isfinite(a.positions).all()
    assert np.isfinite(a.global_rotations).all()
    np.testing.assert_allclose(a.positions, b.positions, atol=1e-12, rtol=0)
    np.testing.assert_allclose(a.global_rotations, b.global_rotations, atol=1e-12, rtol=0)


def test_unmasked_invalid_rotation_is_rejected_before_fit(model_path):
    model = load_body22_model(model_path, shape_components=2)
    observed = _synthetic_observations(model_path)
    source = observed["global_rotations"].copy()
    source[3, 5] = 2 * np.eye(3)
    with pytest.raises(ValueError, match="SO\\(3\\)|rotation"):
        fit_body22(model, observed["positions"], np.ones((6, 22), dtype=bool),
                   observed["timestamps"], source_global_rotations=source,
                   rotation_mask=np.ones((6, 22), dtype=bool),
                   config=_short_fit_config(1))
