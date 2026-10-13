"""Sequence invariants, masking, independent FK and hierarchical calibration."""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from egomotion_pipeline.schema import PARENTS
from egomotion_pipeline.smpl_fit import so3_exp
from egomotion_pipeline.smplh_retarget import CenterCalibration, load_smplh_body22_model
from egomotion_pipeline.smplh_sequence import (
    GroupCalibration, SequenceConfig, SourceCenterPrior, _blend_so3, _rotation_vectors,
    _torch_log, _hinge_reference, _hinge_scores, calibrate_group, fit_sequence, palm_reference_from_smplh_asset,
    train_shared_calibration,
)


def _rz(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


@pytest.fixture
def model(model_path):
    return load_smplh_body22_model(model_path, shape_components=2, allow_test_fixture=True)


def _config(**kwargs):
    return replace(SequenceConfig(pose_iterations=0, shape_iterations=0,
                                 min_calibration_frames=8, block_frames=12, overlap_frames=4), **kwargs)


def _fk(model, local, root, beta):
    rest = model.rest_zero.numpy() + np.einsum("jcb,b->jc", model.shape_basis.numpy(), beta)
    p, g = np.zeros((len(local), 22, 3)), np.empty_like(local)
    p[:, 0], g[:, 0] = root, local[:, 0]
    for joint in range(1, 22):
        parent = PARENTS[joint]
        p[:, joint] = p[:, parent] + np.einsum("tij,j->ti", g[:, parent], rest[joint] - rest[parent])
        g[:, joint] = g[:, parent] @ local[:, joint]
    return p, g, rest[0]


def _scene(model, n=25):
    beta = np.array([.1, -.2])
    local = np.broadcast_to(np.eye(3), (n, 22, 3, 3)).copy()
    local[:, 0] = np.stack([_rz(.01 * frame) for frame in range(n)])
    root = np.stack((np.arange(n) * .003, np.zeros(n), np.ones(n)), -1)
    p, g, _ = _fk(model, local, root, beta)
    native = g @ np.stack([_rz(.17 * joint) for joint in range(22)])
    return dict(positions=p, position_mask=np.ones((n, 22), bool), timestamps=np.arange(n) / 30.,
                source_global_rotations=native, rotation_mask=np.ones((n, 22), bool)), beta


def _calibration(model, beta, *, group="person", shift=0.):
    anchors = np.arange(22)
    anchors[[12, 13, 14, 15]] = [9, 9, 9, 12]
    offsets, valid = np.zeros((22, 3)), np.zeros(22, bool)
    if shift:
        offsets[12] = [shift, 0, 0]
        valid[12] = True
    center = CenterCalibration(offsets, anchors, valid, beta, model.metadata["model_sha256"], "0" * 64)
    return GroupCalibration(center, "egosuite", group)


def test_shared_prior_allows_different_frozen_betas_and_preserves_equal_person_weight(model):
    a = _calibration(model, np.array([1., 0.]), group="a", shift=.02)
    b = _calibration(model, np.array([-1., .5]), group="b", shift=.04)
    shared = train_shared_calibration([a, b], ["a", "b"])
    np.testing.assert_allclose(shared.offsets[12], [.03, 0, 0])
    assert shared.metadata["per_joint_offset_scatter_rms_m"][12] == pytest.approx(.01)
    assert shared.metadata["shape_independent_offset_semantics_verified"] is False
    np.testing.assert_equal(SourceCenterPrior.from_dict(shared.as_dict()).offsets, shared.offsets)
    np.testing.assert_equal(GroupCalibration.from_dict(a.as_dict()).betas, a.betas)
    with pytest.raises(ValueError, match="distinct"):
        train_shared_calibration([a, a])
    with pytest.raises(ValueError, match="identities"):
        train_shared_calibration([a, b], ["b", "a"])


def test_config_json_roundtrip_is_exact_and_invalid_overlap_rejected():
    config = _config()
    restored = SequenceConfig.from_dict(json.loads(json.dumps(config.__dict__)))
    assert restored == config
    with pytest.raises(ValueError, match="overlap_frames"):
        replace(config, overlap_frames=config.block_frames).validate()


def test_native_rate_chunks_recompute_exact_fk_and_translation(model):
    scene, beta = _scene(model)
    result = fit_sequence(model, **scene, calibration=_calibration(model, beta), config=_config())
    p, g, j0 = _fk(model, result.local_rotations, result.root_world, beta)
    np.testing.assert_allclose(result.positions, p, atol=1e-12)
    np.testing.assert_allclose(result.global_rotations, g, atol=1e-12)
    np.testing.assert_allclose(result.smpl_trans, result.root_world - j0, atol=1e-12)
    np.testing.assert_allclose(so3_exp(torch.tensor(result.local_axis_angle)).numpy(), result.local_rotations, atol=1e-9)
    assert len(result.metadata["blocks"]) == 3
    assert len(result.quality["seams"]) == 2
    np.testing.assert_array_equal(result.source_position_valid, scene["position_mask"])
    assert result.estimated_position_valid.all()
    assert np.all((result.fit_quality >= 0) & (result.fit_quality <= 1))
    assert result.metadata["absolute_rotation_accuracy_established"] is False
    assert result.as_mapping()["body_world"] is result.positions
    with pytest.raises(ValueError, match="native fps"):
        fit_sequence(model, **dict(scene, timestamps=np.arange(25) / 15.),
                     calibration=_calibration(model, beta), config=_config())


def test_half_turn_log_and_geodesic_overlap_are_proper_rotations():
    a = np.broadcast_to(np.eye(3), (1, 22, 3, 3)).copy()
    b = np.broadcast_to(_rz(np.pi), a.shape).copy()
    fused = _blend_so3(a, b, np.array([.5]))
    np.testing.assert_allclose(fused, np.broadcast_to(_rz(np.pi / 2), a.shape), atol=1e-9)
    np.testing.assert_allclose(so3_exp(torch.tensor(_rotation_vectors(b))).numpy(), b, atol=1e-9)
    vector = torch.zeros((2, 22, 3), dtype=torch.float64, requires_grad=True)
    log = _torch_log(so3_exp(vector))
    log.sum().backward()
    np.testing.assert_allclose(vector.grad.numpy(), 1., atol=1e-8)
    np.testing.assert_allclose(_torch_log(torch.tensor(b)).numpy()[0, :, 2], np.pi, atol=1e-8)


def test_long_gaps_short_completion_and_reset_do_not_become_observations(model):
    scene, beta = _scene(model, 20)
    original = scene["position_mask"].copy()
    original[5:9] = False
    original[12] = False
    positions = scene["positions"].copy()
    positions[~original] = np.nan
    reset = np.ones(19, bool)
    reset[14] = False
    result = fit_sequence(model, **dict(scene, positions=positions, position_mask=original),
                          transition_mask=reset, calibration=_calibration(model, beta), config=_config())
    np.testing.assert_array_equal(result.source_position_valid, original)
    assert not result.estimated_position_valid[5:9].any()
    assert result.estimated_position_valid[12].all()
    assert result.estimate_interpolated[12].all()
    assert result.metadata["segments"] == [[0, 5], [9, 15], [15, 20]]
    assert not result.fit_quality[5:9].any()
    assert np.isfinite(result.positions).all()


def test_low_calibration_coverage_is_explicit_prior_fallback(model):
    scene, _ = _scene(model, 14)
    config = _config(min_calibration_frames=240)
    result = calibrate_group(model, **scene, group_id="short", config=config)
    assert result.metadata["calibration_status"] == "low_coverage_prior_fallback"
    assert result.metadata["shape_trusted_for_anthropometric_estimate"] is False
    assert not result.center.offset_valid.any()
    np.testing.assert_array_equal(result.betas, np.zeros(2))
    assert not result.metadata["subject_offsets_optimized"]


def test_shared_d_is_frozen_when_subject_has_insufficient_coverage(model):
    a = _calibration(model, np.array([.1, -.2]), group="a", shift=.04)
    b = _calibration(model, np.array([.3, .4]), group="b", shift=.02)
    shared = train_shared_calibration([a, b])
    scene, _ = _scene(model, 14)
    result = calibrate_group(model, **scene, shared_offsets=shared, group_id="short",
                             config=_config(min_calibration_frames=240))
    np.testing.assert_array_equal(result.center.offsets, shared.offsets)
    np.testing.assert_array_equal(result.center.offset_valid, shared.offset_valid)


def test_subject_residual_is_bounded_and_beta_stays_frozen(model):
    scene, beta = _scene(model, 14)
    a = _calibration(model, beta, group="a", shift=.02)
    b = _calibration(model, -beta, group="b", shift=.03)
    shared = train_shared_calibration([a, b])
    shifted = scene["positions"].copy()
    shifted[:, 12, 0] += .10
    config = _config(pose_iterations=12, subject_offset_bound_m=.003)
    result = calibrate_group(model, **dict(scene, positions=shifted), fixed_betas=beta,
                             shared_offsets=shared, group_id="held", config=config)
    assert np.linalg.norm(result.center.offsets[12] - shared.offsets[12]) <= .003 + 1e-10
    np.testing.assert_array_equal(result.betas, beta)
    frozen = GroupCalibration.from_dict(result.as_dict())
    fit = fit_sequence(model, **scene, calibration=frozen, config=_config())
    np.testing.assert_array_equal(fit.betas, beta)
    np.testing.assert_allclose(fit.reconstructed_observations - fit.positions,
                               np.einsum("tjik,jk->tji", fit.global_rotations[:, frozen.center.anchors], frozen.center.offsets), atol=1e-12)


def test_hand_wrist_merging_is_single_constraint_and_nan_mask_safe(model):
    scene, beta = _scene(model, 14)
    hands = scene["positions"][:, [20, 21]].copy()
    hands[:, 0, 0] += .03
    valid = np.ones((14, 2), bool)
    valid[3, 0] = False
    hands[3, 0] = np.nan
    result = fit_sequence(model, **scene, hand_wrist_world=hands, hand_wrist_valid=valid,
                          calibration=_calibration(model, beta), config=_config())
    assert result.quality["body_to_hand_wrist_error_m"]["count"] == 27
    assert result.metadata["blocks"][0]["fit"]["position_pair_constraints"] == 11 * 22
    np.testing.assert_array_equal(result.source_position_valid, scene["position_mask"])
    assert np.isfinite(result.positions).all()


def test_palm_uses_defined_target_and_noncollinear_valid_points_only(model):
    scene, beta = _scene(model, 14)
    hands = np.repeat(scene["positions"][:, [20, 21], None, :], 21, axis=2)
    reference = np.array([[[.06, -.015, 0], [.08, 0, 0], [.055, .025, 0]]] * 2)
    directions = np.einsum("ij,skj->ski", _rz(.5), reference)
    hands[:, :, [5, 9, 17]] += directions
    valid = np.ones((14, 2, 21), bool)
    valid[0, 0, 5] = False
    hands[1, 1, [5, 9, 17]] = hands[1, 1, 0] + np.array([.05, 0, 0])
    config = _config(pose_iterations=5, palm_direction_weight=.01)
    result = fit_sequence(model, **scene, hand_world=hands, hand_valid=valid,
                          target_palm_reference=reference, calibration=_calibration(model, beta), config=config)
    assert result.metadata["blocks"][0]["fit"]["palm_constraints"] == 22
    assert result.metadata["palm_direction_is_independent_rotation_truth"] is False
    assert np.isfinite(result.positions).all()
    with pytest.raises(ValueError, match="noncollinear"):
        fit_sequence(model, **scene, hand_world=hands, hand_valid=valid,
                     target_palm_reference=np.ones((2, 3, 3)), calibration=_calibration(model, beta), config=_config())


def test_contact_needs_valid_source_feet_and_never_projects_joint_to_ground(model):
    scene, beta = _scene(model, 14)
    validity = scene["position_mask"].copy()
    validity[:, [10, 11]] = False
    result = fit_sequence(model, **dict(scene, position_mask=validity),
                          source_contact=np.ones((14, 2), bool), contact_valid=np.ones((14, 2), bool),
                          calibration=_calibration(model, beta), config=_config(pose_iterations=3))
    assert result.quality["contact_constraints"] == 0
    assert not result.metadata["foot_joint_projected_to_ground"]
    assert np.abs(result.positions[:, [10, 11], 2]).min() > .1
    assert not result.source_position_valid[:, [10, 11]].any()
    assert result.estimated_position_valid[:, [10, 11]].all()
    with pytest.raises(ValueError, match="contact_valid"):
        fit_sequence(model, **scene, source_contact=np.ones((14, 2), bool),
                     calibration=_calibration(model, beta), config=_config())


def test_contact_proxy_preserves_observed_horizontal_motion_instead_of_forcing_static(model):
    scene, beta = _scene(model, 14)
    source_contact = np.ones((14, 2), bool)
    contact_valid = np.ones((14, 2), bool)
    config = _config(pose_iterations=15, contact_velocity_weight=.2)
    result = fit_sequence(model, **scene, source_contact=source_contact, contact_valid=contact_valid,
                          calibration=_calibration(model, beta), config=config)
    assert result.quality["contact_constraints"] == 26
    native = result.quality["raw_contact_horizontal_foot_velocity_m_s"]["mean"]
    fitted = result.quality["source_contact_horizontal_foot_velocity_m_s"]["mean"]
    assert native > .05
    assert fitted >= .8 * native
    assert result.quality["contact_foot_horizontal_motion_matching_error_m_s"]["mean"] < .02
    assert "observed_contact_foot_motion_matching" in result.metadata["source_contact_kind"]
    assert np.isfinite(result.positions).all()
    assert np.abs(result.positions[:, [10, 11], 2]).min() > .1


def test_calibration_samples_contiguous_snippets_and_hashes_contact_inputs(model):
    scene, beta = _scene(model, 160)
    config = _config(calibration_frames=40, calibration_block_frames=10)
    a = calibrate_group(model, **scene, fixed_betas=beta, config=config)
    b = calibrate_group(model, **scene, fixed_betas=beta,
                        source_contact=np.ones((160, 2), bool), contact_valid=np.ones((160, 2), bool), config=config)
    indices = np.asarray(a.metadata["calibration_frame_indices"])
    assert len(indices) == 40
    assert indices[:10].tolist() == list(range(10))
    assert indices[-10:].tolist() == list(range(150, 160))
    # Four ten-frame snippets create 36 adjacent-frame constraints per joint.
    assert a.metadata["calibration_fit"]["rotation_increment_constraints"] == 36 * 22
    assert a.input_sha256 != b.input_sha256


def test_no_pelvis_or_safe_limbs_keeps_world_estimate_separate(model):
    scene, _ = _scene(model, 14)
    valid = np.zeros((14, 22), bool)
    valid[:, [9, 12, 13, 14, 15]] = True
    calibration = calibrate_group(model, **dict(scene, position_mask=valid), group_id="upper",
                                  config=_config())
    assert not calibration.metadata["shape_trusted_for_anthropometric_estimate"]
    result = fit_sequence(model, **dict(scene, position_mask=valid), calibration=calibration, config=_config())
    assert result.estimated_position_valid.all()
    assert not result.source_position_valid[:, 0].any()
    assert np.isfinite(result.root_world).all()


def test_hip_proxy_recovers_missing_pelvis_in_rotated_world_without_upgrading_source_mask(model):
    scene, beta = _scene(model, 14)
    rotation = so3_exp(torch.tensor([.9, -.4, .7], dtype=torch.float64)).numpy()
    positions = scene["positions"] @ rotation.T + np.array([3., -2., .5])
    validity = scene["position_mask"].copy()
    validity[:, 0] = False
    masked = positions.copy()
    masked[:, 0] = np.nan
    source_rotations = rotation @ scene["source_global_rotations"]
    result = fit_sequence(model, **dict(scene, positions=masked, position_mask=validity,
                                        source_global_rotations=source_rotations),
                          calibration=_calibration(model, beta), config=_config())
    np.testing.assert_allclose(result.root_world, positions[:, 0], atol=1e-10)
    assert not result.source_position_valid[:, 0].any()
    np.testing.assert_allclose(result.position_confidence[:, 0], .15)


def test_missing_source_pelvis_short_segment_does_not_use_model_rest_as_world_axes(model):
    scene, beta = _scene(model, 14)
    rotation = so3_exp(torch.tensor([np.pi / 2, 0., .3], dtype=torch.float64)).numpy()
    positions = scene["positions"] @ rotation.T + np.array([0., 0., 2.])
    mask = scene["position_mask"].copy()
    mask[5:8, 0] = False
    masked = positions.copy()
    masked[~mask] = np.nan
    reset = np.ones(13, bool)
    reset[[4, 7]] = False
    result = fit_sequence(model, **dict(scene, positions=masked, position_mask=mask,
                                        source_global_rotations=rotation @ scene["source_global_rotations"]),
                          transition_mask=reset, calibration=_calibration(model, beta), config=_config())
    np.testing.assert_allclose(result.root_world, positions[:, 0], atol=1e-10)
    assert not result.source_position_valid[5:8, 0].any()
    assert result.metadata["segments"] == [[0, 5], [5, 8], [8, 14]]


def test_native_axis_cannot_be_certified_by_constant_basis_or_twist(model):
    scene, beta = _scene(model, 14)
    corrected = scene["source_global_rotations"].copy()
    corrected[:, 15] = corrected[:, 15] @ _rz(np.pi / 2)
    a = fit_sequence(model, **scene, calibration=_calibration(model, beta), config=_config(pose_iterations=3))
    b = fit_sequence(model, **dict(scene, source_global_rotations=corrected),
                     calibration=_calibration(model, beta), config=_config(pose_iterations=3))
    np.testing.assert_allclose(a.positions, b.positions, atol=1e-9)
    np.testing.assert_allclose(a.global_rotations, b.global_rotations, atol=1e-9)
    assert not b.metadata["native_axis_certified"]
    assert not b.metadata["absolute_twist_observed"]


def test_real_palm_reference_uses_actual16_shape_and_52_joint_asset():
    path = Path("/gaozt-test1/datasets/MixEgoMotion/weights/smplh/male/model.npz")
    if not path.exists():
        pytest.skip("real SMPL-H numeric asset unavailable")
    model = load_smplh_body22_model(path, shape_components=16)
    beta = np.linspace(-.2, .3, 16)
    reference = palm_reference_from_smplh_asset(model, beta)
    with np.load(path, allow_pickle=False) as asset:
        vertices = asset["v_template"] + np.einsum("vcb,b->vc", asset["shapedirs"], beta)
        joints = asset["J_regressor"] @ vertices
    expected = np.stack((joints[[22, 25, 28]] - joints[20], joints[[37, 40, 43]] - joints[21]))
    np.testing.assert_allclose(reference, expected, atol=1e-13)
    assert model.shape_components == 16


def test_model_rest_hinge_heuristic_penalizes_reverse_bend_without_native_axis_assumption():
    path = Path("/gaozt-test1/datasets/MixEgoMotion/weights/smplh/male/model.npz")
    if not path.exists():
        pytest.skip("real SMPL-H numeric asset unavailable")
    model = load_smplh_body22_model(path, shape_components=16)
    ref = _hinge_reference(model, np.zeros(16))
    assert ref["indices"].tolist() == [4, 5, 18, 19]
    reference = {key: torch.tensor(value, dtype=torch.long if key == "indices" else torch.float64)
                 for key, value in ref.items()}
    positive, negative = np.zeros((1, 22, 3)), np.zeros((1, 22, 3))
    positive[0, ref["indices"]] = ref["axes"] * .5
    negative[0, ref["indices"]] = ref["axes"] * -.5
    good = _hinge_scores(so3_exp(torch.tensor(positive)), reference)
    bad = _hinge_scores(so3_exp(torch.tensor(negative)), reference)
    assert good.max() < 1e-12
    assert bad.min() > .1
    leaf = torch.tensor(negative, requires_grad=True)
    _hinge_scores(so3_exp(leaf), reference).sum().backward()
    assert torch.isfinite(leaf.grad).all()
    out_of_plane = np.zeros((1, 22, 3))
    normals = np.cross(ref["outgoing"], ref["axes"])
    out_of_plane[0, ref["indices"]] = normals * (np.pi / 2)
    bad_plane = torch.tensor(out_of_plane, requires_grad=True)
    score = _hinge_scores(so3_exp(bad_plane), reference)
    assert score.min() > .99
    score.sum().backward()
    assert torch.isfinite(bad_plane.grad).all()


def test_isolated_invalid_selected_source_rotation_is_rejected(model):
    scene, beta = _scene(model, 14)
    valid = np.zeros((14, 22), bool)
    valid[7, 15] = True
    rotations = np.full((14, 22, 3, 3), np.nan)
    rotations[7, 15] = np.diag([1., 1., 2.])
    with pytest.raises(ValueError, match=r"SO\(3\)"):
        fit_sequence(model, **dict(scene, source_global_rotations=rotations, rotation_mask=valid),
                     calibration=_calibration(model, beta), config=_config())


def test_contact_motion_loss_matches_moving_proxy_and_penalizes_static_target_mismatch():
    from egomotion_pipeline.smplh_sequence import _contact_motion_loss
    n = 4
    observed = torch.zeros((n, 22, 3), dtype=torch.float64)
    observed[:, [10, 11], 0] = torch.arange(n, dtype=torch.float64)[:, None] * .07 / 30
    fitted = observed.clone().requires_grad_()
    pairs = torch.ones((n - 1, 2), dtype=torch.bool)
    dt = torch.full((n - 1,), 1 / 30, dtype=torch.float64)
    up = torch.tensor([0., 0., 1.], dtype=torch.float64)
    loss = _contact_motion_loss(fitted, observed, dt, up, pairs)
    assert loss.item() == pytest.approx(0., abs=1e-20)
    loss.backward()
    np.testing.assert_allclose(fitted.grad.numpy(), 0., atol=1e-12)
    # A genuinely stationary source target penalizes estimated .07 m/s motion.
    stationary = torch.zeros_like(observed)
    wrong = _contact_motion_loss(fitted, stationary, dt, up, pairs)
    assert wrong.item() == pytest.approx(.07 ** 2, abs=1e-12)
    # The foot point height and vertical motion are not projected to ground.
    vertical = observed.clone()
    vertical[:, [10, 11], 2] = torch.arange(n, dtype=torch.float64)[:, None] * .4
    assert _contact_motion_loss(vertical, observed, dt, up, pairs).item() == pytest.approx(0., abs=1e-20)


def test_contact_pair_mask_requires_observations_proxy_and_continuous_edges():
    from egomotion_pipeline.smplh_sequence import _contact_pair_mask, _contact_motion_loss
    times = np.arange(6) / 30.
    times[4:] += .2
    valid = np.ones((6, 22), bool)
    valid[1, 10] = False
    proxy_valid = np.ones((6, 2), bool)
    proxy_valid[2, 1] = False
    explicit = np.ones((5, 22), bool)
    explicit[4, 11] = False
    data = dict(times=times, valid=valid, contact=np.ones((6, 2), bool),
                cvalid=proxy_valid, explicit=explicit)
    pairs = _contact_pair_mask(data, .05)
    np.testing.assert_equal(pairs, [[False, True], [False, False], [True, False],
                                  [False, False], [True, False]])
    # Masked NaN observations and reset/gap endpoints never enter arithmetic.
    observed = torch.zeros((6, 22, 3), dtype=torch.float64)
    observed[1, 10] = float("nan")
    observed[2, 11] = float("nan")
    fitted = torch.zeros_like(observed).requires_grad_()
    loss = _contact_motion_loss(fitted, observed, torch.tensor(np.diff(times)),
                                torch.tensor([0., 0., 1.]), torch.tensor(pairs))
    assert torch.isfinite(loss) and loss.item() == 0
    loss.backward()
    assert torch.isfinite(fitted.grad).all()


def test_continuous_initializer_bridges_body_dropout_with_hands_without_promoting_masks(model):
    scene, beta = _scene(model, 25)
    mask = scene["position_mask"].copy()
    mask[5:8] = False
    masked = scene["positions"].copy()
    masked[~mask] = np.nan
    hand_wrists = scene["positions"][:, [20, 21]].copy()
    result = fit_sequence(model, **dict(scene, positions=masked, position_mask=mask, rotation_mask=mask),
                          hand_wrist_world=hand_wrists, hand_wrist_valid=np.ones((25, 2), bool),
                          calibration=_calibration(model, beta), config=_config())
    np.testing.assert_array_equal(result.source_position_valid, mask)
    np.testing.assert_array_equal(result.source_rotation_valid, mask)
    assert result.estimated_position_valid.all()
    assert result.estimate_interpolated[5:8].all()
    assert result.metadata["segments"] == [[0, 25]]
    relative = np.swapaxes(result.local_rotations[:-1], -1, -2) @ result.local_rotations[1:]
    steps = np.rad2deg(np.linalg.norm(_rotation_vectors(relative), axis=-1))
    assert steps.max() < 10
    p, g, _ = _fk(model, result.local_rotations, result.root_world, beta)
    np.testing.assert_allclose(result.positions, p, atol=1e-12)
    np.testing.assert_allclose(result.global_rotations, g, atol=1e-12)
    initialization = result.metadata["blocks"][0]["fit"]["initialization"]
    assert initialization["source_rotation_usage"].startswith("world_increment_only")
    assert initialization["source_absolute_axes_used"] is False
    assert initialization["initializer_estimates_added_to_source_masks"] is False


def test_continuous_initializer_never_uses_future_geometry_across_reset(model):
    scene, beta = _scene(model, 25)
    valid = scene["position_mask"].copy()
    valid[5:8] = False
    positions = scene["positions"].copy()
    positions[~valid] = np.nan
    explicit = np.ones(24, bool)
    explicit[14] = False
    arguments = dict(scene, positions=positions, position_mask=valid, rotation_mask=valid,
                     transition_mask=explicit, hand_wrist_world=scene["positions"][:, [20, 21]].copy(),
                     hand_wrist_valid=np.ones((25, 2), bool))
    a = fit_sequence(model, **arguments, calibration=_calibration(model, beta), config=_config())
    changed = {name: value.copy() for name, value in arguments.items()}
    Q = _rz(1.4)
    changed["positions"][15:] = changed["positions"][15:] @ Q.T
    changed["hand_wrist_world"][15:] = changed["hand_wrist_world"][15:] @ Q.T
    changed["source_global_rotations"][15:] = Q @ changed["source_global_rotations"][15:]
    b = fit_sequence(model, **changed, calibration=_calibration(model, beta), config=_config())
    assert a.metadata["segments"] == [[0, 15], [15, 25]]
    np.testing.assert_allclose(a.local_rotations[:15], b.local_rotations[:15], atol=1e-12)
    np.testing.assert_allclose(a.root_world[:15], b.root_world[:15], atol=1e-12)


def test_geodesic_rotation_temporal_loss_retains_gradient_near_half_turn():
    theta = torch.tensor(np.deg2rad(179.9), dtype=torch.float64, requires_grad=True)
    vector = torch.stack((theta * 0, theta * 0, theta))
    R = so3_exp(vector)
    chordal = (R - torch.eye(3)).square().sum()
    chordal_grad = torch.autograd.grad(chordal, theta, retain_graph=True)[0]
    geodesic = _torch_log(R).square().sum()
    geodesic_grad = torch.autograd.grad(geodesic, theta)[0]
    assert torch.isfinite(geodesic_grad) and geodesic_grad.abs() > 6.
    assert chordal_grad.abs() < .01
