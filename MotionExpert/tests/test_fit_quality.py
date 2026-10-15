"""Physical final-FK gates do not confuse world movement or missing evidence."""
from dataclasses import asdict
from types import SimpleNamespace
import json
import numpy as np
import pytest

from MotionExpert.fit_quality import (FitQualityPolicy, FitQualityReason as R, FitQualityWarning as W,
    fk_head_world, native_fit_availability, score_fit_quality, summarize_fit_quality)


def score(head, camera=None, observed=None, **kwargs):
    head = np.asarray(head, dtype=float)
    count = len(head)
    options = dict(native_valid=np.ones(count, bool), camera_valid=np.ones(count, bool),
                   head_observation_valid=np.ones(count, bool), timestamps_s=np.arange(count)/30.)
    options.update(kwargs)
    return score_fit_quality(head, np.zeros_like(head) if camera is None else camera,
                             np.zeros_like(head) if observed is None else observed, **options)


def test_shared_world_correction_and_high_camera_speed_do_not_reject_fit():
    camera = np.arange(12).reshape(4, 3) * 100.
    result = score(camera + [.15, 0, 0], camera, camera + [.05, 0, 0])
    assert result.quality_valid.all()
    assert result.transition_valid.tolist() == [False, True, True, True]
    np.testing.assert_allclose(result.relative_jump_m[1:], 0, atol=1e-10)


def test_final_fk_head_is_checked_even_when_source_head_is_close_to_camera():
    result = score([[2.5, 0, 0]], observed=[[.075, 0, 0]])
    assert result.newly_rejected[0]
    assert result.reasons[0] & R.HEAD_CAMERA_FAR
    assert result.reasons[0] & R.HEAD_OBSERVATION_FAR


def test_source_head_consistency_is_independent_of_camera_distance():
    result = score([[.2, 0, 0]], observed=[[-.2, 0, 0]])
    assert not result.quality_valid[0]
    assert not (result.reasons[0] & R.HEAD_CAMERA_FAR)
    assert result.reasons[0] & R.HEAD_OBSERVATION_FAR


@pytest.mark.parametrize('missing', ['camera_valid', 'head_observation_valid'])
def test_unknown_observation_cannot_certify_a_native_fit(missing):
    result = score([[.1, 0, 0]], **{missing: [False]})
    assert result.native_valid[0] and result.fit_finite[0]
    assert not result.quality_valid[0]
    assert result.newly_rejected[0]
    if missing == 'camera_valid':
        assert result.reasons[0] & R.CAMERA_UNAVAILABLE
        assert np.isnan(result.head_camera_distance_m[0])
    else:
        assert result.reasons[0] & R.HEAD_OBSERVATION_UNAVAILABLE
        assert np.isnan(result.head_observation_distance_m[0])


def test_missing_native_fit_is_not_promoted_or_counted_as_new_rejection():
    result = score([[.1, 0, 0]], native_valid=[False])
    assert not result.quality_valid[0]
    assert not result.newly_rejected[0]
    assert result.reasons[0] & R.NATIVE_INVALID


def test_completed_or_missing_head_does_not_hide_other_native_body_labels():
    pos = np.zeros((3, 22), bool)
    rot = pos.copy()
    pos[0, 0] = True
    rot[1, 21] = True
    assert native_fit_availability(pos, rot).tolist() == [True, True, False]


def test_relative_jump_rejects_both_endpoints_without_distance_failure():
    result = score([[0, 0, 0], [.26, 0, 0], [.26, 0, 0]])
    assert result.quality_valid.tolist() == [False, False, True]
    assert (result.reasons[:2] & R.RELATIVE_JUMP).all()
    assert not (result.reasons & (R.HEAD_CAMERA_FAR | R.HEAD_OBSERVATION_FAR)).any()
    assert not result.transition_valid.any()


def test_invalid_or_unobserved_frame_does_not_poison_neighbour():
    result = score([[20, 0, 0], [.1, 0, 0]], native_valid=[False, True])
    assert result.quality_valid.tolist() == [False, True]
    assert not (result.reasons[1] & R.RELATIVE_JUMP)
    assert not result.transition_checked.any()


def test_time_gap_only_removes_edge_comparison():
    result = score([[0, 0, 0], [.3, 0, 0]], timestamps_s=[0, 1])
    assert result.quality_valid.all()
    assert result.reasons[1] & R.TIME_GAP
    assert not result.transition_checked.any()


def test_warnings_are_separate_and_do_not_reject_at_hard_boundaries():
    result = score([[.5, 0, 0]], observed=[[.15, 0, 0]])
    assert result.quality_valid[0]
    assert result.warnings[0] & W.HEAD_CAMERA_DISTANCE
    assert result.warnings[0] & W.HEAD_OBSERVATION_DISTANCE
    assert not result.reasons[0]


def test_nonfinite_parameters_and_observations_remain_independent():
    result = score([[.1, 0, 0], [np.nan, 0, 0]], fit_finite=[False, True],
                   camera=[[np.inf, 0, 0], [0, 0, 0]])
    assert not result.quality_valid.any()
    assert result.reasons[0] & R.FIT_NONFINITE
    assert result.reasons[0] & R.CAMERA_UNAVAILABLE
    assert result.reasons[1] & R.FIT_NONFINITE
    json.dumps(summarize_fit_quality(result), allow_nan=False)


def toy_model():
    rng = np.random.default_rng(83)
    return SimpleNamespace(rest=rng.normal(size=(22, 3))*.2,
                           shape_basis=rng.normal(size=(22, 3, 16))*.01, model_sha256='a'*64)


def test_head_chain_matches_production_fk_with_nonzero_shape16_and_rotations():
    from MotionExpert.dataset.smplh_archive import SMPLHBodyModel
    rng = np.random.default_rng(17)
    model = toy_model()
    arrays = {'timestamp': np.arange(19)/30,
              'smplh.root_orient': rng.normal(size=(19, 3)),
              'smplh.pose_body': rng.normal(size=(19, 63)),
              'smplh.trans': rng.normal(size=(19, 3)),
              'smplh.betas': rng.normal(size=(19, 16))}
    head, finite = fk_head_world(model, arrays)
    full, _ = SMPLHBodyModel.decode(model, arrays)
    assert finite.all()
    np.testing.assert_array_equal(head, full[:, 15])
    arrays['smplh.betas'][0, -1] += 5
    changed, _ = fk_head_world(model, arrays)
    assert not np.array_equal(changed[0], head[0])


def test_nonfinite_parameter_does_not_substitute_a_fake_finite_head():
    model = toy_model()
    arrays = {'timestamp': [0, 1/30], 'smplh.root_orient': np.zeros((2, 3)),
              'smplh.pose_body': np.zeros((2, 63)), 'smplh.trans': np.zeros((2, 3)),
              'smplh.betas': np.zeros((2, 16))}
    arrays['smplh.betas'][1, 15] = np.nan
    head, finite = fk_head_world(model, arrays)
    assert finite.tolist() == [True, False]
    assert np.isnan(head[1]).all()


@pytest.mark.parametrize('kwargs', [{'max_head_camera_distance_m': 0}, {'max_time_gap_s': float('nan')},
                                  {'warning_head_observation_distance_m': .4}])
def test_invalid_policies_are_rejected(kwargs):
    with pytest.raises(ValueError):
        FitQualityPolicy(**kwargs)
