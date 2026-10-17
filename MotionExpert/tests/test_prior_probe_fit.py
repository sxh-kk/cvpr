"""Critical leakage, masked regression and paired group inference checks."""
import numpy as np
import pytest
import torch

from MotionExpert.prior_probe_fit import (
    _bootstrap, _predict, _ridge_weights, connected_group_ids,
    decode_targets, encode_targets, internal_group_split, noise_stability_statistics,
)
from MotionExpert.prior_probe_math import probe_metric_rows


def test_connected_groups_join_transitively_and_reject_split_leakage():
    entries = [{"split": "train", "split_group_ids": names} for names in
               (("person:a",), ("person:a", "capture:b"), ("capture:b",), ("person:c",))]
    labels = connected_group_ids(entries)
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] != labels[0]
    entries[2]["split"] = "val"
    with pytest.raises(ValueError, match="leak"):
        connected_group_ids(entries)


def test_internal_holdout_keeps_groups_and_original_validation_out():
    entries = []
    for source in ("egosuite", "egoexo4d_hm"):
        for group in range(5):
            entries += [{"source": source, "split": "train", "split_group_ids": [f"{source}:{group}"]}] * 3
        entries.append({"source": source, "split": "val", "split_group_ids": [f"{source}:val"]})
    groups = connected_group_ids(entries)
    train, holdout = internal_group_split(entries, groups)
    assert not set(groups[train]) & set(groups[holdout])
    assert train.sum() == 24 and holdout.sum() == 6
    assert not train[15] and not holdout[15]
    np.testing.assert_array_equal(train, internal_group_split(entries, groups)[0])


def targets(count):
    return {"continuous": np.zeros((count, 45), np.float32),
            "continuous_valid": np.ones((count, 45), bool),
            "contact": np.zeros((count, 10), np.float32), "contact_valid": np.ones((count, 10), bool),
            "transition": np.zeros((count, 10), np.float32), "transition_valid": np.ones((count, 10), bool)}


def test_sincos_target_round_trip_and_masks_do_not_become_labels():
    target = targets(3)
    yaw = np.array([np.pi - .01, -np.pi + .02, .4], np.float32)
    target["continuous"].reshape(3, 5, 9)[:, :, 8] = yaw[:, None]
    target["continuous_valid"][0, 8] = False
    encoded, valid = encode_targets(target)
    assert encoded.shape == valid.shape == (3, 70)
    assert not valid[0, 8:10].any()
    assert not encoded[0, 8:10].any()
    decoded = decode_targets(encoded)
    np.testing.assert_allclose(decoded["continuous"][1:], target["continuous"][1:], atol=1e-6)


def test_ridge_masks_intercept_and_mean_loss_alpha_are_correct():
    rng = np.random.default_rng(42)
    x = torch.tensor(rng.normal(size=(80, 3)), dtype=torch.float32)
    y = torch.stack((3 + 2 * x[:, 0], -2 + x[:, 1], torch.zeros(80)), 1)
    valid = torch.ones(80, 3, dtype=torch.bool)
    valid[:10, :2] = False
    valid[:, 2] = False
    y[:10, :2] = 1e6  # Missing observations must have no influence.
    weights = _ridge_weights(x, y, valid, [.001, .1])
    assert weights.shape == (2, 4, 3)
    predicted = _predict(x, weights[0])
    torch.testing.assert_close(predicted[10:, :2], y[10:, :2], atol=.01, rtol=.01)
    assert not bool(weights[:, :, 2].any())
    duplicated = _ridge_weights(x.repeat(2, 1), y.repeat(2, 1), valid.repeat(2, 1), [.001, .1])
    torch.testing.assert_close(weights, duplicated, atol=1e-10, rtol=1e-10)


def test_bootstrap_is_paired_and_counts_connected_groups():
    target = targets(8)
    target["contact"][:, ::2] = 1
    target["transition"][:, ::2] = 1
    prediction = {name: target[name].copy() for name in ("continuous", "contact", "transition")}
    rows = probe_metric_rows(prediction, target)
    entries = [{"source": "egosuite", "split": "val"} for _ in range(8)]
    groups = np.repeat(np.arange(2), 4)
    result = _bootstrap({name: rows for name in ("epoch20", "epoch100", "dino")}, entries, groups, repetitions=100)
    assert result["egosuite"]["independent_connected_groups"] == 2
    for contrast in result["egosuite"]["contrasts"].values():
        for metric in contrast.values():
            assert metric["ci95"] == [0., 0.]
            assert metric["defined_repetitions"] == 100


def test_noise_statistics_use_train_centering_and_fixed_physical_readout():
    target = targets(4)
    entries = [{"source": source} for source in ("egosuite", "egosuite", "egoexo4d_hm", "egoexo4d_hm")]
    mean = np.array([10., 20.])
    noise = np.zeros((4, 3, 2)) + mean
    noise[:, :, 0] += np.array([-1., 1., -2., 2.])[:, None] + np.array([-.1, 0, .1])[None]
    weights = np.zeros((3, 70))
    for horizon in range(5):
        weights[0, horizon * 10 + np.array([0, 3, 6])] = 1
        weights[-1, horizon * 10 + 9] = 1
    result = noise_stability_statistics(noise, mean, np.ones(2), weights,
                                        np.zeros(70), np.ones(70), target, entries, np.arange(4))
    for source in result:
        assert result[source]["wrist_readout_jitter_cm"] == pytest.approx(np.sqrt(2 / 3) * 10)
        assert result[source]["reference_xy_readout_jitter_cm"] == pytest.approx(np.sqrt(2 / 3) * 10)
        assert result[source]["reference_yaw_readout_jitter_deg"] == 0
        assert len(result[source]["draw_metrics"]) == 3
    shifted = noise_stability_statistics(noise + 10000, mean + 10000, np.ones(2), weights,
                                         np.zeros(70), np.ones(70), target, entries, np.arange(4))
    for source in result:
        assert shifted[source]["feature_within_to_between_ratio"] == pytest.approx(result[source]["feature_within_to_between_ratio"])
        assert shifted[source]["train_centered_cosine_draw0_vs"] == result[source]["train_centered_cosine_draw0_vs"]


def test_noise_statistics_do_not_invent_jitter_when_labels_missing():
    target = targets(2)
    target["continuous_valid"][:] = False
    entries = [{"source": "egosuite"}] * 2
    result = noise_stability_statistics(np.ones((2, 3, 2)), np.zeros(2), np.ones(2),
                                        np.zeros((3, 70)), np.zeros(70), np.ones(70),
                                        target, entries, [0, 1])["egosuite"]
    assert result["feature_within_noise_rms"] == 0
    assert result["feature_within_to_between_ratio"] is None
    assert result["wrist_readout_jitter_cm"] is None
    assert result["reference_yaw_readout_jitter_deg"] is None


def test_noise_yaw_jitter_wraps_across_pi_boundary():
    angles = np.array([[np.pi - .01, -np.pi + .01, np.pi - .02]])
    noise = np.stack((np.sin(angles), np.cos(angles)), axis=-1)
    weights = np.zeros((3, 70))
    for horizon in range(5):
        weights[0, horizon * 10 + 8] = 1
        weights[1, horizon * 10 + 9] = 1
    result = noise_stability_statistics(noise, np.zeros(2), np.ones(2), weights,
                                        np.zeros(70), np.ones(70), targets(1),
                                        [{"source": "egosuite"}], [0])["egosuite"]
    assert 0 < result["reference_yaw_readout_jitter_deg"] < 1
