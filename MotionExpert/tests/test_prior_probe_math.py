"""Frozen-prior label/metric checks without production data or model weights."""
import numpy as np
import pytest

from MotionExpert.prior_probe_math import (
    build_probe_targets, compute_probe_metrics, masked_train_statistics,
    probe_metric_rows, reduce_probe_metric_rows, standardize,
)


def fixture(count=1):
    motion = np.zeros((count, 11, 245), dtype=np.float32)
    valid = np.ones_like(motion, dtype=bool)
    floor = np.broadcast_to(np.eye(4), (count, 11, 4, 4)).copy()
    return motion, valid, floor


def z_rotation(angle):
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1.]])


def test_targets_use_current_fixed_reference_and_explicit_wrist():
    motion, valid, floor = fixture()
    floor[:, 5, :3, :3] = z_rotation(np.pi / 2)
    floor[:, 5, :3, 3] = [1, 2, 0]
    floor[:, 6:, :3, :3] = z_rotation(np.pi)
    floor[:, 6:, :3, 3] = [1, 3, 0]
    motion[:, 5, 117:120] = [1, 0, 1]
    motion[:, 6:, 117:120] = [1, 0, 1]
    # Unrelated fitted-body masks cannot suppress independent explicit hands.
    valid[:, :, :108] = False
    targets = build_probe_targets(motion, valid, floor)
    value = targets["continuous"].reshape(1, 5, 9)
    np.testing.assert_allclose(value[0, :, :3], np.tile([0, 1, 0], (5, 1)), atol=1e-6)
    np.testing.assert_allclose(value[0, :, 6:8], np.tile([1, 0], (5, 1)), atol=1e-6)
    np.testing.assert_allclose(value[0, :, 8], np.pi / 2, atol=1e-6)
    assert targets["continuous_valid"].all()


def test_static_world_wrist_does_not_move_when_reference_translates_and_rotates():
    motion, valid, floor = fixture()
    world_point = np.array([3., 2., 1.])
    for index in range(11):
        rotation = z_rotation(.2 * index)
        position = np.array([index * .1, index * .2, .1])
        floor[0, index, :3, :3] = rotation
        floor[0, index, :3, 3] = position
        motion[0, index, 117:120] = rotation.T @ (world_point - position)
        motion[0, index, 180:183] = rotation.T @ (world_point - position)
    targets = build_probe_targets(motion, valid, floor)
    np.testing.assert_allclose(targets["continuous"].reshape(1, 5, 9)[..., :6], 0, atol=5e-7)
    assert targets["continuous_valid"].all()


def test_missing_middle_reference_edge_stops_future_labels_without_affecting_contacts():
    motion, valid, floor = fixture()
    valid[:, 8, 108:117] = False
    targets = build_probe_targets(motion, valid, floor)
    mask = targets["continuous_valid"].reshape(1, 5, 9)
    assert mask[:, :2].all()
    assert not mask[:, 2:].any()
    assert targets["contact_valid"].all()
    assert targets["transition_valid"].all()


def test_contacts_use_endpoint_masks_and_do_not_invent_missing_labels():
    motion, valid, floor = fixture()
    motion[:, 5, 243:245] = [1, 0]
    motion[:, 6:, 243:245] = [0, 1]
    valid[:, 5, 243] = False
    valid[:, 7, 244] = False
    targets = build_probe_targets(motion, valid, floor)
    contact_mask = targets["contact_valid"].reshape(1, 5, 2)
    transition_mask = targets["transition_valid"].reshape(1, 5, 2)
    assert contact_mask[0, 0].all()
    assert not contact_mask[0, 1, 1]
    assert not transition_mask[0, :, 0].any()
    np.testing.assert_array_equal(transition_mask[0, :, 1], [True, False, True, True, True])
    np.testing.assert_array_equal(targets["transition"].reshape(1, 5, 2)[0, :, 1], [1, 0, 1, 1, 1])
    motion[:, 8, 243] = .3
    with pytest.raises(ValueError, match="binary"):
        build_probe_targets(motion, valid, floor)


def test_metrics_are_count_weighted_source_separated_and_yaw_wraps():
    motion, valid, floor = fixture(2)
    motion[0, 6:, 243] = 1
    motion[1, 6:, 244] = 1
    target = build_probe_targets(motion, valid, floor)
    prediction = {key: value.copy() for key, value in target.items() if not key.endswith("valid")}
    reshaped = prediction["continuous"].reshape(2, 5, 9)
    reshaped[0, :, 0] = .01
    reshaped[1, :, 3] = .03
    reshaped[:, :, 8] = 2 * np.pi + .1
    target["continuous_valid"][1, 18:45] = False
    result = compute_probe_metrics(prediction, target, ["EgoSuite", "HM"])
    overall = result["overall"]
    assert overall["wrist_displacement_cm_count"] == 14
    assert overall["wrist_displacement_cm"] == pytest.approx(11 / 14)
    assert result["by_source"]["EgoSuite"]["left_wrist_displacement_cm"] == pytest.approx(1)
    assert result["by_source"]["HM"]["right_wrist_displacement_cm"] == pytest.approx(3)
    assert overall["reference_yaw_deg"] == pytest.approx(np.rad2deg(.1), abs=1e-4)
    assert overall["contact_f1"] == 1
    assert overall["contact_balanced_accuracy"] == 1
    rows = probe_metric_rows(prediction, target)
    # A bootstrap may repeat rows; aggregate supports repeated integer indices.
    duplicate = reduce_probe_metric_rows(rows, np.array([0, 0]))
    assert duplicate["wrist_displacement_cm_count"] == 20
    assert duplicate["wrist_displacement_cm"] == pytest.approx(.5)


def test_undefined_metrics_stay_absent_and_statistics_use_masks():
    motion, valid, floor = fixture()
    valid[:] = False
    target = build_probe_targets(motion, valid, floor)
    prediction = {key: value.copy() for key, value in target.items() if not key.endswith("valid")}
    result = compute_probe_metrics(prediction, target)
    assert result["wrist_displacement_cm_count"] == 0
    assert "wrist_displacement_cm" not in result
    assert "contact_f1" not in result
    assert "contact_balanced_accuracy" not in result
    train = np.array([[1., 0., np.nan], [3., 1000., np.nan]])
    mask = np.array([[True, True, False], [True, False, False]])
    statistics = masked_train_statistics(train, mask)
    np.testing.assert_allclose(statistics["mean"], [2., 0., 0.])
    np.testing.assert_allclose(statistics["std"], [1., 1., 1.])
    np.testing.assert_array_equal(statistics["count"], [2, 1, 0])
    np.testing.assert_allclose(standardize(train, statistics, mask), [[-1, 0, 0], [1, 0, 0]])
