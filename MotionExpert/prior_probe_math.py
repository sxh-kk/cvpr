"""Label geometry and count-aware metrics for frozen motion-prior probes.

Only the label builder receives motion masks and future reference transforms.
The frozen feature extractor and the probe inputs must never receive them.
All displacement targets use the *current*, fixed camera-floor reference, not
the moving reference at each future frame.  Inputs are unnormalized cache data.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

CURRENT_FRAME = 5
FUTURE_FRAMES = (6, 7, 8, 9, 10)
HORIZON_SECONDS = (.2, .4, .6, .8, 1.)
CONTINUOUS_DIM = 45
BINARY_DIM = 10
CONTINUOUS_LAYOUT = {
    "left_wrist_displacement_m": (0, 3),
    "right_wrist_displacement_m": (3, 6),
    "reference_displacement_xy_m": (6, 8),
    "reference_yaw_rad": (8, 9),
}


def _check_inputs(motion, motion_valid, floor_transform):
    motion = np.asarray(motion)
    valid = np.asarray(motion_valid)
    floor = np.asarray(floor_transform, dtype=np.float64)
    if motion.ndim != 3 or motion.shape[1:] != (11, 245):
        raise ValueError("Probe labels require unnormalized N x 11 x 245 motion")
    if valid.shape != motion.shape or valid.dtype != np.bool_:
        raise ValueError("Probe label mask must be boolean and match motion")
    if floor.shape != (len(motion), 11, 4, 4):
        raise ValueError("Probe floor transforms must be N x 11 x 4 x 4")
    if not np.isfinite(motion[valid]).all():
        raise ValueError("Valid probe labels must be finite")
    return motion, valid, floor


def build_probe_targets(motion, motion_valid, floor_transform):
    """Create displacement/contact labels with explicit, target-only masks.

    ``continuous`` is flattened horizon-major N x 45.  Each horizon contains
    left wrist XYZ, right wrist XYZ, reference XY and reference yaw (radians).
    Wrists use each explicit hand's point zero, retaining its independent mask.
    ``contact`` and ``transition`` are horizon-major N x 10 (left/right foot).
    Transition means endpoint contact differs from current-frame contact; it
    does not claim that no earlier intermediate contact change occurred.

    A wrist target requires both endpoints and every intervening reference
    edge.  A reference target requires the same complete edge chain. Contact
    targets are independent of reference availability. Missing labels have a
    false mask; their placeholder zeros must never become observations.
    """
    motion, valid, floor = _check_inputs(motion, motion_valid, floor_transform)
    count = len(motion)
    current = CURRENT_FRAME
    finite_floor = np.isfinite(floor).all(axis=(-2, -1))
    # Invalid transforms may contain NaNs: neutralize before any arithmetic,
    # while preserving their false eligibility in reference_chain below.
    floor_safe = np.where(finite_floor[..., None, None], floor, np.eye(4))
    rotation = floor_safe[:, :, :3, :3]
    position = floor_safe[:, :, :3, 3]
    current_inverse_rotation = np.swapaxes(rotation[:, current], -1, -2)
    relative_rotation = np.einsum("nij,ntjk->ntik", current_inverse_rotation, rotation[:, 6:])
    relative_position = np.einsum("nij,ntj->nti", current_inverse_rotation,
                                  position[:, 6:] - position[:, current, None])
    edge_valid = valid[:, 6:, 108:117].all(-1) & finite_floor[:, 6:]
    reference_chain = np.logical_and.accumulate(edge_valid, axis=1) & finite_floor[:, current, None]
    targets = np.zeros((count, 5, 9), dtype=np.float32)
    target_valid = np.zeros_like(targets, dtype=bool)
    for side_index, begin in enumerate((117, 180)):
        wrist = np.where(valid[:, :, begin:begin + 3], motion[:, :, begin:begin + 3], 0)
        transformed_future = np.einsum("ntij,ntj->nti", relative_rotation, wrist[:, 6:]) + relative_position
        displacement = transformed_future - wrist[:, current, None]
        wrist_valid = valid[:, current, begin:begin + 3].all(-1)[:, None]
        wrist_valid = wrist_valid & valid[:, 6:, begin:begin + 3].all(-1) & reference_chain
        part = slice(side_index * 3, side_index * 3 + 3)
        targets[:, :, part] = displacement
        target_valid[:, :, part] = wrist_valid[..., None]
    targets[:, :, 6:8] = relative_position[:, :, :2]
    targets[:, :, 8] = np.arctan2(relative_rotation[:, :, 1, 0], relative_rotation[:, :, 0, 0])
    target_valid[:, :, 6:9] = reference_chain[..., None]
    contacts = motion[:, :, 243:245]
    contact_mask = valid[:, :, 243:245]
    if not np.isin(contacts[contact_mask], [0, 1]).all():
        raise ValueError("Valid raw contact targets must be binary, not normalized")
    future_contact_valid = contact_mask[:, 6:]
    transition_valid = future_contact_valid & contact_mask[:, current, None]
    transition = contacts[:, 6:] != contacts[:, current, None]
    return {
        "continuous": np.where(target_valid, targets, 0).reshape(count, CONTINUOUS_DIM),
        "continuous_valid": target_valid.reshape(count, CONTINUOUS_DIM),
        "contact": np.where(future_contact_valid, contacts[:, 6:], 0).astype(np.float32).reshape(count, BINARY_DIM),
        "contact_valid": future_contact_valid.reshape(count, BINARY_DIM),
        "transition": np.where(transition_valid, transition, 0).astype(np.float32).reshape(count, BINARY_DIM),
        "transition_valid": transition_valid.reshape(count, BINARY_DIM),
    }


def masked_train_statistics(values, valid=None, *, minimum_std=1e-6):
    """Fit per-coordinate mean/std on train rows only, omitting missing labels.

    The caller must pass the train partition, never concatenated train/val.
    Constant/unobserved coordinates get a unit scale. Unobserved coordinates
    retain count zero and must remain excluded from supervision and metrics.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or minimum_std <= 0:
        raise ValueError("Statistics require a matrix and positive minimum_std")
    valid = np.ones_like(values, dtype=bool) if valid is None else np.asarray(valid)
    if valid.shape != values.shape or valid.dtype != np.bool_:
        raise ValueError("Statistics mask must be boolean and match the matrix")
    if not np.isfinite(values[valid]).all():
        raise ValueError("Valid train values must be finite")
    count = valid.sum(axis=0)
    safe = np.where(valid, values, 0)
    mean = safe.sum(axis=0) / np.maximum(count, 1)
    deviation = np.where(valid, values - mean, 0)
    variance = np.square(deviation).sum(axis=0) / np.maximum(count, 1)
    std = np.sqrt(variance)
    std = np.where((count > 0) & (std >= minimum_std), std, 1.)
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32), "count": count}


def standardize(values, statistics, valid=None):
    """Apply already fitted train statistics; no fitting occurs here."""
    values = np.asarray(values)
    output = (values - np.asarray(statistics["mean"])) / np.asarray(statistics["std"])
    if valid is not None:
        output = np.where(np.asarray(valid, dtype=bool), output, 0)
    return output.astype(np.float32)


def probe_metric_rows(prediction: Mapping, target: Mapping, *, binary_logits=False):
    """Return additive statistics per window for episode-cluster bootstraps.

    Predictions are in physical units (metres/radians, contact probabilities
    unless binary_logits=True). All five horizons are equally scored per valid
    observation. Scalar errors are not averaged over invalid padded targets.
    """
    count = len(target["continuous"])
    if np.asarray(target["continuous"]).shape != (count, CONTINUOUS_DIM):
        raise ValueError("Continuous probe targets must be N x 45")
    rows = {}
    continuous = np.asarray(prediction["continuous"], dtype=np.float64)
    truth = np.asarray(target["continuous"], dtype=np.float64)
    mask = np.asarray(target["continuous_valid"], dtype=bool)
    if continuous.shape != truth.shape or mask.shape != truth.shape:
        raise ValueError("Continuous prediction/target/mask shapes must match")
    if not np.isfinite(continuous).all() or not np.isfinite(truth[mask]).all():
        raise ValueError("Non-finite predictions or valid targets cannot be scored")
    continuous, truth, mask = (array.reshape(count, 5, 9) for array in (continuous, truth, mask))

    def record(name, error, usable):
        rows[name + "_sum"] = np.where(usable, error, 0).reshape(count, -1).sum(axis=1)
        rows[name + "_count"] = usable.reshape(count, -1).sum(axis=1)

    wrist_errors, wrist_masks = [], []
    for side, part in (("left", slice(0, 3)), ("right", slice(3, 6))):
        error = np.linalg.norm(continuous[:, :, part] - truth[:, :, part], axis=-1) * 100
        usable = mask[:, :, part].all(-1)
        record(f"{side}_wrist_displacement_cm", error, usable)
        wrist_errors.append(error)
        wrist_masks.append(usable)
    record("wrist_displacement_cm", np.stack(wrist_errors, -1), np.stack(wrist_masks, -1))
    record("reference_displacement_xy_cm", np.linalg.norm(continuous[:, :, 6:8] - truth[:, :, 6:8], axis=-1) * 100,
           mask[:, :, 6:8].all(-1))
    yaw_difference = continuous[:, :, 8] - truth[:, :, 8]
    record("reference_yaw_deg", np.abs(np.rad2deg(np.arctan2(np.sin(yaw_difference), np.cos(yaw_difference)))),
           mask[:, :, 8])
    for name in ("contact", "transition"):
        predicted = np.asarray(prediction[name])
        binary_truth = np.asarray(target[name])
        usable = np.asarray(target[name + "_valid"], dtype=bool)
        if predicted.shape != (count, BINARY_DIM) or binary_truth.shape != predicted.shape or usable.shape != predicted.shape:
            raise ValueError(f"{name} prediction/target/mask must be N x 10")
        if not np.isfinite(predicted).all() or not np.isfinite(binary_truth[usable]).all():
            raise ValueError(f"Non-finite {name} predictions or valid targets cannot be scored")
        predicted = predicted >= (0. if binary_logits else .5)
        binary_truth = binary_truth >= .5
        for suffix, selected in (("tp", predicted & binary_truth), ("fp", predicted & ~binary_truth),
                                 ("fn", ~predicted & binary_truth), ("tn", ~predicted & ~binary_truth)):
            rows[f"{name}_{suffix}"] = (selected & usable).sum(axis=1)
    return rows


def reduce_probe_metric_rows(rows, selected=None):
    """Pool raw counts; undefined metrics are omitted rather than set to zero."""
    result = {}
    selected = slice(None) if selected is None else selected
    for key, value in rows.items():
        if key.endswith("_sum"):
            name = key[:-4]
            count = int(np.asarray(rows[name + "_count"])[selected].sum())
            result[name + "_count"] = count
            if count:
                result[name] = float(np.asarray(value)[selected].sum() / count)
    for name in ("contact", "transition"):
        tp, fp, fn, tn = (int(np.asarray(rows[f"{name}_{part}"])[selected].sum()) for part in ("tp", "fp", "fn", "tn"))
        result.update({f"{name}_tp": tp, f"{name}_fp": fp, f"{name}_fn": fn, f"{name}_tn": tn,
                       f"{name}_count": tp + fp + fn + tn})
        if tp + fp + fn + tn:
            result[f"{name}_accuracy"] = (tp + tn) / (tp + fp + fn + tn)
        if 2 * tp + fp + fn:
            result[f"{name}_f1"] = 2 * tp / (2 * tp + fp + fn)
        if tp + fn and tn + fp:
            result[f"{name}_balanced_accuracy"] = .5 * (tp / (tp + fn) + tn / (tn + fp))
    return result


def compute_probe_metrics(prediction, target, source=None, *, binary_logits=False):
    """Score overall and, when supplied, each dataset source separately."""
    rows = probe_metric_rows(prediction, target, binary_logits=binary_logits)
    overall = reduce_probe_metric_rows(rows)
    if source is None:
        return overall
    source = np.asarray(source).astype(str)
    if source.shape != (len(target["continuous"]),):
        raise ValueError("source must contain one identity per window")
    return {"overall": overall,
            "by_source": {name: reduce_probe_metric_rows(rows, source == name) for name in np.unique(source)}}
