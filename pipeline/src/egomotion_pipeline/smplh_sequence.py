"""Sequence SMPL-H labels, with frozen group shape and explicit center offsets.

Original positions, rotations, hands and camera trajectories remain observations.
The output is a model-fitted estimate; a defined SMPL-H axis is not a certificate
of the source joint axis or of absolute twist. Matrices act on column vectors.
"""

from dataclasses import asdict, dataclass, field
from hashlib import sha256
from io import BytesIO
import json

import numpy as np
import torch

from .schema import BODY_NAMES, PARENTS
from .smpl_fit import (
    _boolean_mask, _positions_input, _pseudo_huber_squared,
    _timestamps_input, _validate_rotations, _minimal_swing, build_transition_mask,
    initialize_local_rotations, rotation_increment_errors_deg, so3_exp,
    world_rotation_increments,
)
from .smplh_retarget import (
    CenterCalibration, RetargetConfig, _check_model, _fingerprint, _summary,
    fit_safe_limb_betas, reconstruct_source_landmarks,
)


@dataclass(frozen=True)
class SequenceConfig(RetargetConfig):
    """All fitted quantities and thresholds are serialized with each output."""
    fps: float = 30.
    block_frames: int = 300
    overlap_frames: int = 60
    calibration_frames: int = 360
    calibration_block_frames: int = 60
    min_calibration_frames: int = 240
    max_missing_frames: int = 2
    pose_iterations: int = 250
    shape_iterations: int = 150
    shape_regularization: float = 1e-4
    local_rest_regularization: float = 2e-5
    pose_increment_regularization: float = 2e-5
    temporal_regularization: float = 5e-4
    root_temporal_regularization: float = .01
    activity_regularization: float = 1e-3
    twist_regularization: float = 2e-5
    hinge_regularization: float = 5e-4
    subject_offset_bound_m: float = .025
    subject_offset_regularization: float = .1
    hand_wrist_weight: float = 3.
    palm_direction_weight: float = 1e-4
    contact_velocity_weight: float = 1e-4
    seam_regularization: float = 2e-4
    position_quality_scale_m: float = .03
    rotation_quality_scale_deg: float = 10.
    world_up: tuple = (0., 0., 1.)

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        for name in ("offset_joints", "offset_anchors", "world_up"):
            if name in value:
                value[name] = tuple(value[name])
        return cls(**value).validate()

    def validate(self):
        RetargetConfig.validate(self)
        for name in ("block_frames", "calibration_frames", "calibration_block_frames", "min_calibration_frames"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("overlap_frames", "max_missing_frames"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.overlap_frames >= self.block_frames:
            raise ValueError("overlap_frames must be smaller than block_frames")
        for name in ("fps", "subject_offset_bound_m", "hand_wrist_weight",
                     "position_quality_scale_m", "rotation_quality_scale_deg"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("root_temporal_regularization", "activity_regularization",
                     "twist_regularization", "hinge_regularization", "subject_offset_regularization",
                     "palm_direction_weight", "contact_velocity_weight", "seam_regularization"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        up = np.asarray(self.world_up, dtype=float)
        if up.shape != (3,) or not np.isfinite(up).all() or not np.isclose(np.linalg.norm(up), 1.):
            raise ValueError("world_up must be a finite unit vector")
        return self


@dataclass(frozen=True)
class SourceCenterPrior:
    """Train-only equal-group center prior in fixed model anchor axes.

    Unlike the old isolation pilot, different group beta vectors are allowed.
    This assumes residual landmark semantics are comparable across shapes and
    explicitly records that this assumption is unverified.
    """
    offsets: np.ndarray
    anchors: np.ndarray
    offset_valid: np.ndarray
    model_sha256: str
    input_sha256: str
    source_id: str
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        check = CenterCalibration(self.offsets, self.anchors, self.offset_valid, np.zeros(1),
                                  self.model_sha256, self.input_sha256, self.metadata)
        for name in ("offsets", "anchors", "offset_valid", "metadata"):
            object.__setattr__(self, name, getattr(check, name))
        if not isinstance(self.source_id, str) or not self.source_id:
            raise ValueError("source_id must be nonempty")

    def as_dict(self):
        return {"offsets": self.offsets.tolist(), "anchors": self.anchors.tolist(),
                "offset_valid": self.offset_valid.tolist(), "model_sha256": self.model_sha256,
                "input_sha256": self.input_sha256, "source_id": self.source_id,
                "metadata": _json_copy(self.metadata)}

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value["offset_valid"] = np.asarray(value["offset_valid"], dtype=bool)
        return cls(**value)


@dataclass(frozen=True)
class GroupCalibration:
    center: CenterCalibration
    source_id: str
    group_id: str
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if (not isinstance(self.center, CenterCalibration) or not isinstance(self.source_id, str)
                or not self.source_id or not isinstance(self.group_id, str) or not self.group_id):
            raise ValueError("GroupCalibration requires a center, source_id and group_id")
        object.__setattr__(self, "metadata", _json_copy(self.metadata))

    @property
    def betas(self):
        return self.center.betas

    @property
    def input_sha256(self):
        return self.center.input_sha256

    def as_dict(self):
        return {"center": self.center.as_dict(), "source_id": self.source_id,
                "group_id": self.group_id, "metadata": _json_copy(self.metadata)}

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        center = dict(value["center"])
        center["offset_valid"] = np.asarray(center["offset_valid"], dtype=bool)
        value["center"] = CenterCalibration(**center)
        return cls(**value)


@dataclass
class SequenceResult:
    positions: np.ndarray
    global_rotations: np.ndarray
    local_rotations: np.ndarray
    local_axis_angle: np.ndarray
    root_world: np.ndarray
    smpl_trans: np.ndarray
    betas: np.ndarray
    reconstructed_observations: np.ndarray
    source_position_valid: np.ndarray
    source_rotation_valid: np.ndarray
    estimated_position_valid: np.ndarray
    estimated_rotation_valid: np.ndarray
    estimate_interpolated: np.ndarray
    fit_quality: np.ndarray
    position_confidence: np.ndarray
    rotation_confidence: np.ndarray
    quality: dict
    metadata: dict

    def as_mapping(self):
        return {"body_world": self.positions, "positions": self.positions,
                **{name: getattr(self, name) for name in self.__dataclass_fields__ if name != "positions"}}


def _json_copy(value):
    return json.loads(json.dumps(value, allow_nan=False))


def train_shared_calibration(calibration_results, group_ids=None, *, source_id="egosuite"):
    """Pool train groups equally, permitting different fixed shapes.

    Call again after recalibrating groups with this prior for an alternating
    hierarchical estimate. Evaluation groups must never enter this function.
    """
    groups = list(calibration_results)
    if any(not isinstance(item, GroupCalibration) for item in groups):
        raise ValueError("Shared calibration requires GroupCalibration instances")
    ids = [item.group_id for item in groups] if group_ids is None else list(group_ids)
    if (len(groups) < 2 or len(groups) != len(ids) or len(set(ids)) != len(ids)
            or any(not isinstance(item, GroupCalibration) for item in groups)
            or any(not isinstance(value, str) or not value for value in ids)):
        raise ValueError("Shared calibration requires at least two distinct groups")
    first = groups[0].center
    if any(item.source_id != source_id or item.group_id != group_id for item, group_id in zip(groups, ids)):
        raise ValueError("Shared calibration source/group identities differ")
    if any(item.center.model_sha256 != first.model_sha256
           or not np.array_equal(item.center.anchors, first.anchors) for item in groups):
        raise ValueError("Shared calibration model and anchor axes must match")
    if any(len(item.betas) != len(first.betas) for item in groups):
        raise ValueError("Shared calibration shape basis counts differ")
    enabled = np.stack([item.center.offset_valid for item in groups])
    valid = enabled.sum(0) >= 2
    offsets, scatter = np.zeros((22, 3)), np.zeros(22)
    for joint in np.flatnonzero(valid):
        values = np.stack([item.center.offsets[joint] for item in groups])[enabled[:, joint]]
        offsets[joint] = values.mean(0)
        scatter[joint] = np.sqrt(np.mean(np.sum((values - offsets[joint]) ** 2, axis=-1)))
    metadata = {"kind": "train_equal_group_center_prior_with_individual_frozen_shapes",
                "group_ids": ids, "per_joint_calibration_groups": enabled.sum(0).tolist(),
                "per_joint_offset_scatter_rms_m": scatter.tolist(),
                "individual_betas": [item.betas.tolist() for item in groups],
                "individual_input_sha256": [item.input_sha256 for item in groups],
                "shape_independent_offset_semantics_assumed": True,
                "shape_independent_offset_semantics_verified": False,
                "native_axis_certified": False, "independent_center_accuracy_established": False,
                "subject_recalibration_required_for_next_alternating_round": True}
    fingerprint = _fingerprint(offsets, enabled, first.anchors,
                               context={**metadata, "source_id": source_id, "model_sha256": first.model_sha256})
    return SourceCenterPrior(offsets, first.anchors, valid, first.model_sha256, fingerprint,
                             source_id, metadata)


pool_group_centers = train_shared_calibration


def palm_reference_from_smplh_asset(model, betas):
    """Shape-specific model wrist→MCP vectors, not source hand-axis truth."""
    _check_model(model)
    if model.metadata.get("test_fixture"):
        raise ValueError("Synthetic fixture has no defined SMPL-H palm reference")
    beta = np.asarray(betas, dtype=float)
    if beta.shape != (model.shape_components,) or not np.isfinite(beta).all():
        raise ValueError("Palm reference beta basis must match model")
    cached = getattr(model, "_sequence_full_rest", None)
    if cached is None:
        from pathlib import Path
        snapshot = Path(model.metadata["model_path"]).read_bytes()
        if sha256(snapshot).hexdigest() != model.metadata["model_sha256"]:
            raise ValueError("Palm reference asset changed after model validation")
        with np.load(BytesIO(snapshot), allow_pickle=False) as asset:
            reg = np.asarray(asset["J_regressor"], dtype=float)
            template = np.asarray(asset["v_template"], dtype=float)
            basis = np.asarray(asset["shapedirs"], dtype=float)[..., :model.shape_components]
        if reg.shape != (52, 6890) or not np.isfinite(reg).all() or not np.isfinite(basis).all():
            raise ValueError("Palm reference requires numerical full SMPL-H joint regressor")
        cached = reg @ template, np.einsum("jv,vcb->jcb", reg, basis)
        model._sequence_full_rest = cached
    rest = cached[0] + np.einsum("jcb,b->jc", cached[1], beta)
    return np.stack((rest[[22, 25, 28]] - rest[20], rest[[37, 40, 43]] - rest[21]))


def _inputs(positions, position_mask, timestamps, source_global_rotations, rotation_mask,
            hand_wrist_world, hand_wrist_valid, hand_world, hand_valid, transition_mask,
            source_contact, contact_valid, config):
    observed, valid = _positions_input(positions, position_mask)
    count = len(observed)
    times = _timestamps_input(timestamps, count)
    rotations = np.broadcast_to(np.eye(3), (count, 22, 3, 3)).copy()
    rvalid = np.zeros((count, 22), dtype=bool)
    if source_global_rotations is not None:
        raw = np.asarray(source_global_rotations, dtype=float)
        if raw.shape != rotations.shape:
            raise ValueError("source_global_rotations must have shape (T,22,3,3)")
        rvalid = (np.isfinite(raw).all((-2, -1)) if rotation_mask is None
                  else _boolean_mask(rotation_mask, (count, 22), "rotation_mask"))
        _validate_rotations(raw[rvalid], "selected source rotations")
        rotations[rvalid] = raw[rvalid]
    elif rotation_mask is not None:
        raise ValueError("rotation_mask requires source_global_rotations")
    explicit = build_transition_mask(times, np.ones((count, 22), dtype=bool),
                                     transition_mask, max_dt=config.max_dt)
    hands = np.zeros((count, 2, 21, 3))
    hvalid = np.zeros((count, 2, 21), dtype=bool)
    if hand_world is not None:
        hands = np.asarray(hand_world, dtype=float).copy()
        if hands.shape != (count, 2, 21, 3) or hand_valid is None:
            raise ValueError("hand_world requires shape (T,2,21,3) and hand_valid")
        hvalid = _boolean_mask(hand_valid, (count, 2, 21), "hand_valid")
        if not np.isfinite(hands[hvalid]).all():
            raise ValueError("Valid hand observations must be finite")
        hands = np.where(hvalid[..., None], hands, 0.)
        if hand_wrist_world is None:
            hand_wrist_world, hand_wrist_valid = hands[:, :, 0], hvalid[:, :, 0]
    elif hand_valid is not None:
        raise ValueError("hand_valid requires hand_world")
    target, merged_valid = observed.copy(), valid.copy()
    weights = np.ones((count, 22))
    wrist_valid = np.zeros((count, 2), dtype=bool)
    wrists = np.zeros((count, 2, 3))
    if hand_wrist_world is not None:
        wrists = np.asarray(hand_wrist_world, dtype=float).copy()
        if wrists.shape != (count, 2, 3) or hand_wrist_valid is None:
            raise ValueError("hand_wrist_world requires shape (T,2,3) and hand_wrist_valid")
        wrist_valid = _boolean_mask(hand_wrist_valid, (count, 2), "hand_wrist_valid")
        if not np.isfinite(wrists[wrist_valid]).all():
            raise ValueError("Valid hand wrists must be finite")
        wrists = np.where(wrist_valid[..., None], wrists, 0.)
        for side, joint in enumerate((20, 21)):
            selected = wrist_valid[:, side]
            # One target per wrist; do not duplicate body and hand observations.
            target[selected, joint] = wrists[selected, side]
            merged_valid[selected, joint] = True
            weights[selected, joint] = config.hand_wrist_weight
    elif hand_wrist_valid is not None:
        raise ValueError("hand_wrist_valid requires hand_wrist_world")
    contact = np.zeros((count, 2), dtype=bool)
    cvalid = np.zeros((count, 2), dtype=bool)
    if source_contact is not None:
        contact = _boolean_mask(source_contact, (count, 2), "source_contact")
        if contact_valid is None:
            raise ValueError("source_contact requires independent contact_valid")
        cvalid = _boolean_mask(contact_valid, (count, 2), "contact_valid")
        cvalid &= valid[:, [10, 11]]
    elif contact_valid is not None:
        raise ValueError("contact_valid requires source_contact")
    return dict(observed=observed, valid=valid, times=times, source=rotations, rvalid=rvalid,
                target=target, merged_valid=merged_valid, weights=weights, hands=hands,
                hvalid=hvalid, wrists=wrists, wrist_valid=wrist_valid, explicit=explicit,
                contact=contact, cvalid=cvalid)


def _select_calibration(data, frame_mask, config):
    eligible = data["merged_valid"].any(1)
    if frame_mask is not None:
        eligible &= _boolean_mask(frame_mask, (len(eligible),), "calibration_frame_mask")
    candidates = np.flatnonzero(eligible)
    if not len(candidates):
        raise ValueError("No eligible calibration frames")
    if len(candidates) <= config.calibration_frames:
        return candidates
    block = min(config.calibration_block_frames, config.calibration_frames)
    nblocks = int(np.ceil(config.calibration_frames / block))
    starts = np.linspace(0, len(candidates) - block, nblocks, dtype=int)
    chosen = np.unique(np.concatenate([candidates[start:start + block] for start in starts]))
    return chosen[:config.calibration_frames]


def _slice_data(data, indices):
    indices = np.asarray(indices)
    result = {key: value[indices].copy() for key, value in data.items() if key != "explicit"}
    # Preserve both source resets and all skipped calibration-frame gaps.
    contiguous = np.diff(indices) == 1
    result["explicit"] = np.zeros((len(indices) - 1, 22), dtype=bool)
    result["explicit"][contiguous] = data["explicit"][indices[:-1][contiguous]]
    return result


def _initialize_segment(model, data, betas, world_up=(0., 0., 1.)):
    """Seed missing pelvis without assuming model-rest axes are world axes."""
    with torch.no_grad():
        rest = model.rest_joints(betas).cpu().numpy()
    count = len(data["times"])
    selected_root = data["merged_valid"][:, 0]
    seed = data["target"].copy()
    seed_mask = data["merged_valid"].copy()
    if selected_root.any():
        for axis in range(3):
            seed[:, 0, axis] = np.interp(data["times"], data["times"][selected_root],
                                         data["target"][selected_root, 0, axis])
        kind = "source_pelvis_timestamp_interpolation_within_reset_segment"
    else:
        hip_mid = (rest[1] + rest[2]) / 2
        model_up = rest[9] - hip_mid
        if np.linalg.norm(model_up) < 1e-8:
            model_up = np.array([0., 1., 0.])
        model_up /= np.linalg.norm(model_up)
        gravity_rotation = _minimal_swing(model_up, np.asarray(world_up))
        hip_valid = data["merged_valid"][:, 1] & data["merged_valid"][:, 2]
        if hip_valid.any():
            model_axis = rest[2] - rest[1]
            for frame in np.flatnonzero(hip_valid):
                midpoint = data["target"][frame, [1, 2]].mean(0)
                observed_axis = data["target"][frame, 2] - data["target"][frame, 1]
                upward = np.asarray(world_up)
                upward_model = model_up
                for joint in (9, 6, 3, 12, 15):
                    if data["merged_valid"][frame, joint]:
                        upward = data["target"][frame, joint] - midpoint
                        upward_model = rest[joint] - hip_mid
                        break
                else:
                    if data["merged_valid"][frame, [4, 5]].all():
                        upward = midpoint - data["target"][frame, [4, 5]].mean(0)
                        upward_model = hip_mid - rest[[4, 5]].mean(0)
                source_dirs = np.stack((model_axis, upward_model))
                observed_dirs = np.stack((observed_axis, upward))
                lengths = np.linalg.norm(source_dirs, axis=-1) * np.linalg.norm(observed_dirs, axis=-1)
                if (np.min(lengths) > 1e-12 and np.linalg.matrix_rank(source_dirs, tol=1e-7) >= 2
                        and np.linalg.matrix_rank(observed_dirs, tol=1e-7) >= 2):
                    source_dirs /= np.linalg.norm(source_dirs, axis=-1, keepdims=True)
                    observed_dirs /= np.linalg.norm(observed_dirs, axis=-1, keepdims=True)
                    left, _, right = np.linalg.svd(observed_dirs.T @ source_dirs)
                    correction = np.eye(3)
                    correction[-1, -1] = np.linalg.det(left @ right)
                    rotation = left @ correction @ right
                elif np.linalg.norm(observed_axis) > 1e-8 and np.linalg.norm(model_axis) > 1e-8:
                    rotation = _minimal_swing(gravity_rotation @ model_axis, observed_axis) @ gravity_rotation
                else:
                    rotation = gravity_rotation
                seed[frame, 0] = midpoint - rotation @ (hip_mid - rest[0])
            for axis in range(3):
                seed[:, 0, axis] = np.interp(data["times"], data["times"][hip_valid], seed[hip_valid, 0, axis])
            kind = "bilateral_hips_plus_body_up_model_offset_pelvis_proxy"
        else:
            relative = (rest - rest[0]) @ gravity_rotation.T
            root_valid = data["merged_valid"].any(1)
            for frame in np.flatnonzero(root_valid):
                selected = data["merged_valid"][frame]
                seed[frame, 0] = np.median(data["target"][frame, selected] - relative[selected], axis=0)
            for axis in range(3):
                seed[:, 0, axis] = np.interp(data["times"], data["times"][root_valid], seed[root_valid, 0, axis])
            kind = "gravity_oriented_rest_robust_translation_pelvis_proxy"
    # These temporary pelvis points are initializer inputs only. They never
    # enter the optimizer observation mask or the preserved source masks.
    seed_mask[:, 0] = True
    result = initialize_local_rotations(model, seed, seed_mask, betas)
    result["metadata"].update(root_initialization_kind=kind,
                              source_pelvis_observations=int(selected_root.sum()),
                              initializer_pelvis_proxy_frames=int((~selected_root).sum()),
                              initializer_proxy_added_to_observation_loss=False)
    return result


def _geometry_initialize(model, data, betas, world_up=(0., 0., 1.)):
    # Calibration selects separated snippets: never propagate a root/twist seed
    # across those skips, source resets, or actual timestamp gaps.
    cuts = np.flatnonzero(~data["explicit"].all(-1)) + 1
    edges = np.r_[0, cuts, len(data["times"])]
    if len(edges) == 2:
        return _initialize_segment(model, data, betas, world_up)
    local = np.broadcast_to(np.eye(3), (len(data["times"]), 22, 3, 3)).copy()
    root = np.zeros((len(data["times"]), 3))
    diagnostics = []
    for start, end in zip(edges[:-1], edges[1:]):
        snippet = _slice_data(data, np.arange(start, end))
        initialized = _initialize_segment(model, snippet, betas, world_up)
        local[start:end], root[start:end] = initialized["local_rotations"], initialized["root_world"]
        diagnostics.append({"start": int(start), "end": int(end), **initialized["metadata"]})
    return {"local_rotations": local, "root_world": root, "betas": betas.copy(),
            "metadata": {"method": "gap_reset_segmented_geometry_initializer",
                         "initialization_segments": np.stack((edges[:-1], edges[1:]), -1).tolist(),
                         "segment_initializations": diagnostics,
                         "absolute_twist_observed": False, "native_axis_certified": False}}


def _initialize(model, data, betas, world_up=(0., 0., 1.)):
    old = _geometry_initialize(model, data, betas, world_up)
    rest = model.rest_joints(betas).detach().cpu().numpy()
    with torch.no_grad():
        _, seed_G = model(torch.as_tensor(old['local_rotations'], dtype=model.dtype, device=model.device),
                          torch.as_tensor(old['root_world'], dtype=model.dtype, device=model.device),
                          torch.as_tensor(betas, dtype=model.dtype, device=model.device))
    seed_G = seed_G.detach().cpu().numpy()
    children = [np.flatnonzero(PARENTS == joint) for joint in range(22)]
    G = seed_G.copy()
    cuts = np.flatnonzero(~data['explicit'].all(-1)) + 1
    edges = np.r_[0, cuts, len(G)]
    anchor_frames = []
    projection_error_max = 0.

    def proper(rotation):
        # Repeated parent transport and almost-antiparallel swing corrections
        # can amplify tiny numerical errors. Projection only affects this
        # initializer estimate and never the original source rotations.
        nonlocal projection_error_max
        rotation = np.asarray(rotation, dtype=np.float64)
        left, _, right = np.linalg.svd(rotation)
        correction = np.eye(3)
        correction[-1, -1] = np.linalg.det(left @ right)
        projected = left @ correction @ right
        projection_error_max = max(projection_error_max, float(np.linalg.norm(projected - rotation)))
        return projected

    def geometry(frame, joint):
        active = [int(child) for child in children[joint]
                  if data['merged_valid'][frame, joint] and data['merged_valid'][frame, child]]
        source = np.array([rest[child] - rest[joint] for child in active]).reshape(-1, 3)
        target = np.array([data['target'][frame, child] - data['target'][frame, joint]
                           for child in active]).reshape(-1, 3)
        lengths = np.linalg.norm(source, axis=-1) * np.linalg.norm(target, axis=-1)
        source, target = source[lengths > 1e-12], target[lengths > 1e-12]
        if len(source):
            source /= np.linalg.norm(source, axis=-1)[:, None]
            target /= np.linalg.norm(target, axis=-1)[:, None]
        multi = len(source) >= 2 and np.linalg.matrix_rank(source, tol=.02) >= 2 and np.linalg.matrix_rank(target, tol=.02) >= 2
        return source, target, multi

    def step(frame, previous):
        edge = min(frame, previous)
        for joint in range(22):
            parent = int(PARENTS[joint])
            source_increment_available = bool(data['rvalid'][frame, joint] and data['rvalid'][previous, joint] and data['explicit'][edge, joint])
            if source_increment_available:
                predicted = data['source'][frame, joint] @ data['source'][previous, joint].T @ G[previous, joint]
            elif parent >= 0:
                predicted = G[frame, parent] @ G[previous, parent].T @ G[previous, joint]
            else:
                predicted = G[previous, joint]
            source, target, multi = geometry(frame, joint)
            if multi:
                # Multiple directions define geometry; no native absolute R.
                left, _, right = np.linalg.svd(target.T @ source)
                correction = np.eye(3)
                correction[-1, -1] = np.linalg.det(left @ right)
                measured = left @ correction @ right
                predicted = proper(predicted)
                # Geometry is a soft model-axis estimate. New hip/torso geometry
                # must not replace an already continuous world-delta seed.
                # The transported prediction retains actual large native
                # increments; this never imposes a hard velocity cap.
                fraction = .15 if source_increment_available else .35
                vector = _rotation_vectors(proper(predicted.T @ measured))
                G[frame, joint] = predicted @ so3_exp(torch.tensor(fraction * vector, dtype=torch.float64)).numpy()
            elif len(source):
                G[frame, joint] = _minimal_swing(predicted @ source[0], target[0]) @ predicted
            else:
                # Preserve the previous relative pose, not parent identity.
                G[frame, joint] = predicted
            G[frame, joint] = proper(G[frame, joint])

    for start, end in zip(edges[:-1], edges[1:]):
        anchor = next((frame for frame in range(start, end)
                       if data['valid'][frame, 0] and data['valid'][frame, 9]), start)
        anchor_frames.append(int(anchor))
        G[anchor] = seed_G[anchor]
        for frame in range(anchor + 1, end):
            step(frame, frame - 1)
        for frame in range(anchor - 1, start - 1, -1):
            step(frame, frame + 1)
    local = G.copy()
    for joint in range(1, 22):
        local[:, joint] = G[:, PARENTS[joint]].swapaxes(-1, -2) @ G[:, joint]
    # Offline local SO(3) bridges use reliable geometry on each side. These
    # are initializer estimates, never source observations or new edges.
    for start, end in zip(edges[:-1], edges[1:]):
        for joint in range(22):
            parent = int(PARENTS[joint])
            good = data['valid'][start:end, joint].copy()
            if parent >= 0:
                good &= data['valid'][start:end, parent]
            anchors = np.flatnonzero(good) + start
            if not len(anchors):
                continue
            local[start:anchors[0], joint] = local[anchors[0], joint]
            local[anchors[-1] + 1:end, joint] = local[anchors[-1], joint]
            for first, last in zip(anchors[:-1], anchors[1:]):
                if last - first <= 1:
                    continue
                relative = local[first, joint].T @ local[last, joint]
                relative = proper(relative)
                vector = _rotation_vectors(relative)
                fractions = (data['times'][first + 1:last] - data['times'][first]) / (data['times'][last] - data['times'][first])
                increment = so3_exp(torch.tensor(fractions[:, None] * vector, dtype=torch.float64)).numpy()
                local[first + 1:last, joint] = local[first, joint] @ increment
    return {**old, 'local_rotations': local,
            'metadata': {**old.get('metadata', {}), 'method': 'continuous_world_delta_transport_soft_geometry_and_local_SO3_gap_bridge_v2',
                         'anchor_frames': anchor_frames, 'source_absolute_axes_used': False,
                         'source_rotations_used': bool((data['rvalid'][1:] & data['rvalid'][:-1] & data['explicit']).any()),
                         'source_rotation_usage': 'world_increment_only; unknown_constant_right_basis_not_certified',
                         'multi_geometry_SO3_correction_fraction': {'source_increment_available': .15, 'otherwise': .35},
                         'proper_SO3_projection_max_frobenius_correction': projection_error_max,
                         'initializer_estimates_added_to_source_masks': False, 'offline_future_geometry_seed': True}}


def _hinge_reference(model, betas):
    """Soft model-rest bend planes, NOT measured anatomical hinge axes.

    SMPL-H rest spine defines up; its two ankle→foot directions define forward.
    A degenerate reference disables the heuristic rather than inventing an axis.
    """
    if model.metadata.get("test_fixture"):
        return None
    with torch.no_grad():
        rest = model.rest_joints(betas).detach().cpu().numpy()
    up = rest[9] - rest[0]
    if np.linalg.norm(up) < 1e-8:
        return None
    up /= np.linalg.norm(up)
    forward = (rest[10] - rest[7] + rest[11] - rest[8]) / 2
    forward -= np.dot(forward, up) * up
    if np.linalg.norm(forward) < 1e-6:
        return None
    forward /= np.linalg.norm(forward)
    indices, outgoing, desired, axes, max_angles = [], [], [], [], []
    for joint, child, sign, max_degrees in ((4, 7, -1, 155), (5, 8, -1, 155),
                                           (18, 20, 1, 165), (19, 21, 1, 165)):
        direction = rest[child] - rest[joint]
        if np.linalg.norm(direction) < 1e-8:
            continue
        direction /= np.linalg.norm(direction)
        bend = sign * forward - np.dot(sign * forward, direction) * direction
        if np.linalg.norm(bend) < .1:
            continue
        bend /= np.linalg.norm(bend)
        indices.append(joint)
        outgoing.append(direction)
        desired.append(bend)
        axes.append(np.cross(direction, bend))
        max_angles.append(np.deg2rad(max_degrees))
    if not indices:
        return None
    return dict(indices=np.asarray(indices), outgoing=np.asarray(outgoing), desired=np.asarray(desired),
                axes=np.asarray(axes), max_angles=np.asarray(max_angles))


def _hinge_scores(local, reference):
    direction = (local[:, reference["indices"]] @ reference["outgoing"][..., None])[..., 0]
    sine = (direction * reference["desired"]).sum(-1)
    cosine = (direction * reference["outgoing"]).sum(-1)
    planar = sine.square() + cosine.square()
    angle = torch.atan2(torch.where(planar < 1e-8, torch.zeros_like(sine), sine),
                        torch.where(planar < 1e-8, torch.ones_like(cosine), cosine))
    # Five degrees of extension tolerance accommodates rest/landmark semantics.
    reverse = torch.relu(-angle - np.deg2rad(5.))
    overbend = torch.relu(angle - reference["max_angles"])
    plane = (direction * reference["axes"]).sum(-1)
    return plane.square() + reverse.square() + overbend.square()


def _rotation_vectors(matrix):
    """Stable principal SO(3) logarithm, including half turns."""
    matrix = np.asarray(matrix, dtype=float)
    _validate_rotations(matrix, "rotations to logarithm")
    flat = matrix.reshape(-1, 3, 3)
    cosine = np.clip((np.trace(flat, axis1=-2, axis2=-1) - 1) / 2, -1., 1.)
    angles = np.arccos(cosine)
    skew = np.stack((flat[:, 2, 1] - flat[:, 1, 2], flat[:, 0, 2] - flat[:, 2, 0],
                     flat[:, 1, 0] - flat[:, 0, 1]), -1)
    scale = np.full(len(flat), .5)
    half_turn = np.pi - angles < 1e-5
    ordinary = (angles >= 1e-7) & ~half_turn
    scale[ordinary] = angles[ordinary] / (2 * np.sin(angles[ordinary]))
    out = skew * scale[:, None]
    for i in np.flatnonzero(half_turn):
        values, vectors = np.linalg.eigh((flat[i] + flat[i].T) / 2)
        axis = vectors[:, np.argmax(values)]
        if axis[np.argmax(np.abs(axis))] < 0:
            axis = -axis
        out[i] = axis * angles[i]
    return out.reshape(matrix.shape[:-2] + (3,))


def _torch_log(matrix):
    # Quaternion largest-diagonal branch avoids the SO(3) log singularity at pi.
    m00, m11, m22 = matrix[..., 0, 0], matrix[..., 1, 1], matrix[..., 2, 2]
    scores = torch.stack((1 + m00 + m11 + m22, 1 + m00 - m11 - m22,
                          1 - m00 + m11 - m22, 1 - m00 - m11 + m22), -1).clamp_min(1e-12)
    terms = torch.stack((
        torch.stack((scores[..., 0], matrix[..., 2, 1] - matrix[..., 1, 2], matrix[..., 0, 2] - matrix[..., 2, 0], matrix[..., 1, 0] - matrix[..., 0, 1]), -1),
        torch.stack((matrix[..., 2, 1] - matrix[..., 1, 2], scores[..., 1], matrix[..., 0, 1] + matrix[..., 1, 0], matrix[..., 0, 2] + matrix[..., 2, 0]), -1),
        torch.stack((matrix[..., 0, 2] - matrix[..., 2, 0], matrix[..., 0, 1] + matrix[..., 1, 0], scores[..., 2], matrix[..., 1, 2] + matrix[..., 2, 1]), -1),
        torch.stack((matrix[..., 1, 0] - matrix[..., 0, 1], matrix[..., 0, 2] + matrix[..., 2, 0], matrix[..., 1, 2] + matrix[..., 2, 1], scores[..., 3]), -1)), -2)
    branch = scores.argmax(-1)
    quat = torch.gather(terms, -2, branch[..., None, None].expand(*branch.shape, 1, 4))[..., 0, :]
    quat = quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True).clamp_min(1e-12)
    quat = torch.where(quat[..., :1] < 0, -quat, quat)
    norm = torch.linalg.vector_norm(quat[..., 1:], dim=-1, keepdim=True).clamp_min(1e-12)
    scale = 2 * torch.atan2(norm, quat[..., :1].clamp_min(0)) / norm
    return quat[..., 1:] * scale


def _palm_targets(data, reference):
    reference = np.asarray(reference, dtype=float)
    if reference.shape != (2, 3, 3) or not np.isfinite(reference).all():
        raise ValueError("target_palm_reference must be finite (2,3,3) wrist-local MCP vectors")
    lengths = np.linalg.norm(reference, axis=-1)
    if np.any(lengths <= 1e-8):
        raise ValueError("Target MCP references must be nonzero")
    ref = reference / lengths[..., None]
    if np.any(np.linalg.norm(np.cross(ref[:, 0], ref[:, 2]), axis=-1) < .05):
        raise ValueError("Target MCP references must define a noncollinear palm")
    vectors = data["hands"][:, :, [5, 9, 17]] - data["hands"][:, :, :1]
    lengths = np.linalg.norm(vectors, axis=-1)
    selected = data["hvalid"][:, :, [0, 5, 9, 17]].all(-1) & (lengths > 1e-8).all(-1)
    directions = vectors / np.maximum(lengths[..., None], 1e-12)
    selected &= np.linalg.norm(np.cross(directions[:, :, 0], directions[:, :, 2]), axis=-1) > .05
    return ref, directions, selected


def _contact_pair_mask(data, max_dt):
    # The caller supplies ground/proxy validity; native foot observations and
    # explicit gap/reset validity are also mandatory at both endpoints.
    observed_pairs = build_transition_mask(data["times"], data["valid"][:, [10, 11]],
                                           data["explicit"][:, [10, 11]], max_dt=max_dt)
    return (observed_pairs & data["contact"][1:] & data["contact"][:-1]
            & data["cvalid"][1:] & data["cvalid"][:-1])


def _contact_motion_loss(fitted, observed, dt, up, pair_mask):
    """Match observed foot motion under a contact proxy; never force foot Z=0.

    A speed-threshold proxy may include measured horizontal movement. Only
    source-stationary feet get a zero-velocity target. Select valid endpoints
    before arithmetic so masked missing observations cannot contaminate loss.
    """
    if not pair_mask.any():
        return fitted[:, :0].sum()
    delta_time = torch.broadcast_to(dt[:, None], pair_mask.shape)[pair_mask][:, None]
    fitted_velocity = (fitted[1:, [10, 11]][pair_mask] - fitted[:-1, [10, 11]][pair_mask]) / delta_time
    source_velocity = (observed[1:, [10, 11]][pair_mask] - observed[:-1, [10, 11]][pair_mask]) / delta_time
    difference = fitted_velocity - source_velocity
    horizontal = difference - (difference * up).sum(-1, keepdim=True) * up
    return horizontal.square().sum(-1).mean()


def _fit_block(model, data, betas, offsets, anchors, offset_valid, config, *,
               initial=None, estimate_offsets=False, shared_offsets=None, palm_reference=None,
               seam_local=None, seam_root=None, seam_mask=None, offset_variable_valid=None):
    count = len(data["times"])
    initial = _initialize(model, data, betas, config.world_up) if initial is None else initial
    local_start = np.asarray(initial["local_rotations"], dtype=float)
    root_start = np.asarray(initial["root_world"], dtype=float)
    _validate_rotations(local_start, "initial local rotations")
    if local_start.shape != (count, 22, 3, 3) or root_start.shape != (count, 3) or not np.isfinite(root_start).all():
        raise ValueError("Invalid block initialization")
    tensor = lambda value: torch.as_tensor(np.array(value, copy=True), dtype=model.dtype, device=model.device)
    mask = torch.as_tensor(data["merged_valid"], device=model.device)
    weights = tensor(data["weights"][data["merged_valid"]])
    target, beta_tensor, start = tensor(data["target"]), tensor(betas), tensor(local_start)
    enabled = torch.as_tensor(np.array(offset_valid, copy=True), device=model.device)
    variable = enabled if offset_variable_valid is None else torch.as_tensor(np.array(offset_variable_valid, copy=True), device=model.device)
    anchor_tensor = torch.as_tensor(anchors, dtype=torch.long, device=model.device)
    delta = torch.nn.Parameter(torch.zeros((count, 22, 3), dtype=model.dtype, device=model.device))
    root = torch.nn.Parameter(tensor(root_start))
    prior = np.zeros((22, 3)) if shared_offsets is None else shared_offsets.offsets.copy()
    prior_valid = np.zeros(22, dtype=bool) if shared_offsets is None else shared_offsets.offset_valid.copy()
    residual = torch.nn.Parameter(tensor(offsets - prior), requires_grad=estimate_offsets)
    prior_tensor = tensor(prior)
    source_delta = world_rotation_increments(tensor(data["source"]))
    rpairs = build_transition_mask(data["times"], data["rvalid"], data["explicit"], max_dt=config.max_dt)
    rpair_tensor = torch.as_tensor(rpairs, device=model.device)
    ppairs = build_transition_mask(data["times"], data["merged_valid"], data["explicit"], max_dt=config.max_dt)
    # Temporal priors support short explicitly declared completion, never reset/gap edges.
    temporal_mask = torch.as_tensor(data["explicit"], device=model.device)
    root_pairs = torch.as_tensor(data["explicit"].all(-1), device=model.device)
    contact_pairs = _contact_pair_mask(data, config.max_dt)
    cpairs = torch.as_tensor(contact_pairs, device=model.device)
    source_observations = tensor(data["observed"])
    dt = tensor(np.diff(data["times"]))
    up = tensor(config.world_up)
    if palm_reference is None:
        palm_mask, palm_ref, palm_observed = None, None, None
    else:
        ref, directions, selected = _palm_targets(data, palm_reference)
        palm_mask, palm_ref, palm_observed = torch.as_tensor(selected, device=model.device), tensor(ref), tensor(directions)
    with torch.no_grad():
        rest = model.rest_joints(beta_tensor)
    children = [np.flatnonzero(PARENTS == joint) for joint in range(22)]
    single = np.asarray([joint for joint in range(1, 22) if len(children[joint]) == 1])
    twist_axis = rest[[int(children[joint][0]) for joint in single]] - rest[single]
    twist_axis = twist_axis / torch.linalg.vector_norm(twist_axis, dim=-1, keepdim=True).clamp_min(1e-12)
    # Isotropic rest-relative activity envelopes, NOT learned anatomical limits.
    degrees = np.full(22, 175.)
    degrees[[3, 6, 9]] = 65.
    degrees[[12, 15]] = 100.
    degrees[[7, 8]] = 110.
    degrees[[13, 14]] = 100.
    limits = tensor(np.deg2rad(degrees[1:]))
    hinge = _hinge_reference(model, betas)
    if hinge is not None:
        hinge = {name: torch.as_tensor(value, dtype=torch.long if name == "indices" else model.dtype,
                                      device=model.device) for name, value in hinge.items()}
    seam_selected = None if seam_mask is None else torch.as_tensor(seam_mask, device=model.device)
    seam_l = None if seam_local is None else tensor(seam_local)
    seam_r = None if seam_root is None else tensor(seam_root)
    params = [delta, root] + ([residual] if estimate_offsets else [])
    groups = [{"params": [delta], "lr": config.pose_lr}, {"params": [root], "lr": config.root_lr}]
    if estimate_offsets:
        groups.append({"params": [residual], "lr": config.offset_lr})
    optimizer = torch.optim.Adam(groups)

    def objective():
        local = so3_exp(delta) @ start
        p, g = model(local, root, beta_tensor)
        effective_residual = residual * variable[:, None] if estimate_offsets else residual
        d = (prior_tensor + effective_residual) * enabled[:, None]
        predicted = p + (g[:, anchor_tensor] @ d[..., None])[..., 0]
        errors = (predicted[mask] - target[mask]).square().sum(-1)
        position_loss = (_pseudo_huber_squared(errors, config.position_huber_delta) * weights).sum() / weights.sum()
        zero = p.new_zeros(())
        increment_loss = (.5 * (world_rotation_increments(g)[rpair_tensor] - source_delta[rpair_tensor]).square().sum((-2, -1)).mean()
                          if rpair_tensor.any() else zero)
        log_local = _torch_log(local[:, 1:])
        rest_prior = log_local.square().sum(-1).mean()
        activity = torch.relu(torch.linalg.vector_norm(log_local, dim=-1) - limits).square().mean()
        twists = (_torch_log(local[:, single]) * twist_axis).sum(-1)
        twist_prior = twists.square().mean() if len(single) else zero
        hinge_prior = _hinge_scores(local, hinge).mean() if hinge is not None else zero
        temporal = (_torch_log(local[1:] @ local[:-1].transpose(-1, -2)).square().sum(-1)[temporal_mask].mean()
                    if temporal_mask.any() else zero)
        triples = root_pairs[1:] & root_pairs[:-1]
        root_velocity = (root[1:] - root[:-1]) / dt[:, None] if count > 1 else root[:0]
        root_smooth = ((root_velocity[1:] - root_velocity[:-1]).square().sum(-1)[triples].mean()
                       if triples.any() else zero)
        contact_loss = _contact_motion_loss(p, source_observations, dt, up, cpairs)
        palm_loss = ((g[:, [20, 21], None] @ palm_ref[..., None])[..., 0] - palm_observed).square().sum(-1)[palm_mask].mean() if palm_mask is not None and palm_mask.any() else zero
        if estimate_offsets:
            pvalid = torch.as_tensor(prior_valid & offset_valid, device=model.device)
            nvalid = enabled & ~pvalid
            offset_loss = (config.subject_offset_regularization * residual[pvalid].square().sum(-1).mean() if pvalid.any() else zero)
            offset_loss = offset_loss + (config.offset_regularization * d[nvalid].square().sum(-1).mean() if nvalid.any() else zero)
        else:
            offset_loss = zero
        seam_loss = ((local[seam_selected] - seam_l[seam_selected]).square().sum((-2, -1)).mean()
                     + (root[seam_selected] - seam_r[seam_selected]).square().sum(-1).mean()
                     if seam_selected is not None and seam_selected.any() else zero)
        total = (position_loss + config.rotation_increment_weight * increment_loss
                 + config.pose_increment_regularization * delta.square().mean()
                 + config.local_rest_regularization * rest_prior
                 + config.activity_regularization * activity + config.twist_regularization * twist_prior
                 + config.hinge_regularization * hinge_prior
                 + config.temporal_regularization * temporal
                 + config.root_temporal_regularization * root_smooth
                 + config.contact_velocity_weight * contact_loss
                 + config.palm_direction_weight * palm_loss
                 + config.seam_regularization * seam_loss + offset_loss)
        terms = {"loss": total, "position_loss": position_loss, "rotation_increment_loss": increment_loss,
                 "rest_prior": rest_prior, "activity_prior": activity, "twist_prior": twist_prior,
                 "model_rest_hinge_heuristic": hinge_prior,
                 "root_temporal_prior": root_smooth, "contact_velocity_loss": contact_loss,
                 "palm_direction_loss": palm_loss, "offset_prior_loss": offset_loss, "seam_prior": seam_loss}
        return total, terms, (p, g, local, predicted, d)

    best, best_loss, history = None, float("inf"), []
    for step in range(config.pose_iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, terms, _ = objective()
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite sequence fitting loss")
        if float(loss.detach()) < best_loss:
            best_loss = float(loss.detach())
            best = [item.detach().clone() for item in (delta, root, residual)]
        if step % config.history_every == 0 or step == config.pose_iterations:
            history.append({"iteration": step, **{key: float(value.detach()) for key, value in terms.items()}})
        if step == config.pose_iterations:
            break
        loss.backward()
        if any(item.grad is None or not torch.isfinite(item.grad).all() for item in params):
            raise RuntimeError("Non-finite sequence fitting gradient")
        torch.nn.utils.clip_grad_norm_(params, config.grad_clip)
        scale = config.lr_final_ratio + (1 - config.lr_final_ratio) * .5 * (1 + np.cos(np.pi * step / max(config.pose_iterations, 1)))
        for group, rate in zip(optimizer.param_groups, (config.pose_lr, config.root_lr, config.offset_lr)):
            group["lr"] = rate * scale
        optimizer.step()
        if estimate_offsets:
            with torch.no_grad():
                bounds = np.where(prior_valid, config.subject_offset_bound_m, config.offset_bound_m)
                lengths = torch.linalg.vector_norm(residual, dim=-1, keepdim=True)
                residual.mul_(torch.minimum(torch.ones_like(lengths), tensor(bounds[:, None]) / lengths.clamp_min(1e-12)))
                combined = prior_tensor + residual
                lengths = torch.linalg.vector_norm(combined, dim=-1, keepdim=True)
                combined *= torch.minimum(torch.ones_like(lengths), config.offset_bound_m / lengths.clamp_min(1e-12))
                residual.copy_(combined - prior_tensor)
                residual.mul_((enabled & variable)[:, None])
    with torch.no_grad():
        for parameter, value in zip((delta, root, residual), best):
            parameter.copy_(value)
        _, final_terms, output = objective()
    output = [value.detach().cpu().numpy().copy() for value in output]
    output.extend((root.detach().cpu().numpy().copy(), history,
                   {"best_objective": best_loss, "rotation_increment_constraints": int(rpairs.sum()),
                    "initialization": _json_copy(initial.get("metadata", {})),
                    "position_pair_constraints": int(ppairs.sum()), "contact_pair_constraints": int(contact_pairs.sum()),
                    "palm_constraints": 0 if palm_mask is None else int(palm_mask.sum()),
                    "model_rest_hinge_joints": [] if hinge is None else hinge["indices"].cpu().tolist(),
                    "final": {key: float(value) for key, value in final_terms.items()}}))
    return output


def calibrate_group(model, positions, position_mask, timestamps, source_global_rotations=None,
                    rotation_mask=None, *, hand_wrist_world=None, hand_wrist_valid=None,
                    hand_world=None, hand_valid=None, shared_offsets=None,
                    calibration_frame_mask=None, transition_mask=None, source_contact=None,
                    contact_valid=None, group_id="group", source_id="egosuite", fixed_betas=None,
                    target_palm_reference=None, config=None):
    """Fit safe limb shape once, then centers with shape frozen for the group."""
    _check_model(model)
    config = (config or SequenceConfig()).validate()
    if not isinstance(config, SequenceConfig):
        raise ValueError("config must be SequenceConfig")
    data = _inputs(positions, position_mask, timestamps, source_global_rotations, rotation_mask,
                   hand_wrist_world, hand_wrist_valid, hand_world, hand_valid, transition_mask,
                   source_contact, contact_valid, config)
    indices = _select_calibration(data, calibration_frame_mask, config)
    chosen = _slice_data(data, indices)
    low_coverage = len(indices) < config.min_calibration_frames
    shape_trusted = not low_coverage
    safe_children = np.array([4, 5, 7, 8, 10, 11, 18, 19, 20, 21])
    safe_pairs = chosen["merged_valid"][:, safe_children] & chosen["merged_valid"][:, PARENTS[safe_children]]
    if fixed_betas is None and not low_coverage:
        if safe_pairs.any():
            shape = fit_safe_limb_betas(model, chosen["target"], chosen["merged_valid"], config=config)
            beta = shape["betas"]
            shape_metadata = shape["metadata"]
        else:
            beta = np.zeros(model.shape_components)
            shape_metadata = {"kind": "zero_beta_prior_no_safe_limb_observations", "shape_optimized": False}
            shape_trusted = False
    elif fixed_betas is None:
        beta = np.zeros(model.shape_components)
        shape_metadata = {"kind": "zero_beta_prior_insufficient_calibration_coverage", "shape_optimized": False}
    else:
        beta = np.asarray(fixed_betas, dtype=float).copy()
        shape_metadata = {"kind": "caller_frozen_group_shape", "shape_optimized": False}
    if beta.shape != (model.shape_components,) or not np.isfinite(beta).all() or np.any(np.abs(beta) > config.beta_bound):
        raise ValueError("Frozen beta must match the model shape basis and bound")
    anchors = np.arange(22)
    anchors[list(config.offset_joints)] = config.offset_anchors
    if shared_offsets is not None:
        if (not isinstance(shared_offsets, SourceCenterPrior) or shared_offsets.source_id != source_id
                or shared_offsets.model_sha256 != model.metadata["model_sha256"]
                or not np.array_equal(shared_offsets.anchors, anchors)):
            raise ValueError("Source prior model/source/anchor contract mismatch")
        if np.any(np.linalg.norm(shared_offsets.offsets, axis=-1) > config.offset_bound_m):
            raise ValueError("Source offsets exceed configured bound")
        if np.any(shared_offsets.offset_valid & ~np.isin(np.arange(22), config.offset_joints)):
            raise ValueError("Source prior enables undeclared offset joints")
    counts = np.array([(chosen["merged_valid"][:, joint] & chosen["merged_valid"][:, anchors[joint]]).sum() for joint in range(22)])
    enabled = np.isin(np.arange(22), config.offset_joints) & (counts >= config.min_offset_samples)
    if low_coverage and fixed_betas is None:
        enabled[:] = False
    if shared_offsets is not None:
        enabled |= shared_offsets.offset_valid
    offsets = np.zeros((22, 3)) if shared_offsets is None else shared_offsets.offsets.copy()
    initial = _initialize(model, chosen, beta, config.world_up)
    if shared_offsets is None and enabled.any():
        with torch.no_grad():
            p, g = model(torch.as_tensor(initial["local_rotations"], dtype=model.dtype, device=model.device),
                         torch.as_tensor(initial["root_world"], dtype=model.dtype, device=model.device),
                         torch.as_tensor(beta, dtype=model.dtype, device=model.device))
        p, g = p.cpu().numpy(), g.cpu().numpy()
        for joint in np.flatnonzero(enabled):
            selected = chosen["merged_valid"][:, joint] & chosen["merged_valid"][:, anchors[joint]]
            error = chosen["target"][selected, joint] - p[selected, joint]
            offsets[joint] = np.einsum("tji,tj->ti", g[selected, anchors[joint]], error).mean(0)
        lengths = np.linalg.norm(offsets, axis=-1, keepdims=True)
        offsets *= np.minimum(1., config.offset_bound_m / np.maximum(lengths, 1e-12))
    palm_reference_derived = target_palm_reference is None and chosen["hvalid"].any() and not model.metadata.get("test_fixture")
    if palm_reference_derived:
        target_palm_reference = palm_reference_from_smplh_asset(model, beta)
    result = _fit_block(model, chosen, beta, offsets, anchors, enabled, config,
                        initial=initial, estimate_offsets=not (low_coverage and fixed_betas is None), shared_offsets=shared_offsets,
                        palm_reference=target_palm_reference,
                        offset_variable_valid=enabled & (counts >= config.min_offset_samples))
    offsets = result[4]
    prior = np.zeros((22, 3)) if shared_offsets is None else shared_offsets.offsets
    residual_norm = np.linalg.norm(offsets - prior, axis=-1)
    source_prior_valid = np.zeros(22, dtype=bool) if shared_offsets is None else shared_offsets.offset_valid
    fingerprint = _fingerprint(chosen["observed"], chosen["valid"], chosen["target"],
                               chosen["merged_valid"], chosen["weights"], chosen["times"],
                               chosen["source"], chosen["rvalid"], chosen["explicit"],
                               chosen["hands"], chosen["hvalid"], chosen["wrists"], chosen["wrist_valid"],
                               chosen["contact"], chosen["cvalid"], beta,
                               np.empty(0) if target_palm_reference is None else np.asarray(target_palm_reference),
                               context={"model": model.metadata["model_sha256"], "config": asdict(config),
                                        "group": group_id, "source": source_id,
                                        "shared_prior": None if shared_offsets is None else shared_offsets.input_sha256})
    diagnostics = {"mapping": "source_landmark = FK_center + G_anchor @ fixed_d",
                   "offset_norms_m": np.linalg.norm(offsets, axis=-1).tolist(),
                   "offset_bound_m": config.offset_bound_m,
                   "offset_bound_saturated": (enabled & (np.linalg.norm(offsets, axis=-1) >= .995 * config.offset_bound_m)).tolist(),
                   "subject_residual_norms_m": residual_norm.tolist(),
                   "subject_residual_bound_m": config.subject_offset_bound_m,
                   "subject_residual_bound_saturated": (source_prior_valid & (residual_norm >= .995 * config.subject_offset_bound_m)).tolist(),
                   "paired_observation_counts": counts.tolist(), "absolute_twist_observed": False,
                   "center_accuracy_established": False, "native_axis_certified": False}
    center = CenterCalibration(offsets, anchors, enabled, beta, model.metadata["model_sha256"], fingerprint, diagnostics)
    metadata = {"schema": "egomotion-smplh-group-calibration-v1", "config": asdict(config),
                "model": dict(model.metadata), "calibration_frame_indices": indices.tolist(),
                "calibration_frames": len(indices), "shape": shape_metadata,
                "safe_limb_observation_counts": safe_pairs.sum(0).tolist(),
                "beta_bound_saturated": (np.abs(beta) >= .995 * config.beta_bound).tolist(),
                "betas_frozen_after_limb_calibration": True, "beta_max_abs": float(np.abs(beta).max()),
                "source_prior_input_sha256": None if shared_offsets is None else shared_offsets.input_sha256,
                "subject_offsets_optimized": not (low_coverage and fixed_betas is None),
                "shape_absorbs_torso_center_loss": False,
                "calibration_status": "low_coverage_prior_fallback" if low_coverage and fixed_betas is None else "fitted_estimate",
                "shape_trusted_for_anthropometric_estimate": shape_trusted,
                "palm_reference_kind": ("shape_specific_SMPLH_asset_MCP_vectors" if palm_reference_derived
                                         else "caller_declared_target_wrist_local_MCP_vectors" if target_palm_reference is not None else "none"),
                "same_group_betas_and_centers_frozen_across_sequences": True,
                "calibration_fit": result[-1], "history": result[-2],
                "label_kind": "model_fitted_estimate", "native_axis_certified": False,
                "absolute_rotation_accuracy_established": False}
    return GroupCalibration(center, source_id, group_id, metadata)


def _segments(data, config):
    available = data["merged_valid"].any(1)
    filled = available.copy()
    count = len(available)
    start = 0
    while start < count:
        if available[start]:
            start += 1
            continue
        end = start + 1
        while end < count and not available[end]:
            end += 1
        if (start > 0 and end < count and end - start <= config.max_missing_frames
                and data["explicit"][start - 1:end].all()):
            filled[start:end] = True
        start = end
    segments = []
    start = 0
    while start < count:
        if not filled[start]:
            start += 1
            continue
        end = start + 1
        while end < count and filled[end] and data["explicit"][end - 1].all():
            end += 1
        segments.append((start, end))
        start = end
    return segments, filled


def _blend_so3(first, second, fraction):
    relative = np.swapaxes(first, -1, -2) @ second
    vector = _rotation_vectors(relative) * np.asarray(fraction)[..., None, None]
    increment = so3_exp(torch.as_tensor(vector, dtype=torch.float64)).numpy()
    return first @ increment


def fit_sequence(model, positions, position_mask, timestamps, source_global_rotations=None,
                 rotation_mask=None, *, calibration, hand_wrist_world=None, hand_wrist_valid=None,
                 hand_world=None, hand_valid=None, transition_mask=None, source_contact=None,
                 contact_valid=None, target_palm_reference=None, source_id=None, group_id=None, config=None):
    """Fit native-rate segments, fuse local SO(3)/root overlaps, then recompute FK."""
    _check_model(model)
    config = (config or SequenceConfig()).validate()
    if not isinstance(config, SequenceConfig) or not isinstance(calibration, GroupCalibration):
        raise ValueError("SequenceConfig and frozen GroupCalibration are required")
    if ((source_id is not None and source_id != calibration.source_id)
            or (group_id is not None and group_id != calibration.group_id)):
        raise ValueError("Sequence source/group differs from its frozen calibration")
    center, beta = calibration.center, calibration.betas.copy()
    anchors = np.arange(22)
    anchors[list(config.offset_joints)] = config.offset_anchors
    if (center.model_sha256 != model.metadata["model_sha256"]
            or beta.shape != (model.shape_components,) or not np.array_equal(center.anchors, anchors)
            or np.any(np.linalg.norm(center.offsets, axis=-1) > config.offset_bound_m + 1e-10)):
        raise ValueError("Frozen calibration model/beta/center contract mismatch")
    data = _inputs(positions, position_mask, timestamps, source_global_rotations, rotation_mask,
                   hand_wrist_world, hand_wrist_valid, hand_world, hand_valid, transition_mask,
                   source_contact, contact_valid, config)
    dt = np.diff(data["times"])
    regular_dt = dt[dt <= config.max_dt]
    if regular_dt.size and not np.isclose(np.median(regular_dt), 1 / config.fps, atol=1e-4, rtol=.02):
        raise ValueError("Sequence must be fitted at configured native fps; resample outside fitter")
    palm_reference_derived = target_palm_reference is None and data["hvalid"].any() and not model.metadata.get("test_fixture")
    if palm_reference_derived:
        target_palm_reference = palm_reference_from_smplh_asset(model, beta)
    segments, frame_available = _segments(data, config)
    count = len(data["times"])
    local = np.broadcast_to(np.eye(3), (count, 22, 3, 3)).copy()
    root = np.zeros((count, 3))
    mass = np.zeros(count)
    blocks, seams = [], []
    for segment_id, (start, end) in enumerate(segments):
        segment_data = _slice_data(data, np.arange(start, end))
        initial = _initialize(model, segment_data, beta, config.world_up)
        block_start = start
        while block_start < end:
            block_end = min(block_start + config.block_frames, end)
            indices = np.arange(block_start, block_end)
            chunk = _slice_data(data, indices)
            initial_chunk = {"local_rotations": initial["local_rotations"][indices - start].copy(),
                             "root_world": initial["root_world"][indices - start].copy(),
                             "metadata": {**initial.get("metadata", {}), "warm_start_overlap_frames": int((mass[indices] > 0).sum())}}
            overlap = mass[indices] > 0
            warm_local, warm_root = local[indices].copy(), root[indices].copy()
            initial_chunk["local_rotations"][overlap] = warm_local[overlap]
            initial_chunk["root_world"][overlap] = warm_root[overlap]
            result = _fit_block(model, chunk, beta, center.offsets, anchors, center.offset_valid,
                                config, initial=initial_chunk, palm_reference=target_palm_reference,
                                seam_local=warm_local, seam_root=warm_root, seam_mask=overlap)
            chunk_local, chunk_root = result[2], result[5]
            if overlap.any():
                angles = np.linalg.norm(_rotation_vectors(np.swapaxes(warm_local[overlap], -1, -2) @ chunk_local[overlap]), axis=-1)
                seams.append({"segment": segment_id, "start": block_start,
                              "pre_fusion_local_disagreement_deg": _summary(np.rad2deg(angles)),
                              "pre_fusion_root_disagreement_m": _summary(np.linalg.norm(warm_root[overlap] - chunk_root[overlap], axis=-1))})
            weights = np.ones(len(indices))
            ramp = min(config.overlap_frames, len(indices))
            if ramp:
                if block_start > start:
                    weights[:ramp] = np.arange(1, ramp + 1) / (ramp + 1)
                if block_end < end:
                    weights[-ramp:] = np.minimum(weights[-ramp:], np.arange(ramp, 0, -1) / (ramp + 1))
            fraction = weights / (mass[indices] + weights)
            local[indices] = _blend_so3(local[indices], chunk_local, fraction)
            root[indices] = root[indices] * (1 - fraction[:, None]) + chunk_root * fraction[:, None]
            mass[indices] += weights
            blocks.append({"segment": segment_id, "start": block_start, "end": block_end,
                           "fit": result[-1], "history": result[-2]})
            if block_end == end:
                break
            block_start = block_end - config.overlap_frames
    with torch.no_grad():
        p, g = model(torch.as_tensor(local, dtype=model.dtype, device=model.device),
                     torch.as_tensor(root, dtype=model.dtype, device=model.device),
                     torch.as_tensor(beta, dtype=model.dtype, device=model.device))
        j0 = model.rest_joints(beta)[0].cpu().numpy()
    p, g = p.cpu().numpy(), g.cpu().numpy()
    predicted = reconstruct_source_landmarks(p, g, center.offsets, anchors)
    estimate = np.broadcast_to(frame_available[:, None], (count, 22)).copy()
    errors = np.linalg.norm(predicted - data["target"], axis=-1)
    position_confidence = np.where(data["merged_valid"], np.exp(-errors / config.position_quality_scale_m), .15) * estimate
    rotation_confidence = .2 * estimate.astype(float)
    rpairs = build_transition_mask(data["times"], data["rvalid"], data["explicit"], max_dt=config.max_dt)
    rpairs &= frame_available[:-1, None] & frame_available[1:, None]
    inc_errors = rotation_increment_errors_deg(g, data["source"], rpairs)
    edge_scores = np.zeros(rpairs.shape)
    edge_scores[rpairs] = np.exp(-inc_errors / config.rotation_quality_scale_deg)
    scores_sum, scores_count = np.zeros((count, 22)), np.zeros((count, 22))
    scores_sum[:-1] += edge_scores
    scores_sum[1:] += edge_scores
    scores_count[:-1] += rpairs
    scores_count[1:] += rpairs
    # World increments support dynamics, never absolute orientation certification.
    rotation_confidence = np.where(scores_count > 0, .6 * scores_sum / np.maximum(scores_count, 1), rotation_confidence)
    rotation_confidence *= estimate
    fit_quality = np.minimum(position_confidence, rotation_confidence)
    contact_pairs = (_contact_pair_mask(data, config.max_dt)
                     & frame_available[1:, None] & frame_available[:-1, None])
    foot_velocity = (p[1:, [10, 11]] - p[:-1, [10, 11]]) / dt[:, None, None] if count > 1 else p[:0, [10, 11]]
    up = np.asarray(config.world_up)
    horizontal = foot_velocity - (foot_velocity * up).sum(-1, keepdims=True) * up
    source_foot_velocity = (data["observed"][1:, [10, 11]] - data["observed"][:-1, [10, 11]]) / dt[:, None, None] if count > 1 else p[:0, [10, 11]]
    source_horizontal = source_foot_velocity - (source_foot_velocity * up).sum(-1, keepdims=True) * up
    wrist_errors = np.linalg.norm(p[:, [20, 21]] - data["wrists"], axis=-1)
    local_vectors = _rotation_vectors(local)
    degrees = np.full(22, 175.)
    degrees[[3, 6, 9]], degrees[[12, 15]], degrees[[7, 8]], degrees[[13, 14]] = 65., 100., 110., 100.
    local_angles = np.rad2deg(np.linalg.norm(local_vectors, axis=-1))
    temporal_pairs = data["explicit"] & frame_available[:-1, None] & frame_available[1:, None]
    local_changes = np.linalg.norm(_rotation_vectors(np.swapaxes(local[:-1], -1, -2) @ local[1:]), axis=-1) if count > 1 else np.empty((0, 22))
    quality = {"observation_error_m": _summary(errors[data["merged_valid"] & estimate]),
               "original_body_landmark_error_m": _summary(np.linalg.norm(predicted - data["observed"], axis=-1)[data["valid"] & estimate]),
               "per_joint_observation_error_m": {name: _summary(errors[:, joint][data["merged_valid"][:, joint] & estimate[:, joint]]) for joint, name in enumerate(BODY_NAMES)},
               "rotation_increment_error_deg": _summary(inc_errors),
               "body_to_hand_wrist_error_m": _summary(wrist_errors[data["wrist_valid"] & frame_available[:, None]]),
               "source_contact_horizontal_foot_velocity_m_s": _summary(np.linalg.norm(horizontal, axis=-1)[contact_pairs]),
               "raw_contact_horizontal_foot_velocity_m_s": _summary(np.linalg.norm(source_horizontal, axis=-1)[contact_pairs]),
               "contact_foot_horizontal_motion_matching_error_m_s": _summary(np.linalg.norm(horizontal - source_horizontal, axis=-1)[contact_pairs]),
               "source_contact_vertical_foot_velocity_m_s": _summary(np.abs((foot_velocity * up).sum(-1))[contact_pairs]),
               "local_rotation_change_deg": _summary(np.rad2deg(local_changes[temporal_pairs])),
               "broad_activity_envelope_exceeded_count": int(((local_angles[:, 1:] > degrees[None, 1:]) & estimate[:, 1:]).sum()),
               "local_half_turn_estimate_count": int(((local_angles > 179.) & estimate).sum()),
               "contact_constraints": int(contact_pairs.sum()), "seams": seams,
               "position_confidence_kind": "observed_residual_proxy; completion_prior_0.15",
               "rotation_confidence_kind": "world_increment_residual_proxy_capped_0.6; no_absolute_accuracy",
               "source_observed_positions": int(data["valid"].sum()),
               "source_observed_rotations": int(data["rvalid"].sum()),
               "numeric_estimated_positions": int(estimate.sum()),
               "unavailable_frames": np.flatnonzero(~frame_available).tolist()}
    fingerprint = _fingerprint(data["observed"], data["valid"], data["weights"], data["times"], data["source"],
                               data["rvalid"], data["explicit"], data["hands"], data["hvalid"],
                               data["wrists"], data["wrist_valid"], data["contact"], data["cvalid"],
                               np.empty(0) if target_palm_reference is None else np.asarray(target_palm_reference),
                               context={"model": model.metadata["model_sha256"], "config": asdict(config),
                                        "calibration": calibration.input_sha256})
    metadata = {"schema": "egomotion-smplh-sequence-labels-v2", "model": dict(model.metadata),
                "config": asdict(config), "input_sha256": fingerprint,
                "calibration_input_sha256": calibration.input_sha256,
                "source_id": calibration.source_id, "group_id": calibration.group_id,
                "segments": [list(item) for item in segments], "blocks": blocks,
                "label_kind": "model_fitted_estimate", "target_joint_basis": model.metadata["target_joint_basis"],
                "target_model_axes_defined": True, "native_axis_certified": False,
                "source_rotation_basis_verified": False, "absolute_rotation_accuracy_established": False,
                "absolute_twist_observed": False, "betas_frozen": True, "center_offsets_frozen": True,
                "temporal_prior_kind": "principal_local_SO3_log_squared_and_root_velocity; reset_and_gap_aware",
                "pose_prior_kind": "rest_broad_activity_rotation_log_twist_and_model_rest_bend_plane_heuristics; no_learned_prior",
                "hinge_reference_kind": "target_rest_spine_up_and_ankle_to_foot_forward; soft_heuristic_not_anatomical_truth",
                "learned_anatomical_pose_prior_used": False,
                "palm_reference_kind": ("shape_specific_SMPLH_asset_MCP_soft_direction_estimate" if palm_reference_derived
                                         else "caller_declared_target_wrist_local_MCP_soft_direction_estimate" if target_palm_reference is not None else "none"),
                "palm_direction_is_independent_rotation_truth": False,
                "source_hand_landmark_semantics_verified_against_model": False,
                "source_contact_kind": "observed_contact_foot_motion_matching; caller_ground_validity_and_native_foot_masks; proxy_not_certified_sole_contact",
                "foot_joint_projected_to_ground": False,
                "overlap_fusion": "local_SO3_geodesic_and_root_weighted_then_FK",
                "source_inputs_modified": False, "camera_trajectory_modified": False,
                "estimate_validity_meaning": "finite_model_completion_available; never_source_observed",
                "smpl_translation_convention": "root_world_minus_rest_pelvis_J0"}
    return SequenceResult(p, g, local, local_vectors, root, root - j0, beta, predicted,
                          data["valid"].copy(), data["rvalid"].copy(), estimate.copy(), estimate.copy(),
                          estimate & ~data["valid"], fit_quality, position_confidence,
                          rotation_confidence, quality, metadata)
