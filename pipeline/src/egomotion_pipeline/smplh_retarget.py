"""Experimental SMPL-H body22 retargeting with explicit landmark offsets.

The output joint centers are always model FK centers.  Source landmarks are
modelled separately as p_observed[j] = p_FK[j] + G[anchor[j]] @ d[j].  Offsets
are constant during calibration and frozen on subsequent windows.  Neither
these offsets nor world rotation increments authenticate absolute joint twist.
This module is not a production converter and never changes an archive.
"""

from dataclasses import asdict, dataclass, field, fields
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import torch

from .schema import BODY_NAMES, PARENTS
from .smpl_fit import (
    FitConfig, FitResult, _boolean_mask, _positions_input, _pseudo_huber_squared,
    _timestamps_input, _validate_rotations, build_transition_mask,
    fit_shape_from_positions, initialize_local_rotations, load_body22_model,
    rotation_increment_errors_deg, so3_exp, world_rotation_increments,
)


@dataclass(frozen=True)
class RetargetConfig(FitConfig):
    offset_bound_m: float = .12
    offset_regularization: float = .01
    offset_lr: float = .01
    min_offset_samples: int = 8
    offset_joints: tuple = (12, 13, 14, 15)  # neck, collars, head
    offset_anchors: tuple = (9, 9, 9, 12)   # parent body segment

    def validate(self):
        FitConfig(**{item.name: getattr(self, item.name) for item in fields(FitConfig)}).validate()
        for name in ("offset_bound_m", "offset_lr", "offset_regularization"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0 or (name != "offset_regularization" and value == 0):
                raise ValueError(f"{name} must be finite and {'nonnegative' if name == 'offset_regularization' else 'positive'}")
        if (isinstance(self.min_offset_samples, bool) or not isinstance(self.min_offset_samples, int)
                or self.min_offset_samples < 1):
            raise ValueError("min_offset_samples must be a positive integer")
        if (not isinstance(self.offset_joints, tuple) or not isinstance(self.offset_anchors, tuple)
                or not self.offset_joints or len(self.offset_joints) != len(self.offset_anchors)
                or len(set(self.offset_joints)) != len(self.offset_joints)):
            raise ValueError("offset_joints and offset_anchors must be equal nonempty tuples with unique joints")
        for index in self.offset_joints + self.offset_anchors:
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < 22:
                raise ValueError("offset joints and anchors must be body22 indices")
        if 0 in self.offset_joints:
            raise ValueError("Pelvis offsets are excluded to avoid a free translation gauge")
        return self


@dataclass(frozen=True)
class CenterCalibration:
    offsets: np.ndarray
    anchors: np.ndarray
    offset_valid: np.ndarray
    betas: np.ndarray
    model_sha256: str
    input_sha256: str
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        values = {"offsets": np.asarray(self.offsets, dtype=np.float64),
                  "anchors": np.asarray(self.anchors),
                  "offset_valid": np.asarray(self.offset_valid),
                  "betas": np.asarray(self.betas, dtype=np.float64)}
        if (values["offsets"].shape != (22, 3) or values["anchors"].shape != (22,)
                or values["anchors"].dtype.kind not in "iu"
                or np.any((values["anchors"] < 0) | (values["anchors"] >= 22))
                or values["offset_valid"].shape != (22,)
                or values["offset_valid"].dtype != np.bool_
                or values["betas"].ndim != 1 or not len(values["betas"])
                or not np.isfinite(values["offsets"]).all() or not np.isfinite(values["betas"]).all()):
            raise ValueError("Invalid center calibration numerical arrays")
        if values["offset_valid"][0] or np.any(values["offsets"][~values["offset_valid"]] != 0):
            raise ValueError("Unavailable offsets and the pelvis offset must remain zero")
        for name in ("model_sha256", "input_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"{name} must be a lowercase SHA256")
        for name, value in values.items():
            value = value.copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        # An independent copy prevents held-out fitting from editing calibration evidence.
        object.__setattr__(self, "metadata", json.loads(json.dumps(self.metadata, allow_nan=False)))

    def as_dict(self):
        return {"offsets": self.offsets.tolist(), "anchors": self.anchors.tolist(),
                "offset_valid": self.offset_valid.tolist(), "betas": self.betas.tolist(),
                "model_sha256": self.model_sha256, "input_sha256": self.input_sha256,
                "metadata": json.loads(json.dumps(self.metadata, allow_nan=False))}


@dataclass
class RetargetResult(FitResult):
    reconstructed_observations: np.ndarray
    center_calibration: CenterCalibration | None
    observation_position_valid: np.ndarray
    rotation_increment_valid: np.ndarray
    estimate_available: np.ndarray


def _full_smplh_parents():
    parents = PARENTS.tolist()
    for wrist in (20, 21):
        for _finger in range(5):
            first = len(parents)
            parents.extend([wrist, first, first + 1])
    return np.asarray(parents)


def load_smplh_body22_model(path, *, shape_components=16, device="cpu",
                          dtype=torch.float64, allow_test_fixture=False):
    """Require a numerical 6890-vertex, 52-joint SMPL-H layout before taking body22.

    Layout checks do not establish asset authenticity or matching HM provenance.
    A 22-vertex/22-joint synthetic fixture is accepted only with the explicitly
    named test flag and remains labelled test_fixture, never SMPL-H.
    No implicit beta padding, pickle, or object arrays are supported.
    """
    path = Path(path)
    model = load_body22_model(path, shape_components=shape_components, device=device, dtype=dtype)
    snapshot = path.read_bytes()
    if sha256(snapshot).hexdigest() != model.metadata["model_sha256"]:
        raise ValueError("Model asset changed during required SMPL-H layout validation")
    with np.load(BytesIO(snapshot), allow_pickle=False) as asset:
        template = np.asarray(asset["v_template"])
        regressor = np.asarray(asset["J_regressor"])
        tree = np.asarray(asset["kintree_table"])
    fixture = (allow_test_fixture and len(template) == 22
               and regressor.shape == (22, 22) and tree.shape == (2, 22))
    if not fixture and (len(template) != 6890 or regressor.shape != (52, 6890) or tree.shape != (2, 52)):
        raise ValueError("Required SMPL-H asset must have 6890 vertices and 52 regressed/tree joints")
    if not np.isfinite(regressor).all():
        raise ValueError("Full SMPL-H joint regressor must be finite")
    lookup = {int(identifier): index for index, identifier in enumerate(tree[1])}
    try:
        parents = np.asarray([-1] + [lookup[int(value)] for value in tree[0, 1:]])
    except KeyError as error:
        raise ValueError("Full SMPL-H tree has an unknown parent") from error
    if not np.array_equal(parents, PARENTS if fixture else _full_smplh_parents()):
        raise ValueError("Required SMPL-H full kinematic hierarchy differs from the body-plus-hands layout")
    model.metadata.update(required_model_kind="SMPL-H", required_layout_verified=not fixture,
                          test_fixture=bool(fixture), implicit_shape_padding=False,
                          target_joint_basis="synthetic_body22_fixture" if fixture else "smplh_body22_axes_v1",
                          asset_authenticity_verified=False)
    return model


def fit_safe_limb_betas(model, positions, position_mask, *, config=None, frame_mask=None):
    """Optional limb-length calibration; exclude torso/head/hip-root center edges.

    This is an anthropometric estimate, not a surface-shape measurement.  It
    cannot solve source/model semantic differences even on the selected limbs.
    """
    _check_model(model)
    observed, valid = _positions_input(positions, position_mask)
    children = np.asarray([4, 5, 7, 8, 10, 11, 18, 19, 20, 21])
    selected = np.zeros(22, dtype=bool)
    selected[children] = True
    selected[PARENTS[children]] = True
    result = fit_shape_from_positions(model, observed, valid & selected[None],
                                     config=config, frame_mask=frame_mask)
    result["metadata"].update(allowed_child_joints=children.tolist(),
                               kind="safe_limb_only_shape_calibration",
                               source_model_limb_centers_assumed_corresponding=True,
                               torso_head_centers_used=False)
    return result


def _check_model(model):
    meta = model.metadata
    if (meta.get("required_model_kind") != "SMPL-H"
            or not (meta.get("required_layout_verified") is True or meta.get("test_fixture") is True)):
        raise ValueError("Retargeting requires load_smplh_body22_model and its explicit layout contract")


def _fingerprint(*arrays, context=None):
    digest = sha256(json.dumps(context or {}, sort_keys=True, allow_nan=False).encode())
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(str(array.dtype).encode())
        digest.update(json.dumps(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def reconstruct_source_landmarks(positions, global_rotations, offsets, anchors):
    """Keep model joint centers distinct from predicted source landmarks."""
    positions = np.asarray(positions, dtype=np.float64)
    rotations = np.asarray(global_rotations, dtype=np.float64)
    offsets = np.asarray(offsets, dtype=np.float64)
    anchors = np.asarray(anchors)
    if (positions.ndim != 3 or positions.shape[1:] != (22, 3)
            or rotations.shape != (*positions.shape[:2], 3, 3)
            or offsets.shape != (22, 3) or anchors.shape != (22,) or anchors.dtype.kind not in "iu"
            or np.any((anchors < 0) | (anchors >= 22))
            or not np.isfinite(positions).all() or not np.isfinite(offsets).all()):
        raise ValueError("Invalid model-center / source-landmark mapping inputs")
    _validate_rotations(rotations, "landmark anchor rotations")
    return positions + np.einsum("tjik,jk->tji", rotations[:, anchors], offsets)


def _summary(values):
    values = np.asarray(values, dtype=float)
    return {"count": int(values.size),
            "mean": float(values.mean()) if values.size else None,
            "p95": float(np.percentile(values, 95)) if values.size else None,
            "max": float(values.max()) if values.size else None}


def calibrate_center_offsets(model, positions, position_mask, timestamps,
                             source_global_rotations=None, rotation_mask=None, *,
                             fixed_betas=None, config=None, initialization=None,
                             transition_mask=None):
    """Estimate fixed landmark offsets and calibration-window poses jointly.

    Betas default to zero and are never fitted here.  This default isolates
    center effects; it does not complete anthropometric retargeting.  Offsets
    start from mean anchor-local residuals of the shared geometry-only pose
    initialization.  Nothing here establishes absolute twist ground truth.
    """
    return _fit(model, positions, position_mask, timestamps, source_global_rotations,
                rotation_mask, fixed_betas=fixed_betas, config=config,
                initialization=initialization, transition_mask=transition_mask,
                calibration=None, estimate_offsets=True)


def fit_smplh_retarget(model, positions, position_mask, timestamps,
                      source_global_rotations=None, rotation_mask=None, *,
                      fixed_betas=None, center_calibration=None, config=None,
                      initialization=None, transition_mask=None):
    """Fit poses only; optional center_calibration freezes both offsets and beta.

    Without a calibration this is the no-offset baseline.  Supply the same
    fixed beta and geometry initialization for a controlled comparison.
    """
    return _fit(model, positions, position_mask, timestamps, source_global_rotations,
                rotation_mask, fixed_betas=fixed_betas, config=config,
                initialization=initialization, transition_mask=transition_mask,
                calibration=center_calibration, estimate_offsets=False)


def _fit(model, positions, position_mask, timestamps, source_global_rotations,
         rotation_mask, *, fixed_betas, config, initialization, transition_mask,
         calibration, estimate_offsets):
    _check_model(model)
    config = (config or RetargetConfig()).validate()
    if not isinstance(config, RetargetConfig):
        raise ValueError("config must be RetargetConfig")
    observed, valid = _positions_input(positions, position_mask)
    count = len(observed)
    timestamps = _timestamps_input(timestamps, count)
    position_pairs = build_transition_mask(timestamps, valid, transition_mask, max_dt=config.max_dt)
    anchors = np.arange(22)
    anchors[list(config.offset_joints)] = config.offset_anchors
    if calibration is not None:
        if not isinstance(calibration, CenterCalibration):
            raise ValueError("center_calibration must be CenterCalibration")
        if (calibration.model_sha256 != model.metadata["model_sha256"]
                or not np.array_equal(calibration.anchors, anchors)):
            raise ValueError("Frozen calibration model/anchor contract mismatch")
        betas = calibration.betas.copy()
        if fixed_betas is not None and not np.array_equal(np.asarray(fixed_betas), betas):
            raise ValueError("Held-out betas must match the frozen center calibration")
        offset_valid = calibration.offset_valid.copy()
        offsets = calibration.offsets.copy()
        if np.any(np.linalg.norm(offsets, axis=-1) > config.offset_bound_m + 1e-10):
            raise ValueError("Frozen offsets exceed the configured offset bound")
        if np.any(offset_valid & ~np.isin(np.arange(22), config.offset_joints)):
            raise ValueError("Frozen calibration enables an undeclared offset joint")
        beta_kind = "frozen_calibration_betas"
    else:
        betas = np.zeros(model.shape_components) if fixed_betas is None else np.asarray(fixed_betas, dtype=np.float64).copy()
        offsets, offset_valid = np.zeros((22, 3)), np.zeros(22, dtype=bool)
        beta_kind = "zero_shape_center_isolation_experiment" if fixed_betas is None or not np.any(betas) else "caller_supplied_fixed_anthropometric_estimate"
    if (betas.shape != (model.shape_components,) or not np.isfinite(betas).all()
            or np.any(np.abs(betas) > config.beta_bound)):
        raise ValueError("fixed_betas must be finite, model-matched and within beta_bound")

    rotation_valid = np.zeros((count, 22), dtype=bool)
    source = np.broadcast_to(np.eye(3), (count, 22, 3, 3)).copy()
    if source_global_rotations is not None:
        raw_rotations = np.asarray(source_global_rotations, dtype=np.float64)
        if raw_rotations.shape != source.shape:
            raise ValueError("source_global_rotations must have shape (T,22,3,3)")
        rotation_valid = (np.isfinite(raw_rotations).all((-2, -1)) if rotation_mask is None
                          else _boolean_mask(rotation_mask, (count, 22), "rotation_mask"))
        # Validate all selected rotations, including isolated selected frames.
        _validate_rotations(raw_rotations[rotation_valid], "selected source rotations")
        source = np.where(rotation_valid[..., None, None], raw_rotations, source)
    elif rotation_mask is not None:
        raise ValueError("rotation_mask requires source_global_rotations")
    rotation_pairs = build_transition_mask(timestamps, rotation_valid, transition_mask, max_dt=config.max_dt)
    if initialization is None:
        initialization = initialize_local_rotations(model, observed, valid, betas)
    if not isinstance(initialization, dict) or not {"local_rotations", "root_world"} <= initialization.keys():
        raise ValueError("initialization must contain local_rotations and root_world")
    local_start = np.asarray(initialization["local_rotations"], dtype=np.float64).copy()
    root_start = np.asarray(initialization["root_world"], dtype=np.float64).copy()
    if local_start.shape != (count, 22, 3, 3) or root_start.shape != (count, 3) or not np.isfinite(root_start).all():
        raise ValueError("Initialization shape/finite-value mismatch")
    if "betas" in initialization and not np.array_equal(np.asarray(initialization["betas"]), betas):
        raise ValueError("Initialization betas differ from the frozen beta contract")
    _validate_rotations(local_start, "initial rotations")
    init_metadata = json.loads(json.dumps(initialization.get("metadata", {"method": "caller_supplied"}), allow_nan=False))
    tensor = lambda value: torch.as_tensor(np.array(value, copy=True), dtype=model.dtype, device=model.device)
    input_sha = _fingerprint(observed, valid, timestamps, source, rotation_valid,
                             position_pairs, rotation_pairs, betas,
                             context={"model_sha256": model.metadata["model_sha256"]})
    initialization_sha = _fingerprint(local_start, root_start, betas)
    offset_counts = np.asarray([(valid[:, joint] & valid[:, anchors[joint]]).sum() for joint in range(22)])
    if estimate_offsets:
        requested = np.zeros(22, dtype=bool)
        requested[list(config.offset_joints)] = True
        offset_valid = requested & (offset_counts >= config.min_offset_samples)
        if not offset_valid.any():
            raise ValueError("No requested center has enough paired landmark/anchor observations")
        with torch.no_grad():
            initial_p, initial_g = model(tensor(local_start), tensor(root_start), tensor(betas))
        initial_p, initial_g = initial_p.cpu().numpy(), initial_g.cpu().numpy()
        for joint in np.flatnonzero(offset_valid):
            selected = valid[:, joint] & valid[:, anchors[joint]]
            residual = observed[selected, joint] - initial_p[selected, joint]
            offsets[joint] = np.einsum("tji,tj->ti", initial_g[selected, anchors[joint]], residual).mean(0)
        norms = np.linalg.norm(offsets, axis=-1, keepdims=True)
        offsets *= np.minimum(1., config.offset_bound_m / np.maximum(norms, 1e-12))

    selected_positions = torch.as_tensor(valid, device=model.device)
    selected_position_pairs = torch.as_tensor(position_pairs, device=model.device)
    selected_rotation_pairs = torch.as_tensor(rotation_pairs, device=model.device)
    offset_enabled = torch.as_tensor(offset_valid, device=model.device)
    anchor_tensor = torch.as_tensor(anchors, dtype=torch.long, device=model.device)
    target, beta_tensor, local_tensor = tensor(observed), tensor(betas), tensor(local_start)
    source_deltas = world_rotation_increments(tensor(source))
    pose_delta = torch.nn.Parameter(torch.zeros((count, 22, 3), dtype=model.dtype, device=model.device))
    root = torch.nn.Parameter(tensor(root_start))
    offset_tensor = torch.nn.Parameter(tensor(offsets), requires_grad=estimate_offsets)
    groups = [{"params": [pose_delta], "lr": config.pose_lr}, {"params": [root], "lr": config.root_lr}]
    if estimate_offsets:
        groups.append({"params": [offset_tensor], "lr": config.offset_lr})
    optimizer = torch.optim.Adam(groups)
    eye = torch.eye(3, dtype=model.dtype, device=model.device)
    if count > 1:
        dt = np.diff(timestamps)
        temporal_scale = tensor(np.minimum((np.median(dt) / dt) ** 2, 100.))[:, None]
    else:
        temporal_scale = tensor(np.empty((0, 1)))

    def objective():
        local = so3_exp(pose_delta) @ local_tensor
        fitted, global_rotations = model(local, root, beta_tensor)
        effective_offsets = offset_tensor * offset_enabled[:, None]
        predicted = fitted + (global_rotations[:, anchor_tensor] @ effective_offsets[..., None])[..., 0]
        residual = predicted[selected_positions] - target[selected_positions]
        position_loss = _pseudo_huber_squared(residual.square().sum(-1), config.position_huber_delta).mean()
        if selected_rotation_pairs.any():
            delta_error = world_rotation_increments(global_rotations)[selected_rotation_pairs] - source_deltas[selected_rotation_pairs]
            increment_loss = .5 * delta_error.square().sum((-2, -1)).mean()
        else:
            increment_loss = fitted.new_zeros(())
        increment_prior = pose_delta.square().mean()
        local_rest_prior = .5 * (local[:, 1:] - eye).square().sum((-2, -1)).mean()
        if selected_position_pairs.any():
            changes = .5 * (local[1:] - local[:-1]).square().sum((-2, -1))
            temporal_prior = (changes * temporal_scale)[selected_position_pairs].mean()
        else:
            temporal_prior = fitted.new_zeros(())
        offset_prior = effective_offsets[offset_enabled].square().sum(-1).mean() if offset_enabled.any() else fitted.new_zeros(())
        total = (position_loss + config.rotation_increment_weight * increment_loss
                 + config.pose_increment_regularization * increment_prior
                 + config.local_rest_regularization * local_rest_prior
                 + config.temporal_regularization * temporal_prior
                 + (config.offset_regularization * offset_prior if estimate_offsets else 0))
        terms = {"loss": total, "position_loss": position_loss,
                 "rotation_increment_loss": increment_loss, "offset_prior_loss": offset_prior,
                 "pose_increment_prior_loss": increment_prior, "local_rest_prior_loss": local_rest_prior,
                 "temporal_prior_loss": temporal_prior,
                 "observation_rmse_m": residual.square().sum(-1).mean().sqrt()}
        return total, terms, (fitted, global_rotations, local, predicted)

    history, best_loss, best_iteration = [], float("inf"), 0
    best_state = None
    parameters = [pose_delta, root] + ([offset_tensor] if estimate_offsets else [])
    for iteration in range(config.pose_iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, terms, _ = objective()
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite offset-aware retargeting objective")
        value = float(loss.detach())
        if value < best_loss:
            best_loss, best_iteration = value, iteration
            best_state = [item.detach().clone() for item in (pose_delta, root, offset_tensor)]
        if iteration % config.history_every == 0 or iteration == config.pose_iterations:
            history.append({"iteration": iteration, **{key: float(item.detach()) for key, item in terms.items()}})
        if iteration == config.pose_iterations:
            break
        loss.backward()
        if any(item.grad is None or not torch.isfinite(item.grad).all() for item in parameters):
            raise RuntimeError("Non-finite offset-aware retargeting gradient")
        torch.nn.utils.clip_grad_norm_(parameters, config.grad_clip)
        annealing = config.lr_final_ratio + (1 - config.lr_final_ratio) * .5 * (1 + np.cos(np.pi * iteration / max(config.pose_iterations, 1)))
        for group, learning_rate in zip(optimizer.param_groups, (config.pose_lr, config.root_lr, config.offset_lr)):
            group["lr"] = learning_rate * annealing
        optimizer.step()
        if estimate_offsets:
            with torch.no_grad():
                offset_tensor.mul_(offset_enabled[:, None])
                lengths = torch.linalg.vector_norm(offset_tensor, dim=-1, keepdim=True)
                offset_tensor.mul_(torch.minimum(torch.ones_like(lengths), config.offset_bound_m / lengths.clamp_min(1e-12)))
    with torch.no_grad():
        for item, best in zip((pose_delta, root, offset_tensor), best_state):
            item.copy_(best)
        _, final_terms, (fitted, global_rotations, local, predicted) = objective()
        smpl_trans = root - model.rest_joints(beta_tensor)[0]
    numpy = lambda value: value.detach().cpu().numpy().copy()
    p, g, local_np, root_np, predicted_np, offset_np = [numpy(value) for value in (fitted, global_rotations, local, root, predicted, offset_tensor)]
    observation_errors = np.linalg.norm(predicted_np - observed, axis=-1)
    center_distances = np.linalg.norm(p - observed, axis=-1)
    increments = rotation_increment_errors_deg(g, source, rotation_pairs)
    lengths = np.linalg.norm(offset_np, axis=-1)
    saturated = offset_valid & (lengths >= config.offset_bound_m * .995)
    offset_metadata = {"mapping": "source_landmark = model_fk_center + model_anchor_world_rotation @ fixed_local_offset",
                       "offset_bound_m": config.offset_bound_m, "offset_regularization": config.offset_regularization,
                       "offset_norms_m": lengths.tolist(), "offset_bound_saturated": saturated.tolist(),
                       "offset_bound_saturated_count": int(saturated.sum()),
                       "paired_window_observation_counts": offset_counts.tolist(),
                       "offset_regularization_applied": bool(estimate_offsets),
                       "offsets_identifiable_from_independent_reference": False,
                       "absolute_twist_observed": False,
                       "initial_offset_method": ("mean_anchor_local_residual_of_shared_pose_initialization"
                                                 if estimate_offsets else "frozen_calibration_artifact" if calibration else "zero_baseline"),
                       "source_or_subject_offset_estimate_only": True}
    if estimate_offsets:
        calibration = CenterCalibration(offset_np, anchors, offset_valid, betas,
                                        model.metadata["model_sha256"], input_sha, offset_metadata)
    metadata = {"schema": "egomotion-smplh-center-aware-pilot-v1", "model": dict(model.metadata),
                "config": asdict(config), "input_sha256": input_sha,
                "initialization_sha256": initialization_sha,
                "label_kind": "model_fitted_estimate", "target_model_axes_defined": True,
                "target_joint_basis": model.metadata["target_joint_basis"],
                "native_axis_certified": False, "source_rotation_basis_verified": False,
                "absolute_rotation_accuracy_established": False, "absolute_twist_observed": False,
                "anatomical_pose_prior_used": False,
                "shape_optimized": False, "shape_kind": beta_kind,
                "zero_beta_shape_isolation": bool(not np.any(betas)),
                "center_offsets_optimized": estimate_offsets, "center_offsets_frozen": not estimate_offsets,
                "calibration_input_sha256": calibration.input_sha256 if calibration else None,
                "mode": "center_calibration" if estimate_offsets else "frozen_offsets" if calibration else "no_offset_baseline",
                "output_position_kind": "model_fk_joint_centers; source_landmarks_are_separate",
                "camera_trajectory_modified": False, "source_inputs_modified": False,
                "position_metrics_world_alignment": "none", "position_constraints": int(valid.sum()),
                "rotation_increment_constraints": int(rotation_pairs.sum()),
                "position_neighbor_constraints": int(position_pairs.sum()),
                "weak_absolute_orientation_joints": [BODY_NAMES[index] for index in (10, 11, 15, 20, 21)],
                "iterations": config.pose_iterations, "best_iteration": best_iteration,
                "returned_best_objective": best_loss,
                "final": {key: float(value) for key, value in final_terms.items()},
                "observation_error_m": _summary(observation_errors[valid]),
                "model_center_to_source_landmark_distance_m": _summary(center_distances[valid]),
                "rotation_increment_error_deg": _summary(increments), "offset_diagnostics": offset_metadata,
                "per_joint": {name: {"observation_error_m": _summary(observation_errors[:, joint][valid[:, joint]]),
                                      "model_center_to_source_landmark_distance_m": _summary(center_distances[:, joint][valid[:, joint]])}
                              for joint, name in enumerate(BODY_NAMES)}}
    init = {"local_rotations": local_start, "root_world": root_start, "betas": betas.copy(), "metadata": init_metadata}
    estimate_available = np.broadcast_to(valid.any(1)[:, None], (count, 22)).copy()
    return RetargetResult(p, g, local_np, root_np, numpy(smpl_trans), betas.copy(), init, history, metadata,
                          predicted_np, calibration, valid.copy(), rotation_pairs.copy(), estimate_available)


def pool_center_calibrations(calibrations, group_ids):
    """Equal-person mean of calibration offsets, with frozen-model/beta checks.

    This pools estimated calibration-window offsets; it is not a joint global
    optimum or independent center validation.  A group may appear only once.
    Hold-out groups must never be passed to this function.
    """
    calibrations, group_ids = list(calibrations), list(group_ids)
    if (len(calibrations) < 2 or len(calibrations) != len(group_ids)
            or any(not isinstance(group, str) or not group for group in group_ids)
            or len(set(group_ids)) != len(group_ids)
            or any(not isinstance(item, CenterCalibration) for item in calibrations)):
        raise ValueError("Pooling requires at least two distinct nonempty calibration groups")
    first = calibrations[0]
    for item in calibrations[1:]:
        if (item.model_sha256 != first.model_sha256 or not np.array_equal(item.betas, first.betas)
                or not np.array_equal(item.anchors, first.anchors)):
            raise ValueError("Pooled center calibration model, betas or anchors differ")
    available = np.stack([item.offset_valid for item in calibrations])
    # A shared source correction requires more than one calibration person.
    valid = available.sum(0) >= 2
    offsets, scatter, counts = np.zeros((22, 3)), np.zeros(22), available.sum(0)
    for joint in np.flatnonzero(valid):
        vectors = np.stack([item.offsets[joint] for item in calibrations])[available[:, joint]]
        offsets[joint] = vectors.mean(0)
        scatter[joint] = np.sqrt(np.mean(np.sum((vectors - offsets[joint]) ** 2, axis=-1)))
    metadata = {"kind": "equal_group_mean_of_estimated_calibration_offsets",
                "group_ids": group_ids, "calibration_groups": len(group_ids),
                "per_joint_calibration_groups": counts.tolist(),
                "per_joint_offset_scatter_rms_m": scatter.tolist(),
                "individual_input_sha256": [item.input_sha256 for item in calibrations],
                "individual_offset_diagnostics": [item.metadata for item in calibrations],
                "native_axis_certified": False, "absolute_rotation_accuracy_established": False,
                "independent_reference_identifiability_established": False,
                "shape_kind": "same_fixed_beta_for_center_isolation"}
    input_sha = _fingerprint(offsets, available, first.betas,
                             context={"groups": group_ids, "inputs": metadata["individual_input_sha256"], "model": first.model_sha256})
    return CenterCalibration(offsets, first.anchors, valid, first.betas,
                             first.model_sha256, input_sha, metadata)
