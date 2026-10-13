"""SMPL-H full22 FK selected to K12; explicit hands, no shape prediction."""
from copy import deepcopy

import numpy as np
import torch

from .representation import MotionWindow, build_motion_window, decode_motion as _decode_motion
from .motion_layout import SMPLH_REPRE_TYPE, motion_layout

REPRE_TYPE = SMPLH_REPRE_TYPE
MOTION_DIM = 245
TARGET_JOINT_BASIS = "smplh_body22_axes_v1"
PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)


def build_smplh_window(arrays, metadata, *, task="fore", forecast_prefix=6, fps=5,
                       contact_speed_m_s=.2, contact_height_m=.1):
    """Select K12 from the reader's shape-conditioned world FK.

    Shape16 belongs to the source FK and sample metadata. It is neither part
    of the generated 245-channel state nor an external model condition.
    Ground fallback and contact labels use the retained source feet.
    """
    if (metadata.get("target_model_axes_defined") is not True
            or metadata.get("target_joint_basis") != TARGET_JOINT_BASIS
            or metadata.get("rotation_supervision_policy") != "smplh_model_defined"):
        raise ValueError("SMPL-H supervision requires explicitly defined model body22 axes")
    if metadata.get("body_basis_correction") is not None:
        raise ValueError("Model-defined SMPL-H poses must not receive a native basis correction")
    betas = np.asarray(arrays["smplh.betas"])
    count = len(arrays["timestamp"])
    if betas.shape != (count, 16) or not np.isfinite(betas).all():
        raise ValueError("SMPL-H source FK requires finite per-frame sixteen actual shape coefficients")
    basis = np.asarray(arrays["observation.valid.head_reference_basis"])
    head_valid = np.asarray(arrays["observation.valid.head_reference_rotation"])
    for name, mask in (("head_reference_basis", basis), ("head_reference_rotation", head_valid)):
        if mask.dtype != np.bool_ or mask.shape not in ((count,), (count, 1)):
            raise ValueError(f"{name} must contain per-frame boolean validity")
    checked_arrays = dict(arrays)
    checked_arrays["observation.valid.head_reference_rotation"] = (
        head_valid.reshape(count) & basis.reshape(count)).reshape(head_valid.shape)
    window = build_motion_window(
        checked_arrays, metadata, task=task, forecast_prefix=forecast_prefix, fps=fps,
        rotation_basis_verified=True, joint_indices=motion_layout(REPRE_TYPE).joint_indices,
        source_feet_world=arrays["source.feet_world"],
        source_feet_valid=np.asarray(arrays["source.feet_valid"]),
        contact_speed_m_s=contact_speed_m_s, contact_height_m=contact_height_m)
    meta = deepcopy(window.metadata)
    meta["training_window"].update(
        representation=REPRE_TYPE, target_joint_basis=TARGET_JOINT_BASIS,
        rotation_basis_mode="smplh_model_defined", target_model_axes_defined=True,
        label_kind="model_fitted_estimate", shape_policy="source_beta16_for_full_fk_only",
        shape_is_prediction_target=False, shape_is_condition=False,
        body_fk_joint_count=22, body_joint_indices=list(motion_layout(REPRE_TYPE).joint_indices),
        ground_fallback_body="native_source_feet", contact_body="native_source_feet")
    return MotionWindow(window.motion, window.motion_valid, window.trajectory, window.trajectory_valid, meta)


def decode_smplh_motion(motion, **kwargs):
    """Decode only K12 global poses and independent hands21.

    Sparse predictions cannot determine the omitted joints, a complete SMPL-H
    local pose, shape or transl. No complete parametric body is invented here.
    """
    value = torch.as_tensor(motion)
    if value.ndim != 2 or value.shape[-1] != MOTION_DIM:
        raise ValueError("SMPL-H sparse decode requires T x 245 motion")
    return _decode_motion(value, **kwargs)


decode_motion = decode_smplh_motion
