"""Compare the dataset coordinates against the actual MotionExpert functions.

Torch is optional for the pipeline. The isolated namespace loads only numerical
reference modules, without importing MotionExpert's training/model packages.
"""

import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from egomotion_pipeline.body_model import BodyModel
from egomotion_pipeline.geometry import axis_angle_to_quat, matrix_to_quat, quat_to_matrix
from egomotion_pipeline.head_reference import EE4D_CAMERA_FROM_REFERENCE
from egomotion_pipeline.resampling import resample, transform_episode
from egomotion_pipeline.schema import Episode, PARENTS
from egomotion_pipeline.windows import prepare_window


@pytest.fixture
def motionexpert_reference(monkeypatch):
    torch = pytest.importorskip("torch")
    root = Path(__file__).resolve().parents[2]/"MotionExpert"
    if not root.is_dir():
        pytest.skip("MotionExpert reference source is not included in this checkout")
    namespace = "_pipeline_motionexpert_parity"
    # Pre-register package shells so no root/utils/dataset __init__ executes.
    for suffix in ["", ".utils", ".dataset"]:
        package = ModuleType(namespace+suffix)
        package.__path__ = [str(root/suffix.lstrip("."))] if suffix else [str(root)]
        monkeypatch.setitem(sys.modules, namespace+suffix, package)
    try:
        canonicalization = importlib.import_module(namespace+".dataset.canonicalization")
        representation = importlib.import_module(namespace+".dataset.representation_utils")
        yield torch, canonicalization, representation
    finally:
        # Child modules imported normally are outside monkeypatch's bookkeeping.
        for name in list(sys.modules):
            if name.startswith(namespace+".") and name not in {
                namespace+".utils", namespace+".dataset",
            }:
                sys.modules.pop(name, None)


@pytest.mark.parametrize("basis", ["aria_rgb_native", "egosuite_opencv_camera_to_world"])
def test_dense22_matches_motionexpert_after_window_canonicalization(
        motionexpert_reference, model_path, basis, record_property):
    torch, canonicalization, representation = motionexpert_reference
    count, floor = 8, 2.
    time = np.arange(count)/30
    # Camera reference axes at level: left, up, optical-forward.
    level_reference = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
    yaw = quat_to_matrix(axis_angle_to_quat(np.stack([
        np.zeros(count), np.zeros(count), .4 + time*1.2], axis=-1)))
    pitch = quat_to_matrix(axis_angle_to_quat(np.stack([
        .12 + time*.7, np.zeros(count), np.zeros(count)], axis=-1)))
    roll = quat_to_matrix(axis_angle_to_quat(np.stack([
        np.zeros(count), np.zeros(count), -.1 + time*.8], axis=-1)))
    reference_rotation = yaw @ level_reference @ pitch @ roll
    native_from_reference = (EE4D_CAMERA_FROM_REFERENCE if basis == "aria_rgb_native"
                             else np.diag([-1., -1., 1.]))
    camera_rotation = reference_rotation @ native_from_reference.T
    camera_position = np.array([.5, -1.8, floor+1.65]) + time[:, None]*[.8, -.3, .1]

    # Real FK in the existing synthetic body fixture provides mutually
    # consistent joint positions and local/root rotations for canonicalization.
    trans = np.array([-.5, -4., -1.2]) + time[:, None]*[.3, .2, .05]
    root_aa = np.stack([time*.1, time*-.2, .6 + time*.5], axis=-1)
    local_aa = np.zeros((count, 21, 3))
    local_aa[:, 17, 0] = time*.4
    local_aa[:, 18, 1] = -.15 - time*.3
    body, body_rotation, valid = BodyModel(model_path).decode(
        trans, root_aa, local_aa.reshape(count, 63), np.zeros((count, 2)),
        np.ones(count, bool))
    episode = Episode("motionexpert-parity", "fixture", time,
                      {"body": body, "camera": camera_position[:, None]},
                      {"body": body_rotation,
                       "camera": matrix_to_quat(camera_rotation)[:, None]},
                      position_valid={"body": valid},
                      metadata={"camera_pose_basis": basis})
    arrays = resample(transform_episode(episode, {
        "coordinate_verified": True,
        "ground": {"mode": "manual", "height_m": floor, "verified": True},
    }))
    assert arrays["observation.valid.body_head_floor_position"].all()

    tensor = lambda value: torch.tensor(value, dtype=torch.float32)
    head = torch.eye(4)[None].repeat(count, 1, 1)
    head[:, :3, :3] = tensor(quat_to_matrix(
        arrays["observation.state.head_reference_world_rotation"]))
    head[:, :3, 3] = tensor(arrays["observation.state.head_reference_world"])
    joints = tensor(arrays["observation.state.body_world"].reshape(count, 22, 3))
    params = {
        "global_orient": tensor(quat_to_matrix(axis_angle_to_quat(root_aa))),
        "body_pose": tensor(quat_to_matrix(axis_angle_to_quat(local_aa))),
        "transl": tensor(trans),
        "betas": torch.zeros(count, 10),
        "left_hand_pose": torch.eye(3)[None, None].repeat(count, 15, 1, 1),
        "right_hand_pose": torch.eye(3)[None, None].repeat(count, 15, 1, 1),
    }
    # Only parents and PCA bases are needed by the actual representation path;
    # no licensed mesh asset or complete training dependencies are required.
    skeleton = SimpleNamespace(parents=list(PARENTS),
                               left_hand_components=torch.eye(45)[:12],
                               right_hand_components=torch.eye(45)[:12])
    can_head, can_params, can_joints = canonicalization.canonicalize_trajectory(
        head, params, joints)
    assert torch.max(torch.abs(can_head[0, :2, 3])).item() < 1e-6
    # Use all canonicalized inputs together, including the rotated root pose.
    motion, trajectory = representation.saved_sequence_to_repre_v4_beta(
        can_head, can_params, can_joints, floor, skeleton)
    assert motion.shape == (count, 243)
    assert trajectory.shape == (count, 18)
    reference_positions = motion[:, :198].reshape(count, 22, 9)[..., 6:9].numpy()
    pipeline_positions = arrays["observation.state.body_head_floor"].reshape(count, 22, 3)
    maximum_error = float(np.max(np.abs(reference_positions-pipeline_positions)))
    record_property("maximum_position_error_m", maximum_error)
    np.testing.assert_allclose(pipeline_positions, reference_positions, atol=2e-6, rtol=1e-6)
    # Both retain height relative to the explicit nonzero floor.
    np.testing.assert_allclose(trajectory[:, 8].numpy(),
                               arrays["observation.state.head_reference_height"].reshape(-1), atol=1e-6)
    # The pipeline reader must select FPS before recomputing head deltas.
    window = prepare_window(arrays, episode.metadata, start_frame=0, num_frames=4, fps=15)
    selected_head, selected_params, selected_joints = canonicalization.canonicalize_trajectory(
        head[::2], {key: value[::2] for key, value in params.items()}, joints[::2])
    selected_motion, selected_trajectory = representation.saved_sequence_to_repre_v4_beta(
        selected_head, selected_params, selected_joints, floor, skeleton)
    np.testing.assert_allclose(window.trajectory, selected_trajectory.numpy(), atol=2e-6)
    np.testing.assert_allclose(window.arrays["observation.state.body_head_floor"].reshape(4, 22, 3),
                               selected_motion[:, :198].reshape(4, 22, 9)[..., 6:9].numpy(), atol=2e-6)
    record_property("maximum_selected_head_error", float(np.max(
        np.abs(window.trajectory-selected_trajectory.numpy()))))
    # Also compare the actual legacy zero-sentinel fallback, with all feet valid.
    # The pipeline uses an explicit false mask instead of assuming zero is absent.
    fallback_meta = {**episode.metadata, "ground": {
        **episode.metadata["ground"], "mode": "estimate", "height_m": 0.,
        "valid": False, "verified": False, "reason": "no_stationary_feet",
        "window_fallback": {"mode": "mean_valid_feet", "offset_m": .02}}}
    fallback = prepare_window(arrays, fallback_meta, start_frame=0, num_frames=4, fps=15)
    fallback_motion, fallback_trajectory = representation.saved_sequence_to_repre_v4_beta(
        selected_head, selected_params, selected_joints, 0., skeleton)
    np.testing.assert_allclose(fallback.trajectory, fallback_trajectory.numpy(), atol=2e-6)
    np.testing.assert_allclose(fallback.arrays["observation.state.body_head_floor"].reshape(4, 22, 3),
                               fallback_motion[:, :198].reshape(4, 22, 9)[..., 6:9].numpy(), atol=2e-6)
    record_property("maximum_fallback_head_error", float(np.max(
        np.abs(fallback.trajectory-fallback_trajectory.numpy()))))
