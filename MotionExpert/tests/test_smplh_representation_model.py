"""Sparse K12 geometry from complete SMPL-H FK; no shape/pose prediction."""
import copy
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn

from MotionExpert.config import smplh_motion_expert_config, motion_expert_config
from MotionExpert.dataset.motion_layout import motion_layout, SMPLH_REPRE_TYPE, K12_JOINT_INDICES
from MotionExpert.dataset.smplh_representation import build_smplh_window, decode_smplh_motion
from MotionExpert.model import uniegomotion as model_module
from MotionExpert.motion_expert import MotionExpert
from MotionExpert.mydiffusion.flow_matching import FlowMatching
from MotionExpert.tests.fixtures import conditioning
from MotionExpert.tests.test_dense_model import InspectDecoder
from MotionExpert.utils.rotation_conversions import axis_angle_to_matrix
from MotionExpert.utils.task_conditioning import apply_task_conditioning


def world_fixture():
    count = 11
    arrays = {"timestamp": np.arange(count, dtype=np.float32) / 5}
    body = np.zeros((count, 22, 3), np.float64)
    body[..., 0] = 2 + np.arange(22)[None] * .02
    body[..., 1] = 3 + np.arange(count)[:, None] * .02
    body[..., 2] = .3 + np.arange(22)[None] * .02
    angles = np.arange(22)[None] * .03 + np.arange(count)[:, None] * .01
    axes = np.stack([np.cos(np.arange(22)), np.sin(np.arange(22)), np.ones(22)], axis=-1)
    axes /= np.linalg.norm(axes, axis=-1, keepdims=True)
    bodyq = np.zeros((count, 22, 4))
    bodyq[..., 0], bodyq[..., 1:] = np.cos(angles/2), axes[None]*np.sin(angles[..., None]/2)
    head = np.broadcast_to([2., 3., 1.7], (count, 1, 3)).copy()
    head[:, 0, 0] += np.arange(count) * .04
    yaw = np.arange(count) * .04
    headq = np.zeros((count, 1, 4))
    headq[:, 0, 0], headq[:, 0, 3] = np.cos(yaw/2), np.sin(yaw/2)
    values = {"body": (body, bodyq), "head_reference": (head, headq)}
    for side in ("hand_left", "hand_right"):
        hand = np.repeat(head, 21, axis=1)
        hand[..., 2] -= .25
        hand[..., 0] += np.arange(21)[None] * .01
        if side == "hand_right": hand[..., 1] += .3
        values[side] = (hand, None)
    for name, (position, quaternion) in values.items():
        arrays[f"observation.state.{name}_world"] = position.reshape(count, -1)
        arrays[f"observation.valid.{name}_position"] = np.ones(position.shape[:2], bool)
        if quaternion is not None:
            arrays[f"observation.state.{name}_world_rotation"] = quaternion.reshape(count, -1)
            arrays[f"observation.valid.{name}_rotation"] = np.ones(position.shape[:2], bool)
    arrays["source.feet_world"] = np.broadcast_to([0, 0, .22, .1, 0, .22], (count, 6)).copy()
    arrays["source.feet_valid"] = np.ones((count, 2), bool)
    arrays["observation.valid.head_reference_basis"] = np.ones(count, bool)
    arrays["smplh.betas"] = np.arange(count*16, dtype=np.float32).reshape(count, 16)*.001
    metadata = {"coordinate_verified": True, "head_reference": {"basis_verified": True},
                "ground": {"valid": True, "height_m": .2, "verified": False},
                "rotation_basis_verified": False, "target_model_axes_defined": True,
                "target_joint_basis": "smplh_body22_axes_v1",
                "rotation_supervision_policy": "smplh_model_defined"}
    return arrays, metadata


def test_layout_same_245_channels_but_distinct_source_semantics():
    legacy, smplh = motion_layout(245), motion_layout(SMPLH_REPRE_TYPE)
    assert smplh.motion_dim == 245 and smplh.joint_indices == K12_JOINT_INDICES
    assert legacy.representation != smplh.representation
    for name in ("body_slice", "global_slice", "left_hand_slice", "right_hand_slice", "contact_slice"):
        assert getattr(legacy, name) == getattr(smplh, name)
    assert smplh.betas_slice is None
    regions = [smplh.body_slice, smplh.global_slice, smplh.left_hand_slice, smplh.right_hand_slice, smplh.contact_slice]
    assert [i for region in regions for i in range(region.start, region.stop)] == list(range(245))
    for invalid in (351, 335, True, "smplh22_rot6d_hands21_contacts_betas16_v1", None):
        with pytest.raises(ValueError): motion_layout(invalid)


def test_sparse_world_roundtrip_preserves_exact_k12_and_hands():
    arrays, meta = world_fixture()
    before = {key: value.copy() for key, value in arrays.items()}
    window = build_smplh_window(arrays, meta)
    assert window.motion.shape == (11, 245) and window.motion_valid.all()
    assert window.metadata["rotation_basis_verified"] is False
    assert not window.metadata["training_window"]["shape_is_prediction_target"]
    assert not window.metadata["training_window"]["shape_is_condition"]
    result = decode_smplh_motion(torch.tensor(window.motion), floor_height=.2,
        window_to_world=window.metadata["training_window"]["archive_world_from_window"])
    np.testing.assert_allclose(result["body_positions"].numpy(), arrays["observation.state.body_world"].reshape(11,22,3)[:,K12_JOINT_INDICES,:], atol=1e-6)
    np.testing.assert_allclose(result["left_hand"].numpy(), arrays["observation.state.hand_left_world"].reshape(11,21,3), atol=1e-6)
    source_q = torch.tensor(arrays["observation.state.body_world_rotation"].reshape(11,22,4)[:,K12_JOINT_INDICES,:], dtype=torch.float32)
    axis = source_q[...,1:]/source_q[...,1:].norm(dim=-1,keepdim=True).clamp_min(1e-8)
    angle = 2*torch.atan2(source_q[...,1:].norm(dim=-1), source_q[...,0])
    torch.testing.assert_close(result["body_rotations"], axis_angle_to_matrix(axis*angle[...,None]), atol=2e-6, rtol=0)
    assert not {"betas", "pose_body", "trans", "root_orient", "local_axis_angle"}.intersection(result)
    assert result["body_positions"].shape == (11,12,3)
    for key in arrays: np.testing.assert_array_equal(arrays[key], before[key])


def test_native_feet_prefix_floor_and_contacts_independent_from_fitted_body():
    arrays, meta = world_fixture()
    meta["ground"] = {"valid": False, "mode": "estimate", "window_fallback": {"mode": "mean_valid_feet", "offset_m": .02}}
    expected = build_smplh_window(arrays, meta)
    assert expected.metadata["ground"]["height_m"] == pytest.approx(.2)
    assert expected.motion[:,243:245].all()
    changed = copy.deepcopy(arrays)
    changed["source.feet_world"][6:, [2,5]] = 500
    changed["observation.state.body_world"][:,32] = 900
    actual = build_smplh_window(changed, meta)
    assert actual.metadata["ground"]["height_m"] == expected.metadata["ground"]["height_m"]
    changed["source.feet_valid"][:6] = False
    assert not build_smplh_window(changed, meta).motion_valid.any()


def test_model_axes_masks_and_source_shape_do_not_add_prediction_channels():
    arrays, meta = world_fixture()
    arrays["observation.valid.body_rotation"][:,20] = False
    arrays["observation.valid.hand_left_position"][:,0] = False
    result = build_smplh_window(arrays, meta)
    sparse_wrist = K12_JOINT_INDICES.index(20)
    assert not result.motion_valid[:,sparse_wrist*9:sparse_wrist*9+6].any()
    assert result.motion_valid[:,sparse_wrist*9+6:(sparse_wrist+1)*9].all()
    assert not result.motion_valid[:,117:120].any() and result.motion_valid[:,120:180].all()
    shape_changed = copy.deepcopy(arrays)
    shape_changed["smplh.betas"][:] = 42  # Builder selects the reader's FK; it has no beta target channels.
    np.testing.assert_array_equal(build_smplh_window(shape_changed,meta).motion, result.motion)
    for name, value in (("target_model_axes_defined", False), ("target_joint_basis", "egosuite_native_v1"),
                        ("rotation_supervision_policy", "source_native"), ("body_basis_correction", np.eye(3))):
        with pytest.raises(ValueError): build_smplh_window(arrays, dict(meta, **{name:value}))
    broken = copy.deepcopy(arrays)
    broken["smplh.betas"][4,8] = np.nan
    with pytest.raises(ValueError, match="shape coefficients"): build_smplh_window(broken,meta)


def test_per_frame_camera_basis_never_upgrades_floor_labels():
    arrays, meta = world_fixture()
    arrays["observation.valid.head_reference_basis"][7] = False
    before = arrays["observation.valid.head_reference_rotation"].copy()
    result = build_smplh_window(arrays,meta)
    assert not result.motion_valid[7,:243].any()
    assert not result.trajectory_valid[7].any()
    assert not result.motion_valid[8,108:117].any()
    assert result.motion_valid[8,:108].all()
    assert result.motion_valid[7,243:].all()  # Source feet contact has its own availability.
    np.testing.assert_array_equal(arrays["observation.valid.head_reference_rotation"],before)
    for wrong in (np.ones(11,dtype=int), np.ones((11,2),dtype=bool)):
        broken = dict(arrays, **{"observation.valid.head_reference_basis":wrong})
        with pytest.raises(ValueError, match="boolean"): build_smplh_window(broken,meta)


def test_model_has_original_input_output_sizes_and_joint_image_only_gradients():
    torch.set_num_threads(1)
    with patch.object(model_module,"MoEDecoderBlock",InspectDecoder):
        model = model_module.UniEgoMotion(smplh_motion_expert_config(),dropout=0).train()
        original = model_module.UniEgoMotion(motion_expert_config(),dropout=0)
    assert {k:tuple(v.shape) for k,v in model.state_dict().items()} == {k:tuple(v.shape) for k,v in original.state_dict().items()}
    assert model.input_process.in_features == model.output_process.out_features == 245
    assert model.sparse_joint_indices == K12_JOINT_INDICES
    y = apply_task_conditioning(conditioning(batch=1,frames=11),"fore",forecast_prefix=6,
                                trajectory_condition=False,supervise_history=True)
    x = torch.randn(1,11,245)
    prediction,hidden,global_hidden = model(x,torch.ones(1),y,return_features=True)
    assert prediction.shape == (1,11,245) and hidden.shape == (1,11,768) and global_hidden.shape == (1,768)
    prediction.retain_grad()
    prediction.square().mean().backward()
    assert prediction.grad[:,:6].norm()>0 and prediction.grad[:,6:].norm()>0
    y["traj"][:] = float("nan"); y["img_embs"][:,6:] = float("nan")
    torch.testing.assert_close(model(x,torch.ones(1),y),prediction)


def test_sparse_flow_groups_do_not_include_shape_supervision():
    class Zero(nn.Module):
        def forward(self,x,t,y): return torch.zeros_like(x)
    target = torch.zeros(1,11,245)
    target[:,:,108:117] = 2
    valid = torch.ones_like(target,dtype=torch.bool); valid[:,:,117:180] = False
    y = {"valid_frames":torch.ones(1,11,dtype=torch.bool),"feature_valid_mask":valid}
    result = FlowMatching().training_losses(Zero(),target,model_kwargs={"y":y},t=torch.tensor([.5]))
    assert result["global_mse"].item() == 4 and result["body_rotation_valid_count"].item() == 11*12*6
    assert result["left_hand_valid_count"].item() == 0 and not any("beta" in name for name in result)
    assert result["loss"].item() == pytest.approx((11*9*4*8)/(11*(245-63-9+9*8)))


def test_image_only_api_forecast_decode_stays_sparse_without_model_or_gt_shape():
    class Prediction(nn.Module):
        def __init__(self):
            super().__init__(); template = torch.zeros(245)
            template[:108].view(12,9)[:,:6] = torch.tensor([1.,0,0,0,1,0])
            template[108:114] = torch.tensor([1.,0,0,0,1,0])
            self.template = nn.Parameter(template)
        def forward(self,x,t,y,cond_scale=None,return_features=False,return_hidden=False):
            self.last_y = y; prediction = self.template[None,None].expand_as(x)
            hidden = torch.nn.functional.pad(x,(0,768-x.shape[-1]))
            if return_features: return prediction,hidden,hidden.mean(1)
            if return_hidden: return prediction,hidden
            return prediction
    class IdentityNormalizer:
        @staticmethod
        def denormalize(x,*args): return x
    expert = MotionExpert.__new__(MotionExpert); expert.cfg = smplh_motion_expert_config()
    expert.model,expert.flow,expert.normalizer = Prediction().eval(),FlowMatching(),IdentityNormalizer()
    expert.device,expert.dtype = torch.device("cpu"),torch.float32
    expert.weights_path,expert.weights_kind = "unit-test","model"
    images = conditioning(batch=1,frames=6)["img_embs"]
    result = expert.predict_future({"img_embs":images},seed=62,decode=True,
                                  decode_kwargs={"initial_floor_transform":torch.eye(4)})
    assert result["future_motion"].shape == (1,5,245) and result["motion"].shape == (1,11,245)
    assert result["decoded"]["body_positions"].shape == (1,5,12,3)
    assert not {"betas","pose_body","trans","root_orient","local_axis_angle"}.intersection(result["decoded"])
    assert "betas" not in expert.model.last_y and "motion_generation_mask" not in expert.model.last_y
    assert expert.model.last_y["traj_mask"].all() and not expert.model.last_y["trajectory_valid"].any()
    assert result["metadata"]["representation"] == SMPLH_REPRE_TYPE
