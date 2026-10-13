"""Regression checks for dense motion losses and five-token conditioning."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from MotionExpert.config import motion_expert_config
from MotionExpert.model import uniegomotion as model_module
from MotionExpert.model.core import MoEFeedForward
from MotionExpert.mydiffusion.flow_matching import FlowMatching
from MotionExpert.mydiffusion.mask_utils import feature_mask_for
from MotionExpert.utils.task_conditioning import apply_task_conditioning


class InspectDecoder(nn.Module):
    """Exercise real condition assembly without allocating twelve large MoEs."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.ff = SimpleNamespace(last_aux_loss=None)
        self.call = None

    def forward(self, **kwargs):
        self.call = kwargs
        context, mask = kwargs['context'], kwargs['context_mask'][:, 0, 0]
        pooled = (context * mask[:, :, None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        return kwargs['x'] + pooled[:, None] * 0.001


class ConstantPrediction(nn.Module):
    def __init__(self, value=0.0):
        super().__init__()
        self.value = nn.Parameter(torch.full((245,), float(value)))

    def forward(self, x, t, y, **kwargs):
        return self.value.view(1, 1, 245).expand_as(x)


class DenseModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self):
        with patch.object(model_module, 'MoEDecoderBlock', InspectDecoder):
            return model_module.UniEgoMotion(motion_expert_config(), dropout=0).eval()

    def conditioning(self, frames=4):
        return {'valid_frames': torch.ones(1, frames, dtype=torch.bool),
                'traj': torch.randn(1, frames, 18),
                'trajectory_valid': torch.ones(1, frames, 18, dtype=torch.bool),
                'img_embs': torch.randn(1, frames, 5, 1024),
                'valid_img_embs': torch.ones(1, frames, dtype=torch.bool)}

    def test_register_context_shape_and_temporal_encoding(self):
        model, y = self.model(), self.conditioning()
        prediction, hidden, global_hidden = model(torch.zeros(1, 4, 245), torch.ones(1), y,
                                                  return_features=True)
        self.assertEqual(prediction.shape, (1, 4, 245))
        self.assertEqual(hidden.shape, (1, 4, 768))
        self.assertEqual(global_hidden.shape, (1, 768))
        call = model.tsfm[-1].call
        self.assertEqual(call['context'].shape, (1, 21, 768))
        self.assertEqual(call['context_mask'].shape, (1, 1, 1, 21))
        encoding = model.pos_enc(torch.zeros(1, 4, 5, 768))
        self.assertTrue(torch.equal(encoding[:, :, 0], encoding[:, :, 4]))
        self.assertFalse(torch.equal(encoding[:, 0], encoding[:, 1]))
        self.assertFalse(any('text' in name for name, _ in model.named_parameters()))

    def test_cfg_unconditional_uses_five_token_rank(self):
        model, y = self.model(), self.conditioning()
        x, t = torch.zeros(1, 4, 245), torch.ones(1)
        actual = model(x, t, y, cond_scale=0, return_features=True)
        expected = model(x, t, {'valid_frames': y['valid_frames']}, return_features=True)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b)

    def test_hidden_future_cannot_leak_through_router_or_attention(self):
        model, y = self.model(), self.conditioning()
        y = apply_task_conditioning(y, 'fore', forecast_prefix=2)
        expected = model(torch.zeros(1, 4, 245), torch.ones(1), y)
        y['traj'][:, 2:] = float('nan')
        y['img_embs'][:, 2:] = float('nan')
        actual = model(torch.zeros(1, 4, 245), torch.ones(1), y)
        torch.testing.assert_close(actual, expected)
        self.assertFalse(model.tsfm[-1].call['router_conditions']['ego_valid_frames'][0, 2:].any())
        actual.square().mean().backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_forecast_future_image_availability_is_not_an_input(self):
        model, y = self.model(), self.conditioning()
        y = apply_task_conditioning(y, 'fore', forecast_prefix=2)
        x, t = torch.zeros(1, 4, 245), torch.ones(1)
        expected = model(x, t, y)
        y['valid_img_embs'][:, 2:] = False
        actual = model(x, t, y)
        torch.testing.assert_close(actual, expected)
        context_mask = model.tsfm[-1].call['context_mask'][0, 0, 0]
        self.assertFalse(context_mask[11:].any())

    def test_missing_data_masks_compose_with_original_training_dropout(self):
        model = self.model().train()
        cond = torch.ones(1, 4, 768)
        with patch('torch.rand', side_effect=lambda *a, **kw: torch.zeros(*a, **kw)):
            actual = model.mask_cond(cond, 'traj', torch.zeros(1, 4, dtype=torch.bool))
        torch.testing.assert_close(actual, model.mask_tokens['traj'].view(1, 1, -1).expand_as(cond))
        with patch('torch.rand', side_effect=AssertionError('Explicit tasks must not draw dropout')):
            explicit = model.mask_cond(cond, 'traj', torch.zeros(1, 4, dtype=torch.bool), explicit_task=True)
        torch.testing.assert_close(explicit, cond)

    def test_strict_condition_and_label_mask_shapes(self):
        model, y = self.model(), self.conditioning()
        y['trajectory_valid'] = torch.ones(1, 4, 9)
        with self.assertRaisesRegex(ValueError, 'trajectory_valid'):
            model(torch.zeros(1, 4, 245), torch.ones(1), y)
        with self.assertRaisesRegex(ValueError, 'feature_valid_mask'):
            feature_mask_for(torch.zeros(1, 4, 245), {'feature_valid_mask': torch.ones(1, 4, 153)})

    def test_rotation_loss_gradients_and_invalid_groups(self):
        model, flow = ConstantPrediction(), FlowMatching(global_weight=8)
        target = torch.zeros(1, 2, 245)
        target[:, :, :6] = 2
        target[:, :, 117:180] = 4
        valid = torch.ones_like(target, dtype=torch.bool)
        valid[:, :, 117:180] = False
        y = {'valid_frames': torch.ones(1, 2), 'feature_valid_mask': valid}
        terms = flow.training_losses(model, target, model_kwargs={'y': y},
                                     noise=torch.zeros_like(target), t=torch.full((1,), 0.5))
        terms['loss'].mean().backward()
        self.assertTrue((model.value.grad[:6] != 0).all())
        self.assertTrue((model.value.grad[117:180] == 0).all())
        self.assertGreater(terms['body_rotation_mse'].item(), 0)
        self.assertEqual(terms['left_hand_valid_count'].item(), 0)
        self.assertEqual(terms['left_hand_mse'].item(), 0)

    def test_label_invalidity_never_clamps_sampling(self):
        model, flow = ConstantPrediction(value=3), FlowMatching()
        y = {'valid_frames': torch.ones(1, 4),
             'feature_valid_mask': torch.zeros(1, 4, 245, dtype=torch.bool)}
        actual = flow.sample_loop(model, (1, 4, 245), model_kwargs={'y': y},
                                  noise=torch.zeros(1, 4, 245))
        torch.testing.assert_close(actual, torch.full_like(actual, 3), rtol=1e-5, atol=1e-5)

    def test_sampling_noise_matches_model_precision(self):
        class DtypeCheckedPrediction(ConstantPrediction):
            def forward(self, x, t, y, **kwargs):
                if x.dtype != self.value.dtype:
                    raise AssertionError('Sampler noise dtype does not match model parameters')
                return super().forward(x, t, y, **kwargs)
        model = DtypeCheckedPrediction(value=1).to(dtype=torch.bfloat16)
        output = FlowMatching().sample_loop(model, (1, 4, 245),
            model_kwargs={'y': {'valid_frames': torch.ones(1, 4)}})
        self.assertEqual(output.dtype, torch.bfloat16)

    def test_forecast_preserves_missing_conditions_and_target_validity(self):
        y = self.conditioning()
        y['traj_mask'] = torch.tensor([[False, True, False, False]])
        y['img_mask'] = torch.tensor([[True, False, False, False]])
        y['feature_valid_mask'] = torch.ones(1, 4, 245, dtype=torch.bool)
        result = apply_task_conditioning(y, 'fore', forecast_prefix=2)
        self.assertTrue(result['task_conditioning'])
        torch.testing.assert_close(result['traj_mask'], torch.tensor([[0, 1, 1, 1]]))
        torch.testing.assert_close(result['img_mask'], torch.tensor([[1, 0, 1, 1]]))
        torch.testing.assert_close(result['loss_mask'], torch.tensor([[0, 0, 1, 1]]))
        self.assertTrue(result['feature_valid_mask'].all())
        self.assertFalse(y['traj_mask'][0, 2])

    def test_router_pools_only_visible_video(self):
        moe = MoEFeedForward(2, 4, dropout=0, num_experts=2, top_k=1, chunk_size=4)
        with torch.no_grad():
            moe.router.weight.zero_()
            moe.router.bias.zero_()
            moe.router.weight[0, 4] = 1
        y = {'pose': torch.zeros(1, 4, 2), 'ego': torch.tensor([[[2., 0.], [0., 0.], [0., 0.], [0., 0.]]]),
             'ego_valid_frames': torch.tensor([[True, False, False, False]]),
             'timestep': torch.zeros(1, 2), 'valid_frames': torch.ones(1, 4, dtype=torch.bool)}
        logits = moe._router_logits(torch.zeros(1, 5, 2), y, num_prefix_tokens=1)
        torch.testing.assert_close(logits[:, :, 0], torch.full((1, 5), 2.))


if __name__ == '__main__':
    unittest.main()
