import copy
import unittest
from unittest.mock import patch

import torch

from MotionExpert.checkpoints import checkpoint_config, validate_checkpoint_architecture
from MotionExpert.config import architecture_metadata, as_dict, load_config, motion_expert_config, ROOT
from MotionExpert.dataset.normalization import MotionNormalizer
from MotionExpert.ema import EMAOptimizer
from MotionExpert.model import UniEgoMotion
from MotionExpert.module.uem_module import UEM_Module
from MotionExpert.motion_expert import MotionExpert
from MotionExpert.mydiffusion.flow_matching import FlowMatching


class GlobalTokenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(81)
        cls.cfg = motion_expert_config(global_token=True)
        cls.model = UniEgoMotion(cls.cfg, dropout=0)
        # Two real E7 blocks exercise layer-to-layer prefix gradients on CPU.
        cls.model.tsfm = torch.nn.ModuleList(list(cls.model.tsfm)[:2])

    def setUp(self):
        torch.manual_seed(93)
        self.model.eval().requires_grad_(False)
        self.model.global_token.requires_grad_(True)
        self.model.zero_grad(set_to_none=True)
        self.x = torch.randn(2, 7, 153)
        valid = torch.tensor([[True] * 7, [True] * 3 + [False] * 4])
        self.y = {
            "traj": torch.randn(2, 7, 18), "img_embs": torch.randn(2, 7, 1024),
            "text_pooled": torch.randn(2, 768), "text_tokens": torch.randn(2, 16, 768),
            "text_mask": torch.ones(2, 16, dtype=torch.bool),
            "text_valid": torch.tensor([True, False]),
            "valid_frames": valid, "valid_img_embs": valid.clone(),
        }
        self.t = torch.tensor([0.4, 1.0])

    def expert(self):
        expert = MotionExpert.__new__(MotionExpert)
        expert.cfg, expert.model = self.cfg, self.model
        expert.device, expert.dtype = torch.device("cpu"), torch.float32
        expert.flow = FlowMatching()
        expert.normalizer = MotionNormalizer()
        expert.weights_path, expert.weights_kind = "test-global-model", "model"
        return expert

    def test_only_one_parameter_is_added(self):
        with torch.device("meta"):
            legacy = UniEgoMotion(motion_expert_config())
            global_model = UniEgoMotion(self.cfg)
        self.assertEqual(set(global_model.state_dict()) - set(legacy.state_dict()), {"global_token"})
        self.assertEqual(sum(p.numel() for p in global_model.parameters()), 403790133)
        self.assertEqual(sum(p.numel() for p in legacy.parameters()), 403789365)
        with self.assertRaisesRegex(ValueError, "GLOBAL_TOKEN_COUNT=1"):
            legacy(self.x, self.t, self.y, return_features=True)

    @torch.no_grad()
    def test_return_contracts_and_masks(self):
        motion, hidden, global_hidden = self.model(self.x, self.t, self.y, return_features=True)
        self.assertEqual(motion.shape, (2, 7, 153))
        self.assertEqual(hidden.shape, (2, 7, 768))
        self.assertEqual(global_hidden.shape, (2, 768))
        torch.testing.assert_close(self.model(self.x, self.t, self.y), motion)
        old_motion, old_hidden = self.model(self.x, self.t, self.y, return_hidden=True)
        torch.testing.assert_close(old_motion, motion)
        torch.testing.assert_close(old_hidden, hidden)
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            self.model(self.x, self.t, self.y, return_hidden=True, return_features=True)
        changed = copy.deepcopy(self.y)
        padding = ~self.y["valid_frames"]
        changed_x = self.x.clone()
        changed_x[padding] = 1000
        changed["traj"][padding] = -1000
        changed["img_embs"][padding] = 1000
        actual = self.model(changed_x, self.t, changed, return_features=True)
        torch.testing.assert_close(actual[0][~padding], motion[~padding])
        torch.testing.assert_close(actual[1][~padding], hidden[~padding])
        torch.testing.assert_close(actual[2], global_hidden)

    def test_motion_loss_updates_global_and_ema(self):
        parameter = self.model.global_token
        before = parameter.detach().clone()
        optimizer = EMAOptimizer(torch.optim.AdamW([parameter], lr=1e-3), torch.device("cpu"), decay=0.5)
        loss = FlowMatching().training_losses(self.model, self.x, model_kwargs={"y": self.y}, t=self.t)["loss"].mean()
        loss.backward()
        self.assertIsNotNone(parameter.grad)
        self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(parameter.grad.norm().item(), 0)
        optimizer.step()
        optimizer.join()
        self.assertFalse(torch.equal(before, parameter))
        torch.testing.assert_close(optimizer.ema_params[0], (before + parameter.detach()) / 2)

    def test_full_training_phase_includes_global(self):
        cfg = load_config(ROOT / "configs/e7_k12_400m_global_text.yaml")
        cfg.DATA.STATS_PATH = None
        with patch("MotionExpert.module.uem_module.UniEgoMotion", return_value=self.model):
            module = UEM_Module(cfg)
        module._set_motion_expert_train_phase(0)
        for name, parameter in self.model.named_parameters():
            unused = name.startswith("embed_text_cond.") or name == "mask_tokens.text"
            self.assertEqual(parameter.requires_grad, not unused, name)
        optimizer = module.configure_optimizers()[0][0]
        group = next(group for group in optimizer.param_groups if group["phase"] == "unfrozen")
        self.assertTrue(any(parameter is self.model.global_token for parameter in group["params"]))
        self.assertEqual(group["lr_scale"], 1.0)

    @torch.no_grad()
    def test_cfg_pairs_motion_hidden_and_global(self):
        cond = self.model(self.x, self.t, self.y, return_features=True)
        uncond = self.model(self.x, self.t, {"valid_frames": self.y["valid_frames"]}, return_features=True)
        actual = self.model(self.x, self.t, self.y, cond_scale=2.5, return_features=True)
        for value, conditional, unconditional in zip(actual, cond, uncond):
            torch.testing.assert_close(value, unconditional + 2.5 * (conditional - unconditional))
        with self.assertRaisesRegex(ValueError, "finite"):
            self.model(self.x, self.t, self.y, cond_scale=float("nan"))

    @torch.no_grad()
    def test_prefixes_do_not_shift_frame_routing(self):
        ff = self.model.tsfm[0].ff
        frames = torch.randn(2, 7, 768)
        conditions = {"pose": torch.randn_like(frames), "ego": torch.randn_like(frames),
                      "timestep": torch.randn(2, 768), "valid_frames": self.y["valid_frames"]}
        one = ff._router_logits(torch.cat((frames[:, :1], frames), dim=1), conditions, num_prefix_tokens=1)
        two = ff._router_logits(torch.cat((frames[:, :2], frames), dim=1), conditions, num_prefix_tokens=2)
        torch.testing.assert_close(one[:, 1:], two[:, 2:])
        torch.testing.assert_close(two[:, 0], two[:, 1])
        torch.testing.assert_close(two[:, 2:6], two[:, 2:3].expand(-1, 4, -1))
        with self.assertRaisesRegex(ValueError, "frame count"):
            ff._router_logits(frames, conditions, num_prefix_tokens=2)

    def test_one_call_features_equal_first_of_ten_steps(self):
        expert = self.expert()
        with patch.object(self.model, "forward", wraps=self.model.forward) as forward:
            features = expert.extract_features(self.y, seed=62)
            self.assertEqual(forward.call_count, 1)
        with patch.object(self.model, "forward", wraps=self.model.forward) as forward:
            sampled = expert.sample(self.y, seed=62, denormalize=True, return_all_pred_xstart=True)
            self.assertEqual(forward.call_count, 10)
            self.assertEqual(sum(bool(call.kwargs.get("return_features")) for call in forward.call_args_list), 1)
        self.assertEqual(len(sampled["pred_xstart"]), 10)
        self.assertNotIn("motion", features)
        for key in ("one_step_hidden", "one_step_global", "feature_mask"):
            torch.testing.assert_close(features[key], sampled[key], rtol=0, atol=0)
        self.assertFalse(torch.is_inference(features["one_step_global"]))
        head = torch.nn.Linear(768, 1)
        head(features["one_step_global"]).sum().backward()
        self.assertIsNotNone(head.weight.grad)

    def test_forecast_and_generation_do_not_leak_hidden_conditions(self):
        expert = self.expert()
        for task, start in (("fore", 2), ("gen", 1)):
            features = expert.extract_features(self.y, seed=62, task=task, forecast_prefix=2)
            changed = copy.deepcopy(self.y)
            changed["traj"][:, start:] = 1e4
            changed["img_embs"][:, start:] = -1e4
            if task == "gen":
                changed["traj"][:, 0] = 1e4
            actual = expert.extract_features(changed, seed=62, task=task, forecast_prefix=2)
            torch.testing.assert_close(actual["one_step_global"], features["one_step_global"])
            torch.testing.assert_close(actual["one_step_hidden"], features["one_step_hidden"])
            self.assertTrue(features["feature_mask"].all())

    def test_checkpoint_architecture_is_strict(self):
        checkpoint = {"state_dict": {"model.global_token": torch.zeros(768)},
                      "hyper_parameters": as_dict(self.cfg),
                      "motionexpert_architecture": architecture_metadata(self.cfg)}
        self.assertEqual(checkpoint_config(checkpoint).MODEL.GLOBAL_TOKEN_COUNT, 1)
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_checkpoint_architecture(motion_expert_config(), checkpoint)
        with self.assertRaisesRegex(ValueError, "disagree"):
            checkpoint_config(dict(checkpoint, state_dict={"model.input_process.weight": torch.zeros(2, 2)}))
        bad = dict(checkpoint, motionexpert_architecture=dict(checkpoint["motionexpert_architecture"], num_prefix_tokens=1))
        with self.assertRaisesRegex(ValueError, "metadata"):
            checkpoint_config(bad)
        with self.assertRaisesRegex(ValueError, "metadata"):
            checkpoint_config({"global_token": torch.zeros(1, 1, 768)})
        self.assertEqual(checkpoint_config({"input_process.weight": torch.zeros(2, 2)}).MODEL.GLOBAL_TOKEN_COUNT, 0)

    def test_original_weighted_loss_normalization(self):
        error = torch.zeros(2, 1, 153)
        error[0, 0, 0], error[0, 0, 108] = 1, 2
        valid = torch.zeros_like(error, dtype=torch.bool)
        valid[0, 0, [0, 108]] = True
        loss = FlowMatching()._valid_frame_mse(error, {"y": {
            "valid_frames": torch.ones(2, 1), "feature_valid_mask": valid,
        }})
        torch.testing.assert_close(loss, torch.tensor([33 / 9, 0]))


if __name__ == "__main__":
    unittest.main()
