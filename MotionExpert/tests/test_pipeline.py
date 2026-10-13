import tempfile
import unittest

import torch

from MotionExpert.config import load_config
from MotionExpert.checkpoints import model_state, parameter_phase
from MotionExpert.dataset.ee4d_motion_dataset import EE4D_Motion_DataModule
from MotionExpert.dataset.normalization import MotionNormalizer
from MotionExpert.dataset.text_conditions import TextFeatureStore
from MotionExpert.model.uniegomotion import mask_it
from MotionExpert.motion_expert import MotionExpert
from MotionExpert.mydiffusion.flow_matching import FlowMatching
from MotionExpert.utils.task_conditioning import apply_task_conditioning
from tests.fixtures import write_fixture


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.directory = tempfile.TemporaryDirectory()
        cls.cfg = load_config(write_fixture(cls.directory.name))
        cls.dm = EE4D_Motion_DataModule(cls.cfg)
        cls.dm.setup("fit")

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_two_sources_align_dino_and_mask_targets(self):
        ds = self.dm.train_dataset
        for name in ds.seq_names:
            sample = ds.get_from_seq_and_st(name, 60, 0)
            self.assertEqual(sample["misc"]["motion"].shape, (80, 153))
            self.assertEqual(int(sample["y"]["valid_frames"].sum()), 40)
            torch.testing.assert_close(sample["y"]["img_embs"][:4, 0], torch.tensor([30., 30., 31., 31.]))
            self.assertTrue(sample["y"]["text_valid"])
            self.assertTrue(torch.isfinite(sample["misc"]["motion"]).all())
            if sample["misc"]["source"] == "nymeriaplus":
                self.assertFalse(sample["y"]["feature_valid_mask"][:, 117:141].any())
                self.assertEqual(float(sample["misc"]["motion"][:, 117:141].abs().sum()), 0.)

    def test_missing_text_window_is_not_silently_empty(self):
        store = self.dm.train_dataset.text_features
        with self.assertRaises(KeyError):
            store.get("missing", 0)

    def test_missing_dino_fails_before_training(self):
        from MotionExpert.dataset.feats import ImageFeats
        with self.assertRaises(FileNotFoundError):
            ImageFeats(self.directory.name, "train", "dinov2", nymeria_feat_path="/missing/dino.pt")

    def test_inference_rejects_malformed_input(self):
        expert = MotionExpert.__new__(MotionExpert)
        expert.device, expert.dtype = torch.device("cpu"), torch.float32
        y = {"traj": torch.zeros(1, 80, 18), "img_embs": torch.zeros(1, 80, 1024),
             "text_pooled": torch.zeros(1, 768), "text_tokens": torch.zeros(1, 16, 768)}
        with self.assertRaises(ValueError):
            expert._prepare(dict(y, valid_frames=torch.ones(80)))
        with self.assertRaises(ValueError):
            expert._prepare(dict(y, img_embs=torch.zeros(1, 80, 1, 1024)))
        with self.assertRaises(ValueError):
            expert._prepare(dict(y, text_tokens=torch.full((1, 16, 768), float("nan"))))

    def test_ema_export_obeys_optimizer_group_order(self):
        model = torch.nn.Module()
        model.backbone = torch.nn.Linear(2, 2, bias=False)
        model.text_proj = torch.nn.Linear(2, 2, bias=False)
        model.tsfm = torch.nn.ModuleList([torch.nn.Module()])
        model.tsfm[0].ff = torch.nn.Module()
        model.tsfm[0].ff.router = torch.nn.Linear(2, 2, bias=False)
        phases = ("routed", "text", "unfrozen")
        ema = [torch.full((2, 2), float(i + 1)) for i in range(3)]
        checkpoint = {"state_dict": {f"model.{key}": value for key, value in model.state_dict().items()},
                      "ema_state_format": "original_model_with_ema_optimizer_v1",
                      "optimizer_states": [{"ema": ema, "opt": {"param_groups": [
                          {"phase": phase, "params": [i]} for i, phase in enumerate(phases)]}}]}
        state = model_state(model, checkpoint)
        for name, value in state.items():
            torch.testing.assert_close(value, ema[phases.index(parameter_phase(name))])

    def test_normalization_round_trip(self):
        normalizer = MotionNormalizer(clip=None)
        values = torch.randn(2, 80, 153)
        torch.testing.assert_close(normalizer.denormalize(normalizer.normalize(values, "motion")), values)

    def test_masks_keep_half_precision_and_do_not_mutate_input(self):
        values = torch.randn(2, 80, 16, dtype=torch.float16)
        self.assertEqual(mask_it(torch.zeros(2, 80), values, torch.zeros(16, dtype=torch.float16)).dtype, torch.float16)
        y = {"traj": torch.zeros(2, 80, 18), "img_embs": torch.zeros(2, 80, 1024), "valid_frames": torch.ones(2, 80)}
        masked = apply_task_conditioning(y, "fore", forecast_prefix=20)
        self.assertEqual(int(masked["loss_mask"].sum()), 120)
        self.assertNotIn("traj_mask", y)

    def test_flow_ignores_unobserved_nymeria_features(self):
        class Constant(torch.nn.Module):
            def forward(self, x, t, y):
                return torch.zeros_like(x)
        flow = FlowMatching(prediction_type="x0", global_feature_start=108, global_feature_end=117, global_weight=8)
        x = torch.zeros(1, 80, 153)
        x[..., 117:141] = 1e6
        valid = torch.ones_like(x, dtype=torch.bool)
        valid[..., 117:141] = False
        loss = flow.training_losses(Constant(), x, model_kwargs={"y": {"valid_frames": torch.ones(1, 80), "feature_valid_mask": valid}})
        self.assertEqual(float(loss["loss"].sum()), 0.)


if __name__ == "__main__":
    unittest.main()
