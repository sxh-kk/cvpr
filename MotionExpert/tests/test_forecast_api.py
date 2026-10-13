"""The deployed image-only protocol and incompatible temporal assets."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from MotionExpert.checkpoints import checkpoint_config, validate_checkpoint_architecture
from MotionExpert.config import as_dict, architecture_metadata, config_from_dict, motion_expert_config, validate_architecture
from MotionExpert.tests.fixtures import conditioning, tiny_expert


class ForecastAPITests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.expert = tiny_expert()
        self.images = conditioning(batch=1, frames=6)["img_embs"]

    def test_images_only_predicts_five_physical_future_frames(self):
        output = self.expert.predict_future({"img_embs": self.images}, seed=62)
        self.assertEqual(output["future_motion"].shape, (1, 5, 245))
        self.assertEqual(output["future_motion_denormalized"].shape, (1, 5, 245))
        torch.testing.assert_close(output["future_time_offsets_s"], torch.tensor([.2, .4, .6, .8, 1.]))
        y = self.expert.model.last_y
        self.assertTrue(y["traj_mask"].bool().all())
        self.assertFalse(y["trajectory_valid"].any())
        self.assertNotIn("motion_generation_mask", y)
        self.assertEqual(output["motion"].shape, (1, 11, 245))
        self.assertEqual(output["metadata"]["forecast_protocol"]["history_latent"], "joint_denoising")
        # Deployment provides no future labels. Generation still covers all
        # five future frames even though no supervised-loss region is valid.
        self.assertFalse(y["loss_mask"].bool().any())
        self.assertTrue(y["valid_frames"].bool().all())

    def test_hidden_camera_and_future_images_cannot_change_forecast(self):
        expected = self.expert.predict_future({"img_embs": self.images}, seed=62)
        full = conditioning(batch=1, frames=11)
        full["img_embs"][:, :6] = self.images
        full["img_embs"][:, 6:] = float("nan")
        full["traj"][:] = float("nan")
        full["feature_valid_mask"] = torch.zeros(1, 11, 245, dtype=torch.bool)
        actual = self.expert.sample(full, task="fore", forecast_prefix=6, output_frames=11, seed=62)
        torch.testing.assert_close(actual["future_motion"], expected["future_motion"], rtol=0, atol=0)
        torch.testing.assert_close(actual["one_step_hidden"], expected["one_step_hidden"], rtol=0, atol=0)
        full["valid_img_embs"][:, 6:] = False
        again = self.expert.sample(full, task="fore", forecast_prefix=6, output_frames=11, seed=62)
        torch.testing.assert_close(again["one_step_hidden"], expected["one_step_hidden"], rtol=0, atol=0)

    def test_convenience_method_enforces_observed_duration(self):
        with self.assertRaisesRegex(ValueError, "6 observed"):
            self.expert.predict_future({"img_embs": self.images[:, :5]})
        with self.assertRaisesRegex(ValueError, "fixed task"):
            self.expert.predict_future({"img_embs": self.images}, output_frames=120)

    def test_joint_and_future_only_checkpoints_cannot_be_interchanged(self):
        payload = {"state_dict": {"global_token": torch.zeros(768)},
                   "hyper_parameters": as_dict(self.expert.cfg),
                   "motionexpert_architecture": architecture_metadata(self.expert.cfg)}
        future_cfg = copy.deepcopy(self.expert.cfg)
        future_cfg.DATA.MOTION_SUPERVISION = "future_only"
        with self.assertRaisesRegex(ValueError, "architecture"):
            validate_checkpoint_architecture(future_cfg, payload)
        old_values = as_dict(future_cfg)
        old_values["DATA"].pop("MOTION_SUPERVISION")
        old_payload = dict(payload, hyper_parameters=old_values,
                           motionexpert_architecture=architecture_metadata(future_cfg))
        self.assertEqual(checkpoint_config(old_payload).DATA.MOTION_SUPERVISION, "future_only")

    def test_forecast_world_decode_requires_anchor_and_returns_future_frames(self):
        with self.assertRaisesRegex(ValueError, "last-observed"):
            self.expert.predict_future({"img_embs": self.images}, decode=True)
        self.assertEqual(self.expert.model.calls, 0)
        with self.assertRaisesRegex(ValueError, "first future"):
            self.expert.predict_future({"img_embs": self.images}, decode=True,
                decode_kwargs={"initial_floor_transform": torch.eye(4), "start_frame": 0})
        actual = self.expert.predict_future({"img_embs": self.images}, decode=True,
            decode_kwargs={"initial_floor_transform": torch.eye(4)}, seed=62)
        self.assertEqual(actual["decoded"]["body_positions"].shape, (1, 5, 12, 3))

    def test_legacy_checkpoint_keeps_legacy_task_and_cannot_initialize_new_profile(self):
        legacy = as_dict(motion_expert_config())
        legacy["DATA"].update(FPS=15, WINDOW=120, FORECAST_PREFIX=30, SEGMENT_STRIDE=30)
        for name in ("TRAIN_TASK", "REQUIRE_FULL_WINDOW", "TRAJECTORY_CONDITION_PROB", "MOTION_SUPERVISION"):
            legacy["DATA"].pop(name)
        cfg = config_from_dict(legacy)
        self.assertIsNone(cfg.DATA.TRAIN_TASK)
        self.assertFalse(cfg.DATA.REQUIRE_FULL_WINDOW)
        payload = {"state_dict": {"global_token": torch.zeros(768)}, "hyper_parameters": legacy,
                   "motionexpert_architecture": architecture_metadata(cfg)}
        self.assertEqual(checkpoint_config(payload).DATA.FPS, 15)
        with self.assertRaisesRegex(ValueError, "architecture"):
            validate_checkpoint_architecture(self.expert.cfg, payload)
        changed = copy.deepcopy(payload)
        changed["hyper_parameters"] = as_dict(self.expert.cfg)
        with self.assertRaisesRegex(ValueError, "metadata"):
            checkpoint_config(changed)

    def test_invalid_forecast_profile_is_rejected_early(self):
        for name, value in (("WINDOW", 120), ("FORECAST_PREFIX", 5),
                            ("TRAIN_TASK", "recon"), ("REQUIRE_FULL_WINDOW", False)):
            with self.subTest(name=name):
                cfg = motion_expert_config()
                setattr(cfg.DATA, name, value)
                with self.assertRaisesRegex(ValueError, "5Hz forecasting"):
                    validate_architecture(cfg)
        cfg = motion_expert_config()
        cfg.DATA.FPS = 15
        with self.assertRaisesRegex(ValueError, "Joint history/future"):
            validate_architecture(cfg)

    def test_cli_defaults_to_forecast_for_image_only_prefix(self):
        from MotionExpert.infer import main
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, destination = root / "images.pt", root / "future.pt"
            torch.save({"img_embs": self.images}, source)
            args = ["motionexpert-infer", "--input", str(source), "--weights", "unused.pt",
                    "--output", str(destination)]
            with patch("sys.argv", args), patch("MotionExpert.infer.MotionExpert", return_value=self.expert):
                main()
            output = torch.load(destination, weights_only=True)
            self.assertEqual(output["metadata"]["task"], "fore")
            self.assertEqual(output["future_motion_denormalized"].shape, (1, 5, 245))


if __name__ == "__main__":
    unittest.main()
