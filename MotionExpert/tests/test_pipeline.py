"""Public dense inference contract, validity handling and text removal."""

import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from MotionExpert.tests.fixtures import conditioning, dense_stats, tiny_expert
from MotionExpert.dataset.normalization import MotionNormalizer


class DenseInferenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.expert = tiny_expert()
        self.y = conditioning()

    def test_prefix_only_forecast_generates_future_without_labels(self):
        prefix = {key: value[:, :2].clone() for key, value in self.y.items()}
        prefix["feature_valid_mask"] = torch.zeros(2, 2, 245, dtype=torch.bool)
        output = self.expert.sample(prefix, task="fore", forecast_prefix=2, output_frames=7,
                                    seed=62, denormalize=True)
        self.assertEqual(tuple(output["motion"].shape), (2, 7, 245))
        self.assertTrue(output["feature_mask"].all())
        self.assertTrue((output["motion"][:, 2:] != 0).all())
        y = self.expert.model.last_y
        self.assertNotIn("feature_valid_mask", y)
        self.assertTrue(y["traj_mask"][:, 2:].bool().all())
        self.assertTrue(y["img_mask"][:, 2:].bool().all())
        self.assertFalse(y["trajectory_valid"][:, 2:].any())

    def test_generation_needs_only_the_first_image(self):
        output = self.expert.sample({"img_embs": self.y["img_embs"][:, :1]},
                                    task="gen", output_frames=7, seed=62)
        self.assertEqual(tuple(output["motion"].shape), (2, 7, 245))
        self.assertTrue(self.expert.model.last_y["traj_mask"].bool().all())
        self.assertFalse(self.expert.model.last_y["img_mask"][:, 0].bool().any())

    def test_flat_five_token_input_is_accepted_and_cls_only_is_rejected(self):
        flattened = dict(self.y, img_embs=self.y["img_embs"].flatten(2))
        actual = self.expert.sample(flattened, seed=62, task="recon")
        expected = self.expert.sample(self.y, seed=62, task="recon")
        torch.testing.assert_close(actual["one_step_hidden"], expected["one_step_hidden"])
        with self.assertRaisesRegex(ValueError, "5,1024"):
            self.expert.sample(dict(self.y, img_embs=self.y["img_embs"][:, :, 0]))

    def test_hidden_future_does_not_change_first_call_features(self):
        for task, start in (("fore", 2), ("gen", 1)):
            expected = self.expert.extract_features(self.y, task=task, forecast_prefix=2, seed=62)
            changed = copy.deepcopy(self.y)
            changed["traj"][:, start:] = 1e4
            changed["img_embs"][:, start:] = -1e4
            if task == "gen":
                changed["traj"][:, 0] = 1e4
            actual = self.expert.extract_features(changed, task=task, forecast_prefix=2, seed=62)
            torch.testing.assert_close(actual["one_step_hidden"], expected["one_step_hidden"])
            torch.testing.assert_close(actual["one_step_global"], expected["one_step_global"])

    def test_invalid_values_are_sanitized_but_valid_nan_is_rejected(self):
        changed = copy.deepcopy(self.y)
        changed["trajectory_valid"][:, 1, 0] = False
        changed["traj"][:, 1, 0] = float("nan")
        changed["valid_img_embs"][:, 1] = False
        changed["img_embs"][:, 1] = float("nan")
        self.expert.sample(changed, task="recon")
        self.assertTrue(torch.isfinite(self.expert.model.last_y["traj"]).all())
        self.assertTrue(torch.isfinite(self.expert.model.last_y["img_embs"]).all())
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.expert.sample(dict(self.y, traj=torch.full_like(self.y["traj"], float("nan"))))

    def test_explicit_missing_conditions_survive_task_masks(self):
        y = dict(self.y, img_mask=torch.zeros(2, 7, dtype=torch.bool))
        y["img_mask"][:, 0] = True
        self.expert.sample(y, task="gen")
        self.assertTrue(self.expert.model.last_y["img_mask"].bool().all())

    def test_hidden_nan_conditions_and_old_label_masks(self):
        for task, cutoff in (("fore", 2), ("gen", 1)):
            changed = copy.deepcopy(self.y)
            changed["traj"][:, cutoff:] = float("nan")
            changed["img_embs"][:, cutoff:] = float("nan")
            if task == "gen":
                changed["traj"][:, 0] = float("nan")
            self.expert.sample(changed, task=task, forecast_prefix=2)
            self.assertTrue(torch.isfinite(self.expert.model.last_y["traj"]).all())
            self.assertTrue(torch.isfinite(self.expert.model.last_y["img_embs"]).all())
        with self.assertRaisesRegex(ValueError, "245"):
            self.expert.sample(dict(self.y, feature_valid_mask=torch.ones(2, 7, 153)))

    def test_prefix_only_and_full_masked_forecast_features_match(self):
        prefix = {key: value[:, :2] for key, value in self.y.items()}
        actual = self.expert.extract_features(prefix, task="fore", forecast_prefix=2,
                                             output_frames=7, seed=62)
        expected = self.expert.extract_features(self.y, task="fore", forecast_prefix=2, seed=62)
        torch.testing.assert_close(actual["one_step_hidden"], expected["one_step_hidden"])
        torch.testing.assert_close(actual["one_step_global"], expected["one_step_global"])

    def test_text_removed_and_forecast_requires_future(self):
        with self.assertRaisesRegex(ValueError, "Text conditioning was removed"):
            self.expert.sample(dict(self.y, text="walk"))
        with self.assertRaisesRegex(ValueError, "at least one frame"):
            self.expert.sample(self.y, task="fore", forecast_prefix=7)

    def test_batched_world_decode_and_observed_forecast_anchor(self):
        original = self.expert.model.forward
        identity = torch.tensor([1., 0., 0., 0., 1., 0.])
        def physical_forward(x, t, y, **kwargs):
            result = original(x, t, y, **kwargs)
            value = torch.zeros_like(x)
            body = value[..., :108].reshape(*x.shape[:2], 12, 9)
            body[..., :6] = identity
            value[..., 108:114] = identity
            if isinstance(result, tuple):
                return (value, *result[1:])
            return value
        transform = torch.eye(4).repeat(2, 1, 1)
        transform[:, 0, 3] = torch.tensor([10., 20.])
        anchor = torch.eye(4)
        anchor[0, 3] = 3.
        with patch.object(self.expert.model, "forward", side_effect=physical_forward):
            output = self.expert.sample(self.y, task="fore", forecast_prefix=2, decode=True,
                decode_kwargs={"window_to_world": transform, "initial_floor_transform": anchor,
                               "floor_height": torch.tensor([1., 2.]), "start_frame": 2})
        decoded = output["decoded"]
        self.assertEqual(tuple(decoded["body_positions"].shape), (2, 5, 12, 3))
        torch.testing.assert_close(decoded["body_positions"][:, :, :, 0],
                                  torch.tensor([13., 23.])[:, None, None].expand(2, 5, 12))
        torch.testing.assert_close(decoded["body_positions"][:, :, :, 2],
                                  torch.tensor([1., 2.])[:, None, None].expand(2, 5, 12))
        torch.testing.assert_close(decoded["body_rotations"],
                                  torch.eye(3).expand(2, 5, 12, 3, 3))

    def test_archive_cli_forecast_keeps_trained_stats_and_extends_horizon(self):
        import yaml
        from MotionExpert.config import as_dict
        from MotionExpert.dataset.dense_motion_dataset import DenseMotionDataset
        from MotionExpert.compute_joint_motion_stats import compute_stats
        from MotionExpert.tests.test_dense_dataset import synthetic_arrays, write_archive
        from MotionExpert.infer import main
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)/"archive"
            root.mkdir()
            manifest, _ = write_archive(root, synthetic_arrays())
            raw = DenseMotionDataset(archive_root=root, split_manifest=manifest, window=4,
                forecast_prefix=2, segment_stride=3, cond_img_feat=False, do_normalization=False,
                rotation_basis_mode="source_native", target_joint_basis="egosuite_native_v1")
            stats = compute_stats(raw)
            stats_path = Path(temporary)/"stats.pt"
            torch.save(stats, stats_path)
            self.expert.cfg.DATA.WINDOW = 4
            self.expert.cfg.DATA.FPS = 15
            self.expert.cfg.DATA.TRAIN_TASK = None
            self.expert.cfg.DATA.MOTION_SUPERVISION = "future_only"
            self.expert.cfg.DATA.REQUIRE_FULL_WINDOW = False
            self.expert.cfg.DATA.MIN_VALID_IMAGE_FRACTION = 0.0
            self.expert.cfg.DATA.FORECAST_PREFIX = 2
            self.expert.cfg.DATA.SEGMENT_STRIDE = 3
            self.expert.cfg.DATA.ARCHIVE_ROOT = str(root)
            self.expert.cfg.DATA.SPLIT_MANIFEST = str(manifest)
            self.expert.cfg.DATA.STATS_PATH = str(stats_path)
            self.expert.normalizer = MotionNormalizer(stats)
            config = Path(temporary)/"config.yaml"
            config.write_text(yaml.safe_dump(as_dict(self.expert.cfg)))
            destination = Path(temporary)/"prediction.pt"
            arguments = ["motionexpert-infer", "--config", str(config), "--weights", "unused.pt",
                         "--split", "val", "--task", "fore", "--frames", "8",
                         "--decode", "--output", str(destination)]
            with patch("sys.argv", arguments), patch("MotionExpert.infer.MotionExpert", return_value=self.expert):
                main()
            output = torch.load(destination, weights_only=True)
            self.assertEqual(tuple(output["motion"].shape), (1, 8, 245))
            self.assertEqual(tuple(output["decoded"]["body_positions"].shape), (1, 6, 12, 3))
            self.assertEqual(output["metadata"]["observed_prefix"], 2)
            self.assertTrue(self.expert.model.last_y["img_mask"][:, 2:].bool().all())
            self.assertTrue(self.expert.model.last_y["traj_mask"][:, 2:].bool().all())

    def test_stats_are_mandatory_train_only_and_contact_identity(self):
        with self.assertRaisesRegex(ValueError, "train-only"):
            MotionNormalizer()
        with self.assertRaisesRegex(ValueError, "provenance"):
            MotionNormalizer(dict(dense_stats(), split="val"))
        stats = dense_stats()
        stats["motion_std"][-1] = 2
        with self.assertRaisesRegex(ValueError, "Contact channels"):
            MotionNormalizer(stats)


if __name__ == "__main__":
    unittest.main()
