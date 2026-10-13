"""One-call feature parity, strict dense checkpoint export and EMA compatibility."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from MotionExpert.checkpoints import checkpoint_config, model_state, read_checkpoint, validate_checkpoint_statistics
from MotionExpert.config import architecture_metadata, as_dict, motion_expert_config
from MotionExpert.export_weights import export
from MotionExpert.ema import EMAOptimizer
from MotionExpert.tests.fixtures import TinyFeatureModel, conditioning, dense_stats, tiny_expert


def checkpoint(model):
    return {"state_dict": {f"model.{key}": value for key, value in model.state_dict().items()},
            "hyper_parameters": as_dict(model.cfg),
            "motionexpert_architecture": architecture_metadata(model.cfg),
            "motionexpert_normalization": dense_stats(),
            "motionexpert_normalization_clip": None}


class DenseGlobalTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.expert = tiny_expert()
        self.y = conditioning()

    def test_one_call_features_equal_first_of_ten_steps(self):
        features = self.expert.extract_features(self.y, seed=62, task="recon")
        self.assertEqual(self.expert.model.calls, 1)
        self.expert.model.calls = 0
        sampled = self.expert.sample(self.y, seed=62, task="recon", return_all_pred_xstart=True)
        self.assertEqual(self.expert.model.calls, 10)
        self.assertEqual(len(sampled["pred_xstart"]), 10)
        for key in ("one_step_hidden", "one_step_global", "feature_mask"):
            torch.testing.assert_close(features[key], sampled[key], rtol=0, atol=0)
        self.assertFalse(torch.is_inference(features["one_step_global"]))
        head = torch.nn.Linear(768, 1)
        head(features["one_step_global"]).sum().backward()
        self.assertIsNotNone(head.weight.grad)

    def test_legacy_153_and_text_weights_are_rejected_early(self):
        for state in ({"input_process.weight": torch.zeros(768, 153)},
                      {"text_cross_gates": torch.zeros(12)}):
            with self.assertRaisesRegex(ValueError, "Legacy SMPL/text"):
                checkpoint_config(state)
        with self.assertRaisesRegex(ValueError, "architecture metadata"):
            checkpoint_config({"global_token": torch.zeros(768)})

    def test_dense_metadata_detects_changes_to_sampling_and_features(self):
        payload = checkpoint(self.expert.model)
        self.assertEqual(checkpoint_config(payload).DATA.FPS, 5)
        for key in ("motion_dim", "joint_indices", "num_steps"):
            bad = dict(payload, motionexpert_architecture=dict(payload["motionexpert_architecture"]))
            bad["motionexpert_architecture"][key] = -1
            with self.assertRaisesRegex(ValueError, "metadata"):
                checkpoint_config(bad)
        with self.assertRaisesRegex(ValueError, "shape"):
            checkpoint_config(dict(payload, state_dict={"global_token": torch.zeros(1, 768)}))

    def test_dense_export_roundtrip_preserves_statistics_and_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, destination = Path(temporary)/"train.ckpt", Path(temporary)/"export.pt"
            torch.save(checkpoint(self.expert.model), source)
            with patch("MotionExpert.export_weights.UniEgoMotion", TinyFeatureModel):
                metadata = export(source, destination, use_ema=False)
            payload = read_checkpoint(destination)
            self.assertEqual(payload["format"], "motionexpert_weights_dense_v1")
            self.assertEqual(metadata["motion_dim"], 245)
            self.assertEqual(payload["motionexpert_normalization"]["split_fingerprint"], "unit-test-train-only")
            torch.testing.assert_close(payload["state_dict"]["level"], self.expert.model.level)
            checkpoint_config(payload)
            with patch("MotionExpert.motion_expert.UniEgoMotion", TinyFeatureModel):
                from MotionExpert.motion_expert import MotionExpert
                actual = MotionExpert(destination, device="cpu")
                output = actual.sample(self.y, task="recon")
            self.assertEqual(tuple(output["motion"].shape), (2, 7, 245))

    def test_checkpoint_statistics_reject_override_units_and_joint_axes(self):
        from MotionExpert.dataset.normalization import MotionNormalizer
        payload = checkpoint(self.expert.model)
        changed = dense_stats()
        changed["motion_mean"][0] = 1
        with self.assertRaisesRegex(ValueError, "motion_mean"):
            validate_checkpoint_statistics(self.expert.cfg, payload, MotionNormalizer(changed))
        changed = dict(dense_stats(), target_joint_basis="different_joint_axes")
        with self.assertRaisesRegex(ValueError, "target_joint_basis"):
            validate_checkpoint_statistics(self.expert.cfg, payload, MotionNormalizer(changed))
        validate_checkpoint_statistics(self.expert.cfg, payload, self.expert.normalizer)

    def test_ema_parameters_follow_dense_model_without_a_text_group(self):
        model = self.expert.model
        payload = checkpoint(model)
        named = list(model.named_parameters())
        averages = tuple(parameter.detach()+1 for _, parameter in named)
        payload["ema_state_format"] = "original_model_with_ema_optimizer_v1"
        payload["optimizer_states"] = [{"ema": averages, "opt": {"param_groups": [
            {"params": list(range(len(named))), "phase": "unfrozen"} ]}}]
        state = model_state(model, payload, use_ema=True)
        with self.assertRaisesRegex(ValueError, "Unsupported EMA"):
            model_state(model, dict(payload, ema_state_format="unknown_future_format"))
        for (name, parameter), average in zip(named, averages):
            torch.testing.assert_close(state[name], average)
        optimizer = EMAOptimizer(torch.optim.SGD([model.level], lr=.1),
                                 device=torch.device("cpu"), decay=.5)
        old = model.level.detach().clone()
        model.level.square().backward()
        optimizer.step()
        optimizer.join()
        torch.testing.assert_close(optimizer.ema_params[0], (old+model.level.detach())/2)


if __name__ == "__main__":
    unittest.main()
