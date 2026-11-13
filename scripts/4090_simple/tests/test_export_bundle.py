"""CPU regression checks for save gating, retention protection and portability."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "export_bundle", Path(__file__).resolve().parents[1] / "export_bundle.py")
export = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(export)


class ExportTests(unittest.TestCase):
    @staticmethod
    def motion_export_inputs(root):
        """Real metadata/serialization with only the large network factory replaced."""
        import torch
        from safetensors.torch import save_file
        from MotionExpert.config import architecture_metadata, as_dict, smplh_motion_expert_config
        from MotionExpert.tests.fixtures import smplh_stats
        from MotionExpert.tests.test_visual_compatibility import visual
        from psi.models.frozen_motion import MotionConditioner

        class TinyMotion(torch.nn.Module):
            def __init__(self, cfg=None):
                super().__init__()
                self.fixture = torch.nn.Parameter(torch.tensor([.2]))

        cfg = smplh_motion_expert_config()
        source = root / "velocity_metadata.ckpt"
        contract = visual(local=True)
        state = TinyMotion().state_dict()
        torch.save({"state_dict": state, "hyper_parameters": as_dict(cfg),
                    "motionexpert_architecture": architecture_metadata(cfg),
                    "motionexpert_normalization": smplh_stats(cfg),
                    "motionexpert_visual_conditioning": contract}, source)
        joint = root / "joint.safetensors"
        signature_key = "motion_conditioner._feature_protocol_signature"
        weights = {export.MOTION_PREFIX + name: value.clone() for name, value in state.items()}
        weights[signature_key] = torch.tensor(MotionConditioner._FEATURE_PROTOCOL_SIGNATURE, dtype=torch.int64)
        save_file(weights, str(joint))
        return source, joint, weights, contract["encoder"], TinyMotion

    def test_motion_export_rejects_legacy_joint_before_rewrapping_velocity_metadata(self):
        import torch
        from safetensors.torch import save_file

        for mismatch in ("missing", "wrong_value", "wrong_dtype"):
            with self.subTest(mismatch=mismatch), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source, joint, weights, encoder, factory = self.motion_export_inputs(root)
                key = "motion_conditioner._feature_protocol_signature"
                if mismatch == "missing":
                    weights.pop(key)
                elif mismatch == "wrong_value":
                    weights[key][1] += 1
                else:
                    weights[key] = weights[key].to(torch.float32)
                save_file(weights, str(joint))
                output = root / "exported/new_motion.ckpt"
                with patch("MotionExpert.model.UniEgoMotion", factory) as model_factory:
                    with self.assertRaisesRegex(RuntimeError, "velocity/LaMP two-call feature protocol"):
                        export.export_motion(source, joint, output, 20000, encoder)
                self.assertFalse(output.exists())
                self.assertFalse(output.parent.exists())

    def test_motion_export_roundtrips_new_protocol_and_exact_weights(self):
        import torch
        from MotionExpert.checkpoints import checkpoint_config, read_checkpoint

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, joint, weights, encoder, factory = self.motion_export_inputs(root)
            output = root / "exported/new_motion.ckpt"
            with patch("MotionExpert.model.UniEgoMotion", factory):
                report = export.export_motion(source, joint, output, 20000, encoder)
            exported = read_checkpoint(output)
            self.assertEqual(checkpoint_config(exported).FLOW.PREDICTION_TYPE, "velocity")
            self.assertEqual(report["strict_meta_load"], "passed")
            self.assertEqual(report["exact_serialization_comparison"], "passed")
            for name, value in exported["state_dict"].items():
                torch.testing.assert_close(value, weights[export.MOTION_PREFIX + name], rtol=0, atol=0)

    def test_redundant_dino_identity_is_derived_and_conflicts_rejected(self):
        original = {"encoder": {"local_file_sha256": {"model.safetensors": "a" * 64}}}
        canonical = export.canonical_visual_identity(original)
        self.assertNotIn("model_id", original["encoder"])
        self.assertEqual(export.canonical_visual_identity(canonical), canonical)
        canonical["encoder"]["local_file_sha256"]["model.safetensors"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "weights_sha256"):
            export.canonical_visual_identity(canonical)

    def test_transfer_verification_rejects_changed_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relative = "joint20k/checkpoints/ckpt_20000/model.safetensors"
            model = root / relative
            model.parent.mkdir(parents=True)
            model.write_bytes(b"joint weights")
            digest = export.sha256(model)
            export.write_json(root / "manifest.json", {
                "format": "cvpr-independent-4090-resources-v1", "step": 20000,
                "joint_sha256": digest,
                "files": [{"path": relative, "bytes": model.stat().st_size, "sha256": digest}],
            })
            (root / "MODEL_RESOURCES_READY").write_text(
                f"step=20000\njoint_sha256={digest}\nevaluation_ready=false\n")
            self.assertEqual(export.verify_bundle(root)["state"], "byte_verification_passed")
            model.write_bytes(b"other weights")
            with self.assertRaisesRegex(ValueError, "content mismatch"):
                export.verify_bundle(root)

    def test_gate_requires_full_save_and_exact_completion_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoints/ckpt_20000"
            checkpoint.mkdir(parents=True)
            log = root / "train.log"
            marker = f"Saved state to {checkpoint}"
            log.write_text(marker + "\n")
            (checkpoint / "model.safetensors").write_bytes(b"model")
            self.assertFalse(export.checkpoint_complete(root, log, 20000, 2))
            for name in ("optimizer.bin", "scheduler.bin", "random_states_0.pkl", "random_states_1.pkl"):
                (checkpoint / name).write_bytes(b"saved")
            log.write_text(marker + "0\n")
            self.assertFalse(export.checkpoint_complete(root, log, 20000, 2))
            log.write_text(marker + "\n")
            self.assertTrue(export.checkpoint_complete(root, log, 20000, 2))
            (checkpoint / "random_states_1.pkl").write_bytes(b"")
            self.assertFalse(export.checkpoint_complete(root, log, 20000, 2))

    def test_snapshot_survives_source_retention(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            weights = source / "model.safetensors"
            weights.write_bytes(b"immutable checkpoint")
            destination = root / "protected/ckpt_20000"
            export.protect_checkpoint(source, destination)
            weights.unlink()
            source.rmdir()
            self.assertEqual((destination / weights.name).read_bytes(), b"immutable checkpoint")
            self.assertFalse(destination.with_name("ckpt_20000.partial").exists())
            with self.assertRaises(FileExistsError):
                export.protect_checkpoint(source, destination)

    def test_hf_symlink_becomes_portable_regular_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            blob = root / "blob"
            blob.write_bytes(b"weight bytes")
            source = root / "hf_snapshot_weight"
            source.symlink_to(blob)
            target = root / "export/model.safetensors"
            export.link_or_copy(source, target)
            source.unlink()
            blob.unlink()
            self.assertFalse(target.is_symlink())
            self.assertEqual(target.read_bytes(), b"weight bytes")

    def test_remap_both_argv_forms_preserves_architecture_and_bounds(self):
        original = {
            "model": {"model_name_or_path": "/a100/qwen", "motion_checkpoint": "/a100/motion",
                      "num_blocks": 6, "action_dim": 80, "odim": 45},
            "data": {"root_dir": "/a100/data", "dino_cache_root": "/a100/cache",
                     "transform": {"field": {"stat_path": "/a100/stats", "action_min": [1, 2]}}},
            "train": {"output_dir": "/a100/logs", "resume_from_checkpoint": "/a100/14k",
                      "deepspeed_config": "/a100/zero.json"},
        }
        argv = "train.py\nposttrain_sonic_psi0_config\n--model.model_name_or_path=/a100/qwen\n"
        argv += "--model.motion-checkpoint /a100/motion\n--model.num-blocks=6\n--model.no-rtc\n"
        config, rewritten, changes = export.deployment_config(
            original, argv, Path("/4090/resources"), Path("/4090/repo"))
        self.assertNotIn("/a100", rewritten)
        self.assertIn("--model.motion_checkpoint=/4090/resources/joint20k/motion/joint_motion.ckpt", rewritten)
        self.assertIn("--model.num-blocks=6\n--model.no-rtc", rewritten)
        self.assertIn("--train.resume_from_checkpoint=None", rewritten)
        self.assertEqual(config["model"]["num_blocks"], original["model"]["num_blocks"])
        self.assertEqual(config["data"]["transform"]["field"]["action_min"], [1, 2])
        self.assertEqual(original["model"]["motion_checkpoint"], "/a100/motion")
        self.assertEqual(len(changes), 8)


if __name__ == "__main__":
    unittest.main()
