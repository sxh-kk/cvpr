"""CPU regression checks for save gating, retention protection and portability."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location(
    "export_bundle", Path(__file__).resolve().parents[1] / "export_bundle.py")
export = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(export)


class ExportTests(unittest.TestCase):
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
