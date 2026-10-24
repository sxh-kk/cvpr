"""CPU-only transfer contracts; no SSH, upload, model or data collection."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "finalize_handover_random100.py"
spec = importlib.util.spec_from_file_location("random100_finalizer_test", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def gates():
    collection = {"contract": "cvpr-handover-random100-collection-ready-v1",
        "quality_and_diversity_passed": True, "episodes": 100, "ready_for_pack": True}
    ready = {"contract": "cvpr-handover-random100-lerobot-pack-v1", "train_episodes": 100,
        "train_payload_packed": True, "train_target_satisfied": True,
        "minimum_distinct_sources_satisfied": True, "old_captures_merged": False,
        "normalization_refitted": False, "final_ready": False, "train_windows": 1000}
    production = {"contract": "cvpr-handover-random100-reader-verification-v1",
        "status": "real_DINO_and_production_reader_passed", "train_episodes": 100, "train_windows": 1000,
        "production_ready_for_bound_pack_and_cache": True, "formal100_quality_and_diversity_passed": True,
        "GPU_used": False, "CUDA_initialized": False, "model_constructed": False, "training_started": False,
        "old_packages_modified": False, "VAL_or_TEST_loaded": False, "PACK_READY_modified": False,
        "final_ready_automatically_promoted": False,
        "cache_episodes": [{"episode_index": i, "every_feature_finite": True, "every_frame_valid": True}
                           for i in range(100)],
        "actual_reader_queries": [{"episode_index": i, "actual_current_RGB_and_production_collator_passed": True}
                                  for i in range(100)]}
    return collection, ready, production


def metadata():
    manifest = {"contract": "unifolm-dino-native-frames-v1", "state": "complete",
        "source": {"dataset_root": "/local/new100/train", "fps": 30,
            "meta_files": {"meta/info.json": {"sha256": "meta", "size_bytes": 123, "mtime_ns": 123456789}}},
        "image_process": {"height": 256, "width": 320}, "encoder": {"weights_sha256": "weights"},
        "extraction": {"stage": "validation", "training": False, "sampling_step": 1},
        "storage_dtype": "float32", "shape": [5, 1024], "episode_indices": [0, 1]}
    manifest["signature"] = module.digest({key: manifest[key] for key in module.MANIFEST_KEYS})
    receipts = {}
    for index in manifest["episode_indices"]:
        source = {"parquet": {"sha256": str(index), "mtime_ns": 192837465, "size_bytes": 200},
                  "video": {"sha256": "video", "mtime_ns": 123456789, "size_bytes": 300}}
        receipts[index] = {"contract": manifest["contract"], "state": "complete", "episode_index": index,
            "manifest_signature": manifest["signature"], "source": source,
            "signature": module.digest({"manifest_signature": manifest["signature"], "source": source}),
            "features": {"sha256": "original features"}, "frames": {"sha256": "original native frames"}}
    return manifest, receipts


class GateTests(unittest.TestCase):
    def test_exact100_complete_cpu_gates_pass(self):
        module.validate_gate_reports(*gates())

    def test_incomplete_quality_or_wrong_contract_is_never_ready(self):
        for which, key, value in [(0, "episodes", 99), (0, "quality_and_diversity_passed", False),
            (0, "contract", "old supplement"), (1, "train_episodes", 99),
            (1, "old_captures_merged", True), (1, "normalization_refitted", True),
            (1, "final_ready", True), (2, "production_ready_for_bound_pack_and_cache", False),
            (2, "GPU_used", True), (2, "CUDA_initialized", True), (2, "model_constructed", True),
            (2, "training_started", True), (2, "train_windows", 999)]:
            with self.subTest(which=which, key=key, value=value):
                reports = gates(); reports[which][key] = value
                with self.assertRaises(ValueError): module.validate_gate_reports(*reports)

    def test_every_episode_cache_and_real_query_required(self):
        for mutate in [lambda p: p["cache_episodes"].pop(),
                       lambda p: p["cache_episodes"][0].update(every_feature_finite=False),
                       lambda p: p["cache_episodes"][99].update(episode_index=0),
                       lambda p: p["actual_reader_queries"][0].update(actual_current_RGB_and_production_collator_passed=False),
                       lambda p: p["actual_reader_queries"].pop()]:
            reports = gates(); mutate(reports[2])
            with self.assertRaises(ValueError): module.validate_gate_reports(*reports)

    def test_failed_local_gate_prevents_remote_probe_and_bundle_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "bundle"
            args = SimpleNamespace(collection_root="pending", pack_root="pending", dino_root="pending",
                                   local_reader_report="pending", bundle_root=bundle)
            with patch.object(module, "local_gates", side_effect=ValueError("COLLECTION_READY missing")), \
                 patch.object(module, "remote_python") as remote:
                with self.assertRaisesRegex(ValueError, "COLLECTION_READY"): module.prepare(args)
                remote.assert_not_called(); self.assertFalse(bundle.exists())

    def test_upload_rechecks_gates_before_any_ssh(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            module.write_new(bundle / "TRANSFER_PLAN.json", {"contract": module.CONTRACT,
                "finalizer_source_sha256": module.sha(SCRIPT), "remote": {"root": module.REMOTE_ROOT},
                "local_inputs": {"collection_root": "pending"}})
            with patch.object(module, "local_gates", side_effect=ValueError("not100")), \
                 patch.object(module, "remote_python") as remote, patch.object(module.subprocess, "run") as subprocess:
                with self.assertRaisesRegex(ValueError, "not100"): module.upload(bundle)
                remote.assert_not_called(); subprocess.assert_not_called()


class RelocationTests(unittest.TestCase):
    def test_only_explicit_metadata_fields_change_and_inputs_stay_original(self):
        manifest, receipts = metadata(); original = deepcopy((manifest, receipts))
        after, changed = module.relocate_metadata(manifest, receipts, "/remote/new100/pack/train")
        self.assertEqual((manifest, receipts), original)
        self.assertEqual(after["source"]["dataset_root"], "/remote/new100/pack/train")
        self.assertNotEqual(after["signature"], manifest["signature"])
        check = deepcopy(after); check["source"]["dataset_root"] = manifest["source"]["dataset_root"]
        check["signature"] = manifest["signature"]; self.assertEqual(check, manifest)
        for index, receipt in changed.items():
            self.assertEqual(receipt["source"], receipts[index]["source"])
            self.assertEqual(receipt["features"], receipts[index]["features"])
            self.assertEqual(receipt["frames"], receipts[index]["frames"])
            self.assertEqual(receipt["manifest_signature"], after["signature"])

    def test_incomplete_or_modified_metadata_and_missing_receipts_rejected(self):
        for mutation in [lambda m, r: m.update(state="partial"),
                         lambda m, r: m["encoder"].update(weights_sha256="other"),
                         lambda m, r: r.pop(1),
                         lambda m, r: r[0]["source"].update(newfield="different"),
                         lambda m, r: r[0].update(episode_index=1)]:
            manifest, receipts = metadata(); mutation(manifest, receipts)
            with self.assertRaises(ValueError): module.relocate_metadata(manifest, receipts, "/remote/pack/train")
        for root in ("relative/pack/train", "/remote/../aliased/train"):
            with self.assertRaises(ValueError): module.relocate_metadata(*metadata(), root)

    def test_real_copy_relocation_archives_originals_and_preserves_feature_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); cache = root / "dino"; (root / "pack/train").mkdir(parents=True)
            manifest, receipts = metadata(); module.write_new(cache / "manifest.json", manifest)
            for index, receipt in receipts.items():
                directory = cache / f"episodes/{index:08d}"; module.write_new(directory / "receipt.json", receipt)
                (directory / "features.npy").write_bytes(b"real feature byte placeholder")
                (directory / "frames.npz").write_bytes(b"native frame byte placeholder")
            files = module.inventory_trees({"dino": cache, "pack": root / "pack"})
            plan = {"release_id": "test-owned-release", "files": files}
            receipt = module.relocate_remote_dino(root, plan)
            self.assertFalse(receipt["ready_for_training"]); self.assertFalse(receipt["PACK_READY_modified"])
            self.assertEqual(module.read(root / "dino-original-metadata/manifest.json"), manifest)
            self.assertEqual(len(receipt["metadata_changes"]), 3)
            for name, expected in files.items():
                if name not in receipt["metadata_changes"]:
                    self.assertEqual(module.identity(root / name), expected)
            self.assertEqual(module.read(cache / "manifest.json")["source"]["dataset_root"], str(root / "pack/train"))


class PayloadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for name in ("pack", "dino", "code", "evidence"):
            (self.root / name).mkdir(); (self.root / name / "file.bin").write_bytes(name.encode())
        trees = {name: str(self.root / name) for name in ("pack", "dino", "code", "evidence")}
        self.plan = {"trees": trees, "files": module.inventory_trees(trees)}

    def test_copied_bytes_and_nanosecond_identity_checked_and_restored(self):
        file = self.root / "pack/file.bin"; original = self.plan["files"]["pack/file.bin"]
        os.utime(file, ns=(1, 1))
        with self.assertRaisesRegex(ValueError, "mtime"): module.check_remote_payload(self.root, self.plan)
        module.check_remote_payload(self.root, self.plan, restore_mtimes=True)
        self.assertEqual(file.stat().st_mtime_ns, original["mtime_ns"])

    def test_same_size_corruption_extra_missing_or_alias_payload_rejected(self):
        file = self.root / "pack/file.bin"; file.write_bytes(b"PACK")
        with self.assertRaisesRegex(ValueError, "SHA"): module.check_remote_payload(self.root, self.plan, restore_mtimes=True)
        file.write_bytes(b"pack"); os.utime(file, ns=(1, self.plan["files"]["pack/file.bin"]["mtime_ns"]))
        extra = self.root / "dino/extra"; extra.write_bytes(b"extra")
        with self.assertRaisesRegex(ValueError, "extra"): module.check_remote_payload(self.root, self.plan)
        extra.unlink(); file.unlink()
        with self.assertRaisesRegex(ValueError, "missing"): module.check_remote_payload(self.root, self.plan)
        file.symlink_to(self.root / "code/file.bin")
        with self.assertRaisesRegex(ValueError, "alias"): module.check_remote_payload(self.root, self.plan)

    def test_owned_paths_refuse_escape_and_alias(self):
        for relative in ("../elsewhere", "/absolute", "missing"):
            with self.assertRaises(ValueError): module.owned(self.root, relative)
        (self.root / "alias").symlink_to(self.root / "pack/file.bin")
        with self.assertRaises(ValueError): module.owned(self.root, "alias")

    def test_no_ready_receipt_on_failed_actual_remote_reader(self):
        self.plan.update(contract=module.CONTRACT, release_id="test", remote={"root": str(self.root)},
                         finalizer_source_sha256=module.sha(SCRIPT))
        module.write_new(self.root / "TRANSFER_PLAN.json", self.plan)
        module.write_new(self.root / "TRANSFER_OWNER.json", {"contract": module.CONTRACT, "release_id": "test",
                         "plan_digest": module.digest(self.plan)})
        with patch.object(module, "REMOTE_ROOT", str(self.root)), \
             patch.object(module, "relocate_remote_dino", return_value={"metadata_changes": {}}), \
             patch.object(module, "real_remote_reader", side_effect=ValueError("actual production failed")):
            with self.assertRaisesRegex(ValueError, "actual production failed"): module.remote_finish(self.root)
        self.assertFalse((self.root / "RELEASE_READY.json").exists())
        self.assertEqual(len(list(self.root.glob("REMOTE_VERIFICATION_FAILED-*.json"))), 1)

    def test_resume_refuses_nested_symlink_before_transfer_can_start(self):
        release = self.root / "new-release"
        plan = {"contract": module.CONTRACT, "release_id": "test", "remote": {"root": str(release)}}
        arguments = {"plan": plan, "allowed_root": str(release), "resume": False}
        result = subprocess.run([sys.executable, "-B", "-c", module.REMOTE_CREATE],
                                input=json.dumps(arguments), text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        arguments["resume"] = True
        result = subprocess.run([sys.executable, "-B", "-c", module.REMOTE_CREATE],
                                input=json.dumps(arguments), text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        (release / "pack/unsafe-link").symlink_to(self.root / "pack", target_is_directory=True)
        result = subprocess.run([sys.executable, "-B", "-c", module.REMOTE_CREATE],
                                input=json.dumps(arguments), text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((release / "RELEASE_READY.json").exists())
        (release / "pack/unsafe-link").unlink()
        (release / "dino-original-metadata").symlink_to(self.root / "evidence", target_is_directory=True)
        result = subprocess.run([sys.executable, "-B", "-c", module.REMOTE_CREATE],
                                input=json.dumps(arguments), text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        with self.assertRaisesRegex(ValueError, "alias"): module.check_release_aliases(release)


if __name__ == "__main__": unittest.main()
