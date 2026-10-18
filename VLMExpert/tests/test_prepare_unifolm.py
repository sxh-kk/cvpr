"""Official archive preparation rejects corruption and unsafe ZIP paths."""

import hashlib
from http.client import IncompleteRead
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile


SOURCE = Path(__file__).parents[1] / "scripts/data/prepare_unifolm.py"
SPEC = importlib.util.spec_from_file_location("prepare_unifolm", SOURCE)
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


class PrepareUnifolmTests(unittest.TestCase):
    @staticmethod
    def response(payload, start, end, total, **headers):
        response = io.BytesIO(payload)
        response.status = 206
        response.headers = {"Content-Range": f"bytes {start}-{end}/{total}", **headers}
        return response

    def test_http_parts_survive_failure_and_resume_without_refetching_completed_parts(self):
        payload = b"0123456789abcdef"
        expected = {"size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            def failing(request, **kwargs):
                start, end = map(int, request.get_header("Range")[6:].split("-"))
                if start == 4:
                    raise IncompleteRead(b"partial")
                return self.response(payload[start:end + 1], start, end, len(payload))
            with patch.object(prepare, "urlopen", side_effect=failing):
                with self.assertRaisesRegex(RuntimeError, "Archive part 1 failed"):
                    prepare.download_archive(root, "pack", expected, workers=1, chunk_size=4, attempts=1)
            parts = root / ".pipeline/downloads/pack" / expected["sha256"]
            self.assertEqual((parts / "000000.bin").read_bytes(), payload[:4])
            fetched = []
            def working(request, **kwargs):
                start, end = map(int, request.get_header("Range")[6:].split("-"))
                fetched.append(start)
                return self.response(payload[start:end + 1], start, end, len(payload))
            with patch.object(prepare, "urlopen", side_effect=working):
                result = prepare.download_archive(root, "pack", expected, workers=2, chunk_size=4)
            self.assertEqual(result.read_bytes(), payload)
            self.assertNotIn(0, fetched)

    def test_http_rejects_wrong_range_encoding_and_incomplete_or_excess_body(self):
        payload = b"01234567"
        expected = {"size": 8, "sha256": hashlib.sha256(payload).hexdigest()}
        for kind in ("range", "status", "encoding", "short", "excess"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                body = payload[:-1] if kind == "short" else payload + b"x" if kind == "excess" else payload
                response = self.response(body, 1 if kind == "range" else 0, 7, 8)
                if kind == "status": response.status = 200
                if kind == "encoding": response.headers["Content-Encoding"] = "gzip"
                with patch.object(prepare, "urlopen", return_value=response):
                    with self.assertRaises(RuntimeError):
                        prepare.download_archive(root, "pack", expected, workers=1, chunk_size=8, attempts=1)
                self.assertFalse((root / "archives/official/sonic/pack.zip").exists())
                self.assertFalse(list((root / ".pipeline/downloads").rglob("*.bin")))

    def test_http_local_part_hashes_do_not_replace_the_official_whole_archive_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            expected = {"size": 8, "sha256": "0" * 64}
            response = self.response(b"01234567", 0, 7, 8)
            with patch.object(prepare, "urlopen", return_value=response):
                with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                    prepare.download_archive(root, "pack", expected, workers=1, chunk_size=8)
            self.assertFalse((root / "archives/official/sonic/pack.zip").exists())
            self.assertEqual(len(list((root / ".pipeline/downloads").rglob("*.bin"))), 1)

    def test_http_rejects_symlink_partial_before_touching_external_file(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            external = root / "external"
            external.write_bytes(b"keep")
            expected = {"size": 8, "sha256": "0" * 64}
            parts = root / ".pipeline/downloads/pack" / expected["sha256"]
            parts.mkdir(parents=True)
            (parts / "000000.partial").symlink_to(external)
            with self.assertRaisesRegex(ValueError, "symlink"):
                prepare.download_archive(root, "pack", expected, workers=1, chunk_size=8)
            self.assertEqual(external.read_bytes(), b"keep")

    def test_http_checks_existing_archive_instead_of_trusting_its_filename(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = root / "archives/official/sonic/pack.zip"
            archive.parent.mkdir(parents=True)
            archive.write_bytes(b"01234567")
            with patch.object(prepare, "urlopen") as network:
                with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                    prepare.download_archive(root, "pack", {"size": 8, "sha256": "0" * 64})
            network.assert_not_called()

    def test_http_refetches_a_part_with_non_object_or_changed_receipt(self):
        payload = b"01234567"
        expected = {"size": 8, "sha256": hashlib.sha256(payload).hexdigest()}
        for saved in ("[]", "null", '{"sha256":"wrong"}'):
            with self.subTest(saved=saved), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                parts = root / ".pipeline/downloads/pack" / expected["sha256"]
                parts.mkdir(parents=True)
                (parts / "000000.bin").write_bytes(payload)
                (parts / "000000.json").write_text(saved)
                with patch.object(prepare, "urlopen", return_value=self.response(payload, 0, 7, 8)) as network:
                    result = prepare.download_archive(root, "pack", expected, workers=1, chunk_size=8)
                self.assertEqual(result.read_bytes(), payload)
                network.assert_called_once()

    def test_training_recipe_rejects_incomplete_cache_before_starting_workers(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "vlm").mkdir()
            (root / "motion.ckpt").write_bytes(b"fixture")
            for pack in ("train", "val"):
                (root / "cache" / pack).mkdir(parents=True)
                (root / "cache" / pack / "manifest.json").write_text('{"state":"extracting"}')
                (root / "data" / pack / "meta").mkdir(parents=True)
                for name in ("modality.json", "stats_psi0.json"):
                    (root / "data" / pack / "meta" / name).write_text("{}")
            env = dict(os.environ, EGO_DEX_VLM_PATH=str(root / "vlm"),
                       MOTION_EXPERT_CHECKPOINT=str(root / "motion.ckpt"),
                       UNIFOLM_DINO_ROOT=str(root / "data"), UNIFOLM_DINO_CACHE=str(root / "cache"),
                       UNIFOLM_DINO_TRAIN_ID="train", UNIFOLM_DINO_VAL_ID="val", PSI_PYTHON=sys.executable)
            script = SOURCE.parents[1] / "train/psi0/posttrain-dual-source-unifolm.sh"
            result = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Incomplete DINO cache", result.stderr)
            self.assertNotIn("Command:", result.stdout)

    def test_extract_checks_existing_content_before_reusing_a_pack(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = root / "pack.zip"
            with zipfile.ZipFile(archive, "w") as stream:
                stream.writestr("pack/meta/value.json", b"12345")
            self.assertEqual(prepare.unpack_pack(archive, root, "pack"), 1)
            self.assertEqual(prepare.unpack_pack(archive, root, "pack"), 1)
            (root / "pack/meta/value.json").write_bytes(b"54321")
            with self.assertRaisesRegex(ValueError, "CRC mismatch"):
                prepare.unpack_pack(archive, root, "pack")

    def test_zip_traversal_and_external_pack_paths_are_rejected(self):
        for member in ("pack/../../outside", "/outside", "other/file", "pack\\..\\outside"):
            with self.subTest(member=member), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                archive = root / "pack.zip"
                with zipfile.ZipFile(archive, "w") as stream:
                    stream.writestr(member, b"bad")
                with self.assertRaisesRegex(ValueError, "Unsafe archive"):
                    prepare.unpack_pack(archive, root, "pack")
                self.assertFalse((root / "outside").exists())

    def test_hash_mismatch_is_rejected_even_when_size_matches(self):
        with tempfile.TemporaryDirectory() as folder:
            archive = Path(folder) / "pack.zip"
            archive.write_bytes(b"12345")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                prepare.verify_archive(archive, {"size": 5, "sha256": "0" * 64})

    def test_extraction_rejects_symlink_leftovers_without_touching_their_targets(self):
        for target_kind in ("partial", "parent", "staging", "existing_parent"):
            with self.subTest(kind=target_kind), tempfile.TemporaryDirectory() as folder:
                root = Path(folder) / "output"
                root.mkdir()
                external = Path(folder) / "external"
                external.mkdir()
                original = external / "data.bin"
                original.write_bytes(b"keep")
                archive = root / "pack.zip"
                with zipfile.ZipFile(archive, "w") as stream:
                    stream.writestr("pack/meta/data.bin", b"replace")
                staging = root / ".pack.extracting"
                if target_kind == "staging":
                    staging.symlink_to(external, target_is_directory=True)
                elif target_kind == "existing_parent":
                    (root / "pack").mkdir()
                    (root / "pack/meta").symlink_to(external, target_is_directory=True)
                elif target_kind == "parent":
                    (staging / "pack").mkdir(parents=True)
                    (staging / "pack/meta").symlink_to(external, target_is_directory=True)
                else:
                    (staging / "pack/meta").mkdir(parents=True)
                    (staging / "pack/meta/data.bin.part").symlink_to(original)
                with self.assertRaisesRegex(ValueError, "symlink"):
                    prepare.unpack_pack(archive, root, "pack")
                self.assertEqual(original.read_bytes(), b"keep")

    def test_prepare_commits_receipt_only_for_two_valid_packs_with_shared_stats(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            packs = {}
            for pack in ("train", "val"):
                archive = root / "archives/official/sonic" / (pack + ".zip")
                archive.parent.mkdir(parents=True, exist_ok=True)
                info = {"fps": 30, "total_episodes": 1, "total_frames": 60,
                        "features": {"action.body_token_v1_1": {"shape": [64]},
                                     "observation.state": {"shape": [43]}}}
                with zipfile.ZipFile(archive, "w") as stream:
                    stream.writestr(pack + "/meta/info.json", json.dumps(info))
                    for name in ("modality.json", "stats_psi0.json", "episodes.jsonl"):
                        stream.writestr(pack + "/meta/" + name, "{}")
                packs[pack] = {"size": archive.stat().st_size,
                               "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}
            original = prepare.PACKS
            try:
                prepare.PACKS = packs
                receipt = prepare.prepare(root, download=False)
                self.assertFalse(receipt["training_started"])
                self.assertEqual(set(receipt["packs"]), {"train", "val"})
                self.assertTrue((root / ".pipeline/unifolm_download_receipt.json").is_file())
            finally:
                prepare.PACKS = original


if __name__ == "__main__":
    unittest.main()
