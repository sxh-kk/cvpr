"""Real CPU lossless video/PNG round trips and fail-closed corruption checks."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pico_pick_rgb_storage as storage


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


class ExactRGBStorageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "episode"
        self.capture = self.root / "capture"
        (self.capture / "rgb").mkdir(parents=True)
        (self.root / "deferred_capture").mkdir()
        self.raw = self.root / "deferred_capture/model.mjb"
        self.raw.write_bytes(b"original-physics-must-not-change")
        self.frames = []
        self.original = {}
        rng = np.random.default_rng(20261109)
        for index in range(5):
            pixels = rng.integers(0, 256, size=(24, 32, 3), dtype=np.uint8)
            # Nonuniform exact RGB catches lossy YUV conversion and resizing.
            pixels[0, :, :] = [index, 19, 251]
            path = self.capture / "rgb" / f"frame_{index:06d}.png"
            Image.fromarray(pixels).save(path, format="PNG", compress_level=3)
            self.original[path] = path.read_bytes()
            self.frames.append({"event_index": index, "capture_index30": index,
                "rgb": {"path": str(path.relative_to(self.capture)),
                    "file_sha256": storage.sha(path),
                    "pixel_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
                    "height": 24, "width": 32, "channels": 3, "dtype": "uint8"}})
        self.publish_sources()

    def publish_sources(self):
        events = self.capture / "events.jsonl"
        events.write_text("".join(json.dumps(frame) + "\n" for frame in self.frames))
        raw = self.root / "deferred_capture/capture.json"
        write_json(raw, {"status": "raw_physics_capture_completed_pending_rtx", "command_rate_hz": 30})
        manifest = self.capture / "capture.json"
        write_json(manifest, {"status": "raw_capture_completed_pending_independent_audit",
            "event_count": len(self.frames), "command_rate_hz": 30, "events_sha256": storage.sha(events)})
        write_json(self.root / "rtx_completion.json", {
            "status": "deferred_rtx_completed_pending_independent_audit",
            "archived_physics_steps": 0, "completed_frames": len(self.frames),
            "capture_manifest_sha256": storage.sha(manifest),
            "source_manifest_sha256": storage.sha(raw)})

    def tearDown(self):
        storage.close_readers()
        self.temporary.cleanup()

    def test_exact_video_png_restore_and_original_physics_unchanged(self):
        original_raw = self.raw.read_bytes()
        original_events = (self.capture / "events.jsonl").read_bytes()
        # Unlisted images are never candidates for removal.
        extra = self.capture / "rgb/debug.png"
        extra.write_bytes(b"unlisted debug image")
        report = storage.compact_episode(self.root)
        self.assertTrue(report["derived_pngs_removed"])
        self.assertTrue(all(not path.exists() for path in self.original))
        self.assertEqual(extra.read_bytes(), b"unlisted debug image")
        self.assertTrue(storage.verify_compacted_episode(self.root)["passed"])
        with patch.object(storage, "_verify_master", wraps=storage._verify_master) as full_check:
            for path, expected in self.original.items():
                self.assertEqual(storage.read_png_bytes(path, hashlib.sha256(expected).hexdigest()), expected)
                self.assertTrue(storage.verify_missing_png(path, hashlib.sha256(expected).hexdigest()))
            # Backwards access reopens one decoder; it does not redo full-video
            # independent verification for each PNG binding.
            self.assertEqual(storage.read_png_bytes(next(iter(self.original))), next(iter(self.original.values())))
            self.assertEqual(full_check.call_count, 1)
        restored = storage.restore_episode(self.root)
        self.assertEqual(restored["restored_pngs"], len(self.original))
        self.assertEqual({path: path.read_bytes() for path in self.original}, self.original)
        self.assertEqual(storage.restore_episode(self.root)["restored_pngs"], 0)
        self.assertEqual(self.raw.read_bytes(), original_raw)
        self.assertEqual((self.capture / "events.jsonl").read_bytes(), original_events)
        receipt = json.loads((self.capture / storage.RECEIPT_NAME).read_text())
        self.assertEqual(receipt["master"]["preset"], "veryslow")
        self.assertEqual(receipt["master"]["fps"], 30)
        self.assertEqual(receipt["png_encoder"]["compress_level"], 3)

    def test_retained_png_mode_and_interrupted_removal_resume(self):
        storage.compact_episode(self.root, delete_pngs=False)
        self.assertTrue(all(path.exists() for path in self.original))
        receipt_bytes = (self.capture / storage.RECEIPT_NAME).read_bytes()
        next(iter(self.original)).unlink()
        (self.capture / storage.COMPLETE_NAME).unlink()
        storage.compact_episode(self.root, delete_pngs=True)
        self.assertEqual((self.capture / storage.RECEIPT_NAME).read_bytes(), receipt_bytes)
        self.assertFalse(any(path.exists() for path in self.original))
        self.assertTrue(storage.verify_compacted_episode(self.root)["passed"])

    def test_file_reencoding_mismatch_refuses_all_deletion(self):
        path = next(iter(self.original))
        with Image.open(path) as image:
            pixels = np.asarray(image).copy()
        Image.fromarray(pixels).save(path, format="PNG", compress_level=0)
        self.frames[0]["rgb"]["file_sha256"] = storage.sha(path)
        self.publish_sources()
        with self.assertRaisesRegex(ValueError, "PNG bytes cannot be reproduced"):
            storage.compact_episode(self.root)
        self.assertTrue(all(path.exists() for path in self.original))
        self.assertFalse((self.capture / storage.RECEIPT_NAME).exists())

    def test_corrupt_png_and_changed_source_authority_refuse_deletion(self):
        path = next(iter(self.original))
        path.write_bytes(path.read_bytes() + b"tampered")
        with self.assertRaisesRegex(ValueError, "PNG file SHA"):
            storage.compact_episode(self.root)
        self.assertTrue(all(path.exists() for path in self.original))
        path.write_bytes(self.original[path])
        events = self.capture / "events.jsonl"
        events.write_bytes(events.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "SHA authority differs"):
            storage.compact_episode(self.root)
        self.assertTrue(all(path.exists() for path in self.original))

    def test_master_tamper_and_missing_png_wrong_sha_fail_closed(self):
        storage.compact_episode(self.root)
        path = next(iter(self.original))
        with self.assertRaisesRegex(ValueError, "path/SHA"):
            storage.verify_missing_png(path, "0" * 64)
        master = self.capture / storage.MASTER_NAME
        master.write_bytes(master.read_bytes() + b"tampered")
        with self.assertRaisesRegex(ValueError, "master SHA"):
            storage.verify_compacted_episode(self.root)
        with self.assertRaisesRegex(ValueError, "master SHA"):
            storage.read_png_bytes(path)

    def test_symlink_and_unfinished_render_rejected_before_any_delete(self):
        path = next(iter(self.original))
        moved = path.with_name("saved-source.png")
        path.rename(moved)
        path.symlink_to(moved.name)
        with self.assertRaisesRegex(ValueError, "linked RGB authority"):
            storage.compact_episode(self.root)
        self.assertTrue(moved.is_file())
        path.unlink()
        moved.rename(path)
        receipt = self.root / "rtx_completion.json"
        value = json.loads(receipt.read_text())
        value["status"] = "rendering"
        write_json(receipt, value)
        with self.assertRaisesRegex(ValueError, "completed derived RTX"):
            storage.compact_episode(self.root)
        self.assertTrue(all(path.exists() for path in self.original))


if __name__ == "__main__":
    unittest.main()
