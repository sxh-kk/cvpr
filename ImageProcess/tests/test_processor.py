"""CPU contract tests; run with unittest or pytest, without model weights."""

import copy
import json
from pathlib import Path
import unittest

import numpy as np
from PIL import Image

from ImageProcess import SharedImageProcessor, load_profile


def _test_image(height=91, width=137):
    yy, xx = np.mgrid[:height, :width]
    return np.stack(((xx * 7 + yy) % 256, (yy * 5 + xx) % 256, (xx * 3 + yy * 11) % 256), axis=-1).astype(np.uint8)


class ImageProfileTests(unittest.TestCase):
    def test_builtin_geometry_and_immutable_copy(self):
        for name, resize_hw, output_hw in (("256x480", (270, 480), (256, 480)), ("256x320", (240, 320), (256, 320))):
            with self.subTest(name=name):
                profile = load_profile(name)
                self.assertEqual(profile.resize_hw, resize_hw)
                self.assertEqual(profile.output_hw, output_hw)
                from_mapping = load_profile(profile.to_dict())
                self.assertEqual(profile.fingerprint, from_mapping.fingerprint)
                self.assertIs(profile, load_profile(profile))
                copy_config = profile.to_dict()
                copy_config["geometry"]["output_hw"][0] = 1
                self.assertEqual(profile.output_hw, output_hw)

    def test_file_and_metadata_serialization(self):
        path = Path(__file__).resolve().parents[1] / "configs" / "256x480.json"
        processor = SharedImageProcessor(path)
        self.assertEqual(processor.metadata, json.loads(json.dumps(processor.metadata)))
        self.assertEqual(processor.metadata["height"], 256)
        self.assertEqual(processor.metadata["width"], 480)
        self.assertEqual(processor.metadata["profile_sha256"], load_profile("256x480").fingerprint)
        copy_metadata = processor.metadata
        copy_metadata["profile"]["geometry"]["output_hw"][0] = 1
        self.assertEqual(processor.metadata["height"], 256)

    def test_geometry_mismatch_rejected(self):
        config = load_profile("256x480").to_dict()
        config["geometry"]["output_hw"] = [270, 480]
        with self.assertRaisesRegex(ValueError, "grid"):
            load_profile(config)

    def test_invalid_config_rejected(self):
        for patch in (
            {"unexpected": 0},
            {"profile_id": "../bad"},
            {"schema_version": True},
            {"schema_version": 1.0},
        ):
            config = load_profile("256x480").to_dict()
            config.update(patch)
            with self.assertRaises(ValueError):
                load_profile(config)
        config = load_profile("256x480").to_dict()
        config["augmentation"]["posttrain"]["hue"] = [-1.0, 1.0]
        with self.assertRaises(ValueError):
            load_profile(config)


class SharedImageTests(unittest.TestCase):
    def setUp(self):
        self.image = _test_image()
        self.processor = SharedImageProcessor("256x480")

    def test_two_final_shapes_and_reference_pixels(self):
        for name, intermediate in (("256x480", (480, 270)), ("256x320", (320, 240))):
            processor = SharedImageProcessor(name)
            output = processor.process_image(self.image)
            height, width = processor.profile.output_hw
            reference = Image.fromarray(self.image).resize(intermediate, Image.Resampling.NEAREST)
            reference = np.asarray(reference.resize((width, height), Image.Resampling.BICUBIC))
            self.assertEqual(output.shape, (height, width, 3))
            self.assertEqual(output.dtype, np.uint8)
            np.testing.assert_array_equal(output, reference)

    def test_single_resize_does_not_silently_replace_two_steps(self):
        expected = self.processor.process_image(self.image)
        direct = np.asarray(Image.fromarray(self.image).resize((480, 256), Image.Resampling.BICUBIC))
        self.assertFalse(np.array_equal(expected, direct))

    def test_pretrain_validation_inference_and_training_false_are_unaugmented(self):
        expected = self.processor.process_image(self.image)
        for stage in ("pretrain", "validation", "inference"):
            result = self.processor.process_window([self.image], stage=stage, seed=42)
            np.testing.assert_array_equal(result.images[0], expected)
            self.assertIsNone(result.augmentation["color"])
            self.assertIsNone(result.augmentation["crop"])
        for stage in ("posttrain", "robust"):
            result = self.processor.process_window([self.image], stage=stage, training=False, seed=42)
            np.testing.assert_array_equal(result.images[0], expected)

    def test_window_shares_parameters_and_json_replay(self):
        for stage in ("posttrain", "robust"):
            result = self.processor.process_window([self.image] * 6, stage=stage, seed=17)
            for frame in result.images:
                np.testing.assert_array_equal(frame, result.images[0])
            augmentation = json.loads(json.dumps(result.augmentation))
            replay = self.processor.process_window([self.image], stage=stage, augmentation=augmentation)
            np.testing.assert_array_equal(replay.images[0], result.images[-1])
            self.assertEqual(result.augmentation, replay.augmentation)
            same_seed = self.processor.process_window([self.image], stage=stage, seed=17)
            np.testing.assert_array_equal(same_seed.images[0], result.images[-1])

    def test_seed_uses_local_rng(self):
        before = np.random.get_state()
        self.processor.process_window([self.image], stage="robust", seed=19)
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_colors_change_pixels_and_stay_in_bounds(self):
        result = self.processor.process_window([self.image], stage="posttrain", seed=11)
        self.assertIsNone(result.augmentation["crop"])
        self.assertFalse(np.array_equal(result.images[0], self.processor.process_image(self.image)))
        factors = result.augmentation["color"]["factors"]
        for key in ("brightness", "contrast", "saturation"):
            self.assertGreaterEqual(factors[key], 0.8)
            self.assertLessEqual(factors[key], 1.2)
        self.assertLessEqual(abs(factors["hue"]), 0.05)
        self.assertEqual(set(result.augmentation["color"]["order"]), set(factors))

    def test_robust_crop_bounds_aspect_and_crop_effect(self):
        result = self.processor.process_window([self.image], stage="robust", seed=23)
        crop = result.augmentation["crop"]
        self.assertGreaterEqual(crop["height"], int(270 * 0.85))
        self.assertGreaterEqual(crop["width"], int(480 * 0.85))
        self.assertLessEqual(crop["top"] + crop["height"], 270)
        self.assertLessEqual(crop["left"] + crop["width"], 480)
        self.assertLess(abs(crop["width"] / crop["height"] - 480 / 270), 0.01)
        # Isolate geometry with neutral colors, preserving valid replay metadata.
        neutral = copy.deepcopy(result.augmentation)
        neutral["color"]["factors"] = {"brightness": 1.0, "contrast": 1.0, "saturation": 1.0, "hue": 0.0}
        cropped = self.processor.process_window([self.image], stage="robust", augmentation=neutral)
        self.assertFalse(np.array_equal(cropped.images[0], self.processor.process_image(self.image)))

    def test_wrong_profile_stage_and_invalid_crop_replay_rejected(self):
        params = self.processor.process_window([self.image], stage="robust", seed=2).augmentation
        with self.assertRaisesRegex(ValueError, "profile"):
            SharedImageProcessor("256x320").process_window([self.image], stage="robust", augmentation=params)
        with self.assertRaisesRegex(ValueError, "stage"):
            self.processor.process_window([self.image], stage="posttrain", augmentation=params)
        invalid = copy.deepcopy(params)
        invalid["crop"]["left"] = 480
        with self.assertRaisesRegex(ValueError, "beyond"):
            self.processor.process_window([self.image], stage="robust", augmentation=invalid)

    def test_input_validation(self):
        with self.assertRaises(TypeError):
            self.processor.process_image(self.image.astype(np.float32))
        with self.assertRaises(ValueError):
            self.processor.process_image(np.zeros((4, 4), dtype=np.uint8))
        with self.assertRaises(ValueError):
            self.processor.process_window([])
        with self.assertRaises(TypeError):
            self.processor.process_window(self.image)
        with self.assertRaises(ValueError):
            self.processor.process_window([self.image], stage="unknown")
        with self.assertRaises(ValueError):
            self.processor.process_window([self.image], seed=True)
        params = self.processor.process_window([self.image]).augmentation
        with self.assertRaises(ValueError):
            self.processor.process_window([self.image], seed=1, augmentation=params)

    def test_pil_input_and_input_unchanged(self):
        original = self.image.copy()
        a = self.processor.process_image(self.image)
        b = self.processor.process_image(Image.fromarray(self.image))
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(original, self.image)

    def test_pil_augmentation_matches_torchvision_when_available(self):
        try:
            from torchvision.transforms import functional as F
        except (ImportError, RuntimeError):
            self.skipTest("torchvision unavailable; base dependency is Pillow")
        result = self.processor.process_window([self.image], stage="robust", seed=37)
        reference = F.resize(Image.fromarray(self.image), [270, 480], interpolation=F.InterpolationMode.NEAREST)
        crop = result.augmentation["crop"]
        reference = F.resized_crop(reference, crop["top"], crop["left"], crop["height"], crop["width"], [270, 480], interpolation=F.InterpolationMode.BILINEAR, antialias=True)
        color = result.augmentation["color"]
        for operation in color["order"]:
            reference = getattr(F, f"adjust_{operation}")(reference, color["factors"][operation])
        reference = np.asarray(reference.resize((480, 256), Image.Resampling.BICUBIC))
        np.testing.assert_array_equal(result.images[0], reference)


if __name__ == "__main__":
    unittest.main()
