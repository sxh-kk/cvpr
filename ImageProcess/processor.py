"""Psi0-compatible geometry with a single augmentation draw per RGB window."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, __version__ as pillow_version

_IMPLEMENTATION = "psi0-shared-rgb-pil-v1"
_STAGES = {"pretrain", "posttrain", "robust", "validation", "inference"}
_COLOR_OPERATIONS = ("brightness", "contrast", "saturation", "hue")


def _keys(value: Any, expected: set[str], location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{location} must contain exactly {sorted(expected)}")
    return value


def _integer(value: Any, location: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{location} must be an integer >= {minimum}")
    return value


def _number(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{location} must be a finite number")
    return float(value)


def _size(value: Any, location: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{location} must be [height, width]")
    return tuple(_integer(x, location) for x in value)


def _grid_resize(hw: tuple[int, int], factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    """Qwen smart-resize rounding/area rule for an already geometrically resized image."""
    height, width = hw
    output_height = max(factor, round(height / factor) * factor)
    output_width = max(factor, round(width / factor) * factor)
    if output_height * output_width > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        output_height = math.floor(height / beta / factor) * factor
        output_width = math.floor(width / beta / factor) * factor
    elif output_height * output_width < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        output_height = math.ceil(height * beta / factor) * factor
        output_width = math.ceil(width * beta / factor) * factor
    return output_height, output_width


def _validate_profile(config: Any) -> dict[str, Any]:
    _keys(config, {"schema_version", "profile_id", "source", "geometry", "augmentation"}, "profile")
    if _integer(config["schema_version"], "schema_version") != 1:
        raise ValueError("unsupported image profile schema_version")
    if not isinstance(config["profile_id"], str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", config["profile_id"]):
        raise ValueError("profile_id must be a nonempty identifier")
    source = _keys(config["source"], {"repository", "revision"}, "source")
    if any(not isinstance(v, str) or not v for v in source.values()):
        raise ValueError("source repository and revision must be nonempty strings")
    geometry = _keys(config["geometry"], {
        "resize_hw", "resize_interpolation", "center_crop_hw", "output_hw",
        "output_interpolation", "grid_factor", "min_pixels", "max_pixels",
    }, "geometry")
    resize_hw = _size(geometry["resize_hw"], "resize_hw")
    if _size(geometry["center_crop_hw"], "center_crop_hw") != resize_hw:
        raise ValueError("this profile version requires center_crop_hw equal to resize_hw")
    output_hw = _size(geometry["output_hw"], "output_hw")
    if geometry["resize_interpolation"] != "nearest" or geometry["output_interpolation"] != "bicubic":
        raise ValueError("this profile version requires nearest then bicubic interpolation")
    factor = _integer(geometry["grid_factor"], "grid_factor")
    min_pixels = _integer(geometry["min_pixels"], "min_pixels")
    max_pixels = _integer(geometry["max_pixels"], "max_pixels")
    if max_pixels < min_pixels:
        raise ValueError("max_pixels must be >= min_pixels")
    if output_hw != _grid_resize(resize_hw, factor, min_pixels, max_pixels):
        raise ValueError("output_hw does not match the configured Qwen grid/area rule")
    augmentation = _keys(config["augmentation"], {"posttrain", "robust"}, "augmentation")
    color = _keys(augmentation["posttrain"], {*_COLOR_OPERATIONS, "random_order"}, "posttrain augmentation")
    for key in _COLOR_OPERATIONS:
        bounds = color[key]
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(f"{key} requires lower/upper bounds")
        low, high = (_number(x, key) for x in bounds)
        if low > high or (key != "hue" and low < 0) or (key == "hue" and (low < -0.5 or high > 0.5)):
            raise ValueError(f"invalid {key} bounds")
    if color["random_order"] is not True:
        raise ValueError("color augmentation requires random_order=true")
    crop = _keys(augmentation["robust"], {
        "min_scale", "probability", "keep_aspect_ratio", "interpolation", "antialias",
    }, "robust augmentation")
    if not 0 < _number(crop["min_scale"], "min_scale") <= 1:
        raise ValueError("min_scale must be in (0, 1]")
    if not 0 <= _number(crop["probability"], "probability") <= 1:
        raise ValueError("probability must be in [0, 1]")
    if crop["keep_aspect_ratio"] is not True or crop["interpolation"] != "bilinear" or crop["antialias"] is not True:
        raise ValueError("robust crop requires preserved aspect ratio and antialiased bilinear resize")
    return json.loads(json.dumps(config, allow_nan=False))


@dataclass(frozen=True)
class ImageProfile:
    """Validated immutable profile; to_dict returns a detached JSON-serializable copy."""

    _canonical_json: str

    def __post_init__(self) -> None:
        config = _validate_profile(json.loads(self._canonical_json))
        object.__setattr__(self, "_canonical_json", json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False))

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self._canonical_json)

    @property
    def profile_id(self) -> str:
        return self.to_dict()["profile_id"]

    @property
    def resize_hw(self) -> tuple[int, int]:
        return tuple(self.to_dict()["geometry"]["resize_hw"])

    @property
    def output_hw(self) -> tuple[int, int]:
        return tuple(self.to_dict()["geometry"]["output_hw"])

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self._canonical_json.encode("utf-8")).hexdigest()


def load_profile(value: str | Path | Mapping[str, Any] | ImageProfile = "256x480") -> ImageProfile:
    """Load a built-in name, JSON file, configuration mapping, or validated profile."""
    if isinstance(value, ImageProfile):
        return value
    if isinstance(value, Mapping):
        config = value
    elif isinstance(value, (str, Path)):
        path = Path(value)
        if str(value) in {"256x480", "256x320"}:
            path = Path(__file__).with_name("configs") / f"{value}.json"
        with path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    else:
        raise TypeError("profile must be a built-in name, JSON path, mapping, or ImageProfile")
    return ImageProfile(json.dumps(config, allow_nan=False))


def _rgb_image(value: Image.Image | np.ndarray) -> Image.Image:
    if isinstance(value, Image.Image):
        if value.width < 1 or value.height < 1:
            raise ValueError("image must have positive dimensions")
        return value.convert("RGB")
    if not isinstance(value, np.ndarray) or value.dtype != np.uint8:
        raise TypeError("array input must be an HWC RGB uint8 numpy array")
    if value.ndim != 3 or value.shape[2] != 3 or min(value.shape[:2]) < 1:
        raise ValueError("array input must have nonempty shape [height, width, 3]")
    return Image.fromarray(np.ascontiguousarray(value))


def _adjust_hue(image: Image.Image, factor: float) -> Image.Image:
    hue, saturation, value = image.convert("HSV").split()
    # The integer shift and uint8 wrap reproduce torchvision's PIL hue adjustment.
    hue_array = (np.asarray(hue, dtype=np.int16) + int(factor * 255)) % 256
    return Image.merge("HSV", (Image.fromarray(hue_array.astype(np.uint8)), saturation, value)).convert("RGB")


@dataclass
class ProcessedWindow:
    images: list[np.ndarray]
    augmentation: dict[str, Any]


class SharedImageProcessor:
    """Apply geometry and synchronized augmentation before branching to encoders.

    Inputs and outputs are RGB, with no channel normalization. This object is
    stateless across calls: each window has a local RNG, and its complete sampled
    augmentation is returned for deterministic replay by another branch.
    """

    def __init__(self, profile: str | Path | Mapping[str, Any] | ImageProfile = "256x480") -> None:
        self.profile = load_profile(profile)
        self._config = self.profile.to_dict()
        self._resize_hw = self.profile.resize_hw
        self._output_hw = self.profile.output_hw

    @property
    def metadata(self) -> dict[str, Any]:
        height, width = self._output_hw
        return {
            "profile_id": self.profile.profile_id,
            "height": height,
            "width": width,
            "profile_sha256": self.profile.fingerprint,
            "implementation": _IMPLEMENTATION,
            "pillow_version": pillow_version,
            "numpy_version": np.__version__,
            "input": "RGB uint8 HWC",
            "output": "RGB uint8 HWC",
            "normalization": "encoder_specific_outside_shared_processor",
            "profile": self.profile.to_dict(),
        }

    def process_image(self, image: Image.Image | np.ndarray) -> np.ndarray:
        """Deterministic pretraining geometry, without random augmentation."""
        return self._apply(image, {"crop": None, "color": None})

    def _sample(self, stage: str, training: bool, seed: int | None) -> dict[str, Any]:
        rng = np.random.default_rng(seed)
        result = {
            "schema_version": 1,
            "implementation": _IMPLEMENTATION,
            "profile_sha256": self.profile.fingerprint,
            "stage": stage,
            "training": training,
            "crop": None,
            "color": None,
        }
        if not training or stage not in {"posttrain", "robust"}:
            return result
        if stage == "robust":
            crop_config = self._config["augmentation"]["robust"]
            if rng.random() < crop_config["probability"]:
                height, width = self._resize_hw
                ratio = width / height
                for _ in range(10):
                    area = height * width * rng.uniform(crop_config["min_scale"] ** 2, 1.0)
                    crop_width = round(math.sqrt(area * ratio))
                    crop_height = round(math.sqrt(area / ratio))
                    if 0 < crop_width <= width and 0 < crop_height <= height:
                        break
                else:
                    crop_height, crop_width = height, width
                result["crop"] = {
                    "top": int(rng.integers(0, height - crop_height + 1)),
                    "left": int(rng.integers(0, width - crop_width + 1)),
                    "height": crop_height,
                    "width": crop_width,
                }
        color_config = self._config["augmentation"]["posttrain"]
        result["color"] = {
            "order": list(rng.permutation(_COLOR_OPERATIONS)),
            "factors": {key: float(rng.uniform(*color_config[key])) for key in _COLOR_OPERATIONS},
        }
        return result

    def _validate_replay(self, augmentation: Mapping[str, Any], stage: str, training: bool) -> dict[str, Any]:
        _keys(augmentation, {"schema_version", "implementation", "profile_sha256", "stage", "training", "crop", "color"}, "augmentation")
        replay = json.loads(json.dumps(augmentation, allow_nan=False))
        if _integer(replay["schema_version"], "augmentation schema_version") != 1 or replay["implementation"] != _IMPLEMENTATION:
            raise ValueError("augmentation implementation/schema does not match")
        if replay["profile_sha256"] != self.profile.fingerprint:
            raise ValueError("augmentation profile does not match this processor")
        if replay["stage"] != stage or replay["training"] is not training:
            raise ValueError("augmentation stage/training does not match this call")
        active = training and stage in {"posttrain", "robust"}
        if not active:
            if replay["crop"] is not None or replay["color"] is not None:
                raise ValueError("pretraining/evaluation cannot replay random augmentation")
            return replay
        color = _keys(replay["color"], {"order", "factors"}, "color")
        if not isinstance(color["order"], list) or len(color["order"]) != 4 or set(color["order"]) != set(_COLOR_OPERATIONS):
            raise ValueError("color order must contain each operation exactly once")
        factors = _keys(color["factors"], set(_COLOR_OPERATIONS), "color factors")
        for key in _COLOR_OPERATIONS:
            value = _number(factors[key], key)
            low, high = self._config["augmentation"]["posttrain"][key]
            if not low <= value <= high:
                raise ValueError(f"replayed {key} is outside the profile bounds")
        crop = replay["crop"]
        if stage != "robust" and crop is not None:
            raise ValueError("view cropping is only allowed in robust training")
        if stage == "robust" and self._config["augmentation"]["robust"]["probability"] == 1.0 and crop is None:
            raise ValueError("this robust profile requires a sampled crop")
        if crop is not None:
            _keys(crop, {"top", "left", "height", "width"}, "crop")
            top = _integer(crop["top"], "crop top", 0)
            left = _integer(crop["left"], "crop left", 0)
            height = _integer(crop["height"], "crop height")
            width = _integer(crop["width"], "crop width")
            image_height, image_width = self._resize_hw
            if top + height > image_height or left + width > image_width:
                raise ValueError("replayed crop extends beyond the intermediate image")
            min_scale = self._config["augmentation"]["robust"]["min_scale"]
            if height < math.floor(image_height * min_scale) or width < math.floor(image_width * min_scale):
                raise ValueError("replayed crop is smaller than the profile permits")
            if abs(width - height * image_width / image_height) > 1 + image_width / image_height:
                raise ValueError("replayed crop does not preserve the profile aspect ratio")
        return replay

    def _apply(self, image: Image.Image | np.ndarray, augmentation: Mapping[str, Any]) -> np.ndarray:
        image = _rgb_image(image)
        height, width = self._resize_hw
        image = image.resize((width, height), Image.Resampling.NEAREST)
        # Both profiles center-crop to exactly the resized dimensions: no extra crop.
        crop = augmentation["crop"]
        if crop is not None:
            left, top = crop["left"], crop["top"]
            image = image.crop((left, top, left + crop["width"], top + crop["height"]))
            image = image.resize((width, height), Image.Resampling.BILINEAR)
        color = augmentation["color"]
        if color is not None:
            for operation in color["order"]:
                factor = color["factors"][operation]
                if operation == "hue":
                    image = _adjust_hue(image, factor)
                else:
                    enhancer = {"brightness": ImageEnhance.Brightness, "contrast": ImageEnhance.Contrast, "saturation": ImageEnhance.Color}[operation]
                    image = enhancer(image).enhance(factor)
        output_height, output_width = self._output_hw
        image = image.resize((output_width, output_height), Image.Resampling.BICUBIC)
        return np.array(image, dtype=np.uint8, copy=True)

    def process_window(
        self,
        images: Iterable[Image.Image | np.ndarray],
        *,
        stage: str = "pretrain",
        seed: int | None = None,
        training: bool = True,
        augmentation: Mapping[str, Any] | None = None,
    ) -> ProcessedWindow:
        """Sample once per window, or replay a JSON-compatible augmentation record.

        ``pretrain``, ``validation``, and ``inference`` always disable random
        augmentation. ``training=False`` also disables it for the other stages.
        Send the returned images to both encoders, or pass ``augmentation`` into
        another call with the same profile/stage/training to replay the parameters.
        """
        if not isinstance(stage, str) or stage not in _STAGES:
            raise ValueError(f"stage must be one of {sorted(_STAGES)}")
        if not isinstance(training, bool):
            raise TypeError("training must be a bool")
        if seed is not None:
            _integer(seed, "seed", 0)
        if augmentation is not None and seed is not None:
            raise ValueError("use either seed or augmentation replay")
        if isinstance(images, (np.ndarray, Image.Image)):
            raise TypeError("process_window expects an iterable of individual RGB images")
        images = list(images)
        if not images:
            raise ValueError("window must contain at least one image")
        params = self._sample(stage, training, seed) if augmentation is None else self._validate_replay(augmentation, stage, training)
        return ProcessedWindow([self._apply(image, params) for image in images], params)
