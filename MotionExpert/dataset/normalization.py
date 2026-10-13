"""Strict train-only SMPL-H245 statistics; binary contacts keep their [0,1] scale."""
from pathlib import Path
import math
import torch

from .representation import CONTACT_SLICE, GLOBAL_SLICE, MOTION_DIM, REPRE_TYPE, TRAJECTORY_DIM
from .motion_layout import motion_layout, SMPLH_REPRE_TYPE

STATS_FORMAT = "motionexpert-dense-stats-v1"
_DEFAULT_CLIP = object()


def statistics_motion_supervision(stats):
    if stats.get("motion_supervision") != "joint":
        raise ValueError("Statistics must supervise joint historical and future motion")
    return "joint"


def validate_statistics_profile(stats):
    expected = {"fps": 5, "stats_schema_version": 4, "train_task": "fore", "label_task": "fore",
                "window": 11, "forecast_prefix": 6, "complete_windows_only": True,
                "sampling_step": 6, "ground_estimation_scope": "observed_prefix",
                "image_validity_scope": "observed_prefix", "statistics_scope": "all_valid_frames",
                "motion_supervision": "joint"}
    for key, value in expected.items():
        if stats.get(key) != value:
            raise ValueError(f"SMPL-H statistics {key} must be {value!r}")
    fraction = stats.get("min_valid_image_fraction")
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("Statistics require a valid observed-image threshold")


class MotionNormalizer:
    def __init__(self, stats=None, clip=_DEFAULT_CLIP, *, allow_diagnostic=False):
        if stats is None:
            raise ValueError("MotionExpert requires train-only STATS_PATH")
        if isinstance(stats, (str, Path)):
            stats = torch.load(stats, map_location="cpu", weights_only=True)
        if not isinstance(stats, dict) or stats.get("format") != STATS_FORMAT:
            raise ValueError("Expected versioned motion training statistics")
        self.layout = motion_layout(stats.get("representation"))
        if self.layout.representation == SMPLH_REPRE_TYPE:
            if stats.get("diagnostic",False) and not allow_diagnostic:
                raise ValueError("Diagnostic SMPL-H statistics require explicit read-only diagnostic use")
            from ..config import SMPLH_MODEL_SHA256, SMPLH_SCHEMA
            if (stats.get("model_sha256") != SMPLH_MODEL_SHA256
                    or stats.get("archive_schema_version") != SMPLH_SCHEMA
                    or stats.get("rotation_basis_mode") != "smplh_model_defined"
                    or stats.get("target_joint_basis") != "smplh_body22_axes_v1"
                    or stats.get("joint_indices") != list(self.layout.joint_indices)):
                raise ValueError("SMPL-H statistics require matching model, schema and defined axes")
        if stats.get("split") != "train" or not stats.get("split_fingerprint"):
            raise ValueError("Statistics require train split provenance and split_fingerprint")
        validate_statistics_profile(stats)
        self.stats = dict(stats)
        for kind, dim in (("motion", self.layout.motion_dim), ("traj", TRAJECTORY_DIM)):
            for suffix in ("mean", "std"):
                key = f"{kind}_{suffix}"
                value = torch.as_tensor(stats[key]).float()
                if value.shape != (dim,) or not torch.isfinite(value).all():
                    raise ValueError(f"Invalid normalization tensor {key}")
                if suffix == "std" and bool((value <= 0).any()):
                    raise ValueError(f"Non-positive normalization scale {key}")
                self.stats[key] = value
        for suffix in ("mean", "std"):
            if not torch.equal(self.stats[f"motion_{suffix}"][self.layout.global_slice], self.stats[f"traj_{suffix}"][9:18]):
                raise ValueError("Motion and trajectory floor deltas must share exact normalization statistics")
        if not torch.equal(self.stats["motion_mean"][self.layout.contact_slice], torch.zeros(2)) or not torch.equal(self.stats["motion_std"][self.layout.contact_slice], torch.ones(2)):
            raise ValueError("Contact channels must retain identity normalization")
        self.clip = stats.get("normalization_clip", None) if clip is _DEFAULT_CLIP else clip
        if self.clip is not None and (isinstance(self.clip, bool) or not isinstance(self.clip, (float,int)) or not math.isfinite(self.clip) or self.clip <= 0):
            raise ValueError("Normalization clip must be positive or None")

    def normalize(self, value, kind="motion", valid=None):
        value = torch.as_tensor(value).float()
        mean, std = (self.stats[f"{kind}_{key}"].to(value.device) for key in ("mean", "std"))
        if value.shape[-1] != len(mean): raise ValueError(f"Invalid {kind} feature width")
        result = (value-mean)/std
        if self.clip is not None:
            clipped = result.clamp(-self.clip, self.clip)
            if kind == "motion": clipped[..., self.layout.contact_slice] = result[..., self.layout.contact_slice]
            result = clipped
        if valid is not None: result = result.masked_fill(~torch.as_tensor(valid, device=result.device).bool(), 0)
        return result

    def denormalize(self, value, kind="motion"):
        value = torch.as_tensor(value).float()
        mean, std = (self.stats[f"{kind}_{key}"].to(value.device) for key in ("mean", "std"))
        if value.shape[-1] != len(mean): raise ValueError(f"Invalid {kind} feature width")
        return value*std+mean
