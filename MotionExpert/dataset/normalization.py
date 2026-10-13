"""Strict training-only dense statistics; binary contacts keep their [0,1] scale."""
from pathlib import Path
import math
import torch

from .dense_representation import CONTACT_SLICE, GLOBAL_SLICE, MOTION_DIM, REPRE_TYPE, TRAJECTORY_DIM
from .motion_layout import motion_layout, SMPLH_REPRE_TYPE

STATS_FORMAT = "motionexpert-dense-stats-v1"
_DEFAULT_CLIP = object()


def statistics_motion_supervision(stats):
    """Old schema4 forecasting statistics supervised only future frames."""
    value = stats.get("motion_supervision")
    if value is None:
        return "future_only" if stats.get("train_task") == "fore" else None
    if value not in ("joint", "future_only"):
        raise ValueError("Statistics motion_supervision must be joint or future_only")
    return value


def validate_statistics_profile(stats):
    """Retain legacy15Hz assets, but never infer the new forecast contract."""
    fps = stats.get("fps")
    if type(fps) is not int or fps < 1 or 30 % fps:
        raise ValueError("Statistics FPS must be an integer divisor of native30Hz")
    version = stats.get("stats_schema_version", 3)
    if type(version) is not int or version not in (3, 4):
        raise ValueError("Unsupported dense statistics schema")
    if version == 3:
        if fps != 15:
            raise ValueError("New sampling profiles require statistics schema4; legacy statistics are15Hz only")
        return
    task = stats.get("train_task")
    if task not in ("recon", "fore") or stats.get("label_task") != task:
        raise ValueError("Statistics require matching train_task and label_task")
    scope = "observed_prefix" if task == "fore" else "full_window"
    supervision = statistics_motion_supervision(stats)
    if supervision == "joint" and task != "fore":
        raise ValueError("Joint statistics require forecasting with observed-prefix conditioning")
    statistics_scope = ("all_valid_frames" if supervision == "joint" else
                        "future_only" if task == "fore" else "all_real_frames")
    if stats.get("ground_estimation_scope") != scope or stats.get("image_validity_scope") != scope:
        raise ValueError("Statistics ground/image scope does not match the training task")
    if stats.get("statistics_scope") != statistics_scope:
        raise ValueError("Statistics target scope does not match the training task")
    if type(stats.get("complete_windows_only")) is not bool:
        raise ValueError("Statistics require an explicit complete-window policy")
    if type(stats.get("sampling_step")) is not int or stats["sampling_step"] != 30 // fps:
        raise ValueError("Statistics sampling_step disagrees with native30Hz/FPS")
    fraction = stats.get("min_valid_image_fraction")
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("Statistics require a valid observed-image threshold")
    if fps == 5 and (task != "fore" or not stats["complete_windows_only"]):
        raise ValueError("5Hz statistics require complete forecasting windows and prefix-floor labels")


class MotionNormalizer:
    def __init__(self, stats=None, clip=_DEFAULT_CLIP, *, allow_diagnostic=False):
        if stats is None:
            raise ValueError("MotionExpert requires train-only STATS_PATH; legacy bundled153 statistics cannot be reused")
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
