"""Training-compatible normalization for canonical v4_beta trajectories and K12 motion."""

from pathlib import Path

import torch

from ..config import K12_JOINTS, ROOT
from .representation_utils import sparse_motion_feature_indices

_DEFAULT_CLIP = object()

class MotionNormalizer:
    def __init__(self, stats=None, clip=_DEFAULT_CLIP):
        if stats is None:
            stats = ROOT / "assets/e7_normalization.pt"
        if isinstance(stats, (str, Path)):
            stats = torch.load(stats, map_location="cpu", weights_only=True)
        self.stats = {}
        for kind, dim in (("motion", 153), ("traj", 18)):
            for suffix in ("mean", "std"):
                key = f"{kind}_{suffix}"
                value = torch.as_tensor(stats[key]).float()
                if kind == "motion" and value.shape == (243,):
                    value = value.index_select(0, sparse_motion_feature_indices(K12_JOINTS))
                if value.shape != (dim,) or not torch.isfinite(value).all():
                    raise ValueError(f"Invalid normalization tensor {key}")
                if suffix == "std" and (value <= 0).any():
                    raise ValueError(f"Non-positive normalization scale {key}")
                self.stats[key] = value
        self.clip = stats.get("normalization_clip", 8.0) if clip is _DEFAULT_CLIP else clip

    def normalize(self, value, kind):
        value = torch.as_tensor(value).float()
        mean, std = (self.stats[f"{kind}_{key}"].to(value.device) for key in ("mean", "std"))
        result = (value - mean) / (std + 1e-6)
        return result.clamp(-self.clip, self.clip) if self.clip is not None else result

    def denormalize(self, value, kind="motion"):
        value = torch.as_tensor(value).float()
        mean, std = (self.stats[f"{kind}_{key}"].to(value.device) for key in ("mean", "std"))
        return value * (std + 1e-6) + mean
