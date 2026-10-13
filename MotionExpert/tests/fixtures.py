"""Small dense inference fixtures independent of archives and SMPL assets."""

import torch

from MotionExpert.config import motion_expert_config
from MotionExpert.dataset.normalization import MotionNormalizer, STATS_FORMAT
from MotionExpert.dataset.dense_representation import REPRE_TYPE
from MotionExpert.motion_expert import MotionExpert
from MotionExpert.mydiffusion.flow_matching import FlowMatching


def dense_stats(cfg=None):
    cfg = motion_expert_config() if cfg is None else cfg
    data = cfg.DATA
    task = getattr(data, "TRAIN_TASK", None) or "recon"
    supervision = getattr(data, "MOTION_SUPERVISION", None)
    if supervision is None and task == "fore":
        supervision = "joint" if data.FPS == 5 else "future_only"
    scope = "observed_prefix" if task == "fore" else "full_window"
    return {
        "format": STATS_FORMAT, "representation": REPRE_TYPE,
        "split": "train", "fps": data.FPS, "split_fingerprint": "unit-test-train-only",
        "archive_fingerprint": "unit-test-archive", "basis_mapping_fingerprint": None,
        "window": data.WINDOW, "forecast_prefix": data.FORECAST_PREFIX,
        "segment_stride": data.SEGMENT_STRIDE, "stats_schema_version": 4,
        "sampling_step": 30 // data.FPS, "train_task": task, "label_task": task,
        "complete_windows_only": getattr(data, "REQUIRE_FULL_WINDOW", False),
        "ground_estimation_scope": scope, "image_validity_scope": scope,
        "min_valid_image_fraction": getattr(data, "MIN_VALID_IMAGE_FRACTION", 0.),
        "motion_supervision": supervision,
        "statistics_scope": "all_valid_frames" if supervision == "joint" else "future_only" if task == "fore" else "all_real_frames",
        "target_joint_basis": "egosuite_native_v1", "rotation_basis_mode": "source_native",
        "contact_speed_m_s": .2, "contact_height_m": .1,
        "motion_mean": torch.zeros(245), "motion_std": torch.ones(245),
        "traj_mean": torch.zeros(18), "traj_std": torch.ones(18),
        "normalization_clip": None,
    }


def conditioning(batch=2, frames=7):
    generator = torch.Generator().manual_seed(3)
    return {"traj": torch.randn(batch, frames, 18, generator=generator),
            "img_embs": torch.randn(batch, frames, 5, 1024, generator=generator),
            "valid_frames": torch.ones(batch, frames, dtype=torch.bool),
            "valid_img_embs": torch.ones(batch, frames, dtype=torch.bool),
            "trajectory_valid": torch.ones(batch, frames, 18, dtype=torch.bool)}


class TinyFeatureModel(torch.nn.Module):
    """Deterministic condition-sensitive model to test the public flow API."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.level = torch.nn.Parameter(torch.tensor(.2))
        self.global_token = torch.nn.Parameter(torch.zeros(768))
        self.calls = 0
        self.last_y = None

    def forward(self, x, t, y, cond_scale=None, return_features=False, return_hidden=False):
        self.calls += 1
        self.last_y = y
        trajectory = y["traj"].masked_fill(y["traj_mask"].bool()[..., None], 0)
        image = y["img_embs"].masked_fill(y["img_mask"].bool()[..., None, None], 0)
        summary = trajectory.mean(-1) + image.mean((-1, -2))
        hidden = torch.nn.functional.pad(x, (0, 768-245)) + summary[..., None]
        prediction = self.level.expand_as(x)
        global_hidden = hidden.mean(1)
        if return_features:
            return prediction, hidden, global_hidden
        if return_hidden:
            return prediction, hidden
        return prediction


def tiny_expert():
    expert = MotionExpert.__new__(MotionExpert)
    expert.cfg = motion_expert_config(global_token=True)
    expert.device = torch.device("cpu")
    expert.dtype = torch.float32
    expert.model = TinyFeatureModel(expert.cfg).eval()
    expert.normalizer = MotionNormalizer(dense_stats())
    expert.flow = FlowMatching()
    expert.weights_path = "/unit-test/dense.pt"
    expert.weights_kind = "model"
    return expert
