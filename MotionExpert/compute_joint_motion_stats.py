"""Compute train-only masked motion and trajectory18 statistics without DINO embeddings."""

from __future__ import annotations

import argparse
import logging
import os
from collections import defaultdict
from pathlib import Path

import torch
from tqdm.auto import tqdm

from .dataset.ee4d_motion_dataset import EE4D_Motion_Dataset
from .dataset.dense_representation import CONTACT_SLICE, GLOBAL_SLICE, REPRE_TYPE
from .dataset.motion_layout import motion_layout, SMPLH_REPRE_TYPE
from .dataset.normalization import STATS_FORMAT
from .config import load_config

LOGGER = logging.getLogger(__name__)
STATS_SCHEMA_VERSION = 4
STD_FLOOR = 1.0e-3


def shard_partition(windows, num_shards, shard_index, mode="round_robin"):
    """Select a reproducible partition of the indexed training windows."""
    if (type(windows) is not int or windows < 0 or type(num_shards) is not int
            or num_shards < 1 or type(shard_index) is not int
            or not 0 <= shard_index < num_shards):
        raise ValueError("Invalid statistics window count or shard partition")
    if mode == "contiguous":
        start = windows * shard_index // num_shards
        stop = windows * (shard_index + 1) // num_shards
        indices = range(start, stop)
    elif mode == "round_robin":
        start, stop = shard_index, windows
        indices = range(start, stop, num_shards)
    else:
        raise ValueError("Unsupported shard-mode; use round_robin or contiguous")
    return indices, {"shard_mode": mode, "window_index_start": start,
                     "window_index_stop": stop, "total_index_windows": windows}


def _new_accumulator(dim: int) -> dict[str, torch.Tensor]:
    return {
        "sum": torch.zeros(dim, dtype=torch.float64),
        "sumsq": torch.zeros(dim, dtype=torch.float64),
        "count": torch.zeros(dim, dtype=torch.long),
        "nonfinite_count": torch.zeros(dim, dtype=torch.long),
        "min": torch.full((dim,), float("inf"), dtype=torch.float64),
        "max": torch.full((dim,), float("-inf"), dtype=torch.float64),
    }


def _update_stats(acc, values: torch.Tensor, mask: torch.Tensor) -> None:
    """Update float64 sufficient statistics for a ``T x F`` tensor."""
    values = values.detach().float()
    mask = mask.bool()
    if values.shape != mask.shape:
        raise ValueError(f"stats values/mask shape mismatch: {tuple(values.shape)} vs {tuple(mask.shape)}")
    finite = torch.isfinite(values)
    valid = mask & finite
    acc["nonfinite_count"] += (mask & ~finite).long().sum(dim=0)
    safe_values = torch.where(valid, values, torch.zeros_like(values))
    acc["sum"] += safe_values.double().sum(dim=0)
    acc["sumsq"] += safe_values.double().square().sum(dim=0)
    acc["count"] += valid.long().sum(dim=0)
    if valid.any():
        acc["min"] = torch.minimum(
            acc["min"],
            torch.where(valid, values, torch.full_like(values, float("inf"))).min(dim=0).values.double(),
        )
        acc["max"] = torch.maximum(
            acc["max"],
            torch.where(valid, values, torch.full_like(values, float("-inf"))).max(dim=0).values.double(),
        )


def _validate_accumulator(acc, context="") -> None:
    prefix = f"{context}: " if context else ""
    for key in ("sum", "sumsq"):
        bad = ~torch.isfinite(acc[key])
        if bad.any():
            indices = torch.where(bad)[0].tolist()
            raise ValueError(f"{prefix}non-finite accumulator {key} at dimensions {indices[:12]}")
    if (acc["count"] < 0).any() or (acc["nonfinite_count"] < 0).any():
        raise ValueError(f"{prefix}negative statistics count")
    observed = acc["count"] > 0
    for key in ("min", "max"):
        bad = observed & ~torch.isfinite(acc[key])
        if bad.any():
            indices = torch.where(bad)[0].tolist()
            raise ValueError(f"{prefix}non-finite observed {key} at dimensions {indices[:12]}")
    if (observed & (acc["min"] > acc["max"])).any():
        raise ValueError(f"{prefix}statistics min exceeds max")


def _finalize(acc, context=""):
    _validate_accumulator(acc, context=context)
    count = acc["count"]
    denominator = count.clamp_min(1).double()
    mean = acc["sum"] / denominator
    variance = (acc["sumsq"] / denominator - mean.square()).clamp_min(0.0)
    std = variance.sqrt()
    no_observation = count == 0
    constant = (~no_observation) & (std < STD_FLOOR)
    mean[no_observation] = 0.0
    std[no_observation] = 1.0
    # A nearly constant coordinate is numerical noise, not a useful scale.
    # Keeping a unit scale prevents a tiny EE4D-only variance from turning a
    # small Nymeria calibration residual into a very large training target.
    std[constant] = 1.0
    minimum = acc["min"].clone()
    maximum = acc["max"].clone()
    minimum[no_observation] = 0.0
    maximum[no_observation] = 0.0
    return {
        "mean": mean.float(),
        "std": std.float(),
        "min": minimum.float(),
        "max": maximum.float(),
        "count": count,
        "nonfinite_count": acc["nonfinite_count"],
        "constant": constant,
    }


def _accumulate(dataset: EE4D_Motion_Dataset, indices, *, progress=True) -> dict:
    if dataset.split != "train" or dataset.do_normalization:
        raise ValueError("Raw accumulators require unnormalized train data only")
    motion_dim = int(dataset.motion_feature_dim)
    layout = motion_layout(dataset.repre_type)
    traj_dim = 18
    motion_acc = _new_accumulator(motion_dim)
    traj_acc = _new_accumulator(traj_dim)
    source_frames = defaultdict(int)
    sequence_qc = {}
    indices = list(indices)
    for index in tqdm(indices, desc="joint masked stats", disable=not progress):
        # Match forecast labels and their prefix-only floor even when an
        # auxiliary reconstruction task is also used by the training loop.
        sample = dataset.get_sample(index, task=dataset.train_task)
        valid_frames = sample["y"]["valid_frames"].bool()
        if dataset.train_task == "fore":
            valid_frames = valid_frames & sample["y"]["loss_mask"].bool()
        feature_mask = sample["misc"]["feature_valid_mask"].bool()
        effective_mask = valid_frames[:, None] & feature_mask
        motion = sample["misc"]["motion"]
        traj = sample["misc"]["traj"]
        if not torch.equal(feature_mask[:,layout.global_slice],sample["misc"]["trajectory_valid"][:,9:18].bool()) or not torch.equal(motion[:,layout.global_slice],traj[:,9:18]):
            raise ValueError("Motion/condition floor delta labels and validity must match exactly")
        _update_stats(motion_acc, motion, effective_mask)
        traj_mask = valid_frames[:, None] & sample["misc"]["trajectory_valid"].bool()
        _update_stats(traj_acc, traj, traj_mask)
        source = sample["misc"]["source"]
        source_frames[source] += int(valid_frames.sum())
        seq_name = str(sample["misc"]["seq_name"])
        qc = sequence_qc.setdefault(
            seq_name,
            {
                "source": source,
                "windows": 0,
                "valid_frames": 0,
                "nonfinite_motion_values": 0,
                "nonfinite_traj_frames": 0,
                "max_abs_motion": 0.0,
                "max_abs_traj": 0.0,
                "discard_terrain": bool(sample["misc"].get("discard_terrain", False)),
            },
        )
        qc["windows"] += 1
        qc["valid_frames"] += int(valid_frames.sum())
        motion_nonfinite = sample["misc"].get("motion_nonfinite_mask")
        if motion_nonfinite is not None:
            qc["nonfinite_motion_values"] += int(
                (motion_nonfinite & valid_frames[:, None]).sum()
            )
        traj_nonfinite = sample["misc"].get("traj_nonfinite_mask")
        if traj_nonfinite is not None:
            qc["nonfinite_traj_frames"] += int((traj_nonfinite & valid_frames).sum())
        valid_motion = effective_mask & torch.isfinite(motion)
        if valid_motion.any():
            qc["max_abs_motion"] = max(
                float(qc["max_abs_motion"]), float(motion[valid_motion].abs().max())
            )
        valid_traj = traj_mask & torch.isfinite(traj)
        if valid_traj.any():
            qc["max_abs_traj"] = max(
                float(qc["max_abs_traj"]), float(traj[valid_traj].abs().max())
            )
    return {
        "motion_acc": motion_acc,
        "traj_acc": traj_acc,
        "source_valid_frames": dict(source_frames),
        "num_windows": len(indices),
        "feature_layout": dataset.repre_type,
        "sequence_qc": sequence_qc,
        "stats_schema_version": STATS_SCHEMA_VERSION,
        "format": STATS_FORMAT, "representation": dataset.repre_type, "split": dataset.split,
        **({"model_sha256": dataset.model_sha256,
            "archive_schema_version": dataset.archive_schema_version,
            "joint_indices": list(layout.joint_indices),
            "diagnostic": bool(getattr(dataset,"diagnostic",False))} if dataset.repre_type == SMPLH_REPRE_TYPE else {}),
        "fps": dataset.fps, "window": dataset.window, "forecast_prefix": dataset.forecast_prefix,
        "sampling_step": dataset.sampling_step, "train_task": dataset.train_task,
        "motion_supervision": dataset.motion_supervision,
        "label_task": dataset.train_task, "complete_windows_only": dataset.require_full_window,
        "ground_estimation_scope": dataset.ground_estimation_scope,
        "image_validity_scope": dataset.image_validity_scope,
        "min_valid_image_fraction": dataset.min_valid_image_fraction,
        "statistics_scope": dataset.statistics_scope,
        "segment_stride": dataset.segment_stride, "split_fingerprint": dataset.split_fingerprint,
        "archive_fingerprint": dataset.manifest["archive_fingerprint"], "archive_root": str(dataset.root.resolve()),
        "target_joint_basis": dataset.target_joint_basis, "rotation_basis_mode": dataset.rotation_basis_mode,
        "basis_mapping_fingerprint": dataset.basis_mapping_fingerprint,
        "contact_speed_m_s": dataset.contact_speed_m_s, "contact_height_m": dataset.contact_height_m,
        "normalization_clip": None,
    }


def _finalize_accumulators(raw: dict) -> dict:
    layout = motion_layout(raw["representation"])
    motion = _finalize(raw["motion_acc"], context="motion")
    traj = _finalize(raw["traj_acc"], context="trajectory")
    # Floor deltas are the SAME raw features in motion and trajectory.
    for key in ("mean", "std", "min", "max", "count", "nonfinite_count", "constant"):
        motion[key][layout.global_slice] = traj[key][9:18]
    motion["mean"][layout.contact_slice] = 0
    motion["std"][layout.contact_slice] = 1
    metadata_keys = ("format", "representation", "split", "fps", "window", "forecast_prefix", "segment_stride",
                     "split_fingerprint", "archive_fingerprint", "archive_root", "target_joint_basis", "rotation_basis_mode",
                     "basis_mapping_fingerprint", "contact_speed_m_s", "contact_height_m", "normalization_clip")
    profile_keys = ("sampling_step", "train_task", "label_task", "complete_windows_only", "ground_estimation_scope",
                    "image_validity_scope", "min_valid_image_fraction", "statistics_scope", "motion_supervision")
    return {
        **{key:raw[key] for key in metadata_keys},
        **{key:raw[key] for key in (*profile_keys,"model_sha256","archive_schema_version","diagnostic","joint_indices") if key in raw and raw[key] is not None},
        "motion_mean": motion["mean"],
        "motion_std": motion["std"],
        "motion_min": motion["min"],
        "motion_max": motion["max"],
        "traj_mean": traj["mean"],
        "traj_std": traj["std"],
        "traj_min": traj["min"],
        "traj_max": traj["max"],
        "motion_count": motion["count"],
        "traj_count": traj["count"],
        "motion_nonfinite_count": motion["nonfinite_count"],
        "traj_nonfinite_count": traj["nonfinite_count"],
        "motion_constant_dimensions": torch.where(motion["constant"])[0],
        "traj_constant_dimensions": torch.where(traj["constant"])[0],
        "num_windows": raw["num_windows"],
        "source_valid_frames": dict(raw["source_valid_frames"]),
        "sequence_qc": raw.get("sequence_qc", {}),
        "stats_mode": "joint_masked",
        "feature_layout": raw["feature_layout"],
        "stats_schema_version": raw.get("stats_schema_version", STATS_SCHEMA_VERSION),
    }


def compute_stats(
    dataset: EE4D_Motion_Dataset,
    *,
    limit_windows: int = 0,
    indices=None,
) -> dict:
    if dataset.split != "train": raise ValueError("Normalization statistics must be computed from train only")
    if dataset.do_normalization: raise ValueError("Statistics require do_normalization=False")
    windows = min(len(dataset), limit_windows) if limit_windows > 0 else len(dataset)
    if indices is None:
        indices = range(windows)
    indices = list(indices)
    raw = _accumulate(dataset, indices)
    if dataset.repre_type == SMPLH_REPRE_TYPE and sorted(indices) != list(range(len(dataset))):
        raw["diagnostic"] = True
    return _finalize_accumulators(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--archive-root", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-windows", type=int, default=0, help="0 means all train windows")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-mode", choices=("round_robin", "contiguous"), default="round_robin",
                        help="contiguous reduces repeated Parquet row-group reads across workers")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid num-shards/shard-index")
    cfg = load_config(args.config)
    data = cfg.DATA
    root = args.archive_root or getattr(data,"ARCHIVE_ROOT",None) or data.DATA_DIR
    output = args.output.resolve()
    archive_path = Path(root).resolve()
    if output == archive_path or archive_path in output.parents:
        raise ValueError("Statistics must be saved outside the immutable archive")
    dataset = EE4D_Motion_Dataset(archive_root=root, split="train",
        repre_type=data.REPRE_TYPE,
        split_manifest=args.split_manifest or data.SPLIT_MANIFEST,
        window=data.WINDOW, fps=data.FPS, forecast_prefix=data.FORECAST_PREFIX,
        segment_stride=data.SEGMENT_STRIDE, cond_img_feat=False, do_normalization=False,
        rotation_basis_mode=data.ROTATION_BASIS_MODE, target_joint_basis=data.TARGET_JOINT_BASIS,
        rotation_basis_path=getattr(data,"ROTATION_BASIS_PATH",None),
        require_verified_rotations=getattr(data,"REQUIRE_VERIFIED_ROTATIONS",True),
        contact_speed_m_s=getattr(data,"CONTACT_SPEED_M_S",.2), contact_height_m=getattr(data,"CONTACT_HEIGHT_M",.1),
        row_group_cache_mb=getattr(data,"ROW_GROUP_CACHE_MB",128),
        parquet_file_cache=getattr(data,"PARQUET_FILE_CACHE",8),
        min_valid_image_fraction=getattr(data,"MIN_VALID_IMAGE_FRACTION",0.),
        train_task=getattr(data,"TRAIN_TASK",None) or "recon",require_full_window=getattr(data,"REQUIRE_FULL_WINDOW",False),
        motion_supervision=getattr(data,"MOTION_SUPERVISION",None))
    windows = min(len(dataset), args.limit_windows) if args.limit_windows > 0 else len(dataset)
    indices, partition = shard_partition(windows, args.num_shards, args.shard_index, args.shard_mode)
    raw = _accumulate(dataset, indices)
    if dataset.repre_type == SMPLH_REPRE_TYPE and args.limit_windows > 0 and windows < len(dataset):
        raw["diagnostic"] = True
    if args.num_shards > 1:
        raw.update({"kind":"joint_masked_partial", "num_shards":args.num_shards,"shard_index":args.shard_index,
                    **partition})
        stats = raw
    else: stats = _finalize_accumulators(raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    torch.save(stats, temporary); temporary.replace(args.output)
    LOGGER.info("saved dense train statistics to %s: windows=%d, source_frames=%s",
                args.output, stats["num_windows"], stats["source_valid_frames"])


if __name__ == "__main__": main()
