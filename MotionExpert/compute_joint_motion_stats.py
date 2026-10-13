"""Compute feature-validity-aware statistics for an EE4D + Nymeria pack.

The resulting file keeps the legacy statistics keys and adds per-feature
counts/source counts.  Invalid Nymeria hand-PCA and beta dimensions are never
included in any aggregate.
"""

from __future__ import annotations

import argparse
import logging
import os
from collections import defaultdict
from pathlib import Path

import torch
from tqdm.auto import tqdm

from .dataset.ee4d_motion_dataset import EE4D_Motion_Dataset

LOGGER = logging.getLogger(__name__)
STATS_SCHEMA_VERSION = 2
STD_FLOOR = 1.0e-3


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
    acc["sumsq"] += safe_values.square().double().sum(dim=0)
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
    motion_dim = int(dataset.motion_feature_dim)
    traj_dim = 18
    motion_acc = _new_accumulator(motion_dim)
    traj_acc = _new_accumulator(traj_dim)
    source_frames = defaultdict(int)
    sequence_qc = {}
    indices = list(indices)
    for index in tqdm(indices, desc="joint masked stats", disable=not progress):
        sample = dataset[index]
        valid_frames = sample["y"]["valid_frames"].bool()
        feature_mask = sample["misc"]["feature_valid_mask"].bool()
        effective_mask = valid_frames[:, None] & feature_mask
        motion = sample["misc"]["motion"]
        traj = sample["misc"]["traj"]
        _update_stats(motion_acc, motion, effective_mask)
        _update_stats(traj_acc, traj, valid_frames[:, None].expand_as(traj))
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
        valid_traj = valid_frames[:, None] & torch.isfinite(traj)
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
    }


def _finalize_accumulators(raw: dict) -> dict:
    motion = _finalize(raw["motion_acc"], context="motion")
    traj = _finalize(raw["traj_acc"], context="trajectory")
    return {
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
        "stats_schema_version": STATS_SCHEMA_VERSION,
    }


def compute_stats(
    dataset: EE4D_Motion_Dataset,
    *,
    limit_windows: int = 0,
    indices=None,
) -> dict:
    windows = min(len(dataset), limit_windows) if limit_windows > 0 else len(dataset)
    if indices is None:
        indices = range(windows)
    raw = _accumulate(dataset, indices)
    return _finalize_accumulators(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--ee-pack", type=Path, required=True)
    parser.add_argument("--nymeria-pack", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--repre-type", default="v4_beta", choices=["v4_beta"])
    parser.add_argument("--window", type=int, default=80)
    parser.add_argument("--limit-windows", type=int, default=0, help="0 means all windows")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--sequence-manifest", type=Path, default=None)
    parser.add_argument(
        "--max-nonfinite-values",
        type=int,
        default=-1,
        help="fail if masked raw values are non-finite; -1 keeps them out of aggregates and reports them",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.window <= 0:
        raise ValueError("window must be positive")
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid num-shards/shard-index")
    dataset = EE4D_Motion_Dataset(
        data_dir=str(args.data_dir),
        split=args.split,
        repre_type=args.repre_type,
        cond_traj=True,
        cond_img_feat=False,
        cond_betas=False,
        window=args.window,
        img_feat_type="dinov2",
        do_normalization=False,
        motion_pack_paths=[str(args.ee_pack), str(args.nymeria_pack)],
        dataset_name="ee4d_nymeria",
        sequence_manifest_path=str(args.sequence_manifest) if args.sequence_manifest else None,
        sanitize_nonfinite=False,
        mmap_motion_packs=True,
    )
    windows = min(len(dataset), args.limit_windows) if args.limit_windows > 0 else len(dataset)
    indices = range(args.shard_index, windows, args.num_shards)
    raw = _accumulate(dataset, indices)
    if args.num_shards > 1:
        raw.update({"kind": "joint_masked_partial", "num_shards": args.num_shards, "shard_index": args.shard_index})
        stats = raw
    else:
        stats = _finalize_accumulators(raw)
    nonfinite_total = int(
        stats["motion_acc"]["nonfinite_count"].sum()
        + stats["traj_acc"]["nonfinite_count"].sum()
        if "motion_acc" in stats
        else stats["motion_nonfinite_count"].sum() + stats["traj_nonfinite_count"].sum()
    )
    if args.max_nonfinite_values >= 0 and nonfinite_total > args.max_nonfinite_values:
        raise ValueError(
            f"masked raw data contains {nonfinite_total} non-finite values, "
            f"exceeding --max-nonfinite-values={args.max_nonfinite_values}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    torch.save(stats, temporary)
    temporary.replace(args.output)
    motion_count = stats["motion_count"] if "motion_count" in stats else stats["motion_acc"]["count"]
    LOGGER.info(
        "saved joint masked stats to %s: windows=%d motion_count=%s source_frames=%s",
        args.output,
        stats["num_windows"],
        tuple(int(x) for x in motion_count[:3]),
        stats["source_valid_frames"],
    )


if __name__ == "__main__":
    main()
