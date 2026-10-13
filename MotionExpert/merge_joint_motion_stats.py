"""Merge raw sharded accumulators produced by compute_joint_motion_stats.py."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch

from .compute_joint_motion_stats import (
    STATS_SCHEMA_VERSION,
    _finalize_accumulators,
    _validate_accumulator,
)


def _load(path: Path, *, allow_legacy=False) -> dict:
    value = torch.load(path, weights_only=False)
    if not isinstance(value, dict) or value.get("kind") != "joint_masked_partial":
        raise ValueError(f"not a joint-masked partial stats file: {path}")
    if int(value.get("stats_schema_version", -1)) != STATS_SCHEMA_VERSION and not allow_legacy:
        raise ValueError(
            f"stats schema mismatch in {path}: expected {STATS_SCHEMA_VERSION}, "
            f"got {value.get('stats_schema_version')}"
        )
    if int(value.get("stats_schema_version", -1)) != STATS_SCHEMA_VERSION:
        # Legacy shards predate finite-value accounting.  They are accepted
        # only through the explicit recovery flag, never by the normal
        # pipeline, and are treated as having observed zero non-finite values.
        for key in ("motion_acc", "traj_acc"):
            value[key].setdefault("nonfinite_count", torch.zeros_like(value[key]["count"]))
        value.setdefault("sequence_qc", {})
        value["stats_schema_version"] = STATS_SCHEMA_VERSION
    for key in ("motion_acc", "traj_acc"):
        _validate_accumulator(value[key], context=str(path))
    return value


def _merge_accumulator(target: dict | None, source: dict) -> dict:
    if target is None:
        return {
            key: {name: value.clone() for name, value in acc.items()}
            for key, acc in (("motion_acc", source["motion_acc"]), ("traj_acc", source["traj_acc"]))
        }
    for key in ("motion_acc", "traj_acc"):
        for name in ("sum", "sumsq", "count", "nonfinite_count"):
            target[key][name] += source[key][name]
        target[key]["min"] = torch.minimum(target[key]["min"], source[key]["min"])
        target[key]["max"] = torch.maximum(target[key]["max"], source[key]["max"])
    return target


def _merge_sequence_qc(target: dict, source: dict) -> None:
    for name, value in source.items():
        current = target.get(name)
        if current is None:
            target[name] = dict(value)
            continue
        if current["source"] != value["source"]:
            raise ValueError(f"sequence source changed across shards for {name}")
        for key in ("windows", "valid_frames", "nonfinite_motion_values", "nonfinite_traj_frames"):
            current[key] += int(value[key])
        for key in ("max_abs_motion", "max_abs_traj"):
            current[key] = max(float(current[key]), float(value[key]))
        current["discard_terrain"] = bool(current["discard_terrain"] or value["discard_terrain"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stem", default=None, help="stats shard stem without .shard-XXX-of-YYY.pt")
    parser.add_argument(
        "--allow-legacy-shards",
        action="store_true",
        help="explicit recovery mode for pre-schema-2 finite shards",
    )
    parser.add_argument(
        "--exclude-shard-index",
        type=int,
        action="append",
        default=[],
        help="omit a known-bad shard during explicit recovery",
    )
    args = parser.parse_args()
    if args.num_shards <= 0:
        raise ValueError("num-shards must be positive")

    stem = args.stem or f"v4_beta_joint_masked_{args.split}_stats"
    excluded = set(args.exclude_shard_index)
    if excluded and not args.allow_legacy_shards:
        raise ValueError("--exclude-shard-index requires --allow-legacy-shards recovery mode")
    if any(index < 0 or index >= args.num_shards for index in excluded):
        raise ValueError(f"invalid excluded shard index: {sorted(excluded)}")
    paths = [
        args.input_dir / f"{stem}.shard-{index:03d}-of-{args.num_shards:03d}.pt"
        for index in range(args.num_shards)
        if index not in excluded
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing stats shard(s): " + ", ".join(missing))

    merged_acc = None
    source_frames: defaultdict[str, int] = defaultdict(int)
    num_windows = 0
    layout = None
    sequence_qc = {}
    for path in paths:
        shard = _load(path, allow_legacy=args.allow_legacy_shards)
        expected_index = int(path.name.split(".shard-")[1].split("-")[0])
        if int(shard.get("shard_index", -1)) != expected_index:
            raise ValueError(
                f"shard index mismatch in {path}: expected {expected_index}, "
                f"got {shard.get('shard_index')}"
            )
        if int(shard.get("num_shards", -1)) != args.num_shards:
            raise ValueError(f"shard count mismatch in {path}")
        if layout is None:
            layout = shard.get("feature_layout")
        elif layout != shard.get("feature_layout"):
            raise ValueError(f"inconsistent feature layout in {path}")
        merged_acc = _merge_accumulator(merged_acc, shard)
        num_windows += int(shard["num_windows"])
        for source, count in shard["source_valid_frames"].items():
            source_frames[source] += int(count)
        _merge_sequence_qc(sequence_qc, shard.get("sequence_qc", {}))

    raw = {
        **merged_acc,
        "num_windows": num_windows,
        "source_valid_frames": dict(source_frames),
        "feature_layout": layout or "v4_beta",
        "sequence_qc": sequence_qc,
        "stats_schema_version": STATS_SCHEMA_VERSION,
    }
    stats = _finalize_accumulators(raw)
    stats["num_shards"] = args.num_shards
    stats["excluded_shards"] = sorted(excluded)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    torch.save(stats, temporary)
    temporary.replace(args.output)
    print(
        f"merged {args.num_shards} stats shards into {args.output}; "
        f"windows={num_windows}, source_valid_frames={dict(source_frames)}"
    )


if __name__ == "__main__":
    main()
