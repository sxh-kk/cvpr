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
    shard_partition,
)
from .dataset.normalization import validate_statistics_profile, statistics_motion_supervision
from .dataset.motion_layout import motion_layout, SMPLH_REPRE_TYPE


def _load(path: Path) -> dict:
    value = torch.load(path, weights_only=True)
    if not isinstance(value, dict) or value.get("kind") != "joint_masked_partial":
        raise ValueError(f"not a dense train partial stats file: {path}")
    if type(value.get("stats_schema_version")) is not int or value["stats_schema_version"] not in (3, STATS_SCHEMA_VERSION):
        raise ValueError(f"Dense stats schema mismatch in {path}; legacy153/SMPL shards cannot be reused")
    validate_statistics_profile(value)
    for key,dim in (("motion_acc",motion_layout(value["representation"]).motion_dim),("traj_acc",18)):
        if any(tuple(tensor.shape)!=(dim,) for tensor in value[key].values()):
            raise ValueError(f"Invalid {key} feature shape in {path}")
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


def _validate_partition(shard, *, expected_index, num_shards, path):
    """Validate new coverage metadata while accepting legacy round-robin sets."""
    if type(shard.get("shard_index")) is not int or shard["shard_index"] != expected_index:
        raise ValueError(f"shard index mismatch in {path}: expected {expected_index}")
    if type(shard.get("num_shards")) is not int or shard["num_shards"] != num_shards:
        raise ValueError(f"shard count mismatch in {path}")
    count = shard.get("num_windows")
    if type(count) is not int or count < 0:
        raise ValueError(f"Invalid num_windows in {path}")
    fields = ("window_index_start", "window_index_stop", "total_index_windows")
    if "shard_mode" not in shard and not any(key in shard for key in fields):
        if shard.get("representation") == SMPLH_REPRE_TYPE:
            raise ValueError(f"SMPL-H statistics require explicit complete partition coverage in {path}")
        return {"shard_mode": "round_robin", "total_index_windows": None, "legacy": True}
    if any(type(shard.get(key)) is not int for key in fields):
        raise ValueError(f"Incomplete statistics partition metadata in {path}")
    mode, total = shard.get("shard_mode"), shard["total_index_windows"]
    indices, expected = shard_partition(total, num_shards, expected_index, mode)
    if any(shard.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Statistics shard interval does not match its index in {path}")
    if count != len(indices):
        raise ValueError(f"Statistics shard num_windows does not match its interval in {path}")
    return {"shard_mode": mode, "total_index_windows": total, "legacy": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train",), default="train")
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stem", default=None, help="stats shard stem without .shard-XXX-of-YYY.pt")
    args = parser.parse_args()
    if args.num_shards <= 0:
        raise ValueError("num-shards must be positive")

    stem = args.stem or f"dense245_joint_masked_{args.split}_stats"
    paths = [
        args.input_dir / f"{stem}.shard-{index:03d}-of-{args.num_shards:03d}.pt"
        for index in range(args.num_shards)
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing stats shard(s): " + ", ".join(missing))

    merged_acc = None
    source_frames: defaultdict[str, int] = defaultdict(int)
    num_windows = 0
    layout = None
    provenance = None
    partition_provenance = None
    previous_stop = 0
    metadata_keys = ("format", "representation", "split", "fps", "window", "forecast_prefix", "segment_stride",
                     "split_fingerprint", "archive_fingerprint", "archive_root", "target_joint_basis", "rotation_basis_mode",
                     "basis_mapping_fingerprint", "contact_speed_m_s", "contact_height_m", "normalization_clip",
                     "stats_schema_version", "sampling_step", "train_task", "label_task", "complete_windows_only",
                     "ground_estimation_scope", "image_validity_scope", "min_valid_image_fraction", "statistics_scope",
                     "motion_supervision", "model_sha256", "archive_schema_version", "diagnostic", "joint_indices")
    sequence_qc = {}
    for path in paths:
        shard = _load(path)
        current_metadata = {key:shard.get(key) for key in metadata_keys}
        current_metadata["motion_supervision"] = statistics_motion_supervision(shard)
        if current_metadata["split"] != "train": raise ValueError("Only train normalization shards may be merged")
        if provenance is None: provenance = current_metadata
        elif provenance != current_metadata: raise ValueError(f"Statistics provenance differs across shards: {path}")
        expected_index = int(path.name.split(".shard-")[1].split("-")[0])
        partition = _validate_partition(shard, expected_index=expected_index,
                                        num_shards=args.num_shards, path=path)
        if partition_provenance is None:
            partition_provenance = partition
        elif partition != partition_provenance:
            raise ValueError(f"Statistics shard partition mode/total differs across shards: {path}")
        if partition["shard_mode"] == "contiguous":
            if shard["window_index_start"] != previous_stop:
                raise ValueError(f"Statistics intervals have gaps or overlap: {path}")
            previous_stop = shard["window_index_stop"]
        if layout is None:
            layout = shard.get("feature_layout")
        elif layout != shard.get("feature_layout"):
            raise ValueError(f"inconsistent feature layout in {path}")
        merged_acc = _merge_accumulator(merged_acc, shard)
        num_windows += int(shard["num_windows"])
        for source, count in shard["source_valid_frames"].items():
            source_frames[source] += int(count)
        _merge_sequence_qc(sequence_qc, shard.get("sequence_qc", {}))

    total = partition_provenance["total_index_windows"]
    if total is not None and num_windows != total:
        raise ValueError("Statistics shards do not cover all indexed training windows")
    if partition_provenance["shard_mode"] == "contiguous" and previous_stop != total:
        raise ValueError("Statistics contiguous coverage does not reach the final indexed window")

    archive_root = Path(provenance["archive_root"]).resolve()
    output = args.output.resolve()
    if output == archive_root or archive_root in output.parents:
        raise ValueError("Statistics must be stored outside the immutable archive")
    raw = {
        **provenance,
        **merged_acc,
        "num_windows": num_windows,
        "source_valid_frames": dict(source_frames),
        "feature_layout": layout or "dense12_rot6d_hands21_contacts_v1",
        "sequence_qc": sequence_qc,
        "stats_schema_version": provenance["stats_schema_version"],
    }
    stats = _finalize_accumulators(raw)
    stats["num_shards"] = args.num_shards
    stats["shard_mode"] = partition_provenance["shard_mode"]
    if total is not None:
        stats["window_index_start"] = 0
        stats["window_index_stop"] = total
        stats["total_index_windows"] = total
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
