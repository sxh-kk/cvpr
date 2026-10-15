"""Compute full-train masked statistics from accepted packed raw window mmap.

The window population, overlapping-window frame weighting, metadata and
finalization match compute_joint_motion_stats. No source FK, Parquet window
decoding or DINO embedding reads are performed after cache activation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import logging
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch
from tqdm.auto import tqdm

from .compute_joint_motion_stats import _accumulate, _finalize_accumulators, _update_stats
from .config import load_config
from .dataset.motion_layout import SMPLH_REPRE_TYPE, motion_layout
from .dataset.smplh_motion_dataset import SMPLHMotionDataset
from .dataset.window_cache import CONTRACT, file_sha256

LOGGER = logging.getLogger(__name__)
ACCEPTANCE_FORMAT = "motionexpert-window-cache-acceptance-v1"
RAW_ACCEPTANCE_FORMAT = "motionexpert-raw-window-cache-acceptance-v1"


def _require_finite_block(values, valid, finite, *, field, episode, first_window):
    """Reject cache corruption, including unavailable channels, with bounded context."""
    bad = ~finite
    if not bad.any():
        return
    locations = []
    for flat_frame, channel in torch.nonzero(bad, as_tuple=False)[:8].tolist():
        locations.append({"window_row": first_window + flat_frame // 11,
                          "frame_in_window": flat_frame % 11, "channel": channel,
                          "supervision_valid": bool(valid[flat_frame, channel]),
                          "value": str(float(values[flat_frame, channel]))})
    raise ValueError(f"Nonfinite packed {field} at episode={episode}; values={int(bad.sum())}; "
                     f"first_locations={locations}")


def _acceptance_binding(cache, acceptance_path):
    path = Path(acceptance_path).resolve(strict=True)
    report = json.loads(path.read_text())
    report_format = report.get("format")
    if (report_format not in {ACCEPTANCE_FORMAT, RAW_ACCEPTANCE_FORMAT} or report.get("state") != "complete"
            or report.get("passed") is not True or report.get("diagnostic") is not False
            or report.get("error") or not report.get("finished_at")):
        raise ValueError("Cached train statistics require completed non-diagnostic formal cache acceptance or explicit raw-phase acceptance")
    if (report_format == RAW_ACCEPTANCE_FORMAT
            and (report.get("scope") != "all" or report.get("checks_passed") is not True)):
        raise ValueError("Raw-phase acceptance must declare full scope and passing checks")
    if Path(report.get("cache_root", "")).resolve() != cache.root:
        raise ValueError("Acceptance names a different window cache")
    expected = file_sha256(cache.manifest_path)
    if report.get("cache_manifest_sha256") != expected:
        raise ValueError("Acceptance window-cache manifest SHA256 differs")
    return {"acceptance_path": str(path), "acceptance_sha256": file_sha256(path),
            "cache_root": str(cache.root), "cache_manifest_sha256": expected}


def compute_cached_stats(dataset, *, acceptance_path, block_windows=1024, progress=True):
    """Scan every accepted train window exactly once; bound RAM by block size."""
    if type(block_windows) is not int or block_windows < 1:
        raise ValueError("block_windows must be a positive integer")
    cache = dataset.window_cache
    if (dataset.split != "train" or dataset.do_normalization or dataset.cond_img_feat
            or dataset.repre_type != SMPLH_REPRE_TYPE or dataset.diagnostic
            or dataset.train_task != "fore" or dataset.motion_supervision != "joint"
            or dataset.require_full_window is not True or dataset.window != 11
            or dataset.forecast_prefix != 6):
        raise ValueError("Cached statistics require raw full-train joint eleven-frame data without DINO embeddings")
    if (cache is None or cache.split != "train" or cache._packed_spec is None
            or cache.manifest.get("contract") != CONTRACT or cache.manifest.get("diagnostic") is not False
            or cache.manifest.get("scope") != "all" or cache.manifest.get("state") != "complete"):
        raise ValueError("Cached statistics require committed full packed train windows")
    binding = _acceptance_binding(cache, acceptance_path)
    cache.assert_current()
    cache._open_packed()
    packed = cache._packed
    # Empty accumulation captures the SAME current metadata/identity keys. It
    # intentionally never calls get_sample, which would decode the raw archive.
    raw = _accumulate(dataset, [], progress=False)
    layout = motion_layout(dataset.repre_type)
    source_frames, sequence_qc = defaultdict(int), {}
    scanned = 0
    bar = tqdm(total=len(dataset), desc="packed train masked stats", disable=not progress)
    try:
        for record in cache.records:
            descriptor = cache._descriptors[record["episode_index"]]
            count, first = descriptor["windows"], descriptor["row_offset"]
            if not count:
                continue
            qc = {"source": record["source"], "windows": count, "valid_frames": count * 11,
                  "nonfinite_motion_values": 0, "nonfinite_traj_frames": 0,
                  "max_abs_motion": 0., "max_abs_traj": 0., "discard_terrain": False}
            for start in range(first, first + count, block_windows):
                end = min(start + block_windows, first + count)
                # Copy bounded blocks before torch conversion: the source mmap
                # stays read-only, and torch never sees a writable view of it.
                motion = torch.from_numpy(np.array(packed["motion"][start:end], copy=True)).reshape(-1, 245)
                motion_valid = torch.from_numpy(np.array(packed["motion_valid"][start:end], copy=True)).reshape(-1, 245)
                trajectory = torch.from_numpy(np.array(packed["trajectory"][start:end], copy=True)).reshape(-1, 18)
                trajectory_valid = torch.from_numpy(np.array(packed["trajectory_valid"][start:end], copy=True)).reshape(-1, 18)
                finite_motion = torch.isfinite(motion)
                finite_trajectory = torch.isfinite(trajectory)
                qc["nonfinite_motion_values"] += int((~finite_motion).sum())
                qc["nonfinite_traj_frames"] += int((~finite_trajectory).any(-1).sum())
                _require_finite_block(motion, motion_valid, finite_motion, field="motion",
                                      episode=record["episode_index"], first_window=start)
                _require_finite_block(trajectory, trajectory_valid, finite_trajectory, field="trajectory",
                                      episode=record["episode_index"], first_window=start)
                if not motion_valid.reshape(-1, 11, 245)[:, 6:, layout.body_slice].any(dim=(1, 2)).all():
                    raise ValueError(f"Packed train window has no valid future body supervision at episode {record['episode_index']}, rows {start}:{end}")
                if (not torch.equal(motion_valid[:, layout.global_slice], trajectory_valid[:, 9:18])
                        or not torch.equal(motion[:, layout.global_slice], trajectory[:, 9:18])):
                    raise ValueError(f"Motion/trajectory floor delta labels or masks differ at episode {record['episode_index']}, rows {start}:{end}")
                # Joint forecasting supervises all eleven valid complete frames,
                # including repeated source frames in overlapping windows.
                _update_stats(raw["motion_acc"], motion, motion_valid)
                _update_stats(raw["traj_acc"], trajectory, trajectory_valid)
                valid_motion = motion_valid & finite_motion
                valid_trajectory = trajectory_valid & finite_trajectory
                if valid_motion.any():
                    qc["max_abs_motion"] = max(qc["max_abs_motion"], float(motion[valid_motion].abs().max()))
                if valid_trajectory.any():
                    qc["max_abs_traj"] = max(qc["max_abs_traj"], float(trajectory[valid_trajectory].abs().max()))
                scanned += end - start
                bar.update(end - start)
            source_frames[record["source"]] += count * 11
            sequence_qc[str(record["source_key"])] = qc
    finally:
        bar.close()
    if scanned != len(dataset):
        raise ValueError("Packed statistics did not cover the full train window population")
    cache.assert_current()
    if _acceptance_binding(cache, acceptance_path) != binding:
        raise ValueError("Bound cache acceptance changed during statistics computation")
    raw.update(num_windows=scanned, source_valid_frames=dict(source_frames), sequence_qc=sequence_qc)
    return _finalize_accumulators(raw)


def validate_output(output, *, archive_root, cache_root):
    """Reject an existing destination or one owned by immutable input data."""
    supplied = Path(output)
    if supplied.exists() or supplied.is_symlink():
        raise FileExistsError(f"Statistics output already exists; refusing to replace it: {supplied}")
    output = supplied.resolve()
    for root in (Path(archive_root).resolve(), Path(cache_root).resolve()):
        if output == root or output.is_relative_to(root):
            raise ValueError("Statistics output must be outside immutable archive and cache roots")
    return output


def publish_new_stats(stats, output, *, archive_root, cache_root):
    """Publish a complete new file atomically; never replace an old statistic."""
    output = validate_output(output, archive_root=archive_root, cache_root=cache_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(stats, stream)
            stream.flush()
            os.fsync(stream.fileno())
        # Same-directory hard link publishes complete bytes with O_EXCL-like
        # semantics. Another producer's existing destination is never replaced.
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-windows", type=int, default=1024)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    if args.block_windows < 1 or args.torch_threads < 1:
        parser.error("--block-windows and --torch-threads must be positive")
    validate_output(args.output, archive_root=args.archive_root, cache_root=args.cache_root)
    logging.basicConfig(level=logging.INFO)
    torch.set_num_threads(args.torch_threads)
    cfg = load_config(args.config)
    data = cfg.DATA
    dataset = SMPLHMotionDataset(
        archive_root=args.archive_root, split="train", repre_type=data.REPRE_TYPE,
        split_manifest=args.archive_root / "meta/splits.json", do_normalization=False,
        cond_img_feat=False, window=data.WINDOW, fps=data.FPS, forecast_prefix=data.FORECAST_PREFIX,
        segment_stride=data.SEGMENT_STRIDE, rotation_basis_mode=data.ROTATION_BASIS_MODE,
        target_joint_basis=data.TARGET_JOINT_BASIS, rotation_basis_path=getattr(data, "ROTATION_BASIS_PATH", None),
        require_verified_rotations=getattr(data, "REQUIRE_VERIFIED_ROTATIONS", True),
        contact_speed_m_s=data.CONTACT_SPEED_M_S, contact_height_m=data.CONTACT_HEIGHT_M,
        row_group_cache_mb=data.ROW_GROUP_CACHE_MB, parquet_file_cache=data.PARQUET_FILE_CACHE,
        vision_profile_root=data.VISION_PROFILE_ROOT, image_profile=data.IMAGE_PROFILE,
        min_valid_image_fraction=data.MIN_VALID_IMAGE_FRACTION, train_task=data.TRAIN_TASK,
        require_full_window=data.REQUIRE_FULL_WINDOW, motion_supervision=data.MOTION_SUPERVISION,
        window_cache_root=args.cache_root)
    started = time.monotonic()
    stats = compute_cached_stats(dataset, acceptance_path=args.acceptance, block_windows=args.block_windows)
    output = publish_new_stats(stats, args.output, archive_root=args.archive_root, cache_root=args.cache_root)
    LOGGER.info("Saved full train packed statistics to %s: windows=%d, source_frames=%s, scan_seconds=%.3f",
                output, stats["num_windows"], stats["source_valid_frames"], time.monotonic() - started)


if __name__ == "__main__":
    main()
