"""Contiguous statistics coverage, exact merge provenance and legacy shards."""
from copy import deepcopy
import sys

import pytest
import torch

from MotionExpert.compute_joint_motion_stats import _accumulate, compute_stats, shard_partition
from MotionExpert.dataset.dense_motion_dataset import DenseMotionDataset
from MotionExpert.merge_joint_motion_stats import _validate_partition, main as merge_main
from MotionExpert.tests.test_dense_dataset import synthetic_arrays, write_archive


@pytest.mark.parametrize("windows", [0, 1, 7, 17, 101])
def test_contiguous_partition_covers_each_index_once_with_empty_shards(windows):
    ranges = [shard_partition(windows, 16, index, "contiguous")[0] for index in range(16)]
    assert [value for selected in ranges for value in selected] == list(range(windows))
    assert all(first.stop == second.start for first, second in zip(ranges, ranges[1:]))


@pytest.mark.parametrize("mode", ["contiguous", "round_robin"])
def test_round_robin_default_and_partition_validation(mode):
    assert list(shard_partition(17, 3, 1)[0]) == [1, 4, 7, 10, 13, 16]
    indices, metadata = shard_partition(17, 3, 1, mode)
    shard = {"shard_index": 1, "num_shards": 3, "num_windows": len(indices), **metadata}
    assert _validate_partition(shard, expected_index=1, num_shards=3, path="synthetic")["total_index_windows"] == 17
    mutations = {"window_index_start": -1, "window_index_stop": 18,
                 "num_windows": len(indices)+1, "shard_index": 0,
                 "num_shards": 4, "shard_mode": "unsupported", "total_index_windows": -1}
    for key, value in mutations.items():
        with pytest.raises(ValueError):
            _validate_partition(dict(shard, **{key: value}), expected_index=1, num_shards=3, path="synthetic")
    incomplete = dict(shard); del incomplete["window_index_stop"]
    with pytest.raises(ValueError, match="Incomplete"):
        _validate_partition(incomplete, expected_index=1, num_shards=3, path="synthetic")


def raw_dataset(tmp_path):
    root = tmp_path/"archive"; root.mkdir()
    manifest, _ = write_archive(root, synthetic_arrays(23))
    return DenseMotionDataset(archive_root=root, split_manifest=manifest, window=4,
        forecast_prefix=2, segment_stride=3, cond_img_feat=False, do_normalization=False,
        rotation_basis_mode="source_native", target_joint_basis="egosuite_native_v1")


def make_partials(dataset, count, mode, *, legacy=False):
    partials = []
    for index in range(count):
        indices, metadata = shard_partition(len(dataset), count, index, mode)
        raw = _accumulate(dataset, indices, progress=False)
        raw.update(kind="joint_masked_partial", num_shards=count, shard_index=index)
        if not legacy:
            raw.update(metadata)
        partials.append(raw)
    return partials


def run_merge(tmp_path, monkeypatch, partials):
    source = tmp_path/"shards"; source.mkdir(exist_ok=True)
    count = len(partials)
    for index, partial in enumerate(partials):
        torch.save(partial, source/f"stats.shard-{index:03d}-of-{count:03d}.pt")
    output = tmp_path/"merged.pt"
    monkeypatch.setattr(sys, "argv", ["merge", "--input-dir", str(source), "--stem", "stats",
                                     "--num-shards", str(count), "--output", str(output)])
    merge_main()
    return torch.load(output, weights_only=True)


@pytest.mark.parametrize("mode,count,legacy", [("contiguous", 16, False),
    ("round_robin", 3, False), ("round_robin", 3, True)])
def test_merged_statistics_match_full_pass_and_legacy_round_robin(tmp_path, monkeypatch, mode, count, legacy):
    dataset = raw_dataset(tmp_path)
    expected = compute_stats(dataset)
    actual = run_merge(tmp_path, monkeypatch, make_partials(dataset, count, mode, legacy=legacy))
    for key in ("motion_mean", "motion_std", "traj_mean", "traj_std", "motion_count", "traj_count"):
        torch.testing.assert_close(actual[key], expected[key], atol=1e-6, rtol=0)
    assert actual["num_windows"] == len(dataset)
    assert actual["shard_mode"] == mode
    if not legacy:
        assert actual["total_index_windows"] == len(dataset)
        assert actual["window_index_start"] == 0
        assert actual["window_index_stop"] == len(dataset)


def test_merge_rejects_gaps_counts_totals_and_provenance_mismatch(tmp_path, monkeypatch):
    dataset = raw_dataset(tmp_path)
    original = make_partials(dataset, 3, "contiguous")
    for key in ("window_index_start", "window_index_stop", "num_windows"):
        changed = deepcopy(original); changed[1][key] += 1
        with pytest.raises(ValueError, match="interval"):
            run_merge(tmp_path, monkeypatch, changed)
    changed = deepcopy(original)
    indices, metadata = shard_partition(len(dataset)+3, 3, 1, "contiguous")
    changed[1].update(metadata, num_windows=len(indices))
    with pytest.raises(ValueError, match="partition mode/total"):
        run_merge(tmp_path, monkeypatch, changed)
    changed = deepcopy(original); changed[1]["archive_fingerprint"] = "different_archive"
    with pytest.raises(ValueError, match="provenance differs"):
        run_merge(tmp_path, monkeypatch, changed)
    changed = deepcopy(original)
    for key in ("shard_mode", "window_index_start", "window_index_stop", "total_index_windows"):
        del changed[1][key]
    with pytest.raises(ValueError, match="partition mode/total"):
        run_merge(tmp_path, monkeypatch, changed)
