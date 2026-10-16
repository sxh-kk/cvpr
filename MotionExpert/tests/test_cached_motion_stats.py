"""Small real-cache comparison against the original per-window statistics."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from MotionExpert.compute_cached_motion_stats import (
    ACCEPTANCE_FORMAT, RAW_ACCEPTANCE_FORMAT, compute_cached_stats, publish_new_stats, validate_output,
)
from MotionExpert.compute_joint_motion_stats import compute_stats
from MotionExpert import config as config_module
from MotionExpert.config import smplh_motion_expert_config
from MotionExpert.dataset.smplh_archive import SMPLHParameterArchive
from MotionExpert.dataset.smplh_motion_dataset import SMPLHMotionDataset
from MotionExpert.dataset.window_cache import file_sha256
from MotionExpert.readiness import (CACHED_STATISTICS_PROVENANCE_FORMAT,
    validate_cached_statistics_provenance, window_cache_acceptance_binding)
from MotionExpert.tests.fixtures import owned_archive, smplh_stats
from MotionExpert.tests.test_vision_profile_cache import write_cache


@pytest.fixture
def cached_archive(owned_archive, tmp_path, monkeypatch):
    torch.set_num_threads(1)
    vision = write_cache(owned_archive, tmp_path / "vision")
    archive = SMPLHParameterArchive(owned_archive, vision_profile_root=vision)
    monkeypatch.setattr(config_module, "SMPLH_MODEL_SHA256", archive.model_sha256)
    cfg = smplh_motion_expert_config()
    cfg.DATA.ARCHIVE_ROOT = cfg.DATA.DATA_DIR = str(owned_archive)
    cfg.DATA.SPLIT_MANIFEST = str(owned_archive / "meta/splits.json")
    cfg.DATA.VISION_PROFILE_ROOT = str(vision)
    cfg.DATA.MODEL_SHA256 = archive.model_sha256
    cfg.DATA.STATS_PATH = str(tmp_path / "stats.pt")
    stats = smplh_stats(cfg)
    stats.update(archive_fingerprint=archive.archive_fingerprint, split_fingerprint=archive.split_fingerprint,
                 basis_mapping_fingerprint="none")
    torch.save(stats, cfg.DATA.STATS_PATH)
    from MotionExpert.build_window_cache import build_window_cache
    cfg.DATA.WINDOW_CACHE_ROOT = str(tmp_path / "windows")
    build_window_cache(cfg, cfg.DATA.WINDOW_CACHE_ROOT, workers=0)
    return cfg


def _dataset(cfg, *, cached, normalized, split="train", verified_archive=None):
    return SMPLHMotionDataset(archive_root=cfg.DATA.ARCHIVE_ROOT, split=split,
        stats_path=cfg.DATA.STATS_PATH, do_normalization=normalized, cond_img_feat=False,
        vision_profile_root=cfg.DATA.VISION_PROFILE_ROOT, image_profile=cfg.DATA.IMAGE_PROFILE,
        min_valid_image_fraction=cfg.DATA.MIN_VALID_IMAGE_FRACTION,
        window_cache_root=cfg.DATA.WINDOW_CACHE_ROOT if cached else None, verified_archive=verified_archive)


def accepted(cfg, tmp_path):
    path = tmp_path / "acceptance.json"
    path.write_text(json.dumps({"format": ACCEPTANCE_FORMAT, "state": "complete", "passed": True,
        "diagnostic": False, "finished_at": "2026-10-15T00:00:00+00:00",
        "cache_root": cfg.DATA.WINDOW_CACHE_ROOT,
        "cache_manifest_sha256": file_sha256(Path(cfg.DATA.WINDOW_CACHE_ROOT) / "manifest.json")}))
    return path


def assert_same_stats(left, right):
    assert left.keys() == right.keys()
    for key in left:
        if isinstance(left[key], torch.Tensor):
            torch.testing.assert_close(left[key], right[key], atol=2e-6, rtol=2e-6)
        else:
            assert left[key] == right[key], key


def test_packed_stats_match_original_full_train_metadata_counts_and_values(cached_archive, tmp_path, monkeypatch):
    original = _dataset(cached_archive, cached=False, normalized=False)
    expected = compute_stats(original)
    cached = _dataset(cached_archive, cached=True, normalized=False)
    report = accepted(cached_archive, tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError("Packed statistics attempted per-window decoding/embedding reads")
    monkeypatch.setattr(cached, "get_sample", forbidden)
    monkeypatch.setattr(cached.archive.reader, "read", forbidden)
    monkeypatch.setattr(cached.archive.model, "decode", forbidden)
    monkeypatch.setattr(cached.window_cache, "read_images", forbidden)
    small = compute_cached_stats(cached, acceptance_path=report, block_windows=1, progress=False)
    large = compute_cached_stats(cached, acceptance_path=report, block_windows=1024, progress=False)
    raw_report = json.loads(report.read_text())
    raw_report.update(format=RAW_ACCEPTANCE_FORMAT, scope="all", checks_passed=True)
    report.write_text(json.dumps(raw_report))
    raw_phase = compute_cached_stats(cached, acceptance_path=report, block_windows=2, progress=False)
    for stats in (small, large, raw_phase):
        assert_same_stats(expected, {key: value for key, value in stats.items() if key != "window_cache_provenance"})
    assert small["window_cache_provenance"] == large["window_cache_provenance"]
    assert raw_phase["window_cache_provenance"] == {
        "format": CACHED_STATISTICS_PROVENANCE_FORMAT,
        **window_cache_acceptance_binding(cached.window_cache.root, report)}
    validate_cached_statistics_provenance(raw_phase, cached.window_cache.root)
    # The earlier statistics retain their actual proof, even after a new report
    # accepts the same manifest. Evidence identity is not silently rebound.
    with pytest.raises(ValueError, match="provenance differs"):
        validate_cached_statistics_provenance(small, cached.window_cache.root)
    # Every overlapping window still contributes all eleven supervised frames.
    assert sum(large["source_valid_frames"].values()) == large["num_windows"] * 11
    torch.testing.assert_close(large["motion_mean"][108:117], large["traj_mean"][9:18], rtol=0, atol=0)
    torch.testing.assert_close(large["motion_std"][108:117], large["traj_std"][9:18], rtol=0, atol=0)
    assert not large["motion_mean"][243:245].any()
    assert torch.equal(large["motion_std"][243:245], torch.ones(2))


def test_v4_normalized_cache_requires_its_statistics_provenance_but_raw_source_remains_compatible(
        cached_archive, tmp_path):
    cfg = cached_archive
    raw = _dataset(cfg, cached=True, normalized=False)
    # Reuse the verified tiny archive, changing only the in-memory publication
    # kind to exercise the v4 gate without rebuilding a production revision.
    raw.archive.info["quality_revision"] = {"format": "smplh-camera-quality-revision-v1"}
    with pytest.raises(ValueError, match="versioned window-cache provenance"):
        _dataset(cfg, cached=True, normalized=True, verified_archive=raw.archive)
    _dataset(cfg, cached=False, normalized=True, verified_archive=raw.archive)
    report = accepted(cfg, tmp_path)
    statistics = compute_cached_stats(raw, acceptance_path=report, progress=False)
    torch.save(statistics, cfg.DATA.STATS_PATH)
    _dataset(cfg, cached=True, normalized=True, verified_archive=raw.archive)
    statistics["window_cache_provenance"]["cache_manifest_sha256"] = "old-v4-cache"
    torch.save(statistics, cfg.DATA.STATS_PATH)
    with pytest.raises(ValueError, match="provenance differs"):
        _dataset(cfg, cached=True, normalized=True, verified_archive=raw.archive)


@pytest.mark.parametrize("changed", ["cache_manifest", "acceptance_sha", "acceptance_cache_manifest"])
def test_statistics_proof_cannot_pair_another_cache_or_an_edited_acceptance(cached_archive, tmp_path, changed):
    raw = _dataset(cached_archive, cached=True, normalized=False)
    report = accepted(cached_archive, tmp_path)
    statistics = compute_cached_stats(raw, acceptance_path=report, progress=False)
    if changed == "cache_manifest":
        # Re-accept a new same-archive manifest. The old stats must remain bound
        # to their original cache, even when the new report is internally valid.
        manifest = raw.window_cache.root / "manifest.json"
        manifest.write_text(manifest.read_text() + "\n")
        value = json.loads(report.read_text())
        value["cache_manifest_sha256"] = file_sha256(manifest)
        report.write_text(json.dumps(value))
    elif changed == "acceptance_sha":
        report.write_text(report.read_text() + "\n")
    else:
        value = json.loads(report.read_text())
        value["cache_manifest_sha256"] = "other-v4-cache"
        report.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="provenance differs|manifest SHA256"):
        validate_cached_statistics_provenance(statistics, raw.window_cache.root)


def test_rejects_missing_or_mismatched_formal_acceptance(cached_archive, tmp_path):
    dataset = _dataset(cached_archive, cached=True, normalized=False)
    report = accepted(cached_archive, tmp_path)
    value = json.loads(report.read_text())
    value["cache_manifest_sha256"] = "unaccepted revision"
    report.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="manifest SHA256"):
        compute_cached_stats(dataset, acceptance_path=report, progress=False)
    value["passed"] = False
    report.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="formal cache acceptance"):
        compute_cached_stats(dataset, acceptance_path=report, progress=False)
    value.update(format=RAW_ACCEPTANCE_FORMAT, passed=True,
                 cache_manifest_sha256=file_sha256(Path(cached_archive.DATA.WINDOW_CACHE_ROOT) / "manifest.json"))
    for fields in ({"scope": "diagnostic", "checks_passed": True}, {"scope": "all", "checks_passed": False}):
        value.update(fields)
        report.write_text(json.dumps(value))
        with pytest.raises(ValueError, match="full scope and passing checks"):
            compute_cached_stats(dataset, acceptance_path=report, progress=False)


def test_cached_statistics_refuse_normalized_or_non_train_inputs(cached_archive, tmp_path):
    report = accepted(cached_archive, tmp_path)
    normalized = _dataset(cached_archive, cached=True, normalized=True)
    with pytest.raises(ValueError, match="raw full-train"):
        compute_cached_stats(normalized, acceptance_path=report, progress=False)
    validation = _dataset(cached_archive, cached=True, split="val", normalized=False)
    with pytest.raises(ValueError, match="raw full-train"):
        compute_cached_stats(validation, acceptance_path=report, progress=False)


def test_new_statistics_publication_never_overwrites_old_file_or_immutable_data(tmp_path, monkeypatch):
    archive, cache = tmp_path / "archive", tmp_path / "cache"
    archive.mkdir(); cache.mkdir()
    output = tmp_path / "statistics/revision2.pt"
    stats = {"motion_mean": torch.arange(245), "identity": "new"}
    assert publish_new_stats(stats, output, archive_root=archive, cache_root=cache) == output
    saved = output.read_bytes()
    with pytest.raises(FileExistsError):
        publish_new_stats({"identity": "bad"}, output, archive_root=archive, cache_root=cache)
    assert output.read_bytes() == saved
    for root in (archive, cache):
        with pytest.raises(ValueError, match="outside immutable"):
            validate_output(root / "forbidden.pt", archive_root=archive, cache_root=cache)
    # Simulate an independent writer winning immediately before publication.
    raced = output.parent / "raced.pt"
    real_link = __import__("os").link
    def winning_writer(source, destination):
        Path(destination).write_bytes(b"independent writer")
        return real_link(source, destination)
    monkeypatch.setattr("MotionExpert.compute_cached_motion_stats.os.link", winning_writer)
    with pytest.raises(FileExistsError):
        publish_new_stats(stats, raced, archive_root=archive, cache_root=cache)
    assert raced.read_bytes() == b"independent writer"
    assert not list(output.parent.glob(".*.tmp"))


@pytest.mark.parametrize("field,available,nonfinite", [("motion", True, np.nan), ("trajectory", True, np.inf),
                                                       ("motion", False, -np.inf), ("trajectory", False, np.nan)])
def test_full_cached_scan_rejects_nonfinite_even_outside_representative_or_valid_channels(
        cached_archive, tmp_path, field, available, nonfinite):
    # Inject only into this tiny fixture, then honestly rebind file identities
    # and acceptance.  Metadata checks must not substitute for a finite scan.
    root = Path(cached_archive.DATA.WINDOW_CACHE_ROOT)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    packed = manifest["packed"]["train"]
    for name, value in ((field, nonfinite), (field + "_valid", available)):
        identity = packed["arrays"][name]
        path = root / identity["path"]
        array = np.load(path, mmap_mode="r+", allow_pickle=False)
        array[0, 0, 0] = value
        array.flush()
        array._mmap.close()
        stat = path.stat()
        identity.update(size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=file_sha256(path))
    (root / "packed/train/receipt.json").write_text(json.dumps(packed))
    manifest_path.write_text(json.dumps(manifest))
    acceptance = accepted(cached_archive, tmp_path)
    dataset = _dataset(cached_archive, cached=True, normalized=False)
    episode = dataset.window_cache.records[0]["episode_index"]
    with pytest.raises(ValueError, match=rf"Nonfinite packed {field} at episode={episode}.*window_row.*channel") as error:
        compute_cached_stats(dataset, acceptance_path=acceptance, block_windows=1, progress=False)
    assert "'frame_in_window': 0" in str(error.value)
    assert f"'supervision_valid': {available}" in str(error.value)
