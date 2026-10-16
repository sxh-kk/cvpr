"""A published revised archive is not enough to start a training run."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from MotionExpert import readiness


@pytest.fixture
def revised_paths(tmp_path, monkeypatch):
    root, vision = tmp_path / "archive", tmp_path / "vision"
    (root / "meta").mkdir(parents=True)
    (root / ".pipeline").mkdir()
    vision.mkdir()
    (vision / "manifest.json").write_text("{}")
    (root / "meta/info.json").write_text(json.dumps({
        "quality_revision": {"format": "smplh-camera-quality-revision-v1"}}))
    splits, stats = root / "meta/splits.json", tmp_path / "stats.pt"
    splits.write_text("{}")
    torch.save({"diagnostic": False}, stats)
    state = root / ".pipeline/dataset-state.json"
    state.write_text(json.dumps({"formal_training_ready": False}))
    monkeypatch.setattr(readiness, "assert_smplh_archive_ready", lambda _: None)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "manifest.json").write_text("{}")
    cfg = SimpleNamespace(DATA=SimpleNamespace(
        ARCHIVE_ROOT=str(root), SPLIT_MANIFEST=str(splits), STATS_PATH=str(stats),
        VISION_PROFILE_ROOT=str(vision), WINDOW_CACHE_ROOT=str(cache)))
    return cfg, state


def test_published_revision_requires_completed_training_acceptance(revised_paths):
    cfg, state = revised_paths
    with pytest.raises(ValueError, match="before training"):
        readiness.validate_paths(cfg)
    # Processing may check staged paths, but train.main never passes this opt-in.
    readiness.validate_paths(cfg, allow_pending_fit_quality=True)
    state.write_text(json.dumps({"formal_training_ready": True, "statistics_sha256": "stale"}))
    with pytest.raises(ValueError, match="different training statistics"):
        readiness.validate_paths(cfg)
    cache_acceptance = state.parent / "raw-cache-acceptance.json"
    cache_acceptance.write_text(json.dumps({
        "format": readiness.RAW_WINDOW_CACHE_ACCEPTANCE_FORMAT,
        "state": "complete", "passed": True, "diagnostic": False,
        "scope": "all", "checks_passed": True, "finished_at": "2026-10-16T00:00:00+00:00",
        "cache_root": cfg.DATA.WINDOW_CACHE_ROOT,
        "cache_manifest_sha256": hashlib.sha256(
            (Path(cfg.DATA.WINDOW_CACHE_ROOT) / "manifest.json").read_bytes()).hexdigest(),
    }))
    stats = torch.load(cfg.DATA.STATS_PATH, weights_only=True)
    stats["window_cache_provenance"] = {
        "format": readiness.CACHED_STATISTICS_PROVENANCE_FORMAT,
        **readiness.window_cache_acceptance_binding(cfg.DATA.WINDOW_CACHE_ROOT, cache_acceptance),
    }
    torch.save(stats, cfg.DATA.STATS_PATH)
    statistics_sha256 = hashlib.sha256(Path(cfg.DATA.STATS_PATH).read_bytes()).hexdigest()
    window_cache_manifest_sha256 = hashlib.sha256(
        (Path(cfg.DATA.WINDOW_CACHE_ROOT) / "manifest.json").read_bytes()).hexdigest()
    report = state.parent / "gpu-acceptance.json"
    report.write_text(json.dumps({
        "format": readiness.GPU_ACCEPTANCE_FORMAT, "passed": True,
        "model_architecture": readiness.MODEL_ARCHITECTURE_VERSION,
        "statistics_sha256": statistics_sha256,
        "window_cache_manifest_sha256": window_cache_manifest_sha256,
        "code_binding": readiness.training_code_binding(),
    }))
    ready = {"formal_training_ready": True, "statistics_sha256": statistics_sha256,
             "window_cache_manifest_sha256": window_cache_manifest_sha256,
             "model_architecture": readiness.MODEL_ARCHITECTURE_VERSION,
             "gpu_acceptance": {"path": str(report), "sha256": hashlib.sha256(report.read_bytes()).hexdigest()}}
    state.write_text(json.dumps(ready))
    readiness.validate_paths(cfg)
    report.write_text(report.read_text() + "\n")
    with pytest.raises(ValueError, match="GPU acceptance"):
        readiness.validate_paths(cfg)


def test_parallel_stats_and_cache_hashes_do_not_replace_statistics_origin_proof(revised_paths):
    cfg, state = revised_paths
    state.write_text(json.dumps({
        "formal_training_ready": True,
        "statistics_sha256": hashlib.sha256(Path(cfg.DATA.STATS_PATH).read_bytes()).hexdigest(),
        "window_cache_manifest_sha256": hashlib.sha256(
            (Path(cfg.DATA.WINDOW_CACHE_ROOT) / "manifest.json").read_bytes()).hexdigest(),
        "model_architecture": readiness.MODEL_ARCHITECTURE_VERSION,
    }))
    with pytest.raises(ValueError, match="versioned window-cache provenance"):
        readiness.validate_paths(cfg)


def test_old_fit_archive_cannot_start_new_model(revised_paths):
    cfg, state = revised_paths
    (Path(cfg.DATA.ARCHIVE_ROOT) / "meta/info.json").write_text(json.dumps({
        "quality_revision": {"format": "smplh-fit-quality-revision-v1"}}))
    with pytest.raises(ValueError, match="old masks are unsafe"):
        readiness.validate_paths(cfg)


def test_old_architecture_acceptance_cannot_start_new_model(revised_paths):
    cfg, state = revised_paths
    state.write_text(json.dumps({"formal_training_ready": True,
        "statistics_sha256": hashlib.sha256(Path(cfg.DATA.STATS_PATH).read_bytes()).hexdigest(),
        "model_architecture": "e7_smplh12_hands21_global_v1"}))
    with pytest.raises(ValueError, match="architecture"):
        readiness.validate_paths(cfg)
