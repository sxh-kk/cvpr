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
        "quality_revision": {"format": "smplh-fit-quality-revision-v1"}}))
    splits, stats = root / "meta/splits.json", tmp_path / "stats.pt"
    splits.write_text("{}")
    torch.save({"diagnostic": False}, stats)
    state = root / ".pipeline/dataset-state.json"
    state.write_text(json.dumps({"formal_training_ready": False}))
    monkeypatch.setattr(readiness, "assert_smplh_archive_ready", lambda _: None)
    cfg = SimpleNamespace(DATA=SimpleNamespace(
        ARCHIVE_ROOT=str(root), SPLIT_MANIFEST=str(splits), STATS_PATH=str(stats),
        VISION_PROFILE_ROOT=str(vision)))
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
    state.write_text(json.dumps({"formal_training_ready": True,
        "statistics_sha256": hashlib.sha256(Path(cfg.DATA.STATS_PATH).read_bytes()).hexdigest()}))
    readiness.validate_paths(cfg)
