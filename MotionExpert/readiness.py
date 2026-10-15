"""Fast file/publication prerequisites before allocating the motion network."""
from pathlib import Path
import hashlib
import json
import torch
from .dataset.smplh_archive import assert_smplh_archive_ready


def validate_paths(cfg, *, allow_pending_fit_quality=False):
    root = Path(cfg.DATA.ARCHIVE_ROOT)
    if not root.is_dir():
        raise FileNotFoundError(f"Missing archive root: {root}")
    assert_smplh_archive_ready(root)
    for name in ("SPLIT_MANIFEST", "STATS_PATH"):
        value = getattr(cfg.DATA, name)
        if not value or not Path(value).is_file():
            raise FileNotFoundError(f"Missing DATA.{name}: {value}")
    info = json.loads((root / "meta/info.json").read_text())
    if info.get("quality_revision", {}).get("format") == "smplh-fit-quality-revision-v1":
        state = json.loads((root / ".pipeline/dataset-state.json").read_text())
        if not allow_pending_fit_quality:
            if state.get("formal_training_ready") is not True:
                raise ValueError("Final FK quality revision requires completed cache, statistics and GPU acceptance before training")
            stats_sha = hashlib.sha256(Path(cfg.DATA.STATS_PATH).read_bytes()).hexdigest()
            if state.get("statistics_sha256") != stats_sha:
                raise ValueError("Final FK quality readiness is bound to different training statistics")
    if torch.load(cfg.DATA.STATS_PATH, map_location="cpu", weights_only=True).get("diagnostic", False):
        raise ValueError("Training requires full train statistics")
    cache = cfg.DATA.VISION_PROFILE_ROOT
    if not cache or not (Path(cache) / "manifest.json").is_file():
        raise FileNotFoundError("Training requires the newly extracted shared-profile native30Hz DINO cache")
