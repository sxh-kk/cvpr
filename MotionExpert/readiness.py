"""Fast file/publication prerequisites before allocating the motion network."""
from pathlib import Path
import torch
from .dataset.smplh_archive import assert_smplh_archive_ready


def validate_paths(cfg):
    root = Path(cfg.DATA.ARCHIVE_ROOT)
    if not root.is_dir():
        raise FileNotFoundError(f"Missing archive root: {root}")
    assert_smplh_archive_ready(root)
    for name in ("SPLIT_MANIFEST", "STATS_PATH"):
        value = getattr(cfg.DATA, name)
        if not value or not Path(value).is_file():
            raise FileNotFoundError(f"Missing DATA.{name}: {value}")
    if torch.load(cfg.DATA.STATS_PATH, map_location="cpu", weights_only=True).get("diagnostic", False):
        raise ValueError("Training requires full train statistics")
    cache = cfg.DATA.VISION_PROFILE_ROOT
    if not cache or not (Path(cache) / "manifest.json").is_file():
        raise FileNotFoundError("Training requires the newly extracted shared-profile native30Hz DINO cache")
