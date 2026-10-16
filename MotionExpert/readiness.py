"""Fast file/publication prerequisites before allocating the motion network."""
from pathlib import Path
import hashlib
import json
import torch
from .dataset.smplh_archive import assert_smplh_archive_ready
from .dataset.camera_revision_acceptance import REVISION_FORMAT
from .config import MODEL_ARCHITECTURE_VERSION


GPU_ACCEPTANCE_FORMAT = "motionexpert-pooled-safety-gpu-acceptance-v1"
CACHED_STATISTICS_PROVENANCE_FORMAT = "motionexpert-cached-train-statistics-provenance-v1"
WINDOW_CACHE_ACCEPTANCE_FORMAT = "motionexpert-window-cache-acceptance-v1"
RAW_WINDOW_CACHE_ACCEPTANCE_FORMAT = "motionexpert-raw-window-cache-acceptance-v1"
_TRAINING_SOURCES = (
    "config.py", "model/core.py", "model/uniegomotion.py", "mydiffusion/flow_matching.py",
    "module/uem_module.py", "motion_expert.py", "training_safety.py", "train.py",
    "checkpoints.py", "readiness.py", "utils/task_conditioning.py", "ema.py",
    "precision.py", "evaluate.py", "dataset/normalization.py",
    "dataset/smplh_motion_dataset.py", "dataset/data_module.py",
    "compute_cached_motion_stats.py", "compute_joint_motion_stats.py",
)


def window_cache_acceptance_binding(cache_root, acceptance_path):
    """Resolve exact evidence that accepts the current immutable cache manifest."""
    root = Path(cache_root).resolve(strict=True)
    path = Path(acceptance_path).resolve(strict=True)
    report_bytes = path.read_bytes()
    report = json.loads(report_bytes)
    report_format = report.get("format")
    if (report_format not in {WINDOW_CACHE_ACCEPTANCE_FORMAT, RAW_WINDOW_CACHE_ACCEPTANCE_FORMAT}
            or report.get("state") != "complete" or report.get("passed") is not True
            or report.get("diagnostic") is not False or report.get("error") or not report.get("finished_at")):
        raise ValueError("Cached train statistics require completed non-diagnostic formal cache acceptance or explicit raw-phase acceptance")
    if (report_format == RAW_WINDOW_CACHE_ACCEPTANCE_FORMAT
            and (report.get("scope") != "all" or report.get("checks_passed") is not True)):
        raise ValueError("Raw-phase acceptance must declare full scope and passing checks")
    if Path(report.get("cache_root", "")).resolve() != root:
        raise ValueError("Acceptance names a different window cache")
    manifest_sha = hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()
    if report.get("cache_manifest_sha256") != manifest_sha:
        raise ValueError("Acceptance window-cache manifest SHA256 differs")
    return {"acceptance_path": str(path), "acceptance_sha256": hashlib.sha256(report_bytes).hexdigest(),
            "cache_root": str(root), "cache_manifest_sha256": manifest_sha}


def validate_cached_statistics_provenance(statistics, cache_root):
    """A statistics payload must itself name this exact accepted cache."""
    provenance = statistics.get("window_cache_provenance")
    if not isinstance(provenance, dict) or provenance.get("format") != CACHED_STATISTICS_PROVENANCE_FORMAT:
        raise ValueError("Camera statistics require versioned window-cache provenance")
    acceptance_path = provenance.get("acceptance_path")
    if not isinstance(acceptance_path, str) or not Path(acceptance_path).is_absolute():
        raise ValueError("Statistics window-cache provenance requires an absolute acceptance path")
    expected = window_cache_acceptance_binding(cache_root, acceptance_path)
    if provenance != {"format": CACHED_STATISTICS_PROVENANCE_FORMAT, **expected}:
        raise ValueError("Statistics window-cache provenance differs from the current cache or acceptance")


def training_code_binding():
    root = Path(__file__).resolve().parent
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in _TRAINING_SOURCES}


def _validate_training_acceptance(cfg, root, statistics):
    state = json.loads((root / ".pipeline/dataset-state.json").read_text())
    if state.get("formal_training_ready") is not True:
        raise ValueError("Camera quality revision requires completed cache, statistics and GPU acceptance before training")
    stats_sha = hashlib.sha256(Path(cfg.DATA.STATS_PATH).read_bytes()).hexdigest()
    if state.get("statistics_sha256") != stats_sha:
        raise ValueError("Camera quality readiness is bound to different training statistics")
    if state.get("model_architecture") != MODEL_ARCHITECTURE_VERSION:
        raise ValueError("Camera quality readiness requires current mean-pooling/QKV-normalization architecture")
    cache = getattr(cfg.DATA, "WINDOW_CACHE_ROOT", None)
    manifest = Path(cache) / "manifest.json" if cache else None
    if manifest is None or not manifest.is_file():
        raise ValueError("Camera quality training requires its accepted window cache")
    cache_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if state.get("window_cache_manifest_sha256") != cache_sha:
        raise ValueError("Camera quality readiness is bound to a different window cache")
    validate_cached_statistics_provenance(statistics, cache)
    proof = state.get("gpu_acceptance", {})
    report_path = Path(proof.get("path", ""))
    if (not report_path.is_absolute() or not report_path.is_file()
            or hashlib.sha256(report_path.read_bytes()).hexdigest() != proof.get("sha256")):
        raise ValueError("Missing or changed mean-pooling training GPU acceptance")
    report = json.loads(report_path.read_text())
    if (report.get("format") != GPU_ACCEPTANCE_FORMAT or report.get("passed") is not True
            or report.get("model_architecture") != MODEL_ARCHITECTURE_VERSION
            or report.get("statistics_sha256") != stats_sha
            or report.get("window_cache_manifest_sha256") != cache_sha
            or report.get("code_binding") != training_code_binding()):
        raise ValueError("Mean-pooling training GPU acceptance differs from current code/data")


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
    statistics = torch.load(cfg.DATA.STATS_PATH, map_location="cpu", weights_only=True)
    # Processing/QA may inspect an accepted parent or a staged new revision.
    # train.main never uses this internal opt-in.
    if not allow_pending_fit_quality:
        if info.get("quality_revision", {}).get("format") != REVISION_FORMAT:
            raise ValueError("Training requires the accepted native camera-quality revision; old masks are unsafe")
        _validate_training_acceptance(cfg, root, statistics)
    if statistics.get("diagnostic", False):
        raise ValueError("Training requires full train statistics")
    cache = cfg.DATA.VISION_PROFILE_ROOT
    if not cache or not (Path(cache) / "manifest.json").is_file():
        raise FileNotFoundError("Training requires the newly extracted shared-profile native30Hz DINO cache")
