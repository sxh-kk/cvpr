"""Explicit weight transfer across a quality revision, keeping current statistics."""
from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch

from .checkpoints import (checkpoint_config, model_state, parameter_phase, read_checkpoint,
                          validate_checkpoint_architecture, validate_checkpoint_statistics)
from .dataset.normalization import MotionNormalizer
from .visual_contract import (checkpoint_visual_contract, compare_visual_contracts,
                              training_visual_contract)

WARM_START_FORMAT = "motionexpert-quality-warm-start-v1"
_PHYSICAL_FIELDS = (
    "representation", "window", "forecast_prefix", "fps", "target_joint_basis",
    "rotation_basis_mode", "contact_speed_m_s", "contact_height_m", "model_sha256",
    "archive_schema_version", "stats_schema_version", "sampling_step", "segment_stride",
    "train_task", "label_task", "complete_windows_only", "ground_estimation_scope",
    "image_validity_scope", "min_valid_image_fraction", "statistics_scope",
    "motion_supervision", "joint_indices", "split", "split_fingerprint", "basis_mapping_fingerprint",
)


def _file_identity(path):
    info = path.stat()
    return {"device": info.st_dev, "inode": info.st_ino, "bytes": info.st_size,
            "mtime_ns": info.st_mtime_ns}


def _file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _statistics_identity(normalizer):
    stats = normalizer.stats
    metadata = {key: stats[key] for key in (*_PHYSICAL_FIELDS, "archive_fingerprint")}
    metadata["normalization_clip"] = normalizer.clip
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
    for key in ("motion_mean", "motion_std", "traj_mean", "traj_std"):
        digest.update(key.encode())
        digest.update(stats[key].detach().cpu().contiguous().numpy().tobytes())
    return {"normalization_sha256": digest.hexdigest(), "archive_fingerprint": stats["archive_fingerprint"],
            "split_fingerprint": stats["split_fingerprint"],
            "num_windows": stats.get("num_windows"), "normalization_clip": normalizer.clip}


def _validate_ema(model, checkpoint, source_cfg):
    """EMA ordering is the source AdamW group ordering, not named_parameters order."""
    if checkpoint.get("ema_state_format") != "original_model_with_ema_optimizer_v1":
        raise ValueError("EMA warm start requires an explicit supported EMA format; raw fallback is forbidden")
    optimizers = checkpoint.get("optimizer_states")
    if not isinstance(optimizers, (list, tuple)) or len(optimizers) != 1:
        raise ValueError("EMA warm start requires exactly one source optimizer")
    optimizer = optimizers[0]
    if not isinstance(optimizer, Mapping) or not isinstance(optimizer.get("opt"), Mapping):
        raise ValueError("EMA warm start requires the source optimizer group mapping")
    groups, ema = optimizer["opt"].get("param_groups"), optimizer.get("ema")
    if not isinstance(groups, (list, tuple)) or not isinstance(ema, (list, tuple)):
        raise ValueError("Incomplete source EMA groups or parameters")
    named = list(model.named_parameters())
    expert = source_cfg.MODEL.MOTION_EXPERT.ENABLED
    phases = ["routed", "unfrozen"] if expert else ["main"]
    expected = {phase: [(name, parameter) for name, parameter in named
                        if not expert or parameter_phase(name) == phase] for phase in phases}
    phases = [phase for phase in phases if expected[phase]]
    if any(not isinstance(group, Mapping) for group in groups) or [group.get("phase") for group in groups] != phases:
        raise ValueError("Source EMA optimizer phase/order differs from the complete model")
    cursor = 0
    for group, phase in zip(groups, phases):
        count = len(expected[phase])
        ids = group.get("params")
        if (not isinstance(ids, (list, tuple)) or any(type(index) is not int for index in ids)
                or list(ids) != list(range(cursor, cursor + count))):
            raise ValueError("Source EMA optimizer parameter IDs/count/order are invalid")
        cursor += count
    if len(ema) != cursor or cursor != len(named):
        raise ValueError("Source EMA must cover every network parameter")
    if any(not isinstance(value, torch.Tensor) for value in ema):
        raise ValueError("Source EMA parameters must be tensors")


def apply_warm_start(module, checkpoint_path=None):
    """Load selected network weights only; no Trainer/optimizer state is restored.

    The current normalizer is a plain object outside the network state_dict.
    Network weights are kept numerically unchanged and adapt to the revised
    statistics during the new run; this is not a physical-equivalence transform.
    """
    cfg = module.cfg
    if cfg.MODEL.CKPT_PATH or cfg.MODEL.INIT_CKPT_PATH:
        raise ValueError("Warm start cannot be combined with resume or init-weights")
    configured = getattr(cfg.MODEL, "WARM_START_CKPT_PATH", None)
    if not configured:
        raise ValueError("Warm start requires an explicit source checkpoint")
    path = Path(checkpoint_path or configured).expanduser().resolve(strict=True)
    if path != Path(configured).expanduser().resolve(strict=True):
        raise ValueError("Warm-start path override differs from the configured source")
    if getattr(cfg.MODEL, "TRAINING_ORIGIN", "weights_init") != "weights_init":
        raise ValueError("Configure warm start before constructing the training module")
    if getattr(module, "warm_start_provenance", None) is not None:
        raise ValueError("Warm start may initialize a module only once; use resume for an existing run")
    if not path.is_file():
        raise ValueError("Warm-start checkpoint must be a file")
    before = _file_identity(path)
    checkpoint = read_checkpoint(path)
    validate_checkpoint_architecture(cfg, checkpoint)
    source_cfg = checkpoint_config(checkpoint)
    for key in ("ENABLED", "NUM_EXPERTS", "NUM_ROUTED_EXPERTS", "TOP_K", "SHARED_EXPERT", "CONDITIONED_ROUTER"):
        if getattr(source_cfg.MODEL.MOTION_EXPERT, key) != getattr(cfg.MODEL.MOTION_EXPERT, key):
            raise ValueError(f"Warm-start routed architecture differs: {key}")
    original = checkpoint.get("motionexpert_normalization")
    if not isinstance(original, Mapping):
        raise ValueError("Warm start requires embedded source training statistics")
    trained = MotionNormalizer(dict(original, normalization_clip=checkpoint.get(
        "motionexpert_normalization_clip", original.get("normalization_clip", None))))
    current = module.normalizer
    if source_cfg.DATA.NORMALIZATION_CLIP != trained.clip:
        raise ValueError("Source configuration and embedded normalization clipping differ")
    validate_checkpoint_statistics(source_cfg, checkpoint, trained)
    validate_checkpoint_statistics(cfg, {}, current)
    for key in _PHYSICAL_FIELDS:
        if key not in trained.stats or key not in current.stats or trained.stats[key] != current.stats[key]:
            raise ValueError(f"Warm-start physical/split statistics differ: {key}")
    if trained.clip != current.clip:
        raise ValueError("Warm start cannot change normalization clipping")
    if not trained.stats.get("archive_fingerprint") or not current.stats.get("archive_fingerprint"):
        raise ValueError("Warm start requires both archive fingerprints")
    compare_visual_contracts(checkpoint_visual_contract(checkpoint), training_visual_contract(cfg),
                             allow_visual_adaptation=False)
    epoch, step = checkpoint.get("epoch"), checkpoint.get("global_step")
    if type(epoch) is not int or epoch < 0 or type(step) is not int or step < 0:
        raise ValueError("Warm start requires the source epoch and global_step")
    completed = epoch + 1
    requested = getattr(cfg.MODEL, "WARM_START_EXPECTED_COMPLETED_EPOCHS", None)
    if requested is not None and (type(requested) is not int or requested < 1 or completed != requested):
        raise ValueError("Warm-start source completed epochs differ from the explicit selection")
    kind = getattr(cfg.MODEL, "WARM_START_WEIGHTS", "ema")
    if kind not in {"ema", "raw"}:
        raise ValueError("WARM_START_WEIGHTS must be ema or raw")
    if kind == "ema":
        _validate_ema(module.model, checkpoint, source_cfg)
    state = model_state(module.model, checkpoint, use_ema=kind == "ema")
    expected = module.model.state_dict()
    if state.keys() != expected.keys():
        raise ValueError("Warm-start network tensor names differ from the complete current network")
    for name, target in expected.items():
        value = state[name]
        if value.shape != target.shape or value.dtype != target.dtype or not torch.isfinite(value).all():
            raise ValueError(f"Warm-start tensor shape/dtype/finite check failed: {name}")
    old_identity, new_identity = _statistics_identity(trained), _statistics_identity(current)
    source_sha = _file_sha256(path)
    requested_sha = getattr(cfg.MODEL, "WARM_START_SHA256", None)
    if requested_sha is not None and source_sha != requested_sha:
        raise ValueError("Warm-start checkpoint SHA256 differs from the pinned source")
    if _file_identity(path) != before:
        raise ValueError("Warm-start checkpoint changed while loading")
    module.model.load_state_dict(state, strict=True)
    if module.normalizer is not current or _statistics_identity(current) != new_identity:
        raise RuntimeError("Warm start changed current training statistics")
    report = {"format": WARM_START_FORMAT, "source_checkpoint": str(path),
              "source_checkpoint_sha256": source_sha, "source_file_identity": before,
              "source_epoch": epoch, "source_completed_epochs": completed, "source_global_step": step,
              "weights_kind": kind, "network_tensors_loaded": len(state),
              "source_statistics": old_identity, "target_statistics": new_identity,
              "statistics_replaced": old_identity != new_identity,
              "physical_protocol_verified": True, "visual_identity_verified": True,
              "optimizer_state_restored": False, "scheduler_state_restored": False,
              "ema_training_state_restored": False, "rng_state_restored": False,
              "new_epoch": 0, "new_global_step": 0,
              "new_stage_epochs": int(cfg.TRAIN.NUM_EPOCHS),
              "new_scheduler_epochs": int(cfg.TRAIN.SCHEDULER_TOTAL_EPOCHS),
              "new_warmup_epochs": int(cfg.TRAIN.WARMUP_EPOCHS),
              "base_learning_rate": float(cfg.TRAIN.LR),
              "weight_transform": "unchanged_selected_weights_adapt_to_new_normalization"}
    module.warm_start_provenance = deepcopy(report)
    return report
