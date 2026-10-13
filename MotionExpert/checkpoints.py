"""Strict loading of dense K12 motion weights and Lightning EMA checkpoints."""

from collections.abc import Mapping
from pathlib import Path

import torch

from .config import (
    MOTION_DIM, architecture_metadata, config_from_dict,
    validate_architecture, SMPLH_REPRESENTATION_VERSION,
)


def parameter_phase(name):
    if any(marker in name for marker in (".ff.router", ".ff.routed_experts.", ".ff.routed_gate")):
        return "routed"
    return "unfrozen"


def read_checkpoint(path):
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Expected a MotionExpert checkpoint mapping")
    return checkpoint


def _state_dict(checkpoint):
    source = checkpoint.get("state_dict", checkpoint)
    if not isinstance(source, Mapping):
        raise ValueError("Checkpoint state_dict must be a tensor mapping")
    if not all(isinstance(name, str) for name in source):
        raise ValueError("Checkpoint tensor names must be strings")
    state = {name.removeprefix("model."): value for name, value in source.items()}
    if not state or not all(isinstance(name, str) and isinstance(value, torch.Tensor)
                            for name, value in state.items()):
        raise ValueError("Expected a raw state_dict or a Lightning training checkpoint")
    return state


def checkpoint_config(checkpoint):
    """Reject the old 153D/text model before allocating the new backbone."""
    state = _state_dict(checkpoint)
    input_weight = state.get("input_process.weight")
    if input_weight is not None and input_weight.ndim != 2:
        raise ValueError("Checkpoint input_process.weight must be a matrix")
    legacy_text = any(name.startswith(("text_", "embed_text_cond."))
                      or name == "mask_tokens.text" for name in state)
    if legacy_text or (input_weight is not None and input_weight.shape[-1] != MOTION_DIM):
        raise ValueError(
            "Legacy SMPL/text MotionExpert checkpoint is incompatible with the "
            "245D K12 with explicit hands representation; train a matching model."
        )
    metadata = checkpoint.get("motionexpert_architecture")
    if not isinstance(metadata, Mapping) or "hyper_parameters" not in checkpoint:
        raise ValueError("Dense checkpoints require architecture metadata and hyper_parameters; use motionexpert-export")
    cfg = config_from_dict(checkpoint["hyper_parameters"])
    validate_architecture(cfg)
    has_global = "global_token" in state
    if bool(cfg.MODEL.GLOBAL_TOKEN_COUNT) != has_global:
        raise ValueError("Checkpoint global_token weights and architecture configuration disagree")
    if has_global and tuple(state["global_token"].shape) != (768,):
        raise ValueError("Checkpoint global_token must have shape [768]")
    expected = architecture_metadata(cfg)
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("Checkpoint architecture metadata does not match its configuration and weights")
    dim = cfg.MODEL.INPUT_FEATS
    shapes = {"input_process.weight": (768, dim),
              "output_process.weight": (dim, 768),
              "embed_traj_cond.weight": (768, 18),
              "embed_clip_cond.weight": (768, 1024)}
    for name, shape in shapes.items():
        if name in state and tuple(state[name].shape) != shape:
            raise ValueError(f"Checkpoint tensor {name} must have shape {shape}")
    return cfg


def validate_checkpoint_architecture(cfg, checkpoint):
    expected = architecture_metadata(cfg)
    actual = architecture_metadata(checkpoint_config(checkpoint))
    if actual != expected:
        raise ValueError(f"Checkpoint architecture {actual['architecture']} does not match {expected['architecture']}")


def model_state(model, checkpoint, *, use_ema=True):
    if hasattr(model, "cfg"):
        validate_checkpoint_architecture(model.cfg, checkpoint)
    state = _state_dict(checkpoint)
    ema_format = checkpoint.get("ema_state_format")
    if use_ema and ema_format not in (None, "original_model_with_ema_optimizer_v1"):
        raise ValueError(f"Unsupported EMA checkpoint format: {ema_format}")
    if use_ema and ema_format == "original_model_with_ema_optimizer_v1":
        optimizer = checkpoint["optimizer_states"][0]
        ema = optimizer["ema"]
        named = list(model.named_parameters())
        if len(ema) != len(named):
            named = [(name, value) for name, value in named if value.requires_grad]
        groups = optimizer["opt"]["param_groups"]
        if len(groups) > 1:
            ordered = []
            for group in groups:
                values = [(name, value) for name, value in named
                          if parameter_phase(name) == group.get("phase")]
                if len(values) != len(group["params"]):
                    raise ValueError(f"EMA group mismatch for {group.get('phase')}")
                ordered.extend(values)
            named = ordered
        if len(named) != len(ema):
            raise ValueError("EMA tensor count does not match the dense MotionExpert")
        for (name, parameter), value in zip(named, ema):
            if parameter.shape != value.shape:
                raise ValueError(f"EMA tensor shape mismatch: {name}")
            state[name] = value
    return state


def validate_checkpoint_statistics(cfg, checkpoint, normalizer):
    """Bind physical units and joint axes to the statistics used to train weights."""
    stats = normalizer.stats
    expected = {
        "representation": cfg.DATA.REPRE_TYPE,
        "window": cfg.DATA.WINDOW, "forecast_prefix": cfg.DATA.FORECAST_PREFIX,
        "fps": cfg.DATA.FPS, "target_joint_basis": cfg.DATA.TARGET_JOINT_BASIS,
        "rotation_basis_mode": cfg.DATA.ROTATION_BASIS_MODE,
        "contact_speed_m_s": cfg.DATA.CONTACT_SPEED_M_S,
        "contact_height_m": cfg.DATA.CONTACT_HEIGHT_M,
    }
    if cfg.DATA.REPRE_TYPE == SMPLH_REPRESENTATION_VERSION:
        from .config import SMPLH_SCHEMA
        expected.update(model_sha256=cfg.DATA.MODEL_SHA256, archive_schema_version=SMPLH_SCHEMA)
    if cfg.DATA.FPS == 5:
        expected.update({
            "stats_schema_version": 4, "sampling_step": 6,
            "segment_stride": cfg.DATA.SEGMENT_STRIDE,
            "train_task": cfg.DATA.TRAIN_TASK, "label_task": "fore",
            "complete_windows_only": cfg.DATA.REQUIRE_FULL_WINDOW,
            "ground_estimation_scope": "observed_prefix",
            "image_validity_scope": "observed_prefix",
            "min_valid_image_fraction": cfg.DATA.MIN_VALID_IMAGE_FRACTION,
            "statistics_scope": ("all_valid_frames" if cfg.DATA.MOTION_SUPERVISION == "joint" else "future_only"),
        })
        if cfg.DATA.MOTION_SUPERVISION == "joint" or "motion_supervision" in stats:
            expected["motion_supervision"] = cfg.DATA.MOTION_SUPERVISION
    for key, value in expected.items():
        if stats.get(key) != value:
            raise ValueError(f"Training statistics {key}={stats.get(key)!r} do not match checkpoint/config {value!r}")
    if "joint_indices" in stats and list(stats["joint_indices"]) != list(cfg.SPARSE_JOINTS.INDICES):
        raise ValueError("Training statistics use different dense K12 joints")
    original = checkpoint.get("motionexpert_normalization")
    if original is None:
        return
    from .dataset.normalization import MotionNormalizer
    trained = MotionNormalizer(dict(original, normalization_clip=checkpoint.get(
        "motionexpert_normalization_clip", original.get("normalization_clip", None))))
    for key in (*expected, "split_fingerprint", "archive_fingerprint", "basis_mapping_fingerprint"):
        if stats.get(key) != trained.stats.get(key):
            raise ValueError(f"Provided statistics differ from checkpoint training statistics: {key}")
    for key in ("motion_mean", "motion_std", "traj_mean", "traj_std"):
        if not torch.equal(stats[key], trained.stats[key]):
            raise ValueError(f"Provided statistics differ from checkpoint training statistics: {key}")
    if normalizer.clip != trained.clip:
        raise ValueError("Provided normalization clip differs from checkpoint training statistics")
