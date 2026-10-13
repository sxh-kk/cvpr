"""Load raw E7 weights or the EMA tensors in a Lightning training checkpoint."""

from pathlib import Path
from collections.abc import Mapping

import torch

from .config import architecture_metadata, config_from_dict, motion_expert_config, validate_architecture


def parameter_phase(name):
    if name.startswith("text_"):
        return "text"
    if any(marker in name for marker in (".ff.router", ".ff.routed_experts.", ".ff.routed_gate")):
        return "routed"
    return "unfrozen"


def read_checkpoint(path):
    return torch.load(Path(path), map_location="cpu", weights_only=True, mmap=True)


def _state_dict(checkpoint):
    source = checkpoint.get("state_dict", checkpoint)
    state = {name.removeprefix("model."): value for name, value in source.items()}
    if not state or not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("Expected a raw state_dict or a Lightning training checkpoint")
    return state


def checkpoint_config(checkpoint):
    """Recover architecture without treating old weights as a trained global model."""
    state = _state_dict(checkpoint)
    has_global = "global_token" in state
    if "hyper_parameters" in checkpoint:
        cfg = config_from_dict(checkpoint["hyper_parameters"])
    else:
        cfg = motion_expert_config(global_token=has_global)
    validate_architecture(cfg)
    if bool(cfg.MODEL.GLOBAL_TOKEN_COUNT) != has_global:
        raise ValueError("Checkpoint global_token weights and architecture configuration disagree")
    metadata = checkpoint.get("motionexpert_architecture")
    if has_global and metadata is None:
        raise ValueError("Global-token weights require architecture metadata; use motionexpert-export")
    if has_global and tuple(state["global_token"].shape) != (768,):
        raise ValueError("Checkpoint global_token must have shape [768]")
    if metadata is not None:
        expected = architecture_metadata(cfg)
        if not isinstance(metadata, Mapping) or any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError("Checkpoint architecture metadata does not match its configuration and weights")
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
    if use_ema and checkpoint.get("ema_state_format") == "original_model_with_ema_optimizer_v1":
        optimizer = checkpoint["optimizer_states"][0]
        ema = optimizer["ema"]
        named = list(model.named_parameters())
        if len(ema) != len(named):
            named = [(name, value) for name, value in named if value.requires_grad]
        groups = optimizer["opt"]["param_groups"]
        if len(groups) > 1:
            ordered = []
            for group in groups:
                values = [(name, value) for name, value in named if parameter_phase(name) == group.get("phase")]
                if len(values) != len(group["params"]):
                    raise ValueError(f"EMA group mismatch for {group.get('phase')}")
                ordered.extend(values)
            named = ordered
        if len(named) != len(ema):
            raise ValueError("EMA tensor count does not match E7 K12 400M")
        for (name, parameter), value in zip(named, ema):
            if parameter.shape != value.shape:
                raise ValueError(f"EMA tensor shape mismatch: {name}")
            state[name] = value
    return state
