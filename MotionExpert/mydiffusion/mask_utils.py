"""Shared feature-validity mask handling for motion objectives and samplers."""

from __future__ import annotations

from collections.abc import Mapping

import torch


def feature_mask_for(error: torch.Tensor, conditioning: Mapping) -> torch.Tensor | None:
    """Return a broadcastable label mask; incompatible widths are errors."""
    raw = conditioning.get("feature_valid_mask")
    if raw is None:
        return None
    mask = torch.as_tensor(raw, device=error.device, dtype=torch.bool)
    if mask.ndim < 2 or mask.shape[0] != error.shape[0] or mask.shape[-1] != error.shape[-1]:
        raise ValueError(
            f"feature_valid_mask must match batch and feature widths of {tuple(error.shape)}, "
            f"got {tuple(mask.shape)}"
        )
    if mask.ndim == 2 and error.ndim >= 3:
        mask = mask[:, None, :]
    if mask.ndim > error.ndim:
        raise ValueError(
            f"feature_valid_mask has {mask.ndim} dimensions but model output has {error.ndim}."
        )
    try:
        return mask.expand_as(error)
    except RuntimeError as exc:
        raise ValueError(
            f"feature_valid_mask shape {tuple(mask.shape)} cannot be expanded to "
            f"model output shape {tuple(error.shape)}."
        ) from exc


def apply_feature_mask(value: torch.Tensor, conditioning: Mapping, fill: float = 0.0) -> torch.Tensor:
    mask = feature_mask_for(value, conditioning)
    if mask is None:
        return value
    return torch.where(mask, value, torch.as_tensor(fill, device=value.device, dtype=value.dtype))


__all__ = ["feature_mask_for", "apply_feature_mask"]
