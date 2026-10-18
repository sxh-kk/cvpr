"""Parameter-free summaries of the VLM tokens supplied to the action expert."""

from dataclasses import dataclass
from typing import Optional

import torch


def masked_mean_vlm_tokens(
    tokens: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Pool only N in [B, V, N, D], preserving selected VLM layers and gradients.

    The mask is [B, N], with 1/True for valid observation-prefix tokens. None
    means all tokens are valid. Accumulation uses FP32 (FP64 for FP64 inputs),
    and the result [B, V, D] has the input dtype. Padding values are discarded
    before reduction, including nonfinite padding. Empty prefixes are errors.
    """
    if tokens.ndim != 4 or not tokens.is_floating_point():
        raise ValueError("VLM tokens must be floating-point [B, V, N, D]")
    batch, views, length, width = tokens.shape
    if min(batch, views, length, width) == 0:
        raise ValueError("VLM token dimensions must be nonempty")
    if attention_mask is None:
        valid = torch.ones((batch, length), dtype=torch.bool, device=tokens.device)
    else:
        if attention_mask.shape != (batch, length):
            raise ValueError("VLM attention_mask must have shape [B, N]")
        if attention_mask.device != tokens.device:
            raise ValueError("VLM tokens and attention_mask must be on the same device")
        if not torch.all((attention_mask == 0) | (attention_mask == 1)):
            raise ValueError("VLM attention_mask must contain only 0/1 or bool values")
        valid = attention_mask.bool()
    counts = valid.sum(dim=-1)
    if torch.any(counts == 0):
        raise ValueError("Every VLM prefix must contain at least one valid token")
    values = tokens.masked_fill(~valid[:, None, :, None], 0)
    accumulation_dtype = torch.float64 if tokens.dtype == torch.float64 else torch.float32
    pooled = values.sum(dim=2, dtype=accumulation_dtype) / counts[:, None, None]
    return pooled.to(dtype=tokens.dtype)


@dataclass(frozen=True)
class VLMFeatures:
    """One VLM pass: unchanged action context plus a separate pooling branch.

    tokens: [B, V, N, D], exactly the existing action-header views.
    pooled_tokens: [B, V, D], one summary per selected VLM layer.
    last_hidden_state: original last-layer features, retained for existing logs.
    """

    tokens: torch.Tensor
    pooled_tokens: torch.Tensor
    attention_mask: Optional[torch.Tensor]
    last_hidden_state: torch.Tensor

    @property
    def p_v(self) -> torch.Tensor:
        """[B, D] by default; [B, V, D] for explicit layerwise conditioning."""
        if self.pooled_tokens.shape[1] == 1:
            return self.pooled_tokens[:, 0]
        return self.pooled_tokens
