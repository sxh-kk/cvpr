"""Action residual blocks conditioned on VLM and motion memories with autograd.

This is a new architecture, not a numerically equivalent rewrite of Psi0's
joint attention. Its tensor-only checkpoint signature rejects legacy heads,
including when an enclosing model is loaded with ``strict=False``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from types import SimpleNamespace
from typing import Any, Mapping, Optional

import torch
from torch import nn
from torch.nn import functional as F


ARCHITECTURE_ID = "psi0_dual_source_sa_vlm_motion_ffn_adaln12_v1"
MOTION_TOKEN_COUNT = 11
MOTION_FEATURE_DIM = 768


@dataclass
class DualSourceActionOutput:
    action: torch.Tensor

    def to_tuple(self):
        return (None, self.action)


class _RMSNorm(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        dtype = torch.float64 if value.dtype == torch.float64 else torch.float32
        normed = value.to(dtype)
        normed = normed * torch.rsqrt(normed.square().mean(-1, keepdim=True) + 1e-6)
        return normed.to(value.dtype) * self.weight.to(value.dtype)


class _Attention(nn.Module):
    def __init__(self, width: int, heads: int, qk_norm: Optional[str], dropout: float):
        super().__init__()
        self.heads = heads
        self.head_dim = width // heads
        self.dropout = dropout
        self.to_q = nn.Linear(width, width)
        self.to_k = nn.Linear(width, width)
        self.to_v = nn.Linear(width, width)
        self.to_out = nn.Sequential(nn.Linear(width, width), nn.Dropout(dropout))
        if qk_norm == "rms_norm":
            self.norm_q = _RMSNorm(self.head_dim)
            self.norm_k = _RMSNorm(self.head_dim)
        elif qk_norm == "layer_norm":
            self.norm_q = nn.LayerNorm(self.head_dim, eps=1e-6)
            self.norm_k = nn.LayerNorm(self.head_dim, eps=1e-6)
        else:
            self.norm_q = nn.Identity()
            self.norm_k = nn.Identity()

    def forward(self, queries, memory=None, memory_mask=None):
        memory = queries if memory is None else memory
        batch, query_count, width = queries.shape

        def split(value):
            return value.reshape(batch, -1, self.heads, self.head_dim).transpose(1, 2)

        query = self.norm_q(split(self.to_q(queries)))
        key = self.norm_k(split(self.to_k(memory)))
        value = split(self.to_v(memory))
        mask = None if memory_mask is None else memory_mask[:, None, None, :]
        result = F.scaled_dot_product_attention(
            query, key, value, attn_mask=mask, is_causal=False,
            dropout_p=self.dropout if self.training else 0.0,
        )
        result = result.transpose(1, 2).reshape(batch, query_count, width)
        return self.to_out(result)


class DualSourceActionBlock(nn.Module):
    """SA -> CA(VLM) -> CA(Motion) -> FFN, each with its own AdaLN and gate."""

    def __init__(self, width: int, heads: int, qk_norm: Optional[str], dropout: float):
        super().__init__()
        self.norms = nn.ModuleList([
            nn.LayerNorm(width, elementwise_affine=False, eps=1e-6) for _ in range(4)
        ])
        self.self_attn = _Attention(width, heads, qk_norm, dropout)
        self.vlm_attn = _Attention(width, heads, qk_norm, dropout)
        self.motion_attn = _Attention(width, heads, qk_norm, dropout)
        self.ff_act = nn.Sequential(
            nn.Linear(width, width * 4), nn.GELU(approximate="tanh"), nn.Dropout(dropout),
            nn.Linear(width * 4, width), nn.Dropout(dropout),
        )
        self.mod_head = nn.Sequential(nn.SiLU(), nn.Linear(width, 12 * width))
        # Only modulation is zeroed. A simultaneously zero branch output and
        # zero gate would prevent both sides of that residual from learning.
        nn.init.zeros_(self.mod_head[-1].weight)
        nn.init.zeros_(self.mod_head[-1].bias)

    def forward(self, action, condition, vlm_memory, motion_memory, vlm_mask):
        modulations = self.mod_head(condition).chunk(12, dim=-1)
        for index, branch in enumerate((self.self_attn, self.vlm_attn, self.motion_attn, self.ff_act)):
            gamma, beta, gate = modulations[index * 3:index * 3 + 3]
            normalized = self.norms[index](action) * (1 + gamma) + beta
            if index == 0:
                update = branch(normalized)
            elif index == 1:
                update = branch(normalized, vlm_memory, vlm_mask)
            elif index == 2:
                update = branch(normalized, motion_memory)
            else:
                update = branch(normalized)
            action = action + gate * update
        return action


class _TimeEmbedding(nn.Module):
    def __init__(self, time_dim: int, width: int):
        super().__init__()
        self.time_dim = time_dim
        self.register_buffer("frequencies", self._frequencies(time_dim, device=None, dtype=torch.float32))
        self.out_net = nn.Sequential(nn.Linear(time_dim, width), nn.SiLU(), nn.Linear(width, width))

    @staticmethod
    def _frequencies(time_dim, *, device, dtype):
        return torch.exp(torch.arange(time_dim // 2, device=device, dtype=dtype)
                         * (-math.log(10000) / (time_dim // 2 - 1)))

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        # Module.to(bfloat16/float16) also casts floating buffers. Recompute
        # instead of upcasting the rounded frequencies: near t=1000, BF16
        # phases would collapse distinct flow timesteps into the same value.
        dtype = torch.float64 if self.out_net[0].weight.dtype == torch.float64 else torch.float32
        self.frequencies = self._frequencies(self.time_dim, device=self.frequencies.device, dtype=dtype)
        return self

    def forward(self, time):
        phase = time.to(self.frequencies.dtype)[..., None] * self.frequencies
        encoding = torch.cat((torch.cos(phase), torch.sin(phase)), dim=-1)
        return self.out_net(encoding.to(self.out_net[0].weight.dtype))


class _ActionProjectionIn(nn.Module):
    def __init__(self, action_dim: int, horizon: int, width: int):
        super().__init__()
        self.ac_proj = nn.Sequential(
            nn.Linear(action_dim, action_dim), nn.GELU(approximate="tanh"), nn.Linear(action_dim, width)
        )
        self.dec_pos = nn.Parameter(torch.empty(horizon, width))
        nn.init.xavier_uniform_(self.dec_pos)

    def forward(self, actions):
        return self.ac_proj(actions) + self.dec_pos[None]


class _ActionProjectionOut(nn.Module):
    def __init__(self, width: int, action_dim: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 2 * width))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        self.linear = nn.Linear(width, action_dim)

    def forward(self, actions, condition):
        gamma, beta = self.adaLN_modulation(condition).chunk(2, dim=-1)
        return self.linear(self.norm_final(actions) * (1 + gamma) + beta)


class DualSourceActionTransformer(nn.Module):
    """Pure-Torch action head with a clean proprio state token at position zero.

    ``views`` and ``vlm_features.tokens`` must be the same [B,1,N,Dv] tensor.
    Motion features supply eleven [768]-D tokens and their mean ``p_m``.
    Both memories remain unchanged through the stack. Feature extraction and
    freezing belong to the caller; this module never detaches an input.
    ``pooled_projections`` retains its legacy argument slot but is not consumed:
    the new conditioning uses only p_V, p_M and the action flow timestep.
    """

    architecture_id = ARCHITECTURE_ID

    def __init__(
        self, *, action_dim: int, action_pred_horizon: int, action_hidden_dim: int,
        action_num_blocks: int, action_nheads: int, view_feature_dim: int, odim: int,
        time_dim: int = 256, state_drop_prob: float = 0.0, state_null_token: bool = False,
        qk_norm: Optional[str] = "rms_norm", dropout: float = 0.0,
    ):
        super().__init__()
        dimensions = (action_dim, action_pred_horizon, action_hidden_dim, action_num_blocks,
                      action_nheads, view_feature_dim, odim, time_dim)
        if any(type(value) is not int or value < 1 for value in dimensions):
            raise ValueError("Action, horizon, hidden, block, head, VLM, state and time dimensions must be positive integers")
        if action_hidden_dim % action_nheads:
            raise ValueError("action_hidden_dim must be divisible by action_nheads")
        if time_dim < 4 or time_dim % 2:
            raise ValueError("time_dim must be even and >= 4")
        if qk_norm not in (None, "rms_norm", "layer_norm"):
            raise ValueError("qk_norm must be None, 'rms_norm' or 'layer_norm'")
        if not 0 <= state_drop_prob <= 1 or not 0 <= dropout < 1:
            raise ValueError("state_drop_prob must be in [0,1] and dropout in [0,1)")
        if type(state_null_token) is not bool:
            raise ValueError("state_null_token must be bool")
        if state_drop_prob > 0 and not state_null_token:
            raise ValueError("state_drop_prob > 0 requires state_null_token=True")

        self.action_dim = action_dim
        self.action_pred_horizon = action_pred_horizon
        self.action_hidden_dim = action_hidden_dim
        self.view_feature_dim = view_feature_dim
        self.odim = odim
        self.state_drop_prob = float(state_drop_prob)
        self.last_state_drop_frac = None
        self.config = SimpleNamespace(
            action_dim=action_dim, action_pred_horizon=action_pred_horizon,
            action_hidden_dim=action_hidden_dim, action_num_blocks=action_num_blocks,
            action_nheads=action_nheads, view_feature_dim=view_feature_dim, odim=odim,
            time_dim=time_dim, state_drop_prob=state_drop_prob,
            state_null_token=state_null_token, qk_norm=qk_norm, dropout=dropout,
        )
        width = action_hidden_dim
        self.action_proj_in = _ActionProjectionIn(action_dim, action_pred_horizon, width)
        self.state_proj = nn.Linear(odim, width)
        self.state_pos = nn.Parameter(torch.zeros(1, 1, width))
        self.state_null = nn.Parameter(torch.randn(1, 1, width) * 0.02) if state_null_token else None
        self.time_ins_embed = _TimeEmbedding(time_dim, width)

        def projection(input_width):
            return nn.Sequential(nn.LayerNorm(input_width, eps=1e-6), nn.Linear(input_width, width))

        self.vlm_context_proj = projection(view_feature_dim)
        self.motion_context_proj = projection(MOTION_FEATURE_DIM)
        self.vlm_pool_proj = projection(view_feature_dim)
        self.motion_pool_proj = projection(MOTION_FEATURE_DIM)
        self.time_condition_proj = projection(width)
        self.f_fuse = nn.Sequential(nn.Linear(3 * width, width), nn.SiLU(), nn.Linear(width, width))
        self.transformer_blocks = nn.ModuleList([
            DualSourceActionBlock(width, action_nheads, qk_norm, dropout) for _ in range(action_num_blocks)
        ])
        self.action_proj_out = _ActionProjectionOut(width, action_dim)

        # A persistent Tensor is compatible with safetensors/Accelerate exports.
        def float_bits(value):
            return struct.unpack("q", struct.pack("d", float(value)))[0]

        signature = [0x44534143, 1, *dimensions, int(state_null_token),
                     {None: 0, "rms_norm": 1, "layer_norm": 2}[qk_norm],
                     float_bits(state_drop_prob), float_bits(dropout), MOTION_TOKEN_COUNT, MOTION_FEATURE_DIM]
        self.register_buffer("_architecture_signature", torch.tensor(signature, dtype=torch.int64))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        signature = state_dict.get(prefix + "_architecture_signature")
        if (not isinstance(signature, torch.Tensor) or signature.dtype != torch.int64
                or not torch.equal(signature.detach().cpu(), self._architecture_signature.detach().cpu())):
            error_msgs.append(
                f"{prefix or 'action head'} requires {ARCHITECTURE_ID} with matching configuration; "
                "legacy joint-attention or incompatible dual-source checkpoints cannot be loaded"
            )
        missing = [prefix + key for key in self.state_dict() if prefix + key not in state_dict]
        if missing:
            error_msgs.append("Incomplete dual-source action checkpoint (strict=False is not a warm-start): "
                              + ", ".join(missing[:6]))
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    @staticmethod
    def _check_tensor(value, shape, device, name):
        if (not isinstance(value, torch.Tensor) or not value.is_floating_point()
                or tuple(value.shape) != tuple(shape)):
            raise ValueError(f"{name} must be floating-point {tuple(shape)}")
        if value.device != device:
            raise ValueError(f"{name} must be on the action device")

    @staticmethod
    def _check_mask(mask, batch, length, device):
        if mask is None:
            return torch.ones((batch, length), dtype=torch.bool, device=device)
        if not isinstance(mask, torch.Tensor) or tuple(mask.shape) != (batch, length) or mask.device != device:
            raise ValueError("VLM attention mask must be [B,N] on the action device")
        if not torch.all((mask == 0) | (mask == 1)) or torch.any(mask.bool().sum(-1) == 0):
            raise ValueError("VLM attention mask must be binary with a nonempty prefix per sample")
        return mask.bool()

    def _conditioning(self, timestep, p_v, p_m, batch, device):
        if not isinstance(timestep, torch.Tensor) or timestep.dtype == torch.bool:
            raise ValueError("timestep must be a numeric Tensor [B] or [B,T]")
        if timestep.device != device or tuple(timestep.shape) not in ((batch,), (batch, self.action_pred_horizon)):
            raise ValueError("timestep must be [B] or [B,T] on the action device")
        if not torch.isfinite(timestep).all() or torch.any(timestep < 0):
            raise ValueError("Action timesteps must be finite and nonnegative")
        times = timestep[:, None].expand(-1, self.action_pred_horizon) if timestep.ndim == 1 else timestep
        times = torch.cat((torch.zeros((batch, 1), device=device, dtype=times.dtype), times), dim=1)
        dtype = self.state_proj.weight.dtype
        v = self.vlm_pool_proj(p_v.to(dtype))[:, None].expand(-1, times.shape[1], -1)
        m = self.motion_pool_proj(p_m.to(dtype))[:, None].expand(-1, times.shape[1], -1)
        t = self.time_condition_proj(self.time_ins_embed(times))
        return self.f_fuse(torch.cat((v, m, t), dim=-1))

    def forward(
        self, hidden_states=None, encoder_hidden_states=None, pooled_projections=None,
        timestep=None, block_controlnet_hidden_states=None, joint_attention_kwargs: Optional[Mapping[str, Any]] = None,
        vlm_attn_mask=None, return_dict=True, skip_layers=None, **kwargs,
    ) -> DualSourceActionOutput:
        if not isinstance(joint_attention_kwargs, Mapping):
            raise ValueError("joint_attention_kwargs must supply actions, VLM, motion and state features")
        context = joint_attention_kwargs
        actions = context.get("action_hidden_embeds")
        if not isinstance(actions, torch.Tensor) or actions.ndim != 3:
            raise ValueError("action_hidden_embeds must be [B,T,Da]")
        batch = actions.shape[0]
        if batch == 0:
            raise ValueError("The action batch must be nonempty")
        device = actions.device
        self._check_tensor(actions, (batch, self.action_pred_horizon, self.action_dim), device, "action_hidden_embeds")
        if device != self.state_proj.weight.device:
            raise ValueError("Action inputs and head parameters must be on the same device")
        if context.get("traj2ds") is not None:
            raise ValueError("Dual-source action heads do not consume legacy traj2ds conditions")

        views = context.get("views")
        if not isinstance(views, torch.Tensor) or views.ndim != 4 or views.shape[1] != 1 or views.shape[2] < 1:
            raise ValueError("views must be [B,1,N,Dv] with a nonempty prefix; layerwise VLM fusion is unsupported")
        length = views.shape[2]
        self._check_tensor(views, (batch, 1, length, self.view_feature_dim), device, "views")
        vlm = context.get("vlm_features")
        motion = context.get("motion_features")
        if vlm is None or getattr(vlm, "tokens", None) is not views:
            raise ValueError("vlm_features.tokens must be the exact views supplied to the action head")
        feature_mask = getattr(vlm, "attention_mask", None)
        mask = self._check_mask(feature_mask, batch, length, device)
        if vlm_attn_mask is not None:
            supplied = self._check_mask(vlm_attn_mask, batch, length, device)
            if not torch.equal(mask, supplied):
                raise ValueError("vlm_attn_mask must match the mask used to pool vlm_features")
        p_v = getattr(vlm, "p_v", None)
        self._check_tensor(p_v, (batch, self.view_feature_dim), device, "vlm_features.p_v")
        motion_tokens = getattr(motion, "tokens", None)
        p_m = getattr(motion, "p_m", None)
        self._check_tensor(motion_tokens, (batch, MOTION_TOKEN_COUNT, MOTION_FEATURE_DIM), device, "motion_features.tokens")
        self._check_tensor(p_m, (batch, MOTION_FEATURE_DIM), device, "motion_features.p_m")

        skipped = set() if skip_layers is None else set(skip_layers)
        if any(type(index) is not int or index < 0 or index >= len(self.transformer_blocks) for index in skipped):
            raise ValueError("skip_layers must contain valid block indices")
        dtype = self.state_proj.weight.dtype
        tokens = self.action_proj_in(actions.to(dtype))
        obs = context.get("obs")
        if obs is None:
            if self.state_null is None:
                raise ValueError("obs=None requires state_null_token=True")
            state = tokens.new_zeros((batch, self.odim))
            keep = torch.zeros(batch, dtype=torch.bool, device=device)
        else:
            if not isinstance(obs, torch.Tensor) or obs.ndim != 3 or obs.shape[1] < 1:
                raise ValueError("obs must be [B,To,odim] with To >= 1, or None for explicit missing state")
            self._check_tensor(obs, (batch, obs.shape[1], self.odim), device, "obs")
            keep = torch.ones(batch, dtype=torch.bool, device=device)
            if self.training and self.state_drop_prob:
                keep = torch.rand(batch, device=device) >= self.state_drop_prob
            state = obs[:, -1].to(dtype).masked_fill(~keep[:, None], 0)
        state_token = self.state_proj(state)[:, None]
        if self.state_null is not None:
            # Even when no sample is dropped, keep null in the graph with a zero
            # gradient so DDP need not enable find_unused_parameters.
            state_token = torch.where(keep[:, None, None], state_token, self.state_null.to(state_token.dtype))
        self.last_state_drop_frac = float((~keep).float().mean())
        tokens = torch.cat((state_token + self.state_pos, tokens), dim=1)

        condition = self._conditioning(timestep, p_v, p_m, batch, device)
        clean_views = views[:, 0].masked_fill(~mask[:, :, None], 0).to(dtype)
        vlm_memory = self.vlm_context_proj(clean_views)
        motion_memory = self.motion_context_proj(motion_tokens.to(dtype))
        for index, block in enumerate(self.transformer_blocks):
            if index not in skipped:
                tokens = block(tokens, condition, vlm_memory, motion_memory, mask)
        # State is a conditioning token, never a supervised/predicted action.
        return DualSourceActionOutput(action=self.action_proj_out(tokens[:, 1:], condition[:, 1:]))
