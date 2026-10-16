import math
import torch
from torch import nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MoEFeedForward(nn.Module):
    """Chunk-routed MoE FFN with an always-active per-layer shared expert.

    ``num_experts`` denotes *routed* experts.  When ``shared_expert=True``
    (the Motion Expert configuration), every layer owns one independent
    ``shared`` FFN plus ``num_experts`` routed FFNs.  Routing decisions are
    made once per temporal chunk and broadcast to its frames.  The router can
    consume motion, pose, egovideo and flow-time hidden features without
    changing the ``[B, T, D] -> [B, T, D]`` FFN interface.

    The dispatch path intentionally uses ordinary PyTorch indexing.  This is
    slower than a fused MoE kernel, but keeps the experiment reproducible and
    compatible with the existing DDP setup.
    """

    def __init__(
        self,
        dim,
        hidden_dim,
        dropout,
        num_experts=11,
        top_k=2,
        router_jitter=0.0,
        shared_expert=True,
        chunk_size=4,
        router_conditioned=True,
        routed_gate_init=0.05,
    ):
        super().__init__()
        if num_experts < 2:
            raise ValueError(f"MoE requires at least two experts, got {num_experts}.")
        if top_k < 1 or top_k > num_experts:
            raise ValueError(f"top_k must be in [1, {num_experts}], got {top_k}.")
        self.dim = int(dim)
        self.hidden_dim = int(hidden_dim)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.router_jitter = float(router_jitter)
        self.shared_expert_enabled = bool(shared_expert)
        self.chunk_size = int(chunk_size)
        self.router_conditioned = bool(router_conditioned and self.shared_expert_enabled)
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}.")

        # The conditional router receives four D-dimensional features:
        # current motion hidden, pose input hidden, egovideo hidden and the
        # broadcast flow-time embedding.  Direct logits keep this adapter
        # small (~0.4M parameters for all 12 layers) and preserve the 400M
        # parameter target.
        router_input_dim = self.dim * 4 if self.router_conditioned else self.dim
        self.router = nn.Linear(router_input_dim, self.num_experts)
        self.shared = FeedForward(self.dim, self.hidden_dim, dropout) if self.shared_expert_enabled else None
        self.routed_experts = nn.ModuleList(
            [FeedForward(self.dim, self.hidden_dim, dropout) for _ in range(self.num_experts)]
        )
        # A small initial routed contribution lets the copied dense FFN in the
        # shared branch preserve the original K12 behavior during warm-up.
        gate_logit = math.log(float(routed_gate_init) / max(1.0 - float(routed_gate_init), 1e-6))
        self.routed_gate = nn.Parameter(torch.tensor(gate_logit))
        self.last_aux_loss = None
        self.last_router_probs = None
        self.last_top_indices = None

    @staticmethod
    def _masked_mean(x, mask):
        if mask is None:
            return x.mean(dim=1)
        weights = mask.to(dtype=x.dtype).unsqueeze(-1)
        return (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def _chunk_mean(self, x, mask):
        """Return ``[B, ceil(T/C), D]`` masked means."""
        b, t, d = x.shape
        n_chunks = (t + self.chunk_size - 1) // self.chunk_size
        padded_t = n_chunks * self.chunk_size
        if padded_t != t:
            x = F.pad(x, (0, 0, 0, padded_t - t))
            if mask is not None:
                mask = F.pad(mask, (0, padded_t - t), value=False)
        x = x.view(b, n_chunks, self.chunk_size, d)
        if mask is None:
            return x.mean(dim=2)
        mask = mask.view(b, n_chunks, self.chunk_size).to(dtype=x.dtype)
        return (x * mask.unsqueeze(-1)).sum(dim=2) / mask.sum(dim=2, keepdim=True).clamp_min(1.0)

    def _router_logits(self, x, router_conditions=None, frame_mask=None, num_prefix_tokens=1):
        """Build one router logit vector per prefix/frame token."""
        if type(num_prefix_tokens) is not int or num_prefix_tokens < 0:
            raise ValueError("num_prefix_tokens must be a non-negative integer")
        frame_x = x[:, num_prefix_tokens:]
        if frame_mask is None and router_conditions is not None:
            frame_mask = router_conditions.get("valid_frames")

        if not self.router_conditioned:
            return self.router(x)

        if router_conditions is None:
            raise ValueError("Conditional MoE routing requires router_conditions.")
        if x.shape[1] != router_conditions["pose"].shape[1] + num_prefix_tokens:
            raise ValueError("Motion router length must equal frame count plus prefix count")
        pose = router_conditions["pose"]
        ego = router_conditions["ego"]
        timestep = router_conditions["timestep"]
        if timestep.ndim == 2:
            timestep = timestep[:, None, :].expand(-1, frame_x.shape[1], -1)
        motion = frame_x
        expected = frame_x.shape[:2]
        features = []
        for name, feature in (("motion", motion), ("pose", pose), ("ego", ego), ("timestep", timestep)):
            if feature.ndim != 3 or feature.shape[:2] != expected:
                raise ValueError(
                    f"Router condition {name!r} must be [B,T,D] aligned to {tuple(expected)}, "
                    f"got {tuple(feature.shape)}."
                )
            features.append(feature)
        # Video pooling uses visible observations independently of generated
        # future motion, so hidden frames neither leak nor dilute that summary.
        ego_mask = router_conditions.get("ego_valid_frames", frame_mask)
        if ego_mask is not None:
            if ego_mask.shape != expected:
                raise ValueError("ego_valid_frames must be [B,T] aligned to the router")
            if frame_mask is not None:
                ego_mask = ego_mask.bool() & frame_mask.bool()
        feature_masks = (frame_mask, frame_mask, ego_mask, frame_mask)
        chunk_condition = torch.cat([
            self._chunk_mean(feature, feature_mask)
            for feature, feature_mask in zip(features, feature_masks)
        ], dim=-1)
        chunk_logits = self.router(chunk_condition)
        frame_logits = chunk_logits.repeat_interleave(self.chunk_size, dim=1)[:, : frame_x.shape[1]]

        # Prefixes share sequence-level routing without occupying frame chunks.
        prefix_condition = torch.cat([
            self._masked_mean(feature, feature_mask)
            for feature, feature_mask in zip(features, feature_masks)
        ], dim=-1)
        prefix_logits = self.router(prefix_condition.unsqueeze(1))
        if num_prefix_tokens:
            return torch.cat((prefix_logits.expand(-1, num_prefix_tokens, -1), frame_logits), dim=1)
        return frame_logits

    def forward(self, x, router_conditions=None, frame_mask=None, num_prefix_tokens=1):
        original_shape = x.shape
        flat_x = x.reshape(-1, self.dim)
        if frame_mask is None and router_conditions is not None:
            frame_mask = router_conditions.get("valid_frames")
        router_logits = self._router_logits(x, router_conditions=router_conditions, frame_mask=frame_mask,
                                           num_prefix_tokens=num_prefix_tokens)
        if self.training and self.router_jitter > 0:
            router_logits = router_logits + torch.randn_like(router_logits) * self.router_jitter
        router_probs = torch.softmax(router_logits, dim=-1)
        top_values, top_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_weights = top_values / top_values.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        effective_top_k = top_indices.shape[-1]
        flat_top_indices = top_indices.reshape(-1, effective_top_k)
        flat_top_weights = top_weights.reshape(-1, effective_top_k)

        flat_output = torch.zeros_like(flat_x)
        # Each selected token is evaluated only by the experts assigned to it.
        # The empty branch is valid for small batches and simply contributes no
        # output; DDP is configured with find_unused_parameters for this mode.
        for expert_id, expert in enumerate(self.routed_experts):
            token_index, slot_index = torch.where(flat_top_indices == expert_id)
            if token_index.numel() == 0:
                continue
            expert_output = expert(flat_x.index_select(0, token_index))
            expert_weight = flat_top_weights[token_index, slot_index].unsqueeze(-1)
            # AMP can keep router weights in float32 with BF16 activations.
            # Cast at accumulation without detaching the routing gradients.
            weighted_output = (expert_output * expert_weight).to(dtype=flat_output.dtype)
            flat_output.index_add_(0, token_index, weighted_output)

        # Switch-style load-balancing loss.  The loss is exposed to the outer
        # UEM module and added there, so it is not mixed into the motion output.
        if frame_mask is not None:
            if router_probs.shape[1] != frame_mask.shape[1] + num_prefix_tokens:
                raise ValueError("MoE token mask does not match the prefix/frame lengths")
            token_mask = torch.cat(
                (frame_mask.new_ones((frame_mask.shape[0], num_prefix_tokens)), frame_mask),
                dim=1,
            )
        else:
            token_mask = frame_mask
        if token_mask is None:
            token_mask = torch.ones(router_probs.shape[:2], device=router_probs.device, dtype=torch.bool)
        token_mask = token_mask.reshape(-1).to(dtype=router_probs.dtype)
        denom = token_mask.sum().clamp_min(1.0)
        top1_load = torch.nn.functional.one_hot(
            top_indices[..., 0], num_classes=self.num_experts
        ).to(router_probs.dtype).reshape(-1, self.num_experts)
        top1_load = (top1_load * token_mask.unsqueeze(-1)).sum(dim=0) / denom
        mean_router_prob = (router_probs.reshape(-1, self.num_experts) * token_mask.unsqueeze(-1)).sum(dim=0) / denom
        self.last_aux_loss = self.num_experts * torch.sum(top1_load * mean_router_prob)
        self.last_router_probs = router_probs.detach()
        self.last_top_indices = top_indices.detach()
        if self.shared is not None:
            flat_output = self.shared(flat_x) + torch.sigmoid(self.routed_gate) * flat_output
        return flat_output.reshape(original_shape)


class MoEDecoderBlock(nn.Module):
    """Decoder block equivalent to DecoderBlock with an MoE FFN."""

    def __init__(self, dim, heads, dropout, ff_mult, num_experts=11, top_k=2, router_jitter=0.0, **moe_kwargs):
        super().__init__()
        self.attn1 = Attention(dim, heads, dropout)
        self.attn2 = Attention(dim, heads, dropout, cross_attention=True)
        self.ff = MoEFeedForward(
            dim,
            dim * ff_mult,
            dropout,
            num_experts=num_experts,
            top_k=top_k,
            router_jitter=router_jitter,
            **moe_kwargs,
        )

    def forward(self, x, context, mask=None, context_mask=None, router_conditions=None, num_prefix_tokens=1):
        x = self.attn1(x, kv_x=x, mask=mask) + x
        x = self.attn2(x, kv_x=context, mask=context_mask) + x
        frame_mask = None if router_conditions is None else router_conditions.get("valid_frames")
        return self.ff(x, router_conditions=router_conditions, frame_mask=frame_mask,
                       num_prefix_tokens=num_prefix_tokens) + x


class Attention(nn.Module):
    """Pre-normalized Q/K/V with an independent cross-attention context norm."""

    def __init__(self, dim, heads, dropout, *, cross_attention=False):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.cross_attention = cross_attention
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x, kv_x, mask=None):
        # mask B x 1 x 1 x N denotes which elements of kv_x are valid

        query = self.norm(x)
        if self.cross_attention:
            # Visual/time context has a different distribution from the motion
            # residual. Normalize it independently, without sharing the query's
            # learned affine transform or introducing ambiguous old weights.
            keys = F.layer_norm(kv_x, self.norm.normalized_shape, eps=self.norm.eps)
        else:
            # All three projections consume the same normalized self-attention
            # state; raw residual magnitude must not enter K/V.
            keys = query
        if mask is not None:
            mask = mask.squeeze([1, 2]).bool().logical_not()
        out = self.attn(query, keys, keys, key_padding_mask=mask, need_weights=False)[0]
        out = self.out_dropout(out)
        return out
