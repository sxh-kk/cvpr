"""E7 K12 400M with shared/top-2 routed experts and CLIP cross-attention."""

import math
import numpy as np
import torch
import torch.nn as nn
from ..config import validate_architecture
from .core import MoEDecoderBlock


class TextCrossAttention(nn.Module):
    """A shared MotionLab-style text cross-attention adapter.

    The residual projection is zero initialized, so adding this module does
    not perturb a loaded E7 checkpoint before text-specific learning begins.
    """

    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.to_out = nn.Linear(dim, dim)
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, x, text_tokens, text_mask, text_valid):
        if text_tokens is None:
            return torch.zeros_like(x)
        query = self.norm(x)
        text_mask = text_mask.bool()
        text_valid = text_valid.bool().view(-1)
        # An all-masked row produces NaNs in MultiheadAttention.  Invalid
        # samples instead attend to zero tokens and are removed after the
        # zero-initialized output projection.
        safe_mask = text_mask.clone()
        safe_mask[~text_valid] = True
        safe_mask[~text_valid, 0] = False
        safe_mask[text_valid, 0] = True
        tokens = text_tokens.to(dtype=query.dtype)
        tokens = tokens * text_valid[:, None, None].to(dtype=tokens.dtype)
        key_padding_mask = ~safe_mask
        attended = self.attn(
            query,
            tokens,
            tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        output = self.to_out(attended)
        return output * text_valid[:, None, None].to(dtype=output.dtype)


def mask_it(mask, cond, replace_token):
    # mask: B bool
    # cond: B x ... x D
    # replace_token: D
    expand_dims = [1] * (len(cond.shape) - len(mask.shape))
    mask = mask.view(*mask.shape, *expand_dims).to(dtype=cond.dtype)  # B x ... x 1

    expand_dims = [1] * (len(cond.shape) - 1)
    mask_token = replace_token.view(*expand_dims, -1)  # 1 x ... x D
    cond = cond * (1.0 - mask) + mask * mask_token
    return cond


class UniEgoMotion(nn.Module):
    """Checkpoint-compatible E7 K12 400M text-conditioned Motion Expert."""

    def __init__(self, cfg, dropout=0.1):
        super().__init__()
        validate_architecture(cfg)
        self.cfg = cfg
        self.latent_dim = 768
        self.input_feats = 153
        self.num_layers = self.num_heads = 12
        self.text_feature_dim = 768
        self.text_token_count = 16
        self.text_dropout = float(cfg.DATA.TEXT_DROPOUT)
        self.text_condition_enabled = self.motion_expert_enabled = True
        self.text_global_proj = self.text_router_proj = None
        self.sparse_joint_indices = tuple(cfg.SPARSE_JOINTS.INDICES)
        self.global_feature_start, self.global_feature_end = 108, 117
        self.cond_mask_prob = {"traj": 0.5, "clip": 0.1, "subseq_frames": 0.5}
        self.input_process = nn.Linear(153, 768)
        self.pos_enc = PositionalEncoding(768, dropout)
        self.embed_timestep = TimestepEmbedder(768, self.pos_enc)
        self.embed_traj_cond = nn.Linear(18, 768)
        self.embed_clip_cond = nn.Linear(1024, 768)
        # Unused tensors stay frozen for strict compatibility with original E7 weights.
        self.embed_text_cond = nn.Linear(768, 768).requires_grad_(False)
        self.text_token_proj = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 768))
        nn.init.eye_(self.text_token_proj[-1].weight)
        nn.init.zeros_(self.text_token_proj[-1].bias)
        self.mask_tokens = nn.ParameterDict({
            name: nn.Parameter(torch.randn(768) * 0.05) for name in ("traj", "clip", "text")
        })
        self.mask_tokens["text"].requires_grad_(False)
        expert = cfg.MODEL.MOTION_EXPERT
        self.tsfm = nn.ModuleList([
            MoEDecoderBlock(768, 12, dropout, 2, num_experts=11, top_k=2,
                            router_jitter=float(expert.ROUTER_JITTER), shared_expert=True,
                            chunk_size=4, router_conditioned=True,
                            routed_gate_init=float(expert.ROUTED_GATE_INIT))
            for _ in range(12)
        ])
        self.text_cross_attention = TextCrossAttention(768, 12, dropout)
        self.text_cross_gates = nn.Parameter(torch.ones(12))
        self.output_process = nn.Linear(768, 153)
        self._last_moe_aux_loss = None
        self.num_prefix_tokens = 1 + cfg.MODEL.GLOBAL_TOKEN_COUNT
        if cfg.MODEL.GLOBAL_TOKEN_COUNT:
            # Store one token as [1,D]; the leading token axis is added when
            # expanding over a batch. This keeps the DDP gradient contiguous.
            # Store the single token as a vector so DDP receives a contiguous gradient.
            self.global_token = nn.Parameter(torch.randn(768) * 0.02)
        else:
            self.register_parameter("global_token", None)

    def get_moe_auxiliary_loss(self):
        return self._last_moe_aux_loss

    def mask_cond(self, cond, cond_type, cond_mask=None):
        # cond: B x T x ... x D
        # for inference of recon, fore and gen, masks are already provided.
        if cond_mask is not None:
            cond = mask_it(cond_mask, cond, self.mask_tokens[cond_type])
            return cond

        if not self.training:
            # technically, this should never happen because we always do inference with cond_mask.
            return cond
        inp_shape = cond.shape

        # During training, mask whole condition randomly based on probability.
        # To simulate generation, we additionally mask subsequent frames (after first frame) with some probability.
        # Overall, this will simulate recon and gen tasks with some probability.

        mask = torch.rand(cond.shape[0], device=cond.device) < self.cond_mask_prob[cond_type]  # B
        cond = mask_it(mask, cond, self.mask_tokens[cond_type])

        assert cond_type in ["traj", "clip"]

        # Mask condition of subsequent frames using probability. For trajectory, we also mask the first frame.
        mask = torch.rand(cond.shape[0], device=cond.device) < self.cond_mask_prob["subseq_frames"]  # B
        if cond_type == "traj":
            cond = mask_it(mask, cond, self.mask_tokens[cond_type])
        else:
            subsequent_cond = mask_it(mask, cond[:, 1:], self.mask_tokens[cond_type])
            cond = torch.cat((cond[:, :1], subsequent_cond), axis=1)

        assert cond.shape == inp_shape
        return cond

    def _prepare_text_condition(self, y, batch_size, dtype, device):
        if not self.text_condition_enabled or "text_pooled" not in y or "text_tokens" not in y:
            return None
        pooled = torch.as_tensor(y["text_pooled"], device=device).to(dtype=dtype)
        tokens = torch.as_tensor(y["text_tokens"], device=device).to(dtype=dtype)
        text_mask = torch.as_tensor(
            y.get("text_mask", torch.ones(tokens.shape[:2], device=device)),
            device=device,
            dtype=torch.bool,
        )
        text_valid = torch.as_tensor(
            y.get("text_valid", torch.ones(batch_size, device=device)),
            device=device,
            dtype=torch.bool,
        ).view(-1)
        if pooled.shape != (batch_size, self.text_feature_dim):
            raise ValueError(
                f"text_pooled must have shape ({batch_size}, {self.text_feature_dim}), "
                f"got {tuple(pooled.shape)}"
            )
        if tokens.ndim != 3 or tokens.shape[0] != batch_size or tokens.shape[2] != self.text_feature_dim:
            raise ValueError(
                "text_tokens must have shape [B,L,text_feature_dim], "
                f"got {tuple(tokens.shape)}"
            )
        if text_mask.shape != tokens.shape[:2] or text_valid.shape != (batch_size,):
            raise ValueError(
                f"text mask/valid shapes are incompatible: mask={tuple(text_mask.shape)}, "
                f"valid={tuple(text_valid.shape)}, tokens={tuple(tokens.shape)}"
            )
        if self.training and self.text_dropout > 0:
            keep = torch.rand(batch_size, device=device) >= self.text_dropout
            text_valid = text_valid & keep
        pooled = pooled * text_valid[:, None].to(dtype=pooled.dtype)
        tokens = tokens * text_valid[:, None, None].to(dtype=tokens.dtype)
        text_mask = text_mask & text_valid[:, None]
        # Guarantee at least one safe key for every row in the attention
        # adapter, including rows whose sidecar has no annotation.
        text_mask[:, 0] = text_mask[:, 0] | text_valid
        return pooled, tokens, text_mask, text_valid

    def forward(self, x, timesteps, y, cond_scale=None, return_hidden=False, return_features=False):
        """
        x_t: B x T x F
        timesteps: B
        y: dict
        """
        if return_hidden and return_features:
            raise ValueError("return_hidden and return_features are mutually exclusive")
        if return_features and self.global_token is None:
            raise ValueError("Global features require a model with GLOBAL_TOKEN_COUNT=1")
        if cond_scale is not None:
            if not math.isfinite(cond_scale):
                raise ValueError("cond_scale must be finite")
            x_cond = self.forward(x, timesteps, y, return_hidden=return_hidden,
                                  return_features=return_features)
            uncond_y = {"valid_frames": y["valid_frames"]}
            if "feature_valid_mask" in y:
                uncond_y["feature_valid_mask"] = y["feature_valid_mask"]
            x_uncond = self.forward(x, timesteps, uncond_y, return_hidden=return_hidden,
                                    return_features=return_features)

            # Apply classifier-free guidance to predictions and hidden tokens.
            if return_hidden or return_features:
                return tuple(uncond + (cond - uncond) * cond_scale
                             for cond, uncond in zip(x_cond, x_uncond))
            return x_uncond + (x_cond - x_uncond) * cond_scale

        B, T, F = x.shape
        x = self.input_process(x)  # B x T x D
        x = self.pos_enc(x)  # B x T x D

        traj_mask = y["traj_mask"] if "traj_mask" in y else None
        img_mask = y["img_mask"] if "img_mask" in y else None

        enc_time = self.embed_timestep(timesteps)  # B x D
        text_condition = self._prepare_text_condition(y, B, x.dtype, x.device)
        text_global = None
        text_tokens = None
        text_mask = None
        text_valid = None
        router_text = None
        if text_condition is not None:
            text_pooled, text_tokens_raw, text_mask, text_valid = text_condition
            if self.text_global_proj is not None:
                text_global = self.text_global_proj(text_pooled)
            if self.text_token_proj is not None:
                text_tokens = self.text_token_proj(text_tokens_raw)
            if self.text_router_proj is not None:
                router_text = self.text_router_proj(text_pooled)

        pose_router_cond = x
        if "traj" in y:  # B x T x F
            enc_traj = self.embed_traj_cond(y["traj"])  # B x T x D
            enc_traj = self.mask_cond(enc_traj, "traj", traj_mask)
            x = x + enc_traj
        else:
            x = x + self.mask_tokens["traj"]

        all_cond = [enc_time[:, None]]  # B x 1 x D
        all_cond_mask = [torch.ones(B, 1, dtype=torch.long, device=x.device)]  # B x 1
        if "img_embs" in y:
            enc_imgs = self.embed_clip_cond(y["img_embs"])
            enc_imgs = self.mask_cond(enc_imgs, "clip", img_mask)
            enc_imgs = self.pos_enc(enc_imgs)

            enc_img_mask = y["valid_img_embs"]  # B x C

            # Keep a frame-aligned egovideo condition for the chunk router
            # before flattening DINO register tokens into the attention context.
            router_ego_cond = enc_imgs.mean(dim=2) if enc_imgs.ndim == 4 else enc_imgs
            if router_ego_cond.shape[1] != T:
                router_ego_cond = router_ego_cond.mean(dim=1, keepdim=True).expand(-1, T, -1)

            all_cond.append(enc_imgs)
            all_cond_mask.append(enc_img_mask)
        else:  # if self.cond_img_feat:  # Model was trained with img feats. Use mask tokens here.
            enc_imgs = self.mask_tokens["clip"].view(1, 1, -1).repeat(B, T, 1)
            enc_imgs = self.pos_enc(enc_imgs)
            enc_img_mask = y["valid_frames"]
            router_ego_cond = enc_imgs

            all_cond.append(enc_imgs)
            all_cond_mask.append(enc_img_mask)

        prefixes = [enc_time[:, None]]
        if self.global_token is not None:
            prefixes.append(self.global_token.to(dtype=x.dtype).view(1, 1, -1).expand(B, -1, -1).contiguous())
        x = torch.cat((*prefixes, x), dim=1)
        mask = y["valid_frames"]  # B x T where valid are 1
        mask = torch.cat((mask.new_ones((B, self.num_prefix_tokens)), mask), dim=1)
        mask = mask[:, None, None, :]

        context_mask = torch.cat(all_cond_mask, dim=1)  # B x C
        context_mask = context_mask[:, None, None, :]  # B x 1 x 1 x C
        context = torch.cat(all_cond, dim=1)  # B x C x D

        router_conditions = None
        if self.motion_expert_enabled:
            router_conditions = {
                "pose": pose_router_cond,
                "ego": router_ego_cond,
                "timestep": enc_time,
                "valid_frames": y["valid_frames"].bool(),
            }
            if router_text is not None:
                router_conditions["text"] = router_text

        for layer_index, dec in enumerate(self.tsfm):
            decoder_kwargs = {
                "x": x,
                "context": context,
                "mask": mask,
                "context_mask": context_mask,
                "num_prefix_tokens": self.num_prefix_tokens,
            }
            if self.motion_expert_enabled:
                decoder_kwargs["router_conditions"] = router_conditions
            x = dec(**decoder_kwargs)
            if self.text_cross_attention is not None and text_tokens is not None:
                text_delta = self.text_cross_attention(x, text_tokens, text_mask, text_valid)
                x = x + self.text_cross_gates[layer_index] * text_delta

        if self.motion_expert_enabled:
            aux_losses = [
                block.ff.last_aux_loss
                for block in self.tsfm
                if hasattr(block.ff, "last_aux_loss") and block.ff.last_aux_loss is not None
            ]
            self._last_moe_aux_loss = torch.stack(aux_losses).mean() if aux_losses else None

        global_hidden = x[:, 1, :].contiguous() if return_features else None
        hidden = x[:, self.num_prefix_tokens : self.num_prefix_tokens + T].contiguous()
        x = hidden
        x = self.output_process(x)
        x = x.view(B, T, F)

        if return_features:
            return x, hidden, global_hidden
        if return_hidden:
            return x, hidden
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe[None])  # 1 x Tmax x D

    def forward(self, x):
        # x is B x T x D
        if x.shape[1] > self.pe.shape[1]:
            raise ValueError("Sequence exceeds positional embedding length")
        x = x + self.pe[:, : x.shape[1], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, pos_enc, min_period=4e-3, max_period=4.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.pos_enc = pos_enc  # .pe is 1 x Tmax x D

        if not 0 < min_period < max_period:
            raise ValueError(f"Expected 0 < min_period < max_period, got {min_period} and {max_period}.")
        half_dim = self.latent_dim // 2
        periods = torch.exp(torch.linspace(math.log(min_period), math.log(max_period), half_dim))
        angular_frequencies = (2.0 * math.pi) / periods
        # Non-persistent keeps state_dict keys compatible with existing
        # diffusion checkpoints while still following module device moves.
        self.register_buffer("continuous_angular_frequencies", angular_frequencies, persistent=False)

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        # timesteps is (B,)
        if timesteps.ndim != 1:
            raise ValueError(f"Expected one timestep per batch item, got shape {tuple(timesteps.shape)}.")

        if timesteps.dtype.is_floating_point:
            # OpenPI-style continuous sin/cos features. Flow time is expected
            # in [0, 1], with t=0 at data and t=1 at Gaussian noise.
            continuous_time = timesteps.to(dtype=torch.float32)
            frequencies = self.continuous_angular_frequencies.to(dtype=torch.float32)
            angles = continuous_time[:, None] * frequencies[None, :]
            t = torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)
            if t.shape[-1] < self.latent_dim:
                t = torch.cat((t, torch.zeros_like(t[:, :1])), dim=-1)
            t = t.to(dtype=self.time_embed[0].weight.dtype)
        else:
            # Preserve the original discrete diffusion embedding exactly.
            t = self.pos_enc.pe[0, timesteps.long()]  # B x D
        return self.time_embed(t)  # B x D
