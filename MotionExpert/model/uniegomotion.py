"""Versioned body rotations/positions, explicit hands and image-conditioned MoE."""

import math
import numpy as np
import torch
import torch.nn as nn
from ..config import validate_architecture
from ..dataset.motion_layout import motion_layout
from .core import MoEDecoderBlock


def mask_it(mask, cond, replace_token):
    """Replace conditions without retaining NaNs from an unavailable value."""
    mask = mask.reshape(*mask.shape, *([1] * (cond.ndim - mask.ndim))).bool()
    token = replace_token.view(*([1] * (cond.ndim - 1)), -1)
    return torch.where(mask, token.to(dtype=cond.dtype), cond)


class UniEgoMotion(nn.Module):
    """The retained 768-dimensional E7 MoE with a versioned dense state."""

    def __init__(self, cfg, dropout=0.1):
        super().__init__()
        validate_architecture(cfg)
        self.cfg = cfg
        self.latent_dim = 768
        self.layout = motion_layout(cfg.DATA.REPRE_TYPE)
        self.input_feats = self.layout.motion_dim
        self.num_layers = self.num_heads = 12
        self.image_tokens = 5
        self.motion_expert_enabled = True
        self.sparse_joint_indices = tuple(cfg.SPARSE_JOINTS.INDICES)
        self.global_feature_start = self.layout.global_slice.start
        self.global_feature_end = self.layout.global_slice.stop
        self.cond_mask_prob = {"traj": 0.5, "clip": 0.1, "subseq_frames": 0.5}
        self.input_process = nn.Linear(self.input_feats, self.latent_dim)
        self.pos_enc = PositionalEncoding(self.latent_dim, dropout)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.pos_enc)
        self.embed_traj_cond = nn.Linear(18, self.latent_dim)
        # A missing scalar and a valid zero must remain distinguishable.
        self.embed_traj_valid = nn.Linear(18, self.latent_dim, bias=False)
        self.embed_clip_cond = nn.Linear(1024, self.latent_dim)
        self.mask_tokens = nn.ParameterDict({
            name: nn.Parameter(torch.randn(self.latent_dim) * 0.05)
            for name in ("traj", "clip")
        })
        expert = cfg.MODEL.MOTION_EXPERT
        self.tsfm = nn.ModuleList([
            MoEDecoderBlock(self.latent_dim, self.num_heads, dropout, 2,
                            num_experts=11, top_k=2,
                            router_jitter=float(expert.ROUTER_JITTER), shared_expert=True,
                            chunk_size=4, router_conditioned=True,
                            routed_gate_init=float(expert.ROUTED_GATE_INIT))
            for _ in range(self.num_layers)
        ])
        self.output_process = nn.Linear(self.latent_dim, self.input_feats)
        self._last_moe_aux_loss = None
        self.num_prefix_tokens = 1 + cfg.MODEL.GLOBAL_TOKEN_COUNT
        if cfg.MODEL.GLOBAL_TOKEN_COUNT:
            self.global_token = nn.Parameter(torch.randn(self.latent_dim) * 0.02)
        else:
            self.register_parameter("global_token", None)

    def get_moe_auxiliary_loss(self):
        return self._last_moe_aux_loss

    def _condition_mask(self, batch, frames, device, cond_type, cond_mask=None,
                        explicit_task=False):
        mask = torch.zeros(batch, frames, device=device, dtype=torch.bool)
        if cond_mask is not None:
            supplied = torch.as_tensor(cond_mask, device=device, dtype=torch.bool)
            if supplied.shape == (batch,):
                supplied = supplied[:, None].expand(-1, frames)
            if supplied.shape != mask.shape:
                raise ValueError(f"{cond_type} mask must be [B,T], got {tuple(supplied.shape)}")
            mask = mask | supplied
        # Missing-data masks compose with the original training dropout.
        # A task helper explicitly disables random dropout for its task masks.
        explicit = torch.as_tensor(explicit_task, device=device, dtype=torch.bool)
        if explicit.ndim == 0:
            explicit = explicit.expand(batch)
        if explicit.shape != (batch,):
            raise ValueError("task_conditioning must be a bool or one bool per batch item")
        if self.training and not bool(explicit.all()):
            whole = torch.rand(batch, device=device) < self.cond_mask_prob[cond_type]
            subsequent = torch.rand(batch, device=device) < self.cond_mask_prob["subseq_frames"]
            whole = whole & ~explicit
            subsequent = subsequent & ~explicit
            mask = mask | whole[:, None]
            if cond_type == "traj":
                mask = mask | subsequent[:, None]
            else:
                mask[:, 1:] = mask[:, 1:] | subsequent[:, None]
        return mask

    def mask_cond(self, cond, cond_type, cond_mask=None, explicit_task=False):
        mask = self._condition_mask(cond.shape[0], cond.shape[1], cond.device,
                                    cond_type, cond_mask, explicit_task)
        return mask_it(mask, cond, self.mask_tokens[cond_type])

    @staticmethod
    def _frame_mask(raw, batch, frames, device, name):
        mask = torch.as_tensor(raw, device=device, dtype=torch.bool)
        if mask.shape != (batch, frames):
            raise ValueError(f"{name} must have shape {(batch, frames)}, got {tuple(mask.shape)}")
        return mask

    def forward(self, x, timesteps, y, cond_scale=None, return_hidden=False,
                return_features=False):
        if return_hidden and return_features:
            raise ValueError("return_hidden and return_features are mutually exclusive")
        if return_features and self.global_token is None:
            raise ValueError("Global features require GLOBAL_TOKEN_COUNT=1")
        if x.ndim != 3 or x.shape[-1] != self.input_feats:
            raise ValueError(f"Motion must be [B,T,{self.input_feats}], got {tuple(x.shape)}")
        batch, frames, _ = x.shape
        valid_frames = self._frame_mask(y["valid_frames"], batch, frames, x.device,
                                        "valid_frames")
        if cond_scale is not None:
            if not math.isfinite(cond_scale):
                raise ValueError("cond_scale must be finite")
            conditional = self.forward(x, timesteps, y, return_hidden=return_hidden,
                                       return_features=return_features)
            # Label validity is not an inference condition and must not leak
            # future availability into the unconditional branch.
            unconditional = self.forward(x, timesteps, {"valid_frames": valid_frames},
                                         return_hidden=return_hidden,
                                         return_features=return_features)
            if return_hidden or return_features:
                return tuple(u + (c - u) * cond_scale
                             for c, u in zip(conditional, unconditional))
            return unconditional + (conditional - unconditional) * cond_scale

        explicit_task = y.get("task_conditioning", False)
        x = self.pos_enc(self.input_process(x))
        pose_router_cond = x
        enc_time = self.embed_timestep(timesteps).to(dtype=x.dtype)
        if "traj" in y:
            traj = torch.as_tensor(y["traj"], device=x.device, dtype=x.dtype)
            if traj.shape != (batch, frames, 18):
                raise ValueError(f"traj must be [B,T,18], got {tuple(traj.shape)}")
            raw_valid = y.get("trajectory_valid", y.get("traj_valid"))
            trajectory_valid = (torch.ones_like(traj, dtype=torch.bool) if raw_valid is None
                                else torch.as_tensor(raw_valid, device=x.device, dtype=torch.bool))
            if trajectory_valid.shape != traj.shape:
                raise ValueError("trajectory_valid must match the [B,T,18] trajectory")
            traj_mask = self._condition_mask(batch, frames, x.device, "traj",
                                             y.get("traj_mask"), explicit_task)
            traj_mask = traj_mask | ~trajectory_valid.any(dim=-1)
            visible_valid = trajectory_valid & ~traj_mask[:, :, None]
            clean_traj = torch.where(visible_valid, traj, torch.zeros_like(traj))
            enc_traj = (self.embed_traj_cond(clean_traj) +
                        self.embed_traj_valid(visible_valid.to(dtype=x.dtype)))
            x = x + mask_it(traj_mask, enc_traj, self.mask_tokens["traj"])
        else:
            x = x + self.mask_tokens["traj"]

        if "img_embs" in y:
            images = torch.as_tensor(y["img_embs"], device=x.device, dtype=x.dtype)
            if images.shape != (batch, frames, self.image_tokens, 1024):
                raise ValueError(f"img_embs must be [B,T,5,1024], got {tuple(images.shape)}")
            image_valid = self._frame_mask(y.get("valid_img_embs", valid_frames),
                                           batch, frames, x.device, "valid_img_embs")
            image_valid = image_valid & valid_frames
            img_mask = self._condition_mask(batch, frames, x.device, "clip",
                                            y.get("img_mask"), explicit_task)
            img_mask = img_mask | ~image_valid
            visible_images = image_valid & ~img_mask
            images = torch.where(visible_images[:, :, None, None], images, torch.zeros_like(images))
            enc_imgs = self.embed_clip_cond(images)
            enc_imgs = mask_it(img_mask, enc_imgs, self.mask_tokens["clip"])
            # Every token of a frame gets the same temporal position.
            enc_imgs = self.pos_enc(enc_imgs)
            router_ego_valid = image_valid & ~img_mask
            router_ego_cond = torch.where(router_ego_valid[:, :, None], enc_imgs.mean(dim=2),
                                          torch.zeros_like(enc_imgs[:, :, 0]))
        else:
            # Preserve the register-token rank during CFG and missing image input.
            enc_imgs = self.mask_tokens["clip"].view(1, 1, 1, -1).expand(
                batch, frames, self.image_tokens, -1)
            enc_imgs = self.pos_enc(enc_imgs)
            image_valid = torch.zeros_like(valid_frames)
            visible_images = image_valid
            router_ego_valid = image_valid
            router_ego_cond = torch.zeros(batch, frames, self.latent_dim,
                                          device=x.device, dtype=x.dtype)

        # A time token always provides one valid key, including fully missing video.
        context = torch.cat((enc_time[:, None], enc_imgs.flatten(1, 2)), dim=1)
        context_valid = torch.cat((valid_frames.new_ones(batch, 1),
                                   visible_images.repeat_interleave(self.image_tokens, dim=1)), dim=1)
        prefixes = [enc_time[:, None]]
        if self.global_token is not None:
            prefixes.append(self.global_token.to(dtype=x.dtype).view(1, 1, -1).expand(
                batch, -1, -1).contiguous())
        x = torch.cat((*prefixes, x), dim=1)
        token_valid = torch.cat((valid_frames.new_ones(batch, self.num_prefix_tokens), valid_frames), dim=1)
        router_conditions = {"pose": pose_router_cond, "ego": router_ego_cond,
                             "ego_valid_frames": router_ego_valid, "timestep": enc_time,
                             "valid_frames": valid_frames}
        for decoder in self.tsfm:
            x = decoder(x=x, context=context, mask=token_valid[:, None, None],
                        context_mask=context_valid[:, None, None],
                        num_prefix_tokens=self.num_prefix_tokens,
                        router_conditions=router_conditions)
        aux_losses = [block.ff.last_aux_loss for block in self.tsfm
                      if block.ff.last_aux_loss is not None]
        self._last_moe_aux_loss = torch.stack(aux_losses).mean() if aux_losses else None
        hidden = x[:, self.num_prefix_tokens:self.num_prefix_tokens + frames].contiguous()
        prediction = self.output_process(hidden)
        if return_features:
            return prediction, hidden, x[:, 1].contiguous()
        if return_hidden:
            return prediction, hidden
        return prediction


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
        # Time is axis 1 for both [B,T,D] and [B,T,5,D].
        if x.ndim not in (3, 4) or x.shape[-1] != self.pe.shape[-1]:
            raise ValueError("Positional encoding requires [B,T,D] or [B,T,5,D]")
        if x.shape[1] > self.pe.shape[1]:
            raise ValueError("Sequence exceeds positional embedding length")
        encoding = self.pe[:, : x.shape[1], :]
        if x.ndim == 4:
            encoding = encoding.unsqueeze(2)
        x = x + encoding.to(dtype=x.dtype)
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
