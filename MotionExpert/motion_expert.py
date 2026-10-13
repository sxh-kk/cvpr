from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch

from .config import architecture_metadata
from .model import UniEgoMotion
from .mydiffusion.flow_matching import FlowMatching
from .checkpoints import read_checkpoint, model_state, checkpoint_config
from .mydiffusion.mask_utils import apply_feature_mask
from .dataset.normalization import MotionNormalizer
from .utils.task_conditioning import apply_task_conditioning


class MotionExpert:
    """Inference wrapper for the E7 K12 400M text-conditioned motion expert.

    ``sample`` returns the complete ten-step K12 motion and the hidden token
    produced by the first reverse-flow model call. The hidden token is the
    intended interface for downstream fusion modules.
    """

    def __init__(self, weights=None, device=None, dtype=None, stats=None,
                 text_model="openai/clip-vit-large-patch14", local_files_only=False, use_ema=True):
        root = Path(__file__).resolve().parent
        weights = Path(weights or root / "weights" / "e7_k12_400m_text_ema_fp32.pt")
        if not weights.is_file():
            raise FileNotFoundError(f"Motion Expert weights not found: {weights}")

        checkpoint = read_checkpoint(weights)
        self.cfg = checkpoint_config(checkpoint)
        self.weights_path = str(weights.resolve())
        self.weights_kind = "ema" if use_ema and "ema_state_format" in checkpoint else checkpoint.get("weights", "model")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if dtype is None:
            dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.dtype = dtype if isinstance(dtype, torch.dtype) else getattr(torch, str(dtype))
        if self.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError("dtype must be float32, float16 or bfloat16")
        self.text_encoder = None
        self.text_model = text_model
        self.local_files_only = local_files_only

        self.model = UniEgoMotion(self.cfg).to(device=self.device, dtype=self.dtype).eval()
        state = model_state(self.model, checkpoint, use_ema=use_ema)
        missing, unexpected = self.model.load_state_dict(state, strict=True)
        if missing or unexpected:
            raise RuntimeError(f"Invalid Motion Expert weights: missing={missing}, unexpected={unexpected}")
        if stats is None:
            stats = checkpoint.get("motionexpert_normalization")
            if stats is not None:
                stats = dict(stats, normalization_clip=checkpoint.get("motionexpert_normalization_clip", 8.0))
            elif weights.with_suffix(".stats.pt").is_file():
                stats = weights.with_suffix(".stats.pt")
        self.normalizer = MotionNormalizer(stats)
        self.flow = FlowMatching(
            num_steps=self.cfg.FLOW.NUM_STEPS,
            solver=self.cfg.FLOW.SOLVER,
            beta_alpha=self.cfg.FLOW.BETA_ALPHA,
            beta_beta=self.cfg.FLOW.BETA_BETA,
            t_min=self.cfg.FLOW.T_MIN,
            prediction_type=self.cfg.FLOW.PREDICTION_TYPE,
            global_weight=self.cfg.FLOW.GLOBAL_WEIGHT,
            global_feature_start=self.cfg.FLOW.GLOBAL_FEATURE_START,
            global_feature_end=self.cfg.FLOW.GLOBAL_FEATURE_END,
        )

    def _prepare(self, conditioning: Mapping[str, torch.Tensor], text=None, normalized=True):
        y = dict(conditioning)
        text = y.pop("text", None) if text is None else text
        if "traj" not in y:
            raise KeyError("conditioning is missing traj")
        traj = torch.as_tensor(y["traj"])
        if traj.ndim != 3 or traj.shape[-1] != 18:
            raise ValueError(f"traj must have shape [B,T,18], got {tuple(traj.shape)}")
        batch, frames = traj.shape[:2]
        if batch < 1 or not 1 <= frames <= 5000:
            raise ValueError("Expected B >= 1 and 1 <= T <= 5000")
        if text is not None:
            if "text_pooled" in y or "text_tokens" in y:
                raise ValueError("Provide either raw text or precomputed CLIP features")
            from .text_encoder import CLIPTextEncoder
            texts = [text] * batch if isinstance(text, str) else list(text)
            if len(texts) != batch:
                raise ValueError("Provide one text string per batch item")
            if self.text_encoder is None:
                self.text_encoder = CLIPTextEncoder(self.text_model, self.device, self.local_files_only)
            y.update(self.text_encoder(texts))
        required = ("traj", "img_embs", "text_pooled", "text_tokens")
        missing = [key for key in required if key not in y]
        if missing:
            raise KeyError("conditioning is missing required keys: " + ", ".join(missing))
        for key in required:
            y[key] = torch.as_tensor(y[key])
            if not torch.isfinite(y[key]).all():
                raise ValueError(f"{key} contains non-finite values")
        if not normalized:
            y["traj"] = self.normalizer.normalize(y["traj"], "traj")
        if tuple(y["traj"].shape) != (batch, frames, 18):
            raise ValueError(f"traj must have shape [B,T,18], got {tuple(y['traj'].shape)}")
        if tuple(y["img_embs"].shape) != (batch, frames, 1024):
            raise ValueError(f"img_embs must have shape [B,T,1024], got {tuple(y['img_embs'].shape)}")
        if tuple(y["text_pooled"].shape) != (batch, 768):
            raise ValueError(f"text_pooled must have shape [B,768], got {tuple(y['text_pooled'].shape)}")
        if y["text_tokens"].ndim != 3 or tuple(y["text_tokens"].shape[:2]) != (batch, 16) or y["text_tokens"].shape[-1] != 768:
            raise ValueError(f"text_tokens must have shape [B,16,768], got {tuple(y['text_tokens'].shape)}")
        valid = y.get("valid_frames", torch.ones(batch, frames, dtype=torch.bool))
        valid_img = y.get("valid_img_embs", valid)
        y["valid_frames"] = torch.as_tensor(valid, device=self.device, dtype=torch.bool)
        y["valid_img_embs"] = torch.as_tensor(valid_img, device=self.device, dtype=torch.bool)
        for key in ("traj", "img_embs"):
            y[key] = torch.as_tensor(y[key], device=self.device, dtype=self.dtype)
        for key in ("text_pooled", "text_tokens"):
            y[key] = torch.as_tensor(y[key], device=self.device, dtype=self.dtype)
        y["text_mask"] = torch.as_tensor(y.get("text_mask", torch.ones(batch, 16)), device=self.device, dtype=torch.bool)
        y["text_valid"] = torch.as_tensor(y.get("text_valid", torch.ones(batch)), device=self.device, dtype=torch.bool)
        for key in ("traj_mask", "img_mask"):
            if key in y:
                y[key] = torch.as_tensor(y[key], device=self.device, dtype=torch.bool)
        shapes = {"valid_frames": (batch, frames), "valid_img_embs": (batch, frames),
                  "text_mask": (batch, 16), "text_valid": (batch,),
                  "traj_mask": (batch, frames), "img_mask": (batch, frames)}
        for key, shape in shapes.items():
            if key in y and tuple(y[key].shape) != shape:
                raise ValueError(f"{key} must have shape {shape}, got {tuple(y[key].shape)}")
        if not y["valid_frames"].any(dim=1).all():
            raise ValueError("Each input sequence must contain at least one valid frame")
        for key in required:
            if not torch.isfinite(y[key]).all():
                raise ValueError(f"{key} contains non-finite values")
        if not normalized:
            y["traj"] = y["traj"].masked_fill(~y["valid_frames"][..., None], 0)
        if "feature_valid_mask" in y:
            mask = torch.as_tensor(y["feature_valid_mask"], device=self.device, dtype=torch.bool)
            if mask.shape not in ((batch, 153), (batch, frames, 153)):
                raise ValueError("feature_valid_mask must be [B,153] or [B,T,153]")
            y["feature_valid_mask"] = mask
        return y, batch, frames

    def _sampling_inputs(self, conditioning, *, text, seed, task, forecast_prefix, normalized):
        y, batch, frames = self._prepare(conditioning, text=text, normalized=normalized)
        if task is not None:
            prefix = max(1, frames // 4) if forecast_prefix is None else forecast_prefix
            y = apply_task_conditioning(y, task, forecast_prefix=prefix)
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(seed))
        noise = torch.randn(
            (batch, frames, 153), device=self.device, dtype=self.dtype, generator=generator
        )
        return y, apply_feature_mask(noise, y)

    def _feature_metadata(self, y, *, task, seed, cond_scale):
        return {
            "feature_mask": y["valid_frames"].clone(),
            "metadata": {
                **architecture_metadata(self.cfg), "task": task or "custom",
                "seed": None if seed is None else int(seed),
                "cond_scale": None if cond_scale is None else float(cond_scale),
                "weights": self.weights_path, "weights_kind": self.weights_kind,
            },
        }

    @torch.inference_mode()
    def sample(self, conditioning: Mapping[str, torch.Tensor], *, text=None, seed=None,
               task=None, forecast_prefix=None, cond_scale=None, normalized=True,
               denormalize=False, return_all_pred_xstart=False):
        y, noise = self._sampling_inputs(conditioning, text=text, seed=seed, task=task,
                                         forecast_prefix=forecast_prefix, normalized=normalized)
        has_global = self.cfg.MODEL.GLOBAL_TOKEN_COUNT == 1
        result = self.flow.sample_loop(
            self.model,
            noise.shape,
            model_kwargs={"y": y, "cond_scale": cond_scale},
            noise=noise,
            progress=False,
            return_all_pred_xstart=return_all_pred_xstart,
            return_one_step_hidden=not has_global,
            return_one_step_features=has_global,
        )
        if has_global:
            output = result
        elif return_all_pred_xstart:
            motion, all_pred_xstart, hidden = result
            output = {"motion": motion, "one_step_hidden": hidden, "pred_xstart": all_pred_xstart}
        else:
            motion, hidden = result
            output = {"motion": motion, "one_step_hidden": hidden}
        if denormalize:
            output["motion_denormalized"] = self.normalizer.denormalize(output["motion"])
        output.update(self._feature_metadata(y, task=task, seed=seed, cond_scale=cond_scale))
        return output

    @torch.no_grad()
    def extract_features(self, conditioning: Mapping[str, torch.Tensor], *, text=None, seed=None,
                         task=None, forecast_prefix=None, cond_scale=None, normalized=True):
        """Run only the first flow call; return ordinary tensors for frozen-expert training."""
        y, noise = self._sampling_inputs(conditioning, text=text, seed=seed, task=task,
                                         forecast_prefix=forecast_prefix, normalized=normalized)
        t = torch.ones(noise.shape[0], device=self.device, dtype=torch.float32)
        if self.cfg.MODEL.GLOBAL_TOKEN_COUNT:
            _, hidden, global_hidden = self.model(noise, t, y, cond_scale=cond_scale, return_features=True)
            output = {"one_step_hidden": hidden, "one_step_global": global_hidden}
        else:
            _, hidden = self.model(noise, t, y, cond_scale=cond_scale, return_hidden=True)
            output = {"one_step_hidden": hidden}
        output.update(self._feature_metadata(y, task=task, seed=seed, cond_scale=cond_scale))
        return output

    __call__ = sample
