"""Image-only K12 forecasting from SMPL-H FK or legacy dense observations."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch

from .checkpoints import checkpoint_config, model_state, read_checkpoint, validate_checkpoint_statistics
from .config import architecture_metadata
from .dataset.motion_layout import motion_layout
from .dataset.normalization import MotionNormalizer
from .model import UniEgoMotion
from .mydiffusion.flow_matching import FlowMatching
from .utils.task_conditioning import apply_task_conditioning


class MotionExpert:
    """Generate versioned motion and expose first-call 768D expert features.

    Forecasting accepts only the observed camera/image prefix. Set
    ``output_frames`` to request the complete generated sequence; its future
    requires neither ground-truth poses nor future label validity masks.
    """

    def __init__(self, weights=None, device=None, dtype=None, stats=None, use_ema=True):
        root = Path(__file__).resolve().parent
        weights = Path(weights or root / "weights" / "smplh_k12_global_ema_fp32.pt")
        if not weights.is_file():
            raise FileNotFoundError(f"MotionExpert weights not found: {weights}")
        checkpoint = read_checkpoint(weights)
        self.cfg = checkpoint_config(checkpoint)
        self.weights_path = str(weights.resolve())
        if use_ema and checkpoint.get("ema_state_format") not in (None, "original_model_with_ema_optimizer_v1"):
            raise ValueError(f"Unsupported EMA checkpoint format: {checkpoint['ema_state_format']}")
        self.weights_kind = "ema" if use_ema and checkpoint.get("ema_state_format") == "original_model_with_ema_optimizer_v1" else checkpoint.get("weights", "model")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if dtype is None:
            dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.dtype = dtype if isinstance(dtype, torch.dtype) else getattr(torch, str(dtype))
        if self.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError("dtype must be float32, float16 or bfloat16")
        if stats is None:
            stats = checkpoint.get("motionexpert_normalization")
            if stats is not None:
                stats = dict(stats, normalization_clip=checkpoint.get("motionexpert_normalization_clip", stats.get("normalization_clip", None)))
            elif weights.with_suffix(".stats.pt").is_file():
                stats = weights.with_suffix(".stats.pt")
        # Validate statistics before allocating the roughly 400M-parameter model.
        self.normalizer = MotionNormalizer(stats)
        validate_checkpoint_statistics(self.cfg, checkpoint, self.normalizer)
        self.model = UniEgoMotion(self.cfg).to(device=self.device, dtype=self.dtype).eval()
        self.model.load_state_dict(model_state(self.model, checkpoint, use_ema=use_ema), strict=True)
        self.flow = FlowMatching(
            num_steps=self.cfg.FLOW.NUM_STEPS, solver=self.cfg.FLOW.SOLVER,
            beta_alpha=self.cfg.FLOW.BETA_ALPHA, beta_beta=self.cfg.FLOW.BETA_BETA,
            t_min=self.cfg.FLOW.T_MIN, prediction_type=self.cfg.FLOW.PREDICTION_TYPE,
            global_weight=self.cfg.FLOW.GLOBAL_WEIGHT,
            global_feature_start=self.cfg.FLOW.GLOBAL_FEATURE_START,
            global_feature_end=self.cfg.FLOW.GLOBAL_FEATURE_END,
        )

    @staticmethod
    def _pad_time(value, frames, fill=0):
        if value.shape[1] == frames:
            return value
        shape = (value.shape[0], frames - value.shape[1], *value.shape[2:])
        return torch.cat((value, value.new_full(shape, fill)), dim=1)

    def _image_forecast_profile(self):
        return (self.cfg.DATA.FPS == 5
                and getattr(self.cfg.DATA, "TRAIN_TASK", None) == "fore")

    @property
    def motion_dim(self):
        return motion_layout(self.cfg.DATA.REPRE_TYPE).motion_dim

    def _prepare(self, conditioning: Mapping[str, torch.Tensor], *, normalized=True,
                 task=None, output_frames=None, forecast_prefix=None):
        if not isinstance(conditioning, Mapping):
            raise TypeError("conditioning must be a tensor mapping")
        if any(key == "text" or key.startswith("text_") for key in conditioning):
            raise ValueError("Text conditioning was removed from dense MotionExpert")
        if "img_embs" not in conditioning:
            raise KeyError("conditioning is missing img_embs")
        image = torch.as_tensor(conditioning["img_embs"])
        if image.ndim == 3 and image.shape[-1] == 5120:
            image = image.reshape(*image.shape[:2], 5, 1024)
        if image.ndim != 4 or tuple(image.shape[-2:]) != (5, 1024):
            raise ValueError(f"img_embs must have shape [B,T,5,1024], got {tuple(image.shape)}")
        batch, observed_frames = image.shape[:2]
        if type(output_frames) is not int and output_frames is not None:
            raise ValueError("output_frames must be an integer")
        frames = observed_frames if output_frames is None else output_frames
        if batch < 1 or not 1 <= observed_frames <= frames <= 5000:
            raise ValueError("Expected B >= 1 and 1 <= input frames <= output_frames <= 5000")
        if frames != observed_frames and task not in ("fore", "gen"):
            raise ValueError("Only fore/gen can extend observations to output_frames")
        if "traj" not in conditioning and task not in ("fore", "gen"):
            raise KeyError("conditioning is missing traj")
        trajectory = torch.as_tensor(conditioning.get("traj", torch.zeros(batch, observed_frames, 18)))
        if tuple(trajectory.shape) != (batch, observed_frames, 18):
            raise ValueError(f"traj must have shape [B,T,18], got {tuple(trajectory.shape)}")

        def frame_mask(key, default):
            value = torch.as_tensor(conditioning.get(key, default), device=self.device, dtype=torch.bool)
            if tuple(value.shape) != (batch, observed_frames):
                raise ValueError(f"{key} must have shape {(batch, observed_frames)}")
            return value

        valid = frame_mask("valid_frames", torch.ones(batch, observed_frames, dtype=torch.bool))
        image_valid = frame_mask("valid_img_embs", valid)
        if not valid.any(dim=1).all():
            raise ValueError("Each sequence must contain at least one valid observation")
        trajectory_valid = torch.as_tensor(
            conditioning.get("trajectory_valid", valid[..., None].expand(-1, -1, 18)),
            device=self.device, dtype=torch.bool,
        )
        if tuple(trajectory_valid.shape) != (batch, observed_frames, 18):
            raise ValueError("trajectory_valid must have shape [B,T,18]")
        if "traj" not in conditioning:
            trajectory_valid = torch.zeros_like(trajectory_valid)
        for key in ("feature_valid_mask", "motion_valid"):
            if key in conditioning:
                shape = tuple(torch.as_tensor(conditioning[key]).shape)
                if shape not in ((batch, self.motion_dim), (batch, observed_frames, self.motion_dim)):
                    raise ValueError(f"{key} must be [B,{self.motion_dim}] or [B,T,{self.motion_dim}], got {shape}")
        supplied_masks = {key: frame_mask(key, torch.zeros_like(valid))
                          for key in ("traj_mask", "img_mask")}
        traj_hidden = supplied_masks["traj_mask"] | ~valid
        img_hidden = supplied_masks["img_mask"] | ~valid
        if task == "gen":
            traj_hidden = torch.ones_like(traj_hidden)
            img_hidden = img_hidden | (torch.arange(observed_frames, device=self.device)[None] > 0)
        elif task == "fore":
            prefix = getattr(self.cfg.DATA, "FORECAST_PREFIX", 30) if forecast_prefix is None else forecast_prefix
            if type(prefix) is not int or prefix < 1:
                raise ValueError("forecast_prefix must be a positive integer")
            prefix = min(prefix, observed_frames)
            future = torch.arange(observed_frames, device=self.device)[None] >= prefix
            traj_hidden, img_hidden = traj_hidden | future, img_hidden | future
            if self._image_forecast_profile():
                # The deployed model consumes images only, even if an archive
                # batch includes camera labels for supervision or evaluation.
                traj_hidden = torch.ones_like(traj_hidden)
                trajectory_valid = torch.zeros_like(trajectory_valid)
        visible_trajectory = trajectory_valid & ~traj_hidden[..., None]
        visible_images = image_valid & ~img_hidden
        trajectory = trajectory.to(device=self.device, dtype=torch.float32)
        image = image.to(device=self.device, dtype=torch.float32)
        if not torch.isfinite(trajectory.masked_select(visible_trajectory)).all():
            raise ValueError("Valid traj values contain non-finite values")
        if not torch.isfinite(image[visible_images]).all():
            raise ValueError("Valid img_embs values contain non-finite values")
        trajectory = trajectory.masked_fill(~visible_trajectory, 0)
        image = image.masked_fill(~visible_images[..., None, None], 0)
        if not normalized:
            trajectory = self.normalizer.normalize(trajectory, "traj")
        trajectory = trajectory.masked_fill(~visible_trajectory, 0)
        y = {
            "traj": self._pad_time(trajectory, frames).to(self.dtype),
            "img_embs": self._pad_time(image, frames).to(self.dtype),
            "trajectory_valid": self._pad_time(trajectory_valid, frames, False),
            "valid_img_embs": self._pad_time(image_valid, frames, False),
            "valid_frames": self._pad_time(valid, frames, False),
        }
        for key, mask in supplied_masks.items():
            y[key] = self._pad_time(mask, frames, True)
        # Label masks in dataset batches never constrain generated state.
        return y, batch, frames, observed_frames

    def _sampling_inputs(self, conditioning, *, seed, task, forecast_prefix, normalized, output_frames):
        y, batch, frames, observed_frames = self._prepare(
            conditioning, normalized=normalized, task=task, output_frames=output_frames,
            forecast_prefix=forecast_prefix,
        )
        if task is not None:
            prefix = getattr(self.cfg.DATA, "FORECAST_PREFIX", 30) if forecast_prefix is None else forecast_prefix
            if type(prefix) is not int or prefix < 1:
                raise ValueError("forecast_prefix must be a positive integer")
            prefix = min(prefix, observed_frames)
            if task == "fore" and prefix >= frames:
                raise ValueError("Forecasting needs at least one frame after the observed prefix")
            image_forecast = task == "fore" and self._image_forecast_profile()
            y = apply_task_conditioning(
                y, task, forecast_prefix=prefix,
                trajectory_condition=not image_forecast,
                noise_only_history=image_forecast and self.cfg.DATA.MOTION_SUPERVISION == "future_only")
            y["task_conditioning"] = True
            if task == "fore":
                y["observed_prefix"] = prefix
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(seed))
        noise = torch.randn((batch, frames, self.motion_dim), device=self.device,
                            dtype=self.dtype, generator=generator)
        return y, noise

    def _feature_metadata(self, y, *, task, seed, cond_scale):
        return {
            "feature_mask": y["valid_frames"].clone(),
            "metadata": {
                **architecture_metadata(self.cfg), "task": task or "custom",
                "output_frames": int(y["valid_frames"].shape[1]),
                "observed_prefix": y.get("observed_prefix", 1 if task == "gen" else None),
                "seed": None if seed is None else int(seed),
                "cond_scale": None if cond_scale is None else float(cond_scale),
                "weights": self.weights_path, "weights_kind": self.weights_kind,
            },
        }

    @torch.inference_mode()
    def sample(self, conditioning: Mapping[str, torch.Tensor], *, seed=None, task=None,
               forecast_prefix=None, output_frames=None, cond_scale=None, normalized=True,
               denormalize=False, decode=False, decode_kwargs=None, return_all_pred_xstart=False):
        y, noise = self._sampling_inputs(
            conditioning, seed=seed, task=task, forecast_prefix=forecast_prefix,
            normalized=normalized, output_frames=output_frames,
        )
        parameters = dict(decode_kwargs or {})
        if decode and task == "fore" and self._image_forecast_profile():
            prefix = y["observed_prefix"]
            if parameters.get("start_frame", prefix) != prefix:
                raise ValueError("Forecast decode must start at the first future frame")
            if "initial_floor_transform" not in parameters:
                raise ValueError("Forecast world decode requires the last-observed initial_floor_transform")
            parameters["start_frame"] = prefix
        has_global = self.cfg.MODEL.GLOBAL_TOKEN_COUNT == 1
        result = self.flow.sample_loop(
            self.model, noise.shape, model_kwargs={"y": y, "cond_scale": cond_scale},
            noise=noise, progress=False, return_all_pred_xstart=return_all_pred_xstart,
            return_one_step_hidden=not has_global, return_one_step_features=has_global,
        )
        if has_global:
            output = result
        elif return_all_pred_xstart:
            motion, all_pred_xstart, hidden = result
            output = {"motion": motion, "one_step_hidden": hidden, "pred_xstart": all_pred_xstart}
        else:
            motion, hidden = result
            output = {"motion": motion, "one_step_hidden": hidden}
        if denormalize or decode:
            output["motion_denormalized"] = self.normalizer.denormalize(output["motion"])
        if decode:
            from .dataset.dense_representation import decode_dense_motion
            decoder = decode_dense_motion
            from .dataset.motion_layout import SMPLH_REPRE_TYPE
            if self.cfg.DATA.REPRE_TYPE == SMPLH_REPRE_TYPE:
                from .dataset.smplh_representation import decode_smplh_motion
                decoder = decode_smplh_motion
            physical = output["motion_denormalized"]
            decoded = []
            for index, sample_motion in enumerate(physical):
                sample_parameters = dict(parameters)
                for key, batched_rank in (("window_to_world", 3), ("initial_floor_transform", 3),
                                          ("delta_valid", 2), ("floor_height", 1)):
                    if key in parameters:
                        value = torch.as_tensor(parameters[key])
                        if value.ndim == batched_rank:
                            if value.shape[0] != physical.shape[0]:
                                raise ValueError(f"Batched {key} must match motion batch size")
                            sample_parameters[key] = value[index]
                decoded.append(decoder(sample_motion, **sample_parameters))
            output["decoded"] = {key: torch.stack([item[key] for item in decoded])
                                 for key in decoded[0]}
        output.update(self._feature_metadata(y, task=task, seed=seed, cond_scale=cond_scale))
        if task == "fore":
            prefix = y["observed_prefix"]
            output["future_motion"] = output["motion"][:, prefix:]
            output["future_time_offsets_s"] = torch.arange(
                1, output["motion"].shape[1] - prefix + 1,
                device=self.device, dtype=torch.float32) / self.cfg.DATA.FPS
            if "motion_denormalized" in output:
                output["future_motion_denormalized"] = output["motion_denormalized"][:, prefix:]
        return output

    def predict_future(self, conditioning: Mapping[str, torch.Tensor], **kwargs):
        """Predict the five future frames from six 5Hz historical image features.

        ``future_motion_denormalized`` has five frames of the checkpoint's state. The
        default joint protocol also reconstructs the six historical body
        frames in ``motion``; only the last five frames are returned as future.
        """
        if not self._image_forecast_profile():
            raise ValueError("predict_future requires the 5Hz image-history forecasting profile")
        if not isinstance(conditioning, Mapping) or "img_embs" not in conditioning:
            raise KeyError("conditioning is missing img_embs")
        image = torch.as_tensor(conditioning["img_embs"])
        if image.ndim < 2 or image.shape[1] != self.cfg.DATA.FORECAST_PREFIX:
            raise ValueError(f"predict_future requires {self.cfg.DATA.FORECAST_PREFIX} observed image frames")
        reserved = {"task", "forecast_prefix", "output_frames"}.intersection(kwargs)
        if reserved:
            raise ValueError("predict_future uses the checkpoint's fixed task and temporal window")
        kwargs.setdefault("denormalize", True)
        return self.sample(conditioning, task="fore", forecast_prefix=self.cfg.DATA.FORECAST_PREFIX,
                           output_frames=self.cfg.DATA.WINDOW, **kwargs)

    @torch.no_grad()
    def extract_features(self, conditioning: Mapping[str, torch.Tensor], *, seed=None, task=None,
                         forecast_prefix=None, output_frames=None, cond_scale=None, normalized=True):
        """Run one flow call and return ordinary tensors usable by a trainable head."""
        y, noise = self._sampling_inputs(
            conditioning, seed=seed, task=task, forecast_prefix=forecast_prefix,
            normalized=normalized, output_frames=output_frames,
        )
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
