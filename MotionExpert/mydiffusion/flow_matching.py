"""E7 x0 Flow Matching loss and Euler sampling.

t=0 denotes clean motion and t=1 Gaussian noise. The model predicts x0;
sampling converts it to velocity (x_t - x0) / t and integrates toward t=0.
"""

from __future__ import annotations
import math
from collections.abc import Mapping
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple, Union

import torch
from .mask_utils import apply_feature_mask, feature_mask_for
from ..dataset.motion_layout import motion_layout

Tensor = torch.Tensor


def _append_dims(value, ndim):
    if value.ndim > ndim:
        raise ValueError("Cannot broadcast to fewer dimensions")
    return value.reshape(*value.shape, *((1,) * (ndim - value.ndim)))


class FlowMatching:
    def __init__(self, num_steps=10, solver="euler", beta_alpha=1.5, beta_beta=1.0,
                 t_min=0.001, prediction_type="x0", global_weight=8.0,
                 global_feature_start=None, global_feature_end=None):
        if not isinstance(num_steps, int) or num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if solver != "euler" or prediction_type != "x0":
            raise ValueError("E7 uses x0 prediction and Euler sampling")
        if (not all(math.isfinite(value) for value in (beta_alpha, beta_beta, t_min))
                or beta_alpha <= 0 or beta_beta <= 0 or not 0 <= t_min < 1):
            raise ValueError("Invalid flow timestep distribution")
        if (global_feature_start is None) != (global_feature_end is None):
            raise ValueError("Provide both global feature bounds or neither")
        self._automatic_global_slice = global_feature_start is None
        if self._automatic_global_slice:
            global_feature_start, global_feature_end = 108, 117
        if not math.isfinite(global_weight) or global_weight <= 0 or not 0 <= global_feature_start < global_feature_end:
            raise ValueError("Invalid global feature weighting")
        self.num_steps, self.solver = num_steps, solver
        self.beta_alpha, self.beta_beta, self.t_min = beta_alpha, beta_beta, t_min
        self.prediction_type, self.global_weight = prediction_type, global_weight
        self.global_feature_start, self.global_feature_end = global_feature_start, global_feature_end

    def _global_bounds(self, dimension):
        if self._automatic_global_slice and dimension == 245:
            region = motion_layout(dimension).global_slice
            return region.start, region.stop
        return self.global_feature_start, self.global_feature_end

    @staticmethod
    def _generation_mask(state: Tensor, conditioning: Mapping[str, Any]) -> Optional[Tensor]:
        """Task time domain, independent of availability of supervised labels.

        Joint forecasting omits this mask and denoises the full motion sequence.
        The optional future-only experiment uses false historical frames as
        initial Gaussian context throughout sampling, never as history GT.
        """
        raw = conditioning.get("motion_generation_mask")
        if raw is None:
            return None
        if state.ndim != 3:
            raise ValueError("motion_generation_mask requires a [B,T,D] state")
        mask = torch.as_tensor(raw, device=state.device)
        if mask.dtype != torch.bool or mask.shape != state.shape[:2]:
            raise ValueError("motion_generation_mask must be a [B,T] boolean tensor")
        return mask[:, :, None]

    @staticmethod
    def _check_no_repaint(model_kwargs: Optional[Mapping[str, Any]]) -> None:
        if not model_kwargs:
            return
        y = model_kwargs.get("y")
        if isinstance(y, Mapping) and "repaint_mask" in y:
            raise NotImplementedError(
                "Flow Matching does not support repaint/inpainting. "
                "Remove y['repaint_mask'] before training or sampling."
            )

    @staticmethod
    def _normalize_t(t: Tensor, batch_size: int, device: torch.device) -> Tensor:
        t = torch.as_tensor(t, device=device, dtype=torch.float32)
        if t.ndim == 0:
            t = t.expand(batch_size)
        if t.shape != (batch_size,):
            raise ValueError(f"t must have shape ({batch_size},), got {tuple(t.shape)}.")
        if not bool(torch.all((t >= 0.0) & (t <= 1.0))):
            raise ValueError("All flow timesteps must lie in [0, 1].")
        return t

    def sample_timesteps(self, batch_size: int, device: Union[str, torch.device]) -> Tensor:
        """Draw continuous training times from the OpenPI Beta schedule."""
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        device = torch.device(device)
        alpha = torch.tensor(self.beta_alpha, device=device, dtype=torch.float32)
        beta = torch.tensor(self.beta_beta, device=device, dtype=torch.float32)
        samples = torch.distributions.Beta(alpha, beta).sample((batch_size,))
        return self.t_min + (1.0 - self.t_min) * samples

    @staticmethod
    def interpolate(x_start: Tensor, noise: Tensor, t: Tensor) -> Tensor:
        """Construct ``x_t = (1-t) x_0 + t epsilon``."""
        if x_start.shape != noise.shape:
            raise ValueError(
                f"x_start and noise must have identical shapes, got "
                f"{tuple(x_start.shape)} and {tuple(noise.shape)}."
            )
        t_view = _append_dims(t.to(device=x_start.device, dtype=x_start.dtype), x_start.ndim)
        return (1.0 - t_view) * x_start + t_view * noise

    @staticmethod
    def target_to_velocity(x_t: Tensor, t: Tensor, pred_xstart: Tensor) -> Tensor:
        """Convert a clean-target prediction to the current-convention velocity.

        Along ``x_t = (1-t) x_0 + t epsilon``, the corresponding velocity is
        ``(x_t - x_0) / t``. Sampling never calls this conversion at ``t=0``;
        the final Euler interval ends at zero without a model call there.
        """
        if x_t.shape != pred_xstart.shape:
            raise ValueError(
                f"x_t and pred_xstart must have identical shapes, got "
                f"{tuple(x_t.shape)} and {tuple(pred_xstart.shape)}."
            )
        if bool(torch.any(t <= 0.0)):
            raise ValueError("Target-predictive velocity conversion requires t > 0.")
        t_view = _append_dims(t.to(device=x_t.device, dtype=x_t.dtype), x_t.ndim)
        return (x_t - pred_xstart) / t_view

    def _model_prediction(
        self,
        model: Any,
        x_t: Tensor,
        t: Tensor,
        model_kwargs: Mapping[str, Any],
        return_hidden: bool = False,
        return_features: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """Return ``(velocity, clean_prediction)`` for either parameterization."""
        if return_hidden and return_features:
            raise ValueError("return_hidden and return_features are mutually exclusive")
        if return_features:
            model_output, hidden, global_hidden = model(x_t, t, return_features=True, **model_kwargs)
        elif return_hidden:
            model_output, hidden = model(x_t, t, return_hidden=True, **model_kwargs)
        else:
            model_output = model(x_t, t, **model_kwargs)
            hidden = None
        if model_output.shape != x_t.shape:
            raise ValueError(
                f"The model must output shape {tuple(x_t.shape)}, got "
                f"{tuple(model_output.shape)}."
            )
        pred_xstart = model_output
        velocity = self.target_to_velocity(x_t, t, pred_xstart)
        if return_features:
            return velocity, pred_xstart, hidden, global_hidden
        if return_hidden:
            return velocity, pred_xstart, hidden
        return velocity, pred_xstart

    def _valid_frame_mse(
        self,
        error: Tensor,
        model_kwargs: Mapping[str, Any],
        return_group_mse: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Optional[Tensor], Optional[Tensor]]]:
        """Return valid-frame-masked MSE independently for every sample.

        When requested, the two additional tensors are unweighted Local and
        Global means.  They are diagnostics only and do not change the
        weighted training objective.
        """
        try:
            conditioning = model_kwargs["y"]
            valid_frames = conditioning.get("loss_mask", conditioning["valid_frames"])
        except (KeyError, TypeError) as exc:
            raise KeyError(
                "Flow Matching training requires model_kwargs['y']['valid_frames']; "
                "an optional loss_mask may further restrict the predicted task region."
            ) from exc

        mask = torch.as_tensor(valid_frames, device=error.device, dtype=torch.bool)
        padding = conditioning.get("padding_mask")
        if padding is not None:
            padding = torch.as_tensor(padding, device=error.device, dtype=torch.bool)
            if padding.shape != mask.shape:
                raise ValueError("padding_mask and loss_mask/valid_frames must have identical shapes")
            mask = mask & padding
        generation_mask = self._generation_mask(error, conditioning)
        if generation_mask is not None:
            mask = mask & generation_mask.squeeze(-1)
        if mask.shape[0] != error.shape[0]:
            raise ValueError(
                "valid_frames and the model output must have the same batch size, "
                f"got {mask.shape[0]} and {error.shape[0]}."
            )
        if mask.ndim > error.ndim:
            raise ValueError(
                f"valid_frames has {mask.ndim} dimensions but the model output has {error.ndim}."
            )
        mask = _append_dims(mask, error.ndim)
        try:
            mask = mask.expand_as(error)
        except RuntimeError as exc:
            raise ValueError(
                f"valid_frames shape {tuple(valid_frames.shape)} cannot be expanded to "
                f"model output shape {tuple(error.shape)}."
            ) from exc

        # Accumulate the loss in fp32 even under mixed precision. An all-padding
        # sample contributes zero rather than producing a NaN.
        mask = mask.to(dtype=torch.float32)
        feature_mask = feature_mask_for(error, conditioning)
        if feature_mask is not None:
            mask = mask * feature_mask.to(dtype=torch.float32)
        global_start, global_end = self._global_bounds(error.shape[-1])
        if global_end > error.shape[-1]:
            raise ValueError("Global feature slice exceeds model output dimension")
        feature_weights = torch.ones(error.shape[-1], device=error.device, dtype=torch.float32)
        feature_weights[global_start:global_end] = self.global_weight
        loss_weights = mask * feature_weights

        safe_error = torch.where(mask.bool(), error.float(), torch.zeros_like(error, dtype=torch.float32))
        squared_error = safe_error.square() * loss_weights
        reduce_dims = tuple(range(1, error.ndim))
        numerator = squared_error.sum(dim=reduce_dims)
        denominator = loss_weights.sum(dim=reduce_dims)
        weighted_mse = torch.where(
            denominator > 0,
            numerator / denominator.clamp_min(1.0),
            torch.zeros_like(numerator),
        )
        if not return_group_mse:
            return weighted_mse

        feature_dim = error.shape[-1]
        if global_end > feature_dim:
            return weighted_mse, None, None

        unweighted_squared_error = safe_error.square() * mask
        global_error = unweighted_squared_error[
            ..., global_start : global_end
        ]
        global_mask = mask[..., global_start : global_end]
        global_numerator = global_error.sum(dim=reduce_dims)
        global_denominator = global_mask.sum(dim=reduce_dims)

        total_numerator = unweighted_squared_error.sum(dim=reduce_dims)
        total_denominator = mask.sum(dim=reduce_dims)
        local_numerator = total_numerator - global_numerator
        local_denominator = total_denominator - global_denominator

        local_mse = torch.where(
            local_denominator > 0,
            local_numerator / local_denominator.clamp_min(1.0),
            torch.zeros_like(local_numerator),
        )
        global_mse = torch.where(
            global_denominator > 0,
            global_numerator / global_denominator.clamp_min(1.0),
            torch.zeros_like(global_numerator),
        )
        return weighted_mse, local_mse, global_mse

    def _dense_group_metrics(self, error: Tensor, model_kwargs: Mapping[str, Any]) -> Dict[str, Tensor]:
        """Unweighted component diagnostics, using the same label/time mask."""
        if error.ndim != 3 or error.shape[-1] != 245:
            return {}
        layout = motion_layout(error.shape[-1])
        conditioning = model_kwargs["y"]
        frames = torch.as_tensor(conditioning.get("loss_mask", conditioning["valid_frames"]),
                                 device=error.device, dtype=torch.bool)
        if "padding_mask" in conditioning:
            frames = frames & torch.as_tensor(conditioning["padding_mask"], device=error.device,
                                               dtype=torch.bool)
        generation_mask = self._generation_mask(error, conditioning)
        if generation_mask is not None:
            frames = frames & generation_mask.squeeze(-1)
        mask = frames[:, :, None].expand_as(error)
        feature_mask = feature_mask_for(error, conditioning)
        if feature_mask is not None:
            mask = mask & feature_mask
        body_channels = torch.arange(layout.body_slice.start, layout.body_slice.stop, device=error.device)
        groups = {
            "body_rotation": body_channels[body_channels.remainder(9) < 6],
            "body_position": body_channels[body_channels.remainder(9) >= 6],
            "left_hand": torch.arange(layout.left_hand_slice.start, layout.left_hand_slice.stop, device=error.device),
            "right_hand": torch.arange(layout.right_hand_slice.start, layout.right_hand_slice.stop, device=error.device),
            "global_rotation": torch.arange(layout.global_slice.start, layout.global_slice.start+6, device=error.device),
            "global_translation": torch.arange(layout.global_slice.start+6, layout.global_slice.stop, device=error.device),
            "contacts": torch.arange(layout.contact_slice.start, layout.contact_slice.stop, device=error.device),
        }
        result = {}
        for name, indices in groups.items():
            group_mask = mask.index_select(-1, indices)
            values = torch.where(group_mask, error.index_select(-1, indices).float(),
                                 torch.zeros_like(group_mask, dtype=torch.float32))
            count = group_mask.sum(dim=(1, 2))
            result[f"{name}_mse"] = values.square().sum(dim=(1, 2)) / count.clamp_min(1)
            result[f"{name}_valid_count"] = count
        return result

    def training_losses(
        self,
        model: Any,
        x_start: Tensor,
        model_kwargs: Optional[Dict[str, Any]] = None,
        noise: Optional[Tensor] = None,
        t: Optional[Tensor] = None,
        return_diagnostics: bool = False,
    ) -> Dict[str, Tensor]:
        """Compute a per-sample conditional flow matching loss.

        ``model`` is called as ``model(x_t, t, **model_kwargs)``. In
        particular, classifier-free guidance remains the model's
        responsibility and is not duplicated in this class.
        """
        model_kwargs = {} if model_kwargs is None else model_kwargs
        self._check_no_repaint(model_kwargs)
        if x_start.ndim < 2:
            raise ValueError(f"x_start must include batch and feature dimensions, got {x_start.shape}.")

        batch_size = x_start.shape[0]
        conditioning = model_kwargs.get("y", {})
        generation_mask = self._generation_mask(x_start, conditioning)
        label_mask = feature_mask_for(x_start, conditioning)
        x_start = apply_feature_mask(x_start, conditioning)
        if noise is None:
            noise = torch.randn_like(x_start)
        elif noise.shape != x_start.shape:
            raise ValueError(
                f"noise must have shape {tuple(x_start.shape)}, got {tuple(noise.shape)}."
            )
        else:
            noise = noise.to(device=x_start.device, dtype=x_start.dtype)
        if t is None:
            t = self.sample_timesteps(batch_size, x_start.device)
        else:
            t = self._normalize_t(t, batch_size, x_start.device)

        x_t = self.interpolate(x_start, noise, t)
        target_velocity = noise - x_start
        if label_mask is not None:
            # Missing targets are unknown motion, rather than a known zero
            # state. Keep their independent noise at every flow time; at t=1
            # all channels then match the full Gaussian deployment input.
            x_t = torch.where(label_mask, x_t, noise)
            target_velocity = torch.where(label_mask, target_velocity,
                                          torch.zeros_like(target_velocity))
        if generation_mask is not None:
            x_t = torch.where(generation_mask, x_t, noise)
            target_velocity = torch.where(generation_mask, target_velocity,
                                          torch.zeros_like(target_velocity))
        model_output = model(x_t, t, **model_kwargs)
        if model_output.shape != x_start.shape:
            raise ValueError(
                f"The model must output shape {tuple(x_start.shape)}, "
                f"got {tuple(model_output.shape)}."
            )

        pred_xstart = model_output
        pred_velocity = self.target_to_velocity(x_t, t, pred_xstart)
        if generation_mask is not None:
            pred_velocity = torch.where(generation_mask, pred_velocity,
                                        torch.zeros_like(pred_velocity))
        target = x_start

        mse, local_mse, global_mse = self._valid_frame_mse(
            model_output - target,
            model_kwargs,
            return_group_mse=True,
        )
        terms = {"loss": mse, "mse": mse, "t": t}
        terms.update(self._dense_group_metrics(model_output - target, model_kwargs))
        if local_mse is not None and global_mse is not None:
            terms.update({"local_mse": local_mse, "global_mse": global_mse})
        if return_diagnostics:
            terms.update(
                {
                    "x_t": x_t,
                    "model_output": model_output,
                    "pred_velocity": pred_velocity,
                    "target_velocity": target_velocity,
                    "pred_xstart": pred_xstart,
                }
            )
        return terms

    @staticmethod
    def _infer_device(model: Any, noise: Optional[Tensor], device: Optional[Union[str, torch.device]]) -> torch.device:
        if device is not None:
            return torch.device(device)
        if noise is not None:
            return noise.device
        try:
            return next(model.parameters()).device
        except (AttributeError, StopIteration):
            return torch.device("cpu")

    @torch.no_grad()
    def sample_loop_progressive(
        self,
        model: Any,
        shape: Sequence[int],
        model_kwargs: Optional[Dict[str, Any]] = None,
        noise: Optional[Tensor] = None,
        num_steps: Optional[int] = None,
        solver: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        progress: bool = False,
        return_one_step_hidden: bool = False,
        return_one_step_features: bool = False,
    ) -> Iterator[Dict[str, Tensor]]:
        """Integrate from Gaussian noise at ``t=1`` to data at ``t=0``."""
        model_kwargs = {} if model_kwargs is None else model_kwargs
        self._check_no_repaint(model_kwargs)
        if return_one_step_hidden and return_one_step_features:
            raise ValueError("Choose either hidden-only or global feature output")

        shape = tuple(shape)
        if not shape or shape[0] <= 0:
            raise ValueError(f"shape must contain a positive batch dimension, got {shape}.")
        steps = self.num_steps if num_steps is None else num_steps
        if not isinstance(steps, int) or steps <= 0:
            raise ValueError(f"num_steps must be a positive integer, got {steps!r}.")
        chosen_solver = self.solver if solver is None else solver.lower()
        if chosen_solver != "euler":
            raise ValueError("E7 uses Euler sampling")

        device = self._infer_device(model, noise, device)
        if noise is None:
            try:
                dtype = next(model.parameters()).dtype
            except (AttributeError, StopIteration):
                dtype = torch.float32
            x = torch.randn(*shape, device=device, dtype=dtype)
        else:
            if tuple(noise.shape) != shape:
                raise ValueError(f"noise must have shape {shape}, got {tuple(noise.shape)}.")
            x = noise.to(device=device)
        # Label validity belongs to supervised loss, never to the generated
        # state. Otherwise missing/future labels would oracle-clear sampling.
        conditioning = model_kwargs.get("y", {})
        generation_mask = self._generation_mask(x, conditioning)
        context_noise = x.clone() if generation_mask is not None else None
        if "feature_valid_mask" in conditioning:
            feature_mask_for(x, conditioning)  # Reject stale widths, then discard.
            model_kwargs = dict(model_kwargs)
            model_kwargs["y"] = dict(conditioning)
            model_kwargs["y"].pop("feature_valid_mask")

        times = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=torch.float32)
        indices: Any = range(steps)
        if progress:
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        batch_size = shape[0]
        for index in indices:
            t_scalar, next_t_scalar = times[index], times[index + 1]
            t = t_scalar.expand(batch_size)
            first_features = return_one_step_features and index == 0
            prediction = self._model_prediction(
                model,
                x,
                t,
                model_kwargs,
                return_hidden=return_one_step_hidden,
                return_features=first_features,
            )
            if first_features:
                velocity, pred_xstart, hidden, global_hidden = prediction
            elif return_one_step_hidden:
                velocity, pred_xstart, hidden = prediction
            else:
                velocity, pred_xstart = prediction
                hidden = None
            dt = (next_t_scalar - t_scalar).to(dtype=x.dtype)

            next_x = x + dt * velocity
            if generation_mask is not None:
                next_x = torch.where(generation_mask, next_x, context_noise)

            output = {
                "sample": next_x,
                "pred_xstart": pred_xstart,
                "t": t,
                "next_t": next_t_scalar.expand(batch_size),
            }
            if return_one_step_hidden or first_features:
                output["one_step_hidden"] = hidden
            if first_features:
                output["one_step_global"] = global_hidden
            yield output
            x = next_x

    @torch.no_grad()
    def sample_loop(
        self,
        model: Any,
        shape: Sequence[int],
        model_kwargs: Optional[Dict[str, Any]] = None,
        noise: Optional[Tensor] = None,
        num_steps: Optional[int] = None,
        solver: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        progress: bool = False,
        return_all_pred_xstart: bool = False,
        return_one_step_hidden: bool = False,
        return_one_step_features: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, list[Tensor]]]:
        """Sample a batch and optionally return one clean estimate per step."""
        final: Optional[Dict[str, Tensor]] = None
        all_pred_xstart = []
        one_step_hidden = None
        one_step_global = None
        for output in self.sample_loop_progressive(
            model=model,
            shape=shape,
            model_kwargs=model_kwargs,
            noise=noise,
            num_steps=num_steps,
            solver=solver,
            device=device,
            progress=progress,
            return_one_step_hidden=return_one_step_hidden,
            return_one_step_features=return_one_step_features,
        ):
            final = output
            if (return_one_step_hidden or return_one_step_features) and one_step_hidden is None:
                one_step_hidden = output.get("one_step_hidden")
                one_step_global = output.get("one_step_global")
            if return_all_pred_xstart:
                all_pred_xstart.append(output["pred_xstart"])

        # num_steps is validated as positive by sample_loop_progressive.
        assert final is not None
        if return_one_step_features:
            result = {"motion": final["sample"], "one_step_hidden": one_step_hidden,
                      "one_step_global": one_step_global}
            if return_all_pred_xstart:
                result["pred_xstart"] = all_pred_xstart
            return result
        if return_one_step_hidden:
            if return_all_pred_xstart:
                return final["sample"], all_pred_xstart, one_step_hidden
            return final["sample"], one_step_hidden
        if return_all_pred_xstart:
            return final["sample"], all_pred_xstart
        return final["sample"]

__all__ = ["FlowMatching"]
