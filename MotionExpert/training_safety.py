"""Collective, pre-update training gates and bounded failure-only diagnostics.

A finite but enormous loss is not made safe by gradient clipping. All ranks
make the same decision *before* backward or optimizer.step, including ranks
whose own batch was healthy. Healthy steps perform no diagnostic disk writes.
"""

from collections import deque
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import statistics

import numpy as np
import torch


SAFETY_FORMAT = "motionexpert-training-safety-v1"
# Objective magnitude and diagnostic-group magnitude have different scales.
# Identify the actual objective by name, including singleton per-sample arrays.
OBJECTIVE_LOSS_METRICS = frozenset({"loss", "flow_loss", "per_sample_flow_loss"})


class TrainingSafetyError(FloatingPointError):
    """Every distributed rank aborts before an unsafe optimizer update."""


@dataclass(frozen=True)
class SafetyLimits:
    enabled: bool = True
    max_loss: float = 100.
    max_grad_norm: float = 1000.
    max_activation_rms: float = 100.
    rolling_window: int = 128
    rolling_warmup_steps: int = 32
    rolling_multiplier: float = 50.
    rolling_min_loss: float = 1.
    save_batch: bool = True

    @classmethod
    def from_config(cls, config):
        values = {}
        for key, default in asdict(cls()).items():
            name = "SAFETY_" + key.upper()
            values[key] = config.get(name, default) if isinstance(config, Mapping) else getattr(config, name, default)
        return cls(**values)


def collective_max(values):
    """Exactly one small collective per gate; never raise on a local flag."""
    result = values.detach().clone()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(result, op=torch.distributed.ReduceOp.MAX)
    return result


def validate_safety_precision(config, *, use_gpu=True):
    """Fused FP16 can defer unscale until after the pre-step hook."""
    precision = str(getattr(config, "PRECISION", "32-true"))
    if use_gpu and precision in ("16-mixed", "16") and getattr(config, "FUSED_ADAMW", False):
        raise ValueError("FP16 with fused AdamW cannot guarantee unscaled pre-update safety checks; "
                         "use non-fused AdamW or BF16")


def gradient_norm(optimizer, device):
    """Inspect the actual, unscaled gradients without changing or clipping them."""
    gradients = [parameter.grad for group in optimizer.param_groups for parameter in group["params"]
                 if parameter.grad is not None]
    if not gradients:
        return torch.zeros((), device=device)
    # Foreach avoids one CPU/device synchronization per parameter. DDP has
    # already reduced these gradients when Lightning calls the pre-step hook.
    by_device_dtype = {}
    for gradient in gradients:
        if gradient.is_sparse:
            gradient = gradient.coalesce().values()
        by_device_dtype.setdefault((gradient.device, gradient.dtype), []).append(gradient.detach())
    norms = []
    for group in by_device_dtype.values():
        # Stack before dtype conversion: casting a thousand scalar tensors one
        # by one would add a thousand tiny kernels to every training step.
        norms.append(torch.stack(torch._foreach_norm(group, 2)).to(device=device, dtype=torch.float64))
    return torch.linalg.vector_norm(torch.cat(norms))


class TrainingSafety:
    """Rolling history uses globally agreed healthy training losses only."""

    def __init__(self, config):
        self.limits = SafetyLimits.from_config(config)
        self.history = deque(maxlen=self.limits.rolling_window)

    def state_dict(self):
        return {"format": SAFETY_FORMAT, "limits": asdict(self.limits), "history": list(self.history)}

    def load_state_dict(self, state):
        if not isinstance(state, Mapping) or state.get("format") != SAFETY_FORMAT:
            raise ValueError("Invalid training safety checkpoint state")
        history = state.get("history", [])
        if not isinstance(history, list) or any(
                isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
                for value in history):
            raise ValueError("Invalid training safety loss history")
        # Changing a limit explicitly is allowed; retain the most recent healthy
        # baseline rather than silently disabling it after resume.
        self.history = deque(history[-self.limits.rolling_window:], maxlen=self.limits.rolling_window)

    def check_forward(self, metrics, *, device, activation_rms=None, prediction=None, training=False):
        # Every loss/diagnostic remains subject to the finite gate. The shared
        # magnitude limit applies only to weighted objectives: a narrow,
        # unweighted group may have a legitimate much larger target-energy tail.
        values = {name: value.detach().float().reshape(-1) for name, value in metrics.items()
                  if not name.endswith("_valid_count")}
        joined = torch.cat(list(values.values())) if values else torch.zeros(1, device=device)
        objectives = [value for name, value in values.items() if name in OBJECTIVE_LOSS_METRICS]
        objective = torch.cat(objectives) if objectives else torch.zeros(1, device=device)
        loss = metrics.get("loss", torch.zeros((), device=device)).detach().float().mean()
        activation = (activation_rms.detach().float().reshape(-1) if activation_rms is not None
                      else torch.zeros(1, device=device))
        prediction_nonfinite = ((~torch.isfinite(prediction.detach())).any().float() if prediction is not None
                                else torch.zeros((), device=device))
        packed = torch.stack((
            (~torch.isfinite(joined)).any().float(),
            torch.nan_to_num(joined.abs(), nan=float("inf"), posinf=float("inf"), neginf=float("inf")).amax(),
            torch.nan_to_num(objective.abs(), nan=float("inf"), posinf=float("inf"), neginf=float("inf")).amax(),
            torch.nan_to_num(loss, nan=float("inf"), posinf=float("inf"), neginf=float("inf")),
            (~torch.isfinite(activation)).any().float(),
            torch.nan_to_num(activation.abs(), nan=float("inf"), posinf=float("inf"), neginf=float("inf")).amax(),
            prediction_nonfinite,
        ))
        global_values = collective_max(packed).cpu().tolist()
        (nonfinite, loss_max, objective_max, loss_mean_max,
         activation_nonfinite, activation_max, prediction_nonfinite) = global_values
        reasons = []
        if nonfinite:
            reasons.append("non-finite loss or loss component")
        if activation_nonfinite:
            reasons.append("non-finite hidden activation")
        if prediction_nonfinite:
            reasons.append("non-finite model prediction")
        limits = self.limits
        rolling_limit = None
        if limits.enabled:
            if objective_max > limits.max_loss:
                reasons.append("objective loss exceeds absolute limit")
            if activation_max > limits.max_activation_rms:
                reasons.append("hidden token RMS exceeds absolute limit")
            if training and len(self.history) >= limits.rolling_warmup_steps:
                rolling_limit = max(limits.rolling_min_loss, limits.rolling_multiplier * statistics.median(self.history))
                if loss_mean_max > rolling_limit:
                    reasons.append("loss exceeds healthy rolling baseline")
        result = {"stage": "forward", "reasons": reasons,
                  "global_loss_component_abs_max": loss_max, "global_loss_mean_max": loss_mean_max,
                  "global_objective_loss_abs_max": objective_max,
                  "global_activation_rms_max": activation_max, "rolling_limit": rolling_limit,
                  "healthy_history_count": len(self.history)}
        if training and not reasons:
            self.history.append(loss_mean_max)
        return result

    def check_gradients(self, optimizer, *, device):
        norm = gradient_norm(optimizer, device)
        packed = torch.stack(((~torch.isfinite(norm)).to(norm),
                              torch.nan_to_num(norm, nan=float("inf"), posinf=float("inf"), neginf=float("inf"))))
        nonfinite, maximum = collective_max(packed).cpu().tolist()
        reasons = []
        if nonfinite:
            reasons.append("non-finite unscaled gradient norm")
        if self.limits.enabled and maximum > self.limits.max_grad_norm:
            reasons.append("unscaled gradient norm exceeds absolute limit")
        return {"stage": "before_optimizer_step", "reasons": reasons, "global_unclipped_grad_norm_max": maximum}


def rng_state(device):
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state()}
    if torch.device(device).type == "cuda":
        state["torch_cuda"] = torch.cuda.get_rng_state(device)
    return state


def _cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        return {key: _cpu_tree(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_tree(child) for child in value)
    return value


def _finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _finite_json(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(child) for child in value]
    return value


def _tensor_summary(value):
    if not isinstance(value, torch.Tensor):
        if isinstance(value, Mapping):
            return {key: _tensor_summary(child) for key, child in value.items()}
        if isinstance(value, (list, tuple)):
            return [_tensor_summary(child) for child in value]
        return value
    value = value.detach()
    result = {"shape": list(value.shape), "dtype": str(value.dtype), "numel": value.numel()}
    if value.dtype == torch.bool:
        result["true_count"] = value.sum().item()
    elif value.numel():
        finite = torch.isfinite(value)
        result["finite_count"] = finite.sum().item()
        if finite.any():
            selected = value[finite]
            result.update(min=selected.min().item(), max=selected.max().item())
    return result


def write_failure(module, result, context):
    """Write one bounded rank-local batch, not a 400M-parameter checkpoint."""
    root = Path(module.cfg.TRAIN.EXP_PATH).resolve() / "anomalies"
    repo = Path(__file__).resolve().parents[1]
    if root.is_relative_to(repo):
        root = Path("/tmp/motionexpert-anomalies")
    rank = torch.distributed.get_rank() if torch.distributed.is_available() and torch.distributed.is_initialized() else 0
    epoch, step = int(module.current_epoch), int(module.global_step)
    suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = root / f"epoch{epoch:04d}-step{step:09d}-rank{rank}-{suffix}-pid{os.getpid()}"
    output.mkdir(parents=True, exist_ok=False)
    batch = context.get("batch", {})
    summary = {"format": SAFETY_FORMAT, "rank": rank, "pid": os.getpid(), "epoch": epoch,
               "global_step": step, "batch_idx": context.get("batch_idx"), "mode": context.get("mode"),
               "sampling_seed": context.get("sampling_seed"), "sampling_task": context.get("task"),
               "decision": result, "limits": asdict(module.training_safety.limits),
               "model_weights_saved": False, "optimizer_update_applied": False,
               "batch_summary": _tensor_summary(batch),
               "conditioning_summary": _tensor_summary(context.get("conditioning", {})),
               "activation_rms_summary": _tensor_summary(context.get("activation_rms")),
               "metrics": _tensor_summary(context.get("metrics", {})),
               "parameter_identity": [{"name": name, "shape": list(parameter.shape), "dtype": str(parameter.dtype),
                                       "version": parameter._version, "requires_grad": parameter.requires_grad}
                                      for name, parameter in module.model.named_parameters()]}
    from .config import as_dict
    summary["config"] = as_dict(module.cfg)
    if module.training_safety.limits.save_batch:
        payload = {"format": SAFETY_FORMAT, "batch": _cpu_tree(batch),
                   "conditioning": _cpu_tree(context.get("conditioning")),
                   "flow_diagnostics": _cpu_tree(context.get("flow_diagnostics")),
                   "activation_rms": _cpu_tree(context.get("activation_rms")),
                   "rng_before_step": context.get("rng_before_step"), "rng_at_detection": rng_state(module.device),
                   "decision": result,
                   "sampling_seed": context.get("sampling_seed"), "sampling_task": context.get("task"),
                   "validation_t1_noise": _cpu_tree(context.get("validation_t1_noise"))}
        # Keep every microbatch contributing to this update, not an unrelated
        # next batch. Only references are held on healthy accumulation steps.
        payload["accumulated_microbatches"] = [_cpu_tree(child)
                                               for child in getattr(module, "_safety_microbatches", [])]
        torch.save(payload, output / "batch.pt")
    (output / "summary.json").write_text(json.dumps(_finite_json(summary), ensure_ascii=False, indent=2,
                                                   allow_nan=False) + "\n", encoding="utf-8")
    return output
