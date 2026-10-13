"""Task-aligned conditioning masks for unified egocentric motion training.

The original UniEgoMotion training path masks image and trajectory conditions
independently.  The helpers in this module instead construct one of the three
evaluation tasks explicitly, and keep the task target mask separate from the
model's sequence-attention mask.
"""

from __future__ import annotations

from typing import Dict, Mapping, Union

import torch


TASKS = ("recon", "fore", "gen")
TASK_TO_ID = {task: index for index, task in enumerate(TASKS)}
ID_TO_TASK = {index: task for task, index in TASK_TO_ID.items()}


def _as_batched_frames(value: torch.Tensor) -> tuple[torch.Tensor, bool]:
    value = torch.as_tensor(value)
    if value.ndim == 1:
        return value.unsqueeze(0), False
    if value.ndim == 2:
        return value, True
    raise ValueError(f"Expected a T or BxT frame mask, got shape {tuple(value.shape)}.")


def apply_task_conditioning(
    conditioning: Mapping[str, torch.Tensor],
    task: Union[str, torch.Tensor],
    *,
    forecast_prefix: int,
    trajectory_condition: Union[bool, torch.Tensor] = True,
    noise_only_history: bool = False,
    supervise_history: bool = False,
) -> Dict[str, torch.Tensor]:
    """Return task-specific conditioning without mutating the input mapping.

    ``valid_frames`` remains the sequence-attention mask.  ``loss_mask`` keeps
    the original padding information and additionally removes the observed
    forecasting prefix from the training target unless ``supervise_history``
    requests joint historical reconstruction and future prediction. Generation and forecasting
    attend over the complete output window, while padded ground-truth frames
    still contribute no loss. ``trajectory_condition=False`` hides all head
    trajectory values, including the observed prefix, for image-only deployment.
    ``noise_only_history`` additionally marks forecasting history as Gaussian
    context rather than a supervised motion state. This generation mask is a
    task definition and never depends on label validity.
    """

    if forecast_prefix <= 0:
        raise ValueError(f"forecast_prefix must be positive, got {forecast_prefix}.")
    if type(supervise_history) is not bool:
        raise ValueError("supervise_history must be a bool")
    if type(noise_only_history) is not bool:
        raise ValueError("noise_only_history must be a bool")
    if supervise_history and noise_only_history:
        raise ValueError("Historical supervision requires generated motion, not fixed noise context")
    if "valid_frames" not in conditioning:
        raise KeyError("Task conditioning requires a valid_frames mask.")

    result = dict(conditioning)
    padding_mask, was_batched = _as_batched_frames(
        conditioning.get("padding_mask", conditioning["valid_frames"])
    )
    padding_mask = padding_mask.to(dtype=torch.bool)
    batch_size, window = padding_mask.shape
    device = padding_mask.device

    if isinstance(task, str):
        if task not in TASK_TO_ID:
            raise ValueError(f"Unsupported task {task!r}; expected one of {TASKS}.")
        task_ids = torch.full(
            (batch_size,), TASK_TO_ID[task], device=device, dtype=torch.long
        )
    else:
        task_ids = torch.as_tensor(task, device=device, dtype=torch.long)
        if task_ids.ndim == 0:
            task_ids = task_ids.expand(batch_size)
        if task_ids.shape != (batch_size,):
            raise ValueError(
                f"task IDs must have shape ({batch_size},), got {tuple(task_ids.shape)}."
            )
        if bool(torch.any((task_ids < 0) | (task_ids >= len(TASKS)))):
            raise ValueError(f"task IDs must lie in [0, {len(TASKS) - 1}].")

    if bool(torch.any(task_ids == TASK_TO_ID["fore"])) and forecast_prefix >= window:
        raise ValueError("Forecasting requires forecast_prefix smaller than the output window")

    frame_indices = torch.arange(window, device=device).unsqueeze(0)
    # Observation is a time interval, not a count of valid observations. A
    # missing image/head frame inside the prefix cannot move the cutoff.
    observed_prefix = (frame_indices < forecast_prefix).expand(batch_size, -1)

    recon_rows = task_ids == TASK_TO_ID["recon"]
    fore_rows = task_ids == TASK_TO_ID["fore"]
    gen_rows = task_ids == TASK_TO_ID["gen"]

    condition_mask = torch.zeros_like(padding_mask)
    condition_mask[fore_rows] = ~observed_prefix[fore_rows]
    condition_mask[gen_rows] = True
    attention_mask = padding_mask.clone()
    attention_mask[fore_rows | gen_rows] = True
    loss_mask = padding_mask.clone()
    if not supervise_history:
        loss_mask[fore_rows] &= ~observed_prefix[fore_rows]

    traj_mask = condition_mask.clone()
    img_mask = condition_mask.clone()
    if window > 0:
        img_mask[gen_rows, 0] = False

    def original_mask(name):
        if name not in conditioning:
            return torch.zeros_like(padding_mask)
        value, existing_batched = _as_batched_frames(conditioning[name])
        if value.shape != padding_mask.shape or existing_batched != was_batched:
            raise ValueError(f"{name} must match the shape of valid_frames")
        return value.to(device=device, dtype=torch.bool)

    # Preserve missing-data masks and any stricter mask supplied by callers.
    traj_mask = traj_mask | original_mask("traj_mask") | ~padding_mask
    img_mask = img_mask | original_mask("img_mask") | ~padding_mask

    if isinstance(trajectory_condition, bool):
        trajectory_visible = torch.full((batch_size,), trajectory_condition,
                                        device=device, dtype=torch.bool)
    else:
        trajectory_visible = torch.as_tensor(trajectory_condition, device=device)
        if trajectory_visible.dtype != torch.bool or trajectory_visible.shape != (batch_size,):
            raise ValueError("trajectory_condition must be a bool or a [B] boolean tensor")
    traj_mask |= ~trajectory_visible[:, None]

    def restore_batch_shape(value: torch.Tensor) -> torch.Tensor:
        return value if was_batched else value[0]

    if "traj" in conditioning:
        result["traj_mask"] = restore_batch_shape(traj_mask.to(dtype=torch.long))
    if "img_embs" in conditioning:
        result["img_mask"] = restore_batch_shape(img_mask.to(dtype=torch.long))

    result["valid_frames"] = restore_batch_shape(attention_mask.to(dtype=torch.long))
    result["loss_mask"] = restore_batch_shape(loss_mask.to(dtype=torch.long))
    result["padding_mask"] = restore_batch_shape(padding_mask.to(dtype=torch.long))
    result["task_id"] = task_ids if was_batched else task_ids[0]
    result["task_conditioning"] = True
    if noise_only_history:
        generation_mask = torch.ones_like(padding_mask)
        generation_mask[fore_rows] = ~observed_prefix[fore_rows]
        result["motion_generation_mask"] = restore_batch_shape(generation_mask)
    else:
        # Reapplying a task must not retain a previous future-only experiment's
        # fixed-history state when the caller switches to joint generation.
        result.pop("motion_generation_mask", None)
    return result


__all__ = [
    "TASKS",
    "TASK_TO_ID",
    "ID_TO_TASK",
    "apply_task_conditioning",
]
