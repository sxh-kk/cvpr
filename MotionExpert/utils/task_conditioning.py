"""Historical image/camera masks for joint 5Hz motion forecasting."""
from collections.abc import Mapping
import torch

TASKS = ("fore",)
TASK_TO_ID = {"fore": 0}
ID_TO_TASK = {0: "fore"}


def apply_task_conditioning(conditioning: Mapping[str, torch.Tensor], task="fore", *, forecast_prefix=6,
                            trajectory_condition=True, supervise_history=True, noise_only_history=False):
    if task != "fore" or forecast_prefix != 6 or not supervise_history or noise_only_history:
        raise ValueError("MotionExpert uses joint historical6/future5 forecasting")
    original = torch.as_tensor(conditioning.get("padding_mask", conditioning["valid_frames"]))
    batched = original.ndim == 2
    if original.ndim not in (1, 2):
        raise ValueError("valid_frames must be [T] or [B,T]")
    padding = (original if batched else original.unsqueeze(0)).bool()
    batch, frames = padding.shape
    if frames != 11:
        raise ValueError("MotionExpert requires complete eleven-frame windows")
    future = (torch.arange(frames, device=padding.device)[None] >= forecast_prefix).expand(batch, -1)
    def mask(name):
        raw = torch.as_tensor(conditioning.get(name, torch.zeros_like(original)), device=padding.device).bool()
        if raw.shape != original.shape:
            raise ValueError(f"{name} must match valid_frames")
        return raw if batched else raw.unsqueeze(0)
    if isinstance(trajectory_condition, bool):
        visible = torch.full((batch,), trajectory_condition, dtype=torch.bool, device=padding.device)
    else:
        visible = torch.as_tensor(trajectory_condition, device=padding.device)
        if visible.dtype != torch.bool or visible.shape != (batch,):
            raise ValueError("trajectory_condition must be a bool or [B] bool tensor")
    def restore(value):
        return value if batched else value[0]
    result = dict(conditioning)
    result.update(valid_frames=restore(torch.ones_like(padding)), padding_mask=restore(padding),
                  loss_mask=restore(padding), task_conditioning=True,
                  task_id=torch.zeros(batch, device=padding.device, dtype=torch.long) if batched else torch.tensor(0,device=padding.device))
    if "traj" in result:
        result["traj_mask"] = restore((future | mask("traj_mask") | ~padding | ~visible[:,None]).long())
    if "img_embs" in result:
        result["img_mask"] = restore((future | mask("img_mask") | ~padding).long())
    result.pop("motion_generation_mask", None)
    return result
