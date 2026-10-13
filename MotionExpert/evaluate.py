"""Task-aware sampling metrics in metres/degrees, using predicted floor deltas."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .config import K12_JOINTS, architecture_metadata, load_config, validate_architecture
from .dataset.representation import decode_motion
from .dataset.motion_layout import motion_layout, SMPLH_REPRE_TYPE
from .utils.rotation_conversions import rotation_6d_to_matrix
from .utils.task_conditioning import TASKS, ID_TO_TASK, apply_task_conditioning
from .visual_contract import archive_visual_compatibility


def _rotation_error(prediction, target):
    """A degenerate prediction incurs 180 degrees, rather than disappearing."""
    first, second = prediction[..., :3], prediction[..., 3:]
    first_norm = first.norm(dim=-1)
    first_unit = first / first_norm.clamp_min(1e-8)[..., None]
    second_norm = (second - (second * first_unit).sum(-1)[..., None] * first_unit).norm(dim=-1)
    proper = (first_norm > 1e-7) & (second_norm > 1e-7)
    pred_rotation = rotation_6d_to_matrix(prediction)
    true_rotation = rotation_6d_to_matrix(target)
    relative = pred_rotation @ true_rotation.transpose(-1, -2)
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    angle = torch.rad2deg(torch.acos(cosine))
    return torch.where(proper, angle, torch.full_like(angle, 180)), ~proper


def compute_motion_metrics(prediction, target, valid, *, task="fore", forecast_prefix=6,
                           fps=5, world_from_window=None, floor_transform=None,
                           floor_height=0., representation=None):
    """Score one unnormalized, unpadded T x 245 window.

    Forecast world recovery starts at the last observed reference transform.
    Missing reference edges stop integration; coverage is reported explicitly.
    Neither future camera poses nor ground-truth motion anchor future frames.
    Every mean has a matching ``_count`` for weighted aggregation.
    """
    if task not in TASKS:
        raise ValueError(f"Unsupported task: {task}")
    prediction = torch.as_tensor(prediction).float()
    target = torch.as_tensor(target, device=prediction.device).float()
    valid = torch.as_tensor(valid, device=prediction.device).bool()
    if prediction.ndim != 2 or target.shape != prediction.shape or valid.shape != prediction.shape:
        raise ValueError("Metrics require matching T x D motion and label mask")
    layout = motion_layout(representation or prediction.shape[-1])
    if prediction.shape[-1] != layout.motion_dim:
        raise ValueError("Metric feature width disagrees with the representation")
    joints = list(layout.joint_indices)
    if not torch.isfinite(prediction).all() or not torch.isfinite(target[valid]).all():
        raise ValueError("Non-finite predictions or valid targets cannot be scored")
    start = forecast_prefix if task == "fore" else 0
    if not 0 <= start < len(target):
        raise ValueError("Evaluation region must contain real target frames")
    result = {}

    def record(name, values, mask=None):
        values = torch.as_tensor(values, device=prediction.device).float()
        selected = values.flatten() if mask is None else values[mask]
        result[name + "_count"] = int(selected.numel())
        if selected.numel():
            result[name] = float(selected.mean())

    body_pred, body_true = prediction[:, layout.body_slice].reshape(-1, len(joints), 9), target[:, layout.body_slice].reshape(-1, len(joints), 9)
    body_valid = valid[:, layout.body_slice].reshape(-1, len(joints), 9)
    pos_valid, rot_valid = body_valid[..., 6:].all(-1), body_valid[..., :6].all(-1)
    region = torch.arange(len(target), device=prediction.device) >= start
    first_2s = region & (torch.arange(len(target), device=prediction.device) < start + 2 * fps)
    position_error = (body_pred[..., 6:] - body_true[..., 6:]).norm(dim=-1)
    angle, degenerate = _rotation_error(body_pred[..., :6], body_true[..., :6])
    record("body_local_mpjpe_m", position_error, pos_valid & region[:, None])
    if layout.representation == SMPLH_REPRE_TYPE:
        first_1s = region & (torch.arange(len(target),device=prediction.device) < start+fps)
        record("body_local_1s_mpjpe_m",position_error,pos_valid & first_1s[:,None])
    else:
        record("body_local_2s_mpjpe_m",position_error,pos_valid & first_2s[:,None])
    record("body_rotation_error_deg", angle, rot_valid & region[:, None])
    record("body_degenerate_rotation_fraction", degenerate, rot_valid & region[:, None])
    for side, begin in (("left", layout.left_hand_slice.start), ("right", layout.right_hand_slice.start)):
        hand_pred = prediction[:, begin:begin + 63].reshape(-1, 21, 3)
        hand_true = target[:, begin:begin + 63].reshape(-1, 21, 3)
        hand_valid = valid[:, begin:begin + 63].reshape(-1, 21, 3).all(-1)
        record(f"{side}_hand_local_mpjpe_m", (hand_pred - hand_true).norm(dim=-1), hand_valid & region[:, None])
        wrist = joints.index(20 if side == "left" else 21)
        record(f"{side}_wrist_disagreement_m", (hand_pred[:, 0] - body_pred[:, wrist, 6:]).norm(dim=-1),
               hand_valid[:, 0] & pos_valid[:, wrist] & region)

    delta = layout.global_slice.start
    delta_valid = valid[:, layout.global_slice].all(-1)
    delta_angle, _ = _rotation_error(prediction[:, delta:delta+6], target[:, delta:delta+6])
    record("floor_delta_rotation_error_deg", delta_angle, delta_valid & region)
    record("floor_delta_translation_error_m", (prediction[:, delta+6:delta+9] - target[:, delta+6:delta+9]).norm(dim=-1), delta_valid & region)
    contacts = prediction[:, layout.contact_slice] >= .5
    true_contacts = target[:, layout.contact_slice] >= .5
    contact_valid = valid[:, layout.contact_slice] & region[:, None]
    record("contact_accuracy", contacts == true_contacts, contact_valid)
    record("predicted_contact_fraction", contacts, contact_valid)
    tp = int((contacts & true_contacts & contact_valid).sum())
    fp = int((contacts & ~true_contacts & contact_valid).sum())
    fn = int((~contacts & true_contacts & contact_valid).sum())
    result.update(contact_tp=tp, contact_fp=fp, contact_fn=fn)
    if contact_valid.any():
        result["contact_f1"] = 2 * tp / max(2 * tp + fp + fn, 1)
    feet = [joints.index(10), joints.index(11)]
    record("foot_penetration_m", (-body_pred[:, feet, 8]).clamp_min(0), pos_valid[:, feet] & region[:, None])

    bad = torch.nonzero(~delta_valid[start:], as_tuple=False)
    end = start + int(bad[0]) if len(bad) else len(target)
    record("reference_chain_coverage", (end - start) / (len(target) - start))
    record("world_decode_failure", 0.)
    if end <= start:
        return result
    kwargs = dict(window_to_world=world_from_window, floor_height=floor_height, start_frame=start)
    if start:
        if floor_transform is None:
            raise ValueError("Forecast metrics require the last observed floor transform")
        kwargs["initial_floor_transform"] = torch.as_tensor(floor_transform)[start - 1]
    try:
        pred_world = decode_motion(prediction[:end], **kwargs)
        true_world = decode_motion(target[:end], **kwargs)
    except ValueError as error:
        if "Degenerate decoded floor heading" not in str(error):
            raise
        record("world_decode_failure", 1.)
        return result
    chosen = slice(start, end)
    floor_error = (pred_world["floor_positions"] - true_world["floor_positions"]).norm(dim=-1)
    record("floor_ate_m", floor_error)
    record("floor_final_drift_m", floor_error[-1])
    relative = pred_world["floor_rotations"] @ true_world["floor_rotations"].transpose(-1, -2)
    record("floor_yaw_error_deg", torch.rad2deg(torch.atan2(relative[:, 1, 0], relative[:, 0, 0])).abs())
    world_body_error = (pred_world["body_positions"] - true_world["body_positions"]).norm(dim=-1)
    record("body_world_mpjpe_m", world_body_error, pos_valid[chosen])
    upto = min(2 * fps, end - start)
    if layout.representation == SMPLH_REPRE_TYPE:
        record("body_world_1s_mpjpe_m",world_body_error[:fps],pos_valid[chosen][:fps])
    else:
        record("body_world_2s_mpjpe_m",world_body_error[:upto],pos_valid[chosen][:upto])
    for side, begin in (("left", layout.left_hand_slice.start), ("right", layout.right_hand_slice.start)):
        hand_valid = valid[chosen, begin:begin + 63].reshape(-1, 21, 3).all(-1)
        error = (pred_world[f"{side}_hand"] - true_world[f"{side}_hand"]).norm(dim=-1)
        record(f"{side}_hand_world_mpjpe_m", error, hand_valid)
    if end - start > 1:
        foot_speed = torch.diff(pred_world["body_positions"][:, feet], dim=0).norm(dim=-1) * fps
        foot_valid = pos_valid[chosen][:, feet]
        speed_valid = foot_valid[1:] & foot_valid[:-1] & contacts[chosen][1:] & valid[chosen, layout.contact_slice][1:]
        record("contact_foot_slide_m_s", foot_speed, speed_valid)
    return result


@torch.no_grad()
def sampled_validation(module, dataset, max_samples=16, forecast_prefix=6, tasks=TASKS, seed=None):
    """Fixed window subset, task-specific floor policy and actual flow sampling."""
    if max_samples < 1:
        raise ValueError("max_samples must be positive")
    if forecast_prefix != dataset.forecast_prefix:
        raise ValueError("Forecast prefix must agree with dataset floor policy")
    device = next(module.model.parameters()).device
    sums, counts, confusion = defaultdict(float), defaultdict(int), defaultdict(int)
    selected = np.linspace(0, len(dataset) - 1, min(max_samples, len(dataset)), dtype=int)
    random_seed = int(module.cfg.TRAIN.SEED if seed is None else seed)
    cuda_devices = [device.index or 0] if device.type == "cuda" else []
    normalizer = module.normalizer
    completed, skipped = defaultdict(int), defaultdict(int)
    for task in tasks:
        if task not in TASKS:
            raise ValueError(f"Unsupported evaluation task {task}")
        for index in selected:
            try:
                sample = dataset.get_sample(int(index), task=task)
            except ValueError as error:
                if task == "fore" and str(error).startswith("Forecast window"):
                    skipped[task] += 1
                    continue
                raise
            misc = sample["misc"]
            count = int(misc["num_real_frames"])
            y = {key: value.unsqueeze(0).to(device) if isinstance(value, torch.Tensor) else value
                 for key, value in sample["y"].items()}
            if dataset.normalizer is None:
                y["traj"] = normalizer.normalize(y["traj"], "traj", y["trajectory_valid"])
                target = misc["motion"]
            else:
                target = dataset.denormalize(misc["motion"])
            data = getattr(module.cfg, "DATA", None)
            if data is not None and data.FPS == 5 and getattr(data, "TRAIN_TASK", None) == "fore":
                y = apply_task_conditioning(
                    y, task, forecast_prefix=forecast_prefix,
                    trajectory_condition=False,
                    supervise_history=True)
            # Restoring the RNG avoids changing subsequent training minibatches.
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(random_seed + int(index))
                output = module.sample(y)
            prediction = output["motion"] if isinstance(output, dict) else output
            if isinstance(prediction, tuple):
                prediction = prediction[0]
            prediction = normalizer.denormalize(prediction[0, :count]).cpu()
            metrics = compute_motion_metrics(prediction, target[:count].cpu(), misc["feature_valid_mask"][:count].cpu(),
                task=task, forecast_prefix=forecast_prefix, fps=dataset.fps,
                world_from_window=misc["world_from_window"], floor_transform=misc["floor_transform"],
                floor_height=misc["floor_height"], representation=getattr(dataset, "repre_type", None))
            completed[task] += 1
            for name, value in metrics.items():
                key = f"{task}/{name}"
                if name in ("contact_tp", "contact_fp", "contact_fn"):
                    confusion[key] += value
                elif name.endswith("_count") or name == "contact_f1":
                    continue
                else:
                    n = metrics.get(name + "_count", 1)
                    sums[key] += value * n
                    counts[key] += n
    result = {key: sums[key] / counts[key] for key in sums if counts[key]}
    result.update({key + "_count": value for key, value in counts.items()})
    for task in tasks:
        result[f"{task}/sample_count"] = completed[task]
        result[f"{task}/skipped_window_count"] = skipped[task]
        tp, fp, fn = (confusion[f"{task}/contact_{key}"] for key in ("tp", "fp", "fn"))
        if tp + fp + fn:
            result[f"{task}/contact_f1"] = 2 * tp / max(2 * tp + fp + fn, 1)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config")
    parser.add_argument("--data-dir")
    parser.add_argument("--split-manifest")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=None,
                        help="Forecasting validation")
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=62)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-visual-adaptation", action="store_true",
                        help="Explicitly allow a different shared-RGB size profile; DINO weight and normalization identities must match")
    args = parser.parse_args()
    from .motion_expert import MotionExpert
    from .dataset.smplh_motion_dataset import SMPLHMotionDataset
    expert = MotionExpert(args.checkpoint, device=args.device, dtype=torch.float32)
    cfg = load_config(args.config, data_dir=args.data_dir) if args.config else expert.cfg
    if args.data_dir:
        cfg.DATA.ARCHIVE_ROOT = str(Path(args.data_dir).resolve())
    if args.split_manifest:
        cfg.DATA.SPLIT_MANIFEST = str(Path(args.split_manifest).resolve())
    validate_architecture(cfg)
    if architecture_metadata(cfg) != architecture_metadata(expert.cfg):
        raise ValueError("Evaluation configuration must match the checkpoint architecture and temporal profile")
    data = cfg.DATA
    tasks = args.tasks or ([data.TRAIN_TASK] if getattr(data, "TRAIN_TASK", None) else list(TASKS))
    # Embedded training statistics travel with the checkpoint. Dataset checks
    # their split/basis provenance even when the original stats file moved.
    dataset = SMPLHMotionDataset(archive_root=data.ARCHIVE_ROOT, split=args.split,
        repre_type=data.REPRE_TYPE,
        split_manifest=data.SPLIT_MANIFEST, stats_path=expert.normalizer.stats,
        do_normalization=True, cond_img_feat=data.COND_IMG_FEAT, cond_traj=data.COND_TRAJ,
        window=data.WINDOW, fps=data.FPS, forecast_prefix=data.FORECAST_PREFIX, segment_stride=data.SEGMENT_STRIDE,
        train_task=getattr(data, "TRAIN_TASK", None) or "fore",
        motion_supervision=data.MOTION_SUPERVISION,
        require_full_window=getattr(data, "REQUIRE_FULL_WINDOW", False),
        rotation_basis_mode=data.ROTATION_BASIS_MODE, target_joint_basis=data.TARGET_JOINT_BASIS,
        rotation_basis_path=data.ROTATION_BASIS_PATH, normalization_clip=expert.normalizer.clip,
        contact_speed_m_s=data.CONTACT_SPEED_M_S, contact_height_m=data.CONTACT_HEIGHT_M,
        min_valid_image_fraction=data.MIN_VALID_IMAGE_FRACTION,
        vision_profile_root=getattr(data,"VISION_PROFILE_ROOT",None),
        image_profile=getattr(data,"IMAGE_PROFILE",None))
    visual_metadata = archive_visual_compatibility(dataset, expert.visual_conditioning,
                        allow_visual_adaptation=args.allow_visual_adaptation)
    def sample(y):
        task = ID_TO_TASK[int(y["task_id"][0])]
        return expert.sample(y, task=task, forecast_prefix=data.FORECAST_PREFIX,
                             output_frames=data.WINDOW, normalized=True)
    adapter = SimpleNamespace(model=expert.model, normalizer=expert.normalizer, cfg=cfg, sample=sample)
    metrics = sampled_validation(adapter, dataset, max_samples=args.max_samples,
        forecast_prefix=data.FORECAST_PREFIX, tasks=tasks, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"checkpoint": str(args.checkpoint.resolve()), "split": args.split,
        "split_fingerprint": dataset.split_fingerprint, "seed": args.seed,
        "max_samples": args.max_samples, "tasks": tasks,
        "fps": data.FPS, "window": data.WINDOW, "forecast_prefix": data.FORECAST_PREFIX,
        "conditioning": "images_only",
        **visual_metadata,
        "metrics": metrics}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
