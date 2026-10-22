"""Isolated single-GPU full-task AdamW A/B acceptance, never a training launcher.

Run only after the training job has saved a protected complete checkpoint and
released the GPUs. One GPU holds one FP32 model/optimizer at a time. CPU keeps
the restored source and reference results. No checkpoint or dataset is modified.
This deliberately does not test DDP, the global batch of 32, sampler restoration
or production scheduler advancement; those require a separate resume acceptance.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from adamw_task_resume import select_adamw_backend, optimizer_state_identity
from benchmark_fused_adamw import tensor_comparison


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", copy=True)
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 ** 2), b""):
            digest.update(block)
    return digest.hexdigest()


def restore_optimizer(optimizer, saved, device, *, fused):
    """Do not alias saved CPU counters or transiently retain two GPU moment sets."""
    optimizer.zero_grad(set_to_none=True)
    optimizer.state.clear()
    restored = {"param_groups": copy.deepcopy(saved["param_groups"]), "state": {}}
    for key, entry in saved["state"].items():
        restored["state"][key] = {name: value.clone() if isinstance(value, torch.Tensor)
            and (name == "step" or torch.device(device).type == "cpu") else value
            for name, value in entry.items()}
    # Load the unmodified saved recipe first, then use the production migration.
    optimizer.load_state_dict(restored)
    return select_adamw_backend(optimizer, fused=fused)


def restore_rng(saved, device, *, cuda_index=0):
    required = {"random_state", "numpy_random_seed", "torch_manual_seed"}
    if not required.issubset(saved):
        raise ValueError("Checkpoint is missing Python/NumPy/Torch RNG state")
    random.setstate(saved["random_state"])
    np.random.set_state(saved["numpy_random_seed"])
    torch.set_rng_state(saved["torch_manual_seed"].cpu())
    if torch.device(device).type == "cuda":
        states = saved.get("torch_cuda_manual_seed")
        if not isinstance(states, (tuple, list)) or not 0 <= cuda_index < len(states):
            raise ValueError("Checkpoint lacks the selected rank-local CUDA RNG state")
        torch.cuda.set_rng_state(states[cuda_index].cpu(), device)


def compare_maps(reference, candidate, *, atol, rtol, chunk_elements):
    if set(reference) != set(candidate):
        return {"passed": False, "missing": sorted(set(reference) - set(candidate)),
                "unexpected": sorted(set(candidate) - set(reference))}
    summary = {"passed": True, "tensors": len(reference), "elements": 0,
               "max_abs": 0.0, "max_tolerance_ratio": 0.0, "failures": [], "worst_tensors": []}
    worst = []
    for name, expected in reference.items():
        actual = candidate[name]
        if expected is None or actual is None:
            if expected is not actual:
                summary["passed"] = False
                summary["failures"].append({"name": name, "reason": "None-gradient pattern differs"})
            continue
        exact = not expected.is_floating_point() or name.endswith("/step")
        check = tensor_comparison(expected, actual, atol=0.0 if exact else atol,
            rtol=0.0 if exact else rtol, chunk_elements=chunk_elements)
        summary["elements"] += expected.numel()
        summary["passed"] &= check["passed"]
        summary["max_abs"] = max(summary["max_abs"], check.get("max_abs", 0.0))
        summary["max_tolerance_ratio"] = max(summary["max_tolerance_ratio"], check.get("max_tolerance_ratio", 0.0))
        entry = {"name": name, **check}
        if not check["passed"]:
            summary["failures"].append(entry)
        worst.append(entry)
    summary["worst_tensors"] = sorted(worst, key=lambda item: item.get("max_tolerance_ratio", float("inf")), reverse=True)[:10]
    return summary


def optimizer_tensors(optimizer):
    result = {}
    for group_index, group in enumerate(optimizer.param_groups):
        for parameter_index, parameter in enumerate(group["params"]):
            for name, value in optimizer.state.get(parameter, {}).items():
                if isinstance(value, torch.Tensor):
                    result[f"{group_index}/{parameter_index}/{name}"] = value
    return result


def validate_saved_optimizer(optimizer, saved, scheduler, expected_names):
    groups = saved["param_groups"]
    if [group.get("group_name") for group in groups] != list(expected_names):
        raise ValueError("Saved optimizer must contain the same ordered five task groups")
    if len(groups) != len(optimizer.param_groups):
        raise ValueError("Saved optimizer group count differs")
    initialized = 0
    for actual, group in zip(optimizer.param_groups, groups):
        if len(actual["params"]) != len(group["params"]):
            raise ValueError("Saved optimizer parameter order/count differs")
        for parameter, key in zip(actual["params"], group["params"]):
            entry = saved["state"].get(key)
            if not entry:
                continue  # A never-selected sparse expert is allowed no state.
            initialized += 1
            if set(entry) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError("Require the task's ordinary AdamW state, without AMSGrad/scaler changes")
            for field in ("exp_avg", "exp_avg_sq"):
                if entry[field].shape != parameter.shape or entry[field].dtype != torch.float32:
                    raise ValueError("Saved moment shape/dtype differs from the FP32 model")
            step = float(entry["step"])
            if not np.isfinite(step) or step <= 0 or step != int(step):
                raise ValueError("Expected a nonzero integral saved Adam counter")
        for field in ("betas", "eps", "weight_decay"):
            if actual[field] != group[field]:
                raise ValueError(f"Saved optimizer recipe differs: {field}")
    if not initialized:
        raise ValueError("Cannot validate migration with an empty optimizer")
    if [float(g["lr"]) for g in groups] != [float(lr) for lr in scheduler.get("_last_lr", [])]:
        raise ValueError("Checkpoint optimizer and scheduler learning rates disagree")
    return {"initialized_parameter_states": initialized, "scheduler_last_epoch": scheduler.get("last_epoch"),
            "scheduler_advanced_in_validation": False, "learning_rates": [g["lr"] for g in groups]}


def check_finite_model_optimizer(model, optimizer):
    groups = defaultdict(list)
    for parameter in model.parameters():
        if parameter.is_floating_point():
            groups[(parameter.device, parameter.dtype)].append(parameter.detach())
    for value in optimizer_tensors(optimizer).values():
        if value.is_floating_point():
            groups[(value.device, value.dtype)].append(value.detach())
    for values in groups.values():
        norms = torch._foreach_norm(values, float("inf"))
        if not bool(torch.isfinite(torch.stack(norms)).all()):
            raise FloatingPointError("Nonfinite parameter or Adam state after the actual update")


def changed_groups(model, source, optimizer, chunk_elements):
    """Prove an actual parameter change in every LR group; not just a grad norm."""
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    changes = {}
    for group in optimizer.param_groups:
        witness = None
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            name = names[id(parameter)]
            current, initial = parameter.detach().view(-1), source[name].view(-1)
            for offset in range(0, current.numel(), chunk_elements):
                part = current[offset:offset + chunk_elements].cpu()
                if not torch.equal(part, initial[offset:offset + chunk_elements]):
                    witness = {"parameter": name, "chunk_start": offset}
                    break
            if witness:
                break
        changes[group["group_name"]] = witness
    if any(value is None for value in changes.values()):
        raise RuntimeError("One of the five optimizer groups had no proven parameter update")
    return changes


def full_step(trainer, cpu_batch, safety, step, *, gradient_audit=None):
    """Use production loss, bf16 autocast, backward, finite gates and group clip."""
    from psi.utils import move_to_device
    device = trainer.device
    safety.capture(cpu_batch, global_step=step, local_step=0)
    trainer.optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    begin, backward_end, optimizer_begin, end = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
    wall = time.perf_counter()
    begin.record()
    with trainer.forward_autocast():
        batch = move_to_device(cpu_batch, device)
        losses = trainer.forward_and_loss(trainer.model, batch)
    safety.check_loss(losses["loss"])
    trainer.accelerator.backward(losses["loss"])
    trainer.accelerator.unscale_gradients(trainer.optimizer)
    safety.check_gradients(trainer.model.parameters())
    backward_end.record()
    backward_end.synchronize()
    backward_wall = time.perf_counter() - wall
    audit = gradient_audit(trainer.model) if gradient_audit else None
    # Audit CPU copies/comparison are excluded from both step timing intervals.
    wall = time.perf_counter()
    optimizer_begin.record()
    norms = trainer.clip_grad_norms(already_unscaled=True)
    trainer.optimizer.step()
    end.record()
    end.synchronize()
    update_wall = time.perf_counter() - wall
    memory = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
              "allocated_bytes_after_update": torch.cuda.memory_allocated(device),
              "device_free_bytes_after_update": torch.cuda.mem_get_info(device)[0]}
    check_finite_model_optimizer(trainer.model, trainer.optimizer)
    result = {"loss": float(losses["loss"].detach()), "preclip_group_norms": norms,
        "forward_backward_event_ms": begin.elapsed_time(backward_end),
        "clip_optimizer_event_ms": optimizer_begin.elapsed_time(end),
        "step_wall_seconds_excluding_audit": backward_wall + update_wall,
        "memory": memory, "all_parameters_and_adam_states_finite": True}
    if audit is not None:
        result["gradient_audit"] = audit
    del batch, losses
    return result


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def finish_acceptance(report):
    """A partial run or microbenchmark can never produce the full-step gate."""
    comparisons = report.get("comparisons", [])
    extra = report.get("additional_fused_steps", [])
    warmup = report.get("discarded_warmup", [])
    if (not comparisons or any(not item.get("passed") for item in comparisons)
            or len(extra) < 3 or len(warmup) != 2):
        raise ValueError("Incomplete full-step A/B or continuous-fused evidence")
    steps = [branch["step"] for comparison in comparisons
             for branch in comparison["branches"].values()] + extra
    if (any(not item.get("all_parameters_and_adam_states_finite") for item in steps)
            or any(not item.get("finite") for item in warmup)):
        raise ValueError("Missing finite evidence for a complete-model update")
    required = ("model.safetensors", "optimizer.bin", "scheduler.bin")
    hashes = report.get("source_checkpoint_files_sha256", {})
    if (any(len(hashes.get(name, "")) != 64 for name in required)
            or len(report.get("source_config_sha256", "")) != 64):
        raise ValueError("Full-step acceptance is not bound to the exact checkpoint/config")
    memories = [item["memory"] for item in steps] + [item["memory"] for item in warmup]
    report.update(status="passed", full_step_comparison_passed=True,
        all_model_and_optimizer_tensors_finite=True,
        peak_memory={"peak_allocated_bytes": max(item["peak_allocated_bytes"] for item in memories),
            "peak_reserved_bytes": max(item["peak_reserved_bytes"] for item in memories),
            "minimum_device_free_bytes_after_update": min(item["device_free_bytes_after_update"] for item in memories),
            "gpu_total_bytes": report["runtime"]["gpu_total_bytes"],
            "scope": "PyTorch CUDA allocator peaks plus free-memory samples; includes one model and optimizer"})


def run(args):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or torch.distributed.is_initialized():
        raise ValueError("Validation must run as a separate single-process GPU job")
    if args.repeats < 1 or args.extra_fused_steps < 2 or args.chunk_elements <= 0:
        raise ValueError("Require >=1 exact A/B repeat, >=2 extra fused steps, positive chunks")
    # The script itself refuses to compete with an active training allocation.
    occupied = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
        "--format=csv,noheader"], text=True)
    if occupied.strip():
        raise RuntimeError("GPU processes are still present; save/pause the training job before full-model validation")
    output = args.output_dir.resolve()
    if output.is_relative_to(REPO) or output.exists():
        raise ValueError("Use a new validation output directory outside the repository")
    checkpoint = args.checkpoint.resolve(strict=True)
    if output.is_relative_to(checkpoint):
        raise ValueError("Validation outputs cannot be written inside the source checkpoint")
    required = ["model.safetensors", "optimizer.bin", "scheduler.bin", f"random_states_{args.rng_rank}.pkl"]
    identities = {name: {"bytes": (checkpoint / name).stat().st_size,
        "mtime_ns": (checkpoint / name).stat().st_mtime_ns} for name in required}
    available_kib = int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines()
                             if line.startswith("MemAvailable:")))
    if available_kib < args.minimum_host_free_gib * 1024 ** 2:
        raise MemoryError("Not enough host RAM for source, reference parameters, moments and gradients")
    output.mkdir(parents=True)
    report = {"schema": "cvpr_single_gpu_task_adamw_acceptance_v1", "status": "preparing",
        "started_utc": datetime.now(timezone.utc).isoformat(), "checkpoint": str(checkpoint),
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_files_sha256": {name: file_sha256(checkpoint / name) for name in required},
        "source_config_sha256": file_sha256(args.prepared / "run_config.json"),
        "all_model_and_optimizer_tensors_finite": False, "full_step_comparison_passed": False,
        "checkpoint_files": identities, "prepared": str(args.prepared.resolve()),
        "tolerances": {"parameter_moments_atol": args.atol, "parameter_moments_rtol": args.rtol,
            "gradient_atol": args.grad_atol, "gradient_rtol": args.grad_rtol,
            "loss_atol": args.loss_atol, "adam_steps": "exact"},
        "gpu_count": 1, "per_gpu_batch": 4, "ddp": False, "scheduler_advanced": False,
        "scope": "actual complete task model and real accepted transformed batch; single GPU A/B only",
        "limitations": ["No 8-GPU gradient average, sampler resume or cross-rank parameter test",
            "Restored five learning rates stay fixed for the isolated optimizer comparison",
            "Timing excludes checkpoint restoration, dataset decoding, CPU audits and safety batch snapshot",
            "Cold/warm effects and single-GPU timing do not establish production training speedup"]}
    write_json(output / "report.json", report)
    try:
        for path in (REPO, REPO / "VLMExpert/src", REPO / "VLMExpert"):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        import a100_task_finetune as task
        from accelerate import Accelerator
        from safetensors.torch import load_file
        from psi.models.checkpoint_aliases import restore_shared_tensor_aliases
        from psi.trainers.posttrain import PosttrainTrainer
        from psi.trainers.qwen3vl_mixin import PaddedCollatorForTogether
        from psi.trainers.training_safety import TrainingSafetyGuard
        from psi.utils import batch_str_to_tensor
        config, plan = task.load_prepared(args.prepared)
        task.verify_data_check(args.prepared, plan)
        if (config.train.train_batch_size, config.train.gradient_accumulation_steps,
                config.train.mixed_precision, config.train.max_grad_norm) != (4, 1, "bf16", 1.0):
            raise ValueError("Require the unchanged batch4/accum1/bf16/group-clip1 task recipe")
        if not config.model.gradient_checkpointing or not config.model.tune_vlm or not config.model.tune_motion:
            raise ValueError("Require VLM checkpointing and all three trainable components")
        config = config.model_copy(deep=True)
        config.model.pretrained_joint_checkpoint = str(checkpoint / "model.safetensors")
        config.train.resume_from_checkpoint = None
        config.train.output_dir = str(output / "isolated_trainer")
        config.log.report_to = None
        device = torch.device(args.device)
        if device.type != "cuda":
            raise ValueError("Complete task acceptance requires an explicit CUDA device")
        torch.cuda.set_device(device)
        report["runtime"] = {"torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(device),
            "gpu_total_bytes": torch.cuda.get_device_properties(device).total_memory,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
        accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=1)
        if accelerator.num_processes != 1:
            raise ValueError("An inherited distributed environment is not allowed")
        trainer = PosttrainTrainer(config, torch.device("cpu"))
        trainer.accelerator = accelerator
        trainer.init_models()  # production strict Action/VLM/Motion loading
        source_model = restore_shared_tensor_aliases(trainer.model,
            load_file(str(checkpoint / "model.safetensors"), device="cpu"))
        trainer.model.load_state_dict(source_model, strict=True)
        saved_optimizer = torch.load(checkpoint / "optimizer.bin", map_location="cpu", weights_only=True)
        scheduler = torch.load(checkpoint / "scheduler.bin", map_location="cpu", weights_only=True)
        rng = torch.load(checkpoint / f"random_states_{args.rng_rank}.pkl", map_location="cpu", weights_only=False)
        restore_rng(rng, device, cuda_index=args.saved_cuda_rng_index)
        train, _, window_report = task.datasets(config, trainer.vlm_processor)
        if len(args.batch_indices) != 4 or any(not 0 <= index < len(train) for index in args.batch_indices):
            raise ValueError("Select exactly four real accepted task windows")
        collator = PaddedCollatorForTogether(model_max_length=trainer.tokenizer.model_max_length,
            pad_token_id=trainer.tokenizer.pad_token_id, padding_side="right")
        cpu_batch = batch_str_to_tensor(collator([train[index] for index in args.batch_indices]))
        if tuple(cpu_batch["actions"].shape) != (4, 30, 80):
            raise ValueError("Actual task batch must contain the unchanged 4x30x80 actions")
        report["batch"] = {"accepted_subset_indices": args.batch_indices,
            "source_dataset_rows": [int(train.indices[i]) for i in args.batch_indices],
            "tensor_shapes": {key: list(value.shape) for key, value in cpu_batch.items() if isinstance(value, torch.Tensor)},
            "real_train_windows": window_report["train_windows"],
            "transformed_once_then_replayed": True, "is_next_production_batch": False}
        trainer.model.to(device)
        trainer.model.device = device
        trainer.device = device
        trainer.loss_w = trainer.loss_w.to(device)
        trainer.create_optimizers()
        report["saved_optimizer"] = validate_saved_optimizer(trainer.optimizer, saved_optimizer,
            scheduler, tuple(task.PEAK_LRS))
        if any(p.dtype != torch.float32 for p in trainer.model.parameters() if p.requires_grad):
            raise ValueError("All trainable master parameters must stay FP32")
        safety = TrainingSafetyGuard(output / "diagnostics", device=device, world_size=1)
        checkpoint_step = int(checkpoint.name.removeprefix("ckpt_"))

        def restore(fused):
            trainer.optimizer.zero_grad(set_to_none=True)
            trainer.optimizer.state.clear()
            gc.collect()
            torch.cuda.empty_cache()
            trainer.model.load_state_dict(source_model, strict=True)
            backend = restore_optimizer(trainer.optimizer, saved_optimizer, device, fused=fused)
            trainer.model.train()
            restore_rng(rng, device, cuda_index=args.saved_cuda_rng_index)
            return backend

        # Load kernels/compilers for both implementations, then discard these
        # updates by restoring the complete source and RNG before every trial.
        report["discarded_warmup"] = []
        for fused in (False, True):
            restore(fused)
            warmup = full_step(trainer, cpu_batch, safety, checkpoint_step)
            report["discarded_warmup"].append({"fused": fused, "loss": warmup["loss"],
                "finite": warmup["all_parameters_and_adam_states_finite"], "memory": warmup["memory"]})
        report["comparisons"] = []
        for repeat in range(args.repeats):
            order = ("standard", "fused") if repeat % 2 == 0 else ("fused", "standard")
            branches = {}
            first_name, second_name = order
            first = {"backend": restore(first_name == "fused")}
            gradient_reference = {}

            def capture_gradients(model):
                gradient_reference.update({name: None if p.grad is None else cpu_tree(p.grad)
                    for name, p in model.named_parameters() if p.requires_grad})
                return {"scope": "all preclip trainable gradients copied to CPU",
                        "tensors": len(gradient_reference)}

            first["step"] = full_step(trainer, cpu_batch, safety, checkpoint_step,
                gradient_audit=capture_gradients)
            first["changed_groups"] = changed_groups(trainer.model, source_model,
                trainer.optimizer, args.chunk_elements)
            reference_model = cpu_tree(trainer.model.state_dict())
            reference_adam = cpu_tree(optimizer_tensors(trainer.optimizer))
            branches[first_name] = first
            second = {"backend": restore(second_name == "fused")}

            def audit_gradients(model):
                current = {name: p.grad for name, p in model.named_parameters() if p.requires_grad}
                return compare_maps(gradient_reference, current, atol=args.grad_atol,
                    rtol=args.grad_rtol, chunk_elements=args.chunk_elements)

            second["step"] = full_step(trainer, cpu_batch, safety, checkpoint_step,
                gradient_audit=audit_gradients)
            second["changed_groups"] = changed_groups(trainer.model, source_model,
                trainer.optimizer, args.chunk_elements)
            branches[second_name] = second
            comparison = {"repeat": repeat, "execution_order": list(order), "branches": branches,
                "reference_backend": first_name,
                "loss_abs_difference": abs(first["step"]["loss"] - second["step"]["loss"]),
                "parameters_and_buffers": compare_maps(reference_model, trainer.model.state_dict(),
                    atol=args.atol, rtol=args.rtol, chunk_elements=args.chunk_elements),
                "adam_states_including_steps": compare_maps(reference_adam, optimizer_tensors(trainer.optimizer),
                    atol=args.atol, rtol=args.rtol, chunk_elements=args.chunk_elements),
                "last_backend_optimizer_groups": optimizer_state_identity(trainer.optimizer)}
            comparison["passed"] = (comparison["loss_abs_difference"] <= args.loss_atol
                and second["step"]["gradient_audit"]["passed"]
                and comparison["parameters_and_buffers"]["passed"]
                and comparison["adam_states_including_steps"]["passed"])
            report["comparisons"].append(comparison)
            write_json(output / "report.json", report)
            del gradient_reference, reference_model, reference_adam
            gc.collect()
            if not comparison["passed"]:
                raise RuntimeError("Actual full-model A/B numerical comparison failed; do not switch training")
        report["additional_fused_steps"] = []
        restore(True)
        for index in range(args.extra_fused_steps + 1):
            report["additional_fused_steps"].append(full_step(trainer, cpu_batch, safety,
                checkpoint_step + index))
            trainer.optimizer.zero_grad(set_to_none=True)
        for name, identity in identities.items():
            stat = (checkpoint / name).stat()
            if (stat.st_size, stat.st_mtime_ns) != (identity["bytes"], identity["mtime_ns"]):
                raise ValueError("Protected source checkpoint changed during the audit")
        finish_acceptance(report)
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        write_json(output / "report.json", report)
        raise
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "report.json", report)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--batch-indices", nargs=4, type=int, default=[0, 1, 2, 3])
    parser.add_argument("--rng-rank", type=int, default=0)
    parser.add_argument("--saved-cuda-rng-index", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--extra-fused-steps", type=int, default=2)
    parser.add_argument("--chunk-elements", type=int, default=1048576)
    parser.add_argument("--minimum-host-free-gib", type=int, default=128)
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--grad-atol", type=float, default=1e-7)
    parser.add_argument("--grad-rtol", type=float, default=1e-4)
    parser.add_argument("--loss-atol", type=float, default=1e-6)
    return parser.parse_args(argv)


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"status": result["status"]}))
