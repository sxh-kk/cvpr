"""Start a new fused-AdamW task from the accepted38k model, never resume2345.

The original failed task, failed independent A/B report and all accepted data
remain unchanged. The controlled GPU gate covers identical-gradient optimizer
updates plus bounded real-backward health; it is not an independent-backward
equivalence claim. No command in this entry stops another process.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

import a100_task_finetune as task
from adamw_task_resume import optimizer_state_identity, select_adamw_backend


ARCHIVE = {
    "early_checkpoint_steps": [20] + [step for step in range(1000, 20000, 1000) if step % 4000],
    "max_checkpoints_to_keep": 2,
    "preserve_checkpoint_every": 4000,
}
MODE = "fresh_fused_restart_from38000"


def _json_config(config):
    return config.model_dump(mode="json")


def require_parent(config, plan):
    train = _json_config(config)["train"]
    expected = {"max_training_steps": 20000, "warmup_steps": 1000,
                "checkpointing_steps": 4000, "validation_steps": 1000,
                "early_checkpoint_steps": [], "train_batch_size": 4,
                "gradient_accumulation_steps": 1, "mixed_precision": "bf16",
                "resume_from_checkpoint": None, "skip_resumed_steps": False}
    if plan["source_step"] != 38000 or any(train.get(key) != value for key, value in expected.items()):
        raise ValueError("Require the accepted fresh38k/batch32/task20k parent recipe")


def require_config_delta(config, parent, root, timestamp):
    """Permit naming/output and recovery saves, never training hyperparameters."""
    expected = copy.deepcopy(_json_config(parent))
    expected.update(timestamp=timestamp, exp=f"{expected['exp']}_fused_restart")
    expected["train"].update(output_dir=str(Path(root) / "training"), **ARCHIVE)
    if _json_config(config) != expected:
        raise ValueError("Fresh restart changed fields outside naming/output/recovery checkpoint policy")


def validation_gate(controlled_path, micro_path, *, parent_config_sha256, source_model_sha256):
    controlled_path, micro_path = map(lambda p: Path(p).resolve(strict=True), (controlled_path, micro_path))
    full, micro = task.read_json(controlled_path), task.read_json(micro_path)
    required = ("same_gradient_optimizer_comparison_passed", "postclip_gradients_exactly_equal",
                "all_model_and_optimizer_tensors_finite", "continuous_fused_real_gradient_steps_passed")
    if (full.get("status") != "controlled_optimizer_comparison_passed"
            or any(full.get(key) is not True for key in required)
            or full.get("full_step_comparison_passed") is not False
            or full.get("uncontrolled_full_step_comparison_passed") is not False
            or full.get("source_config_sha256") != parent_config_sha256
            or (full.get("gpu_count"), full.get("per_gpu_batch")) != (1, 4)):
        raise ValueError("Controlled full-model GPU evidence has not passed the fresh-restart contract")
    hashes = full.get("source_checkpoint_files_sha256", {})
    if (hashes.get("model.safetensors") != source_model_sha256
            or any(not re.fullmatch(r"[0-9a-f]{64}", hashes.get(name, ""))
                   for name in ("model.safetensors", "optimizer.bin", "scheduler.bin"))):
        raise ValueError("Controlled GPU evidence belongs to another38k model or is incomplete")
    tolerances = {"parameter_moments_atol": 1e-8, "parameter_moments_rtol": 1e-5,
                  "gradient_atol": 1e-7, "gradient_rtol": 1e-4, "loss_atol": 1e-6, "adam_steps": "exact"}
    if full.get("tolerances") != tolerances:
        raise ValueError("The original numerical tolerances must remain unchanged")
    comparisons = full.get("same_gradient_comparisons", [])
    if len(comparisons) < 2 or full.get("baseline_repetition", {}).get("comparison_complete") is not True:
        raise ValueError("Require repeated controlled comparisons and an independent standard control")
    for index, comparison in enumerate(comparisons):
        order = ["standard", "fused"] if index % 2 == 0 else ["fused", "standard"]
        if comparison.get("passed") is not True or comparison.get("execution_order") != order:
            raise ValueError("Require alternating complete standard/fused comparisons")
        for key in ("preclip_replay", "postclip_gradient_equality"):
            check = comparison.get(key, {})
            if check.get("passed") is not True or check.get("strict_zero_tolerance") is not True:
                raise ValueError("The optimizer must receive exactly identical gradients and None pattern")
    extra = full.get("additional_fused_steps", [])
    if len(extra) < 3 or any(step.get("all_parameters_and_adam_states_finite") is not True
            or step.get("gradient_source") != "own_real_backward"
            or step.get("restored_since_previous_step") is not (index == 0)
            for index, step in enumerate(extra)):
        raise ValueError("Require three consecutive fused steps using their own real backward gradients")
    memory = full.get("peak_memory", {})
    allocated, reserved, total, free = (memory.get(key, 0) for key in (
        "peak_allocated_bytes", "peak_reserved_bytes", "gpu_total_bytes", "minimum_device_free_bytes_after_update"))
    if not 0 < allocated <= reserved < total or free <= 0:
        raise ValueError("Require successful full-model allocated/reserved/free-memory evidence")
    micro_sha = task.shared.sha256(micro_path)
    binding = full.get("microbenchmark_report", {})
    if (micro.get("status") != "optimizer_microbenchmark_passed"
            or micro.get("dtype") != "float32" or not str(micro.get("device", "")).startswith("cuda:")
            or micro.get("candidate") != "AdamW foreach=False fused=True"
            or micro.get("recipe", {}).get("source_sha256") != parent_config_sha256
            or micro.get("recipe", {}).get("lrs") != task.PEAK_LRS
            or micro.get("correctness", {}).get("passed") is not True
            or micro.get("post_timing_correctness", {}).get("passed") is not True
            or binding.get("sha256") != micro_sha or binding.get("status") != micro["status"]):
        raise ValueError("The exact CUDA microbenchmark and parent recipe are not bound to this GPU report")
    return {"controlled_report": {"path": str(controlled_path), "sha256": task.shared.sha256(controlled_path)},
            "micro_report": {"path": str(micro_path), "sha256": micro_sha},
            "parent_config_sha256": parent_config_sha256, "source_model_sha256": source_model_sha256,
            "scope": "identical-gradient optimizer equivalence and bounded real-backward finite/memory checks; independent full-step equivalence remains unproven"}


def prepare(args):
    parent = args.parent.resolve(strict=True)
    config, accepted = task.load_prepared(parent)
    require_parent(config, accepted)
    task.verify_data_check(parent, accepted)
    state = task.read_json(parent / "training_state.json")
    if state.get("state") != "failed":
        raise ValueError("This fresh restart requires the failed original task; it never stops a live run")
    parent_sha = task.shared.sha256(parent / "run_config.json")
    validation_gate(args.controlled_report, args.micro_report,
                    parent_config_sha256=parent_sha, source_model_sha256=accepted["source_checkpoint_sha256"])
    output = args.output.absolute()
    if output.exists() or output.resolve().is_relative_to(task.REPO):
        raise ValueError("Use a new output directory outside the repository")
    timestamp = datetime.now(timezone.utc).strftime("%y%m%d%H%M%S")
    replacements = {"timestamp": timestamp, "exp": f"{config.exp}_fused_restart",
                    "train.output_dir": str(output / "training"),
                    **{f"train.{key}": value for key, value in ARCHIVE.items()}}
    argv = task.shared.rewrite_argv((parent / "argv.txt").read_text().split(), replacements)
    output.mkdir(parents=True)
    (output / "argv.txt").write_text("\n".join(argv) + "\n")
    derived = task.shared.parse_training_argv(output / "argv.txt")
    require_config_delta(derived, config, output, timestamp)
    task.shared.write_json(output / "run_config.json", _json_config(derived))
    (output / "data_check.json").write_bytes((parent / "data_check.json").read_bytes())
    evidence = output / "validation"
    evidence.mkdir()
    for original, name in ((args.controlled_report, "controlled_gpu.json"), (args.micro_report, "micro_gpu.json")):
        (evidence / name).write_bytes(original.read_bytes())
    gate = validation_gate(evidence / "controlled_gpu.json", evidence / "micro_gpu.json",
                          parent_config_sha256=parent_sha, source_model_sha256=accepted["source_checkpoint_sha256"])
    plan = {**accepted, "mode": MODE, "status": "fresh_fused_restart_prepared", "created_utc": task.now(),
        "timestamp": timestamp, "parent_prepared": str(parent),
        "parent_preparation_sha256": task.shared.sha256(parent / "preparation.json"),
        "parent_config_sha256": parent_sha,
        "parent_training_state_sha256": task.shared.sha256(parent / "training_state.json"),
        "accepted_data_check_sha256": task.shared.sha256(parent / "data_check.json"),
        "config_sha256": task.shared.sha256(output / "run_config.json"),
        "argv_sha256": task.shared.sha256(output / "argv.txt"), "optimizer_backend": "fused",
        "fresh_restart": True, "initial_global_step": 0, "unrecoverable_parent_updates": 2345,
        "optimizer_scheduler_rng_restored": False, "task_progress_restored": False,
        "recompute_task_updates_from_zero": True, "controlled_gpu_acceptance": gate,
        "validation_original_paths": {"controlled": str(args.controlled_report.resolve()), "micro": str(args.micro_report.resolve())},
        "independent_full_backward_equivalence_claimed": False,
        "early_checkpoint_steps": ARCHIVE["early_checkpoint_steps"],
        "recovery_checkpoint_policy": {**ARCHIVE, "formal_interval": 4000,
            "reason": "Bound progress loss after the original native CUDA abort; archive policy only, model/training hyperparameters unchanged",
            "deletion_scope": "Only this new run's expired non4k recovery checkpoints; original runs/source weights are read-only"}}
    task.shared.write_json(output / "preparation.json", plan)
    (output / "launch.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\nexec " + shlex.join(
        [sys.executable, str(Path(__file__).resolve()), "run", "--prepared", str(output)]) + "\n")
    load(output)
    print(json.dumps({"prepared": str(output), "mode": MODE, "initial_global_step": 0,
                      "unrecoverable_parent_updates": 2345, "total_task_budget": 20000}, ensure_ascii=False))


def load(root):
    root = Path(root).resolve(strict=True)
    config, plan = task.load_prepared(root)
    if (plan.get("mode") != MODE or plan.get("optimizer_backend") != "fused" or plan.get("fresh_restart") is not True
            or plan.get("initial_global_step") != 0 or plan.get("optimizer_scheduler_rng_restored") is not False
            or plan.get("task_progress_restored") is not False or plan.get("unrecoverable_parent_updates") != 2345):
        raise ValueError("Require an explicit step-zero restart, without inherited optimizer or task progress")
    parent = Path(plan["parent_prepared"])
    for name, key in (("preparation.json", "parent_preparation_sha256"),
                      ("run_config.json", "parent_config_sha256"), ("training_state.json", "parent_training_state_sha256")):
        if task.shared.sha256(parent / name) != plan[key]:
            raise ValueError(f"Accepted failed-parent evidence changed: {name}")
    base, accepted = task.load_prepared(parent)
    require_parent(base, accepted)
    require_config_delta(config, base, root, plan["timestamp"])
    if plan["source_checkpoint_sha256"] != accepted["source_checkpoint_sha256"]:
        raise ValueError("Restart source38k identity changed")
    if (task.shared.sha256(root / "data_check.json") != plan["accepted_data_check_sha256"]
            or task.shared.sha256(parent / "data_check.json") != plan["accepted_data_check_sha256"]):
        raise ValueError("The accepted CPU data evidence changed")
    task.verify_data_check(root, plan, require_launch_gate=False)
    gate = plan["controlled_gpu_acceptance"]
    actual = validation_gate(gate["controlled_report"]["path"], gate["micro_report"]["path"],
                            parent_config_sha256=plan["parent_config_sha256"],
                            source_model_sha256=accepted["source_checkpoint_sha256"])
    if actual != gate:
        raise ValueError("Controlled GPU acceptance bytes changed")
    return config, plan


def trainer_class(plan):
    Base = task.task_trainer_class(plan)
    class FreshFusedTaskTrainer(Base):
        def prepare(self, accelerator):
            loader = super().prepare(accelerator)
            actual = getattr(self.optimizer, "optimizer", self.optimizer)
            if actual.state:
                raise ValueError("Fresh fused task must have no inherited Adam state")
            backend = select_adamw_backend(self.optimizer, fused=True)
            if accelerator.is_main_process:
                task.shared.write_json(Path(self.project_dir) / "task_fused_restart_initialization.json", {
                    "mode": MODE, "source_step": 38000, "source_model_sha256": plan["source_checkpoint_sha256"],
                    "initial_global_step": 0, "optimizer_scheduler_rng_restored": False,
                    "unrecoverable_parent_updates": 2345, "backend": backend,
                    "optimizer_groups": optimizer_state_identity(self.optimizer),
                    "controlled_gpu_acceptance": plan["controlled_gpu_acceptance"],
                    "scope": "rank0 actual backend flags on the empty prepared CUDA optimizer; eight-rank update2/20 checks remain required"})
            return loader

        def step(self, batch_input, global_step, local_step):
            synced, metrics = super().step(batch_input, global_step, local_step)
            completed = global_step + 1
            if synced and completed in (2, 20):
                record_fused_gpu_health(self, completed)
            return synced, metrics
    return FreshFusedTaskTrainer


def record_fused_gpu_health(trainer, completed):
    """Two bounded collectives supplement the base finite/parameter probes."""
    import torch
    optimizer = getattr(trainer.optimizer, "optimizer", trainer.optimizer)
    defaults_ok = optimizer.defaults.get("fused") is True and optimizer.defaults.get("foreach") is False
    groups_ok = bool(optimizer.param_groups) and all(
        group.get("fused") is True and group.get("foreach") is False for group in optimizer.param_groups)
    free, total = torch.cuda.mem_get_info(trainer.device)
    row = torch.tensor([trainer.accelerator.process_index,
        torch.cuda.max_memory_allocated(trainer.device), torch.cuda.max_memory_reserved(trainer.device),
        free, total, int(defaults_ok), int(groups_ok)], dtype=torch.int64, device=trainer.device)
    gathered = trainer.accelerator.gather(row).reshape(8, 7).cpu().tolist()
    if any(rank != index or not (0 < allocated <= reserved < device_total and device_free > 0)
           or default_flags != 1 or group_flags != 1
           for index, (rank, allocated, reserved, device_free, device_total, default_flags, group_flags)
           in enumerate(gathered)):
        raise RuntimeError("Eight-rank fused flags or CUDA allocated/reserved/free memory failed")
    if trainer.accelerator.is_main_process:
        task.shared.write_json(Path(trainer.project_dir) / f"gpu_check_fused_update{completed}.json", {
            "status": "passed", "completed_updates": completed, "gpu_count": 8, "per_gpu_batch": 4,
            "accumulation": 1, "all_rank_actual_fused_flags_passed": True,
            "ranks": [{"rank": row[0], "peak_allocated_bytes": row[1], "peak_reserved_bytes": row[2],
                       "device_free_bytes_after_update": row[3], "device_total_bytes": row[4],
                       "actual_defaults_fused_true_foreach_false": bool(row[5]),
                       "all_actual_groups_fused_true_foreach_false": bool(row[6])} for row in gathered],
            "scope": "Actual eight-rank flags and CUDA allocator peaks since process start at updates2/20 only; base finite and parameter probes remain separate, not full-run memory or throughput acceptance"})


def train(root):
    import torch
    config, plan = load(root)
    task.verify_data_check(root, plan)
    if int(os.environ.get("WORLD_SIZE", "0")) != 8:
        raise ValueError("Fresh fused task requires exactly eight ranks")
    Trainer = trainer_class(plan)
    class Factory:
        @staticmethod
        def instantiate(cfg, device_id):
            return Trainer(cfg, torch.device(f"cuda:{device_id}"))
    spec = importlib.util.spec_from_file_location("task_fused_restart_production", task.REPO / "VLMExpert/scripts/train.py")
    production = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(production)
    production.Trainer = Factory
    sys.argv = (Path(root) / "argv.txt").read_text().split()
    production.train(config)


def diagnostic_directory(plan):
    diagnostic_root = Path("/dev/shm")
    if not any(line.split()[1:3] == [str(diagnostic_root), "tmpfs"] for line in Path("/proc/mounts").read_text().splitlines()):
        raise ValueError("Rolling input diagnostics require the dedicated tmpfs path /dev/shm")
    space = os.statvfs(diagnostic_root)
    free = space.f_bavail * space.f_frsize
    if free < 1024 ** 3:
        raise ValueError("Require at least1GiB free tmpfs for eight-rank rolling input diagnostics")
    diagnostic_dir = diagnostic_root / f"cvpr-task50-fused-{plan['timestamp']}-{os.getpid()}"
    diagnostic_dir.mkdir(mode=0o700)
    return diagnostic_dir, free


def run(root):
    import fcntl
    root = Path(root).resolve(strict=True)
    with (root / "restart_launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any((root / name).exists() for name in ("training_state.json", "restart_launch_provenance.json")):
            raise ValueError("Fresh fused segment was already launched")
        _, plan = load(root)
        return launch(root, plan)


def launch(root, plan):
    diagnostic_dir, free = diagnostic_directory(plan)
    environment = {"CUDA_LAUNCH_BLOCKING": "1", "PSI0_ROLLING_DIAGNOSTICS_DIR": str(diagnostic_dir)}
    paths = ["scripts/4090_simple/a100_task_fused_restart.py", "scripts/4090_simple/a100_task_finetune.py",
             "scripts/4090_simple/adamw_task_resume.py", "scripts/4090_simple/finetune_carrybox.py",
             "VLMExpert/scripts/train.py", "VLMExpert/scripts/analysis/monitor_task_finetune.py",
             "VLMExpert/src/psi/trainers/trainer.py", "VLMExpert/src/psi/trainers/posttrain.py",
             "VLMExpert/src/psi/trainers/training_safety.py", "VLMExpert/src/psi/models/psi0.py",
             "VLMExpert/src/psi/models/dual_source_action.py", "VLMExpert/src/psi/models/frozen_motion.py",
             "VLMExpert/src/psi/config/transform.py", "VLMExpert/src/psi/config/transform_psi0_sonic.py",
             "ImageProcess/processor.py"]
    revision = subprocess.check_output(["git", "-C", str(task.REPO), "rev-parse", "HEAD"], text=True).strip()
    task.shared.write_json(root / "restart_launch_provenance.json", {
        "git_revision": revision,
        "git_status": subprocess.check_output(["git", "-C", str(task.REPO), "status", "--short"], text=True).strip(),
        "captured_utc": task.now(),
        "source_sha256": {name: task.shared.sha256(task.REPO / name) for name in paths},
        "diagnostic_environment": environment, "PSI0_DIAGNOSTIC_BATCH_RANGE": None,
        "tmpfs_available_bytes_at_launch": free, "rolling_slots_per_rank": 2,
        "initial_global_step": 0, "unrecoverable_parent_updates": 2345,
        "scope": "Fresh restart; synchronous CUDA diagnostics and rolling CPU input/RNG dumps add overhead, not a full speedup benchmark"})
    keys = (*environment, "PSI0_DIAGNOSTIC_BATCH_RANGE")
    previous = {key: os.environ.get(key) for key in keys}
    try:
        os.environ.update(environment)
        os.environ.pop("PSI0_DIAGNOSTIC_BATCH_RANGE", None)
        return task.run(root, entrypoint=__file__)
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    p = modes.add_parser("prepare")
    for name in ("parent", "output", "controlled-report", "micro-report"):
        p.add_argument("--" + name, type=Path, required=True)
    for mode in ("train", "run", "name"):
        modes.add_parser(mode).add_argument("--prepared", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args)
    elif args.mode == "train":
        train(args.prepared)
    elif args.mode == "run":
        return run(args.prepared)
    else:
        from psi.trainers.posttrain import PosttrainTrainer
        config, _ = load(args.prepared)
        print(PosttrainTrainer(config, "cpu").project_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
