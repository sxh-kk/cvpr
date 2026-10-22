"""Restore a complete Handover task checkpoint and select AdamW's CUDA backend.

This entry never stops another process. The original task, checkpoint, dataset
and accepted CPU/DINO evidence stay unchanged. A new segment retains the original
20k budget and replays the worker data stream before the next optimizer update.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import a100_task_finetune as task
from adamw_task_resume import (audit_optimizer_restoration, exact_tensor_equal,
                               optimizer_state_identity, select_adamw_backend)


def write(path, value):
    task.shared.write_json(path, value)


def protect_checkpoint(checkpoint, destination, *, log):
    checkpoint = checkpoint.resolve(strict=True)
    if not checkpoint.name.startswith("ckpt_") or not checkpoint.name[5:].isdigit():
        raise ValueError("Use an explicit completed task checkpoint")
    step = int(checkpoint.name[5:])
    if f"Saved state to {checkpoint}" not in Path(log).read_text(errors="replace"):
        raise ValueError("The original production save-complete marker is missing")
    files = ["model.safetensors", "optimizer.bin", "scheduler.bin",
             *[f"random_states_{rank}.pkl" for rank in range(8)]]
    if destination.exists():
        raise FileExistsError("Protect each source snapshot in a fresh directory")
    for name in files:
        if not (checkpoint / name).is_file() or (checkpoint / name).stat().st_size == 0:
            raise ValueError(f"Incomplete task checkpoint: {name}")
    destination.mkdir(parents=True)
    identities = {}
    for name in files:
        source, target = checkpoint / name, destination / name
        os.link(source, target)  # Preserves bytes if rolling retention removes the original name.
        stat = target.stat()
        identities[name] = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                            "sha256": task.shared.sha256(target)}
    write(destination / "snapshot.json", {"source": str(checkpoint), "completed_step": step,
          "files": identities, "protected_utc": task.now(), "protection": "hardlinks; source untouched"})
    return step, identities


def prepare(args):
    parent = args.parent.resolve(strict=True)
    base, accepted = task.load_prepared(parent)
    task.verify_data_check(parent, accepted)
    state = task.read_json(parent / "training_state.json")
    output = args.output.absolute()
    if output.exists() or output.resolve().is_relative_to(task.REPO):
        raise ValueError("Use a fresh output outside the repository")
    checkpoint = args.checkpoint.resolve(strict=True)
    if checkpoint.parent.parent != Path(state["run_directory"]):
        raise ValueError("Checkpoint does not belong to the exact accepted parent task")
    output.mkdir(parents=True)
    protected = output / "protected_source" / "checkpoints" / checkpoint.name
    step, identities = protect_checkpoint(checkpoint, protected, log=state["log"])
    if not 0 < step < accepted["max_optimizer_steps"]:
        raise ValueError("The task checkpoint has no remaining work")
    validation = {}
    if args.backend == "fused":
        if args.full_validation_report is None or args.micro_validation_report is None:
            raise ValueError("Fused continuation requires both GPU numerical/memory acceptance reports")
        full = task.read_json(args.full_validation_report)
        micro = task.read_json(args.micro_validation_report)
        if (full.get("status") != "passed" or full.get("full_step_comparison_passed") is not True
                or full.get("all_model_and_optimizer_tensors_finite") is not True
                or full.get("source_config_sha256") != task.shared.sha256(parent / "run_config.json")
                or micro.get("status") != "optimizer_microbenchmark_passed"):
            raise ValueError("The bound fused GPU acceptance has not passed")
        for name in ("model.safetensors", "optimizer.bin", "scheduler.bin"):
            if full.get("source_checkpoint_files_sha256", {}).get(name) != identities[name]["sha256"]:
                raise ValueError(f"GPU acceptance belongs to a different source checkpoint: {name}")
        validation = {str(path.resolve(strict=True)): task.shared.sha256(path)
                      for path in (args.full_validation_report, args.micro_validation_report)}
    # Keep the standard six-digit date/time name used by the production trainer.
    from datetime import datetime, timezone
    timestamp = datetime.now(timezone.utc).strftime("%y%m%d%H%M%S")
    replacements = {"timestamp": timestamp, "exp": f"{base.exp}_{args.backend}_resume{step}",
        "train.output_dir": str(output / "training"),
        "train.resume_from_checkpoint": str(protected),
        "train.skip_resumed_steps": True, "wandb.id": None, "wandb.resume": "never",
        "wandb.name": None, "wandb.group": None, "wandb.entity": None}
    argv = task.shared.rewrite_argv((parent / "argv.txt").read_text().split(), replacements)
    (output / "argv.txt").write_text("\n".join(argv) + "\n")
    config = task.shared.parse_training_argv(output / "argv.txt")
    write(output / "run_config.json", config.model_dump(mode="json"))
    (output / "data_check.json").write_bytes((parent / "data_check.json").read_bytes())
    plan = {**accepted, "mode": "complete_task_resume", "created_utc": task.now(),
            "parent_prepared": str(parent), "parent_preparation_sha256": task.shared.sha256(parent / "preparation.json"),
            "parent_run_directory": state["run_directory"], "resume_source_checkpoint": str(checkpoint),
            "resume_checkpoint": str(protected), "resume_step": step,
            "resume_checkpoint_files": identities, "optimizer_backend": args.backend,
            "fused_acceptance_reports_sha256": validation,
            "optimizer_scheduler_rng_restored": True, "timestamp": timestamp,
            "argv_sha256": task.shared.sha256(output / "argv.txt"),
            "config_sha256": task.shared.sha256(output / "run_config.json"),
            "data_recovery": "full CPU worker-stream replay from epoch zero; original shuffle/augmentation RNG",
            "cuda_rng_recovery": "checkpoint main-process RNG restored after CPU replay",
            "formal_physical_success": "not_measured"}
    write(output / "preparation.json", plan)
    load(output)
    print(json.dumps({"prepared": str(output), "resume_step": step, "backend": args.backend,
                      "total_task_budget": plan["max_optimizer_steps"]}))


def load(root):
    root = Path(root).resolve(strict=True)
    plan = task.read_json(root / "preparation.json")
    if plan["mode"] != "complete_task_resume":
        raise ValueError("Not a complete task-state continuation")
    if plan["optimizer_backend"] == "fused" and len(plan.get("fused_acceptance_reports_sha256", {})) != 2:
        raise ValueError("Missing fused acceptance gates")
    for path, expected in plan.get("fused_acceptance_reports_sha256", {}).items():
        if task.shared.sha256(path) != expected:
            raise ValueError("Accepted fused verification report changed")
    parent = Path(plan["parent_prepared"])
    if task.shared.sha256(parent / "preparation.json") != plan["parent_preparation_sha256"]:
        raise ValueError("Parent task preparation changed")
    base, original = task.load_prepared(parent)
    if (task.shared.sha256(root / "argv.txt") != plan["argv_sha256"]
            or task.shared.sha256(root / "run_config.json") != plan["config_sha256"]):
        raise ValueError("Continuation configuration changed")
    config = task.shared.parse_training_argv(root / "argv.txt")
    if config.model_dump(mode="json") != task.read_json(root / "run_config.json"):
        raise ValueError("Continuation argv differs from configuration")
    if config.data.model_dump(mode="json") != base.data.model_dump(mode="json"):
        raise ValueError("Data, augmentation, DINO or normalization changed")
    if config.model.model_dump(mode="json") != base.model.model_dump(mode="json"):
        raise ValueError("The model or trainable components changed")
    train, expected = config.train.model_dump(mode="json"), base.train.model_dump(mode="json")
    for key in ("output_dir", "resume_from_checkpoint", "skip_resumed_steps"):
        expected[key] = train[key]
    if train != expected or train["resume_from_checkpoint"] != plan["resume_checkpoint"] or not train["skip_resumed_steps"]:
        raise ValueError("Continuation must preserve batch, optimizer recipe, schedule, safety and save cadence")
    for name, identity in plan["resume_checkpoint_files"].items():
        stat = (Path(plan["resume_checkpoint"]) / name).stat()
        if (stat.st_size, stat.st_mtime_ns) != (identity["bytes"], identity["mtime_ns"]):
            raise ValueError(f"Protected checkpoint changed: {name}")
    if task.shared.normalization_receipt(config.data.transform.field) != original["normalization"]:
        raise ValueError("Original normalization changed")
    task.verify_data_check(root, plan, require_launch_gate=False)
    return config, plan


@contextmanager
def preserve_cuda_rng():
    import torch
    state = torch.cuda.get_rng_state_all()
    try:
        yield
    finally:
        torch.cuda.set_rng_state_all(state)


def resumed_trainer_class(plan):
    import torch
    from accelerate.data_loader import DataLoaderShard
    from psi.trainers.posttrain import PosttrainTrainer
    from replay_task_data import ReplayResumedMapDataLoader
    Base = task.task_trainer_class(plan)

    class ResumedTaskTrainer(Base):
        def init_models(self):
            # Construction uses the accepted38k joint architecture/weights, then
            # the complete task state replaces it before any forward. The
            # production warmstart guard correctly rejects mixing the two modes
            # in its constructor, so hide only the pending resume during build.
            resume = self.cfg.train.resume_from_checkpoint
            self.cfg.train.resume_from_checkpoint = None
            try:
                return super().init_models()
            finally:
                self.cfg.train.resume_from_checkpoint = resume

        @property
        def num_steps_per_epoch(self):
            loader = getattr(self, "train_dataloader", None)
            if loader is None:
                return super().num_steps_per_epoch
            return len(loader) if isinstance(loader, DataLoaderShard) else len(loader) // self.world_size

        def prepare(self, accelerator):
            # The production loop calls resume exactly once after this returns.
            loader = PosttrainTrainer.prepare(self, accelerator)
            if accelerator.num_processes != 8 or len(loader) != 431:
                raise ValueError("Require the original eight-rank431-batch task loader")
            return loader

        def resume_from_checkpoint(self):
            cached = getattr(self, "_task_resume_result", None)
            if cached is not None:
                return cached
            result = PosttrainTrainer.resume_from_checkpoint(self)
            if result != (plan["resume_step"], plan["resume_checkpoint"]):
                raise ValueError("Full state restoration did not reach the expected task update")
            # Check the actual restored tensors before changing backend flags.
            # All ranks loaded the same complete checkpoint; rank zero performs
            # the full36GB comparison, then every rank takes the same gate.
            audit = None
            if self.accelerator.is_main_process:
                try:
                    from safetensors import safe_open
                    saved_optimizer = torch.load(Path(result[1]) / "optimizer.bin", map_location="cpu", weights_only=True, mmap=True)
                    optimizer_audit = audit_optimizer_restoration(self.optimizer, saved_optimizer)
                    del saved_optimizer
                    state = self.unwrap_model().state_dict()
                    tensors, elements = 0, 0
                    with safe_open(str(Path(result[1]) / "model.safetensors"), framework="pt", device="cpu") as saved:
                        for name in saved.keys():
                            tensor = saved.get_tensor(name)
                            if name not in state or not exact_tensor_equal(state[name], tensor):
                                raise ValueError(f"Restored model tensor differs: {name}")
                            tensors += 1
                            elements += tensor.numel()
                    expected_scheduler = torch.load(Path(result[1]) / "scheduler.bin", map_location="cpu", weights_only=True)
                    if self.lr_scheduler.state_dict() != expected_scheduler:
                        raise ValueError("Restored scheduler state differs")
                    audit = {"status": "passed", "model_tensors_compared": tensors,
                             "model_elements_compared": elements, "optimizer": optimizer_audit,
                             "scheduler_exactly_equal": True}
                except Exception as error:
                    audit = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
            broadcast = [audit]
            torch.distributed.broadcast_object_list(broadcast, src=0)
            audit = broadcast[0]
            if audit["status"] != "passed":
                raise RuntimeError(f"Full task-state restoration gate failed: {audit}")
            # Accelerate1.7 catches RNG-load failures as log messages. Make the
            # actual per-rank restoration a mandatory gate instead.
            import random
            import numpy as np
            expected_rng = torch.load(Path(result[1]) / f"random_states_{self.accelerator.process_index}.pkl",
                                      map_location="cpu", weights_only=False)
            actual_numpy = np.random.get_state()
            expected_numpy = expected_rng["numpy_random_seed"]
            numpy_equal = (actual_numpy[0] == expected_numpy[0]
                and np.array_equal(actual_numpy[1], expected_numpy[1])
                and actual_numpy[2:] == expected_numpy[2:])
            cuda_states = torch.cuda.get_rng_state_all()
            cuda_equal = (len(cuda_states) == len(expected_rng["torch_cuda_manual_seed"])
                and all(torch.equal(a, b) for a, b in zip(cuda_states, expected_rng["torch_cuda_manual_seed"])))
            rng_ok = (random.getstate() == expected_rng["random_state"] and numpy_equal
                and torch.equal(torch.get_rng_state(), expected_rng["torch_manual_seed"])
                and cuda_equal and self.accelerator.step == expected_rng["step"])
            agreed = torch.tensor(int(rng_ok), device=self.device)
            torch.distributed.all_reduce(agreed, op=torch.distributed.ReduceOp.MIN)
            if not agreed.item():
                raise RuntimeError("Checkpoint CPU/CUDA/worker-parent RNG or accumulation state was not restored")
            receipt = select_adamw_backend(self.optimizer, fused=plan["optimizer_backend"] == "fused")
            scheduler = self.lr_scheduler.state_dict()
            if self.accelerator.is_main_process:
                write(Path(self.project_dir) / "task_resume_initialization.json", {
                    "origin_source_step": plan["source_step"], "restored_task_step": result[0],
                    "checkpoint": result[1], "optimizer_scheduler_rng_restored": True,
                    "full_restoration_audit": audit,
                    "all_rank_python_numpy_cpu_cuda_rng_exactly_restored": True,
                    "backend": receipt, "optimizer_groups": optimizer_state_identity(self.optimizer),
                    "scheduler": scheduler, "actual_batches_per_epoch": len(self.train_dataloader),
                    "total_task_budget": self.max_training_steps,
                    "scope": "production full load; data replay completion recorded separately"})
            self._task_resume_result = result
            return result

        def create_resume_data_manager(self, completed_updates):
            if completed_updates != plan["resume_step"] or self.gradient_accumulation_steps != 1:
                raise ValueError("Unexpected continuation cursor")
            def progress(done, total):
                if done % 100 and done != total:
                    return
                if self.accelerator.is_main_process:
                    write(Path(self.project_dir) / "data_replay_progress.json", {
                        "completed_microbatches_replayed": done, "target_microbatches": total,
                        "generated_utc": task.now(), "status": "complete" if done == total else "replaying"})
            return ReplayResumedMapDataLoader(self.train_dataloader,
                completed_microbatches=completed_updates, extra_rng_context=preserve_cuda_rng,
                progress=progress)

        def step(self, batch_input, global_step, local_step):
            synced, metrics = super().step(batch_input, global_step, local_step)
            relative = global_step + 1 - plan["resume_step"]
            if synced and relative in (2, 20):
                model = self.unwrap_model()
                probe = torch.cat([p.detach().reshape(-1)[:4].float() for p in model.parameters() if p.requires_grad])
                values = self.accelerator.gather(probe).reshape(8, -1)
                if not torch.isfinite(values).all() or not torch.equal(values, values[0].expand_as(values)):
                    raise RuntimeError("Resumed cross-rank parameter probes disagree")
                if self.accelerator.is_main_process:
                    write(Path(self.project_dir) / f"gpu_check_resumed{relative}.json", {
                        "status": "passed", "completed_task_updates": global_step + 1,
                        "completed_continuation_updates": relative, "gpu_count": 8,
                        "per_gpu_batch": 4, "accumulation": 1,
                        "probe_values_per_rank": probe.numel(), "all_rank_probes_equal": True,
                        "metrics": {k: float(v) for k, v in metrics.items()},
                        "optimizer_groups": optimizer_state_identity(self.optimizer),
                        "scope": "all-rank finite gates and four-value parameter probes; not full equality"})
            return synced, metrics
    return ResumedTaskTrainer


def train(root):
    import torch
    config, plan = load(root)
    if int(os.environ.get("WORLD_SIZE", "0")) != 8:
        raise ValueError("Eight GPU ranks are required")
    Trainer = resumed_trainer_class(plan)
    class Factory:
        @staticmethod
        def instantiate(cfg, device_id):
            return Trainer(cfg, torch.device(f"cuda:{device_id}"))
    spec = importlib.util.spec_from_file_location("task_resume_production", task.REPO / "VLMExpert/scripts/train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.Trainer = Factory
    sys.argv = (Path(root) / "argv.txt").read_text().split()
    module.train(config)


def run(root):
    import fcntl
    root = Path(root).resolve(strict=True)
    with (root / "launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "training_state.json").exists():
            raise FileExistsError("Continuation segment was already launched")
        _, plan = load(root)
        if subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
            raise RuntimeError("GPUs occupied; no process was stopped")
        env = dict(os.environ, WANDB_MODE="offline", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   PYTHONNOUSERSITE="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                   CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7", NCCL_IB_DISABLE="1",
                   NCCL_P2P_DISABLE="0", NCCL_SHM_DISABLE="0")
        env["PYTHONPATH"] = os.pathsep.join(map(str, (task.REPO / "VLMExpert/src", task.REPO, task.REPO / "VLMExpert")))
        command = [sys.executable, str(Path(__file__).resolve()), "name", "--prepared", str(root)]
        naming = subprocess.run(command, env=dict(env, WORLD_SIZE="8", RANK="0", LOCAL_RANK="0"),
            cwd=task.REPO / "VLMExpert", text=True, capture_output=True, check=True)
        (root / "run_name.log").write_text(naming.stdout + naming.stderr)
        run_directory = naming.stdout.strip().splitlines()[-1]
        state = {"state": "starting", "started_utc": task.now(), "timestamp": plan["timestamp"],
                 "run_directory": run_directory, "log": str(root / "train.log"),
                 "source_step": plan["source_step"], "resumed_task_step": plan["resume_step"],
                 "max_optimizer_steps": plan["max_optimizer_steps"], "warmup_steps": plan["warmup_steps"],
                 "optimizer_backend": plan["optimizer_backend"], "parent_run_directory": plan["parent_run_directory"]}
        paths = ["scripts/4090_simple/a100_task_fused_resume.py", "scripts/4090_simple/a100_task_finetune.py",
                 "scripts/4090_simple/adamw_task_resume.py", "scripts/4090_simple/replay_task_data.py",
                 "VLMExpert/scripts/train.py", "VLMExpert/scripts/analysis/monitor_task_finetune.py",
                 "VLMExpert/src/psi/trainers/trainer.py", "VLMExpert/src/psi/trainers/posttrain.py",
                 "VLMExpert/src/psi/trainers/training_safety.py", "VLMExpert/src/psi/models/psi0.py",
                 "VLMExpert/src/psi/models/dual_source_action.py", "VLMExpert/src/psi/models/frozen_motion.py",
                 "VLMExpert/src/psi/config/transform.py", "VLMExpert/src/psi/config/transform_psi0_sonic.py",
                 "ImageProcess/processor.py"]
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=task.REPO, text=True).strip()
        write(root / "SOURCE_AT_LAUNCH.json", {
            "git_commit": revision, "captured_utc": task.now(),
            "git_status": subprocess.check_output(["git", "status", "--short"], cwd=task.REPO, text=True).strip(),
            "files_sha256": {name: task.shared.sha256(task.REPO / name) for name in paths}})
        state["source_revision"] = revision
        with (root / "train.log").open("x") as log, (root / "monitor.log").open("a") as monitor_log:
            process = subprocess.Popen([sys.executable, "-m", "torch.distributed.run", "--standalone",
                "--nproc_per_node=8", str(Path(__file__).resolve()), "train", "--prepared", str(root)],
                cwd=task.REPO / "VLMExpert", env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            state.update(state="running", pid=process.pid)
            write(root / "training_state.json", state)
            monitor = subprocess.Popen([sys.executable, str(task.REPO / "VLMExpert/scripts/analysis/monitor_task_finetune.py"),
                "--state", str(root / "training_state.json"), "--output-dir", str(root / "progress")],
                env=dict(env, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"),
                stdin=subprocess.DEVNULL, stdout=monitor_log, stderr=subprocess.STDOUT)
            result = process.wait()
            state.update(state="complete" if result == 0 else "failed", returncode=result, finished_utc=task.now())
            write(root / "training_state.json", state)
            try:
                state["monitor_returncode"] = monitor.wait(timeout=45)
            except subprocess.TimeoutExpired:
                state["monitor_status"] = "final_refresh_pending"
            write(root / "training_state.json", state)
            return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    prepare_parser = modes.add_parser("prepare")
    for name in ("parent", "checkpoint", "output"):
        prepare_parser.add_argument("--" + name, type=Path, required=True)
    prepare_parser.add_argument("--backend", choices=("single", "fused"), required=True)
    prepare_parser.add_argument("--full-validation-report", type=Path)
    prepare_parser.add_argument("--micro-validation-report", type=Path)
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
