"""Local joint LiftBoxHold pilot, independently initialized from a selected A100 snapshot.

Reuses accepted physical evidence through file-byte binding, the production
history/future reader, trainer/finite guards/per-group clipping, and private-RNG
fixed TEST before/after. Native AdamW masters/moments live on CPU. Only the final
model.safetensors is saved: this entrypoint cannot resume optimizer training.
The local batch/update budget is explicitly distinct from A100 batch256/40k.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import check_liftbox_joint_gpu as joint
import finetune_carrybox as shared


def validate_base_run(base_run, quality_base_run):
    """Bind a newer local weight snapshot while retaining the accepted data anchor."""
    base_run, quality_base_run = (Path(p).resolve(strict=True) for p in (base_run, quality_base_run))
    ready_path = base_run / "LOCAL_RUN_READY.json"
    ready = json.loads(ready_path.read_text())
    step = ready["step"]
    if (type(step) is not int or step <= 0 or ready.get("status") != "local_joint_run_prepared"
            or ready.get("optimizer_rng_restored") is not False
            or ready.get("strict_motion_load") != "passed"):
        raise ValueError("New base must be a locally prepared, weight-only, strict-loaded snapshot")
    snapshot_root = Path(ready["source_snapshot"]).resolve(strict=True)
    snapshot_path = base_run / "provenance/SNAPSHOT_READY.json"
    snapshot = json.loads(snapshot_path.read_text())
    if (shared.sha256(snapshot_path) != shared.sha256(snapshot_root / "SNAPSHOT_READY.json")
            or snapshot.get("step") != step or snapshot.get("model_sha256") != ready["joint_sha256"]
            or snapshot.get("save_completion_marker_verified") is not True or snapshot.get("training_modified") is not False):
        raise ValueError("LOCAL_RUN_READY and the protected SNAPSHOT_READY identities disagree")
    checkpoint = base_run / "checkpoints" / f"ckpt_{step}" / "model.safetensors"
    actual_sha = shared.sha256(checkpoint)
    if actual_sha != snapshot["model_sha256"] or checkpoint.stat().st_size != snapshot["model_bytes"]:
        raise ValueError("Actual new model differs from its protected local snapshot")
    prior_original_path = quality_base_run / "provenance/run_config.json"
    original_path = base_run / "provenance/run_config.json"
    original, prior = (json.loads(p.read_text()) for p in (original_path, prior_original_path))
    if original["model"] != prior["model"] or original["data"] != prior["data"]:
        raise ValueError("Original source model/data configuration changed; new weights cannot reuse this data acceptance")
    if shared.sha256(original_path) != shared.sha256(snapshot_root / "run_config.json"):
        raise ValueError("Copied new source configuration differs from the protected snapshot")
    statistics, prior_statistics = base_run / "dataset_statistics.json", quality_base_run / "dataset_statistics.json"
    dino, prior_dino = base_run / "dino_training_contract.json", quality_base_run / "dino_training_contract.json"
    if (shared.sha256(statistics) != shared.sha256(prior_statistics)
            or shared.sha256(statistics) != shared.sha256(snapshot_root / "dataset_statistics.json")
            or shared.sha256(dino) != shared.sha256(prior_dino)):
        raise ValueError("New base normalization bytes or complete DINO contract differ from the accepted anchor")
    current = json.loads((base_run / "run_config.json").read_text())
    old_local = json.loads((quality_base_run / "run_config.json").read_text())
    def structural_contract(value):
        result = {name: copy.deepcopy(value[name]) for name in ("model", "data")}
        result["model"]["model_name_or_path"] = "local_same_qwen"
        result["model"]["motion_checkpoint"] = "motion_extracted_from_this_snapshot"
        result["data"]["transform"]["field"]["stat_path"] = "identical_statistics_bytes"
        return result
    if structural_contract(current) != structural_contract(old_local):
        raise ValueError("Local base changed model/data beyond the declared immutable path replacements")
    qwen = Path(current["model"]["model_name_or_path"])
    if (not qwen.is_absolute() or qwen.resolve(strict=True) != Path(old_local["model"]["model_name_or_path"]).resolve(strict=True)):
        raise ValueError("New base must use the identical local Qwen construction resources")
    motion = Path(current["model"]["motion_checkpoint"]).resolve(strict=True)
    if motion != (base_run / "motion/joint_motion.ckpt").resolve(strict=True):
        raise ValueError("New base must use its own extracted Motion checkpoint")
    export_path = base_run / "motion/export_validation.json"
    export = json.loads(export_path.read_text())
    if (ready.get("motion_weights") != f"joint_posttrain_step_{step}" or export.get("weights") != ready["motion_weights"]
            or export.get("strict_meta_load") != "passed" or export.get("exact_serialization_comparison") != "passed"):
        raise ValueError("Motion extraction evidence does not belong to the new joint step")
    paths = [ready_path, snapshot_path, original_path, prior_original_path, base_run / "run_config.json", statistics, dino, motion, export_path]
    receipt = {"base_run": str(base_run), "source_step": step, "source_joint_sha256": actual_sha,
               "source_checkpoint": str(checkpoint), "model_bytes": checkpoint.stat().st_size,
               "original_source_model_and_data_equal": True, "statistics_bytes_equal": True, "dino_contract_bytes_equal": True,
               "motion_extraction": "reuse prepare_joint_snapshot strict tensor/serialization comparison; bind actual extracted file SHA",
               "source_sha256": {str(p): shared.sha256(p) for p in paths}}
    replacements = {"model.pretrained_joint_checkpoint": str(checkpoint), "model.motion_checkpoint": str(motion),
                    "model.model_name_or_path": str(qwen.resolve()), "data.transform.field.stat_path": str(statistics)}
    return replacements, receipt


def apply_base_run(config, argv, args, report, original):
    if getattr(args, "base_run", None) is None:
        return config, argv
    quality_base = Path(original["source_checkpoint"]).parents[2]
    replacements, identity = validate_base_run(args.base_run, quality_base)
    target = config.model_dump(mode="json")
    for key, value in replacements.items():
        shared.set_nested(target, key, value)
    updated = type(config).model_validate(target)
    normalization = shared.normalization_receipt(updated.data.transform.field)
    if (normalization["stat_file_sha256"] != report["normalization"]["stat_file_sha256"]
            or normalization["bounds_sha256"] != report["normalization"]["bounds_sha256"]):
        raise ValueError("Rebinding source weights must not refit or change inherited normalization")
    argv = shared.rewrite_argv(argv, replacements)
    (args.output / "argv.txt").write_text("\n".join(argv) + "\n")
    if shared.parse_training_argv(args.output / "argv.txt").model_dump(mode="json") != updated.model_dump(mode="json"):
        raise ValueError("Rebased source argv and JSON disagree")
    shared.write_json(args.output / "run_config.json", updated.model_dump(mode="json"))
    report.update(source_step=identity["source_step"], source_joint_step=identity["source_step"],
                  source_checkpoint=identity["source_checkpoint"], source_joint_sha256=identity["source_joint_sha256"],
                  normalization=normalization, base_run_identity=identity,
                  configuration_sha256=shared.sha256(args.output / "run_config.json"))
    report["source_sha256"].update(identity["source_sha256"])
    for path in identity["source_sha256"]:
        source = Path(path)
        if source.suffix == ".json" and source.stat().st_size < 2 << 20:
            base = Path(identity["base_run"])
            relative = source.relative_to(base) if source.is_relative_to(base) else Path("quality_anchor_original_config.json")
            destination = args.output / "source_at_launch/base_run_identity" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    return updated, argv


def pilot_overrides(args, alignment, output, timestamp):
    if (args.max_steps < 1 or args.batch_size < 1 or args.grad_accum < 1
            or not 0 <= args.warmup_steps < args.max_steps or args.test_windows != 20):
        raise ValueError("Use positive local steps/batches, warmup below updates, and the fixed 20 TEST windows")
    overrides = joint.joint_overrides(alignment, output, timestamp)
    overrides.update({
        "exp": "lift_box_hold_v11_joint_local_pilot", "seed": args.seed,
        "train.train_batch_size": args.batch_size,
        "train.gradient_accumulation_steps": args.grad_accum,
        "train.max_training_steps": args.max_steps,
        "train.warmup_steps": args.warmup_steps, "train.lr_scheduler_type": "cosine",
        "train.checkpointing_steps": args.max_steps + 1,
    })
    return overrides


def prepare(args):
    scope = f"local single GPU pilot: {args.max_steps} updates, b{args.batch_size}, accumulation{args.grad_accum}, effective batch {args.batch_size * args.grad_accum}"
    config, argv, report = joint.prepare(
        args, override_factory=lambda a, o, t: pilot_overrides(args, a, o, t), report_name="preparation.json",
        scope=scope, scheduler_scope=f"fresh cosine schedule, {args.warmup_steps} warmup updates, A100 five peak LRs; no restored scheduler state")
    actual = config.train
    if ((actual.train_batch_size, actual.gradient_accumulation_steps, actual.max_training_steps, actual.warmup_steps)
            != (args.batch_size, args.grad_accum, args.max_steps, args.warmup_steps)
            or actual.lr_scheduler_type != "cosine" or not config.model.tune_vlm or not config.model.tune_motion):
        raise RuntimeError("Actual parsed pilot config does not match the explicitly requested joint budget")
    original = json.loads((args.prepared_run / "preparation.json").read_text())
    report.pop("model_and_optimizer_checkpoints_saved", None)
    launcher_arguments = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    report.update({
        "training_mode": "joint_vlm_motion_action", "source_step": original["source_step"],
        "launcher_arguments": launcher_arguments,
        "quality_gate": original["quality_gate"],
        "accepted_physical_quality": {"preparation_path": str(args.prepared_run / "preparation.json"),
                                      "preparation_sha256": shared.sha256(args.prepared_run / "preparation.json"),
                                      "original_recomputation": original["quality_recomputation"],
                                      "reuse": "accepted numerical evidence retained; all packed payload and DINO feature bytes rebound in this execution"},
        "baseline": "independent weight-only initialization from the original source 28k, not the earlier Action-only300 checkpoint",
        "initial_global_step": 0, "training_updates": args.max_steps,
        "device_batch_size": args.batch_size, "gradient_accumulation_steps": args.grad_accum,
        "effective_local_batch": args.batch_size * args.grad_accum,
        "maximum_training_windows_consumed": args.max_steps * args.batch_size * args.grad_accum,
        "comparison_budget": "local pilot budget is explicitly configured; no claim of A100 batch256/40k or equal budget with earlier Action-only300",
        "model_checkpoint_plan": "one final model.safetensors only; config, statistics and provenance alongside the run",
        "optimizer_checkpoint_plan": "none; fresh CPU-master native torch AdamW, no A100 optimizer/scheduler/RNG restore",
        "test_policy": "same 20 real clean TEST windows before update0 and after final update, private RNG and no_grad; never selection/stopping/statistics",
        "source_image_policy": config.data.transform.model.motion_image_policy,
        "source_joint_step": original["source_step"],
        "data_quality_weight_anchor": {"source_step": original["source_step"], "source_joint_sha256": original["source_joint_sha256"],
                                       "scope": "accepted data quality provenance; distinct from the actual initialization weights"},
    })
    config, argv = apply_base_run(config, argv, args, report, original)
    report["baseline"] = f"independent weight-only initialization from the actual source step {report['source_step']}; not the earlier Action-only300 checkpoint"
    # Include this new entrypoint in the actual launch snapshot; dependencies
    # were already copied/hashed by the shared CPU preparation.
    path = Path(__file__).resolve()
    report["source_sha256"][str(path)] = shared.sha256(path)
    destination = args.output / "source_at_launch" / path.relative_to(shared.REPO)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)
    shared.write_json(args.output / "launcher_arguments.json", launcher_arguments)
    shared.write_json(args.output / "preparation.json", report)
    return config, argv, report


def save_final_model_only(trainer, completed, maximum):
    """Use Accelerate's proven shared-tensor writer, without save_state/optimizer IO."""
    if completed != maximum:
        return None
    root = Path(trainer.project_dir) / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    final = root / f"ckpt_{completed}"
    staging = root / f".ckpt_{completed}.model-only-pending"
    if final.exists() or staging.exists():
        raise FileExistsError("Preserve existing final/pending model artifacts; use a fresh output")
    trainer.accelerator.save_model(trainer.model, str(staging), max_shard_size="100GB", safe_serialization=True)
    if sorted(p.name for p in staging.iterdir()) != ["model.safetensors"]:
        raise RuntimeError("Final model must be exactly one safetensors file, with no optimizer state or shards")
    staging.rename(final)
    return str(final)


def train_joint(config, argv, args, preparation):
    import torch
    torch.set_num_threads(args.cpu_threads)
    from psi.trainers.posttrain import PosttrainTrainer
    from cpu_master_adamw import CpuMasterAdamW
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This local CPU-master pilot requires one GPU process")
    result = {"status": "running", "scope": preparation["scope"], "maximum_updates": args.max_steps,
              "completed_updates": 0, "comparison_complete": False, "optimizer_state_saved": False,
              "source_joint_sha256": preparation["source_joint_sha256"], "updates": [],
              "training_mode": "joint_vlm_motion_action", "normalization": preparation["normalization"]}
    result["runtime"] = {"python": sys.version, "torch": torch.__version__, "torch_cuda_build": torch.version.cuda,
                         "torch_cpu_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
                         "gpu_name": torch.cuda.get_device_properties(0).name}
    started = time.monotonic()
    instance = None

    class JointPilotTrainer(PosttrainTrainer):
        def __init__(self, cfg, device):
            super().__init__(cfg, device)
            self.lifecycle = shared.ComparisonLifecycle(args.max_steps)
            self.completed_updates = 0

        def create_datasets(self):
            train, val = super().create_datasets()
            candidates, columns, history = shared.eligible_dataset_rows(train, 3, source_frame_rule="ceil")
            indices = [i for episode in sorted(candidates) for i in candidates[episode]]
            sources = [int(columns["source_episode_index"][history.source_episode_ranges[0][ep][0]]) for ep in sorted(candidates)]
            if sources != [0, 1, 2]:
                raise ValueError("Only fixed TRAIN source0/1/2 may supply gradients")
            windows = {"source_rows_after_real_history": len(train), "retained_windows": len(indices),
                       "retained_per_episode": {str(ep): len(rows) for ep, rows in candidates.items()}, "source_episode_ids": sources,
                       "policy": "Production six real 5 Hz history frames plus thirty real 30 Hz future labels; ceil source clock, no padded future"}
            shared.write_json(args.output / "train_window_filter.json", windows)
            result["train_window_filter"] = windows
            result["budget_windows_per_filtered_epoch"] = len(indices)
            result["maximum_window_exposures_over_filtered_epoch"] = (
                args.max_steps * args.batch_size * args.grad_accum / len(indices))
            self.train_dataset = torch.utils.data.Subset(train, indices)
            return self.train_dataset, val

        def create_optimizers(self):
            super().create_optimizers()
            result["five_group_peak_contract"] = joint.check_optimizer_groups(
                self.model, self.optimizer, preparation["expected_groups"], device_type="cpu")
            self.cpu_master = CpuMasterAdamW.from_optimizer(self.optimizer)
            self.optimizer = self.cpu_master

        def prepare(self, accelerator):
            loader = super().prepare(accelerator)
            if accelerator.num_processes != 1 or self.resume_from_checkpoint()[0] != 0:
                raise ValueError("Require fresh one-GPU optimizer/scheduler/global step")
            initial = joint.cpu_state_receipt(self.cpu_master)
            if initial["completed_steps"] or initial["native_adamw_state_entries"]:
                raise ValueError("Initial CPU AdamW state must be fresh")
            if int(self.lr_scheduler.state_dict().get("last_epoch", -1)) not in (-1, 0):
                raise ValueError("Scheduler must start fresh")
            if shared.normalization_receipt(self.cfg.data.transform.field) != preparation["normalization"]:
                raise ValueError("Inherited bounds/statistics changed")
            if self._get_training_safety() is None:
                raise ValueError("Production safety guard must stay enabled")
            result.update(initial_global_step=0, initial_optimizer=initial, project_dir=str(self.project_dir),
                          initial_group_lrs=self.group_lrs(), before_group_sha256=joint.group_parameter_hashes(self.unwrap_model(), self.optimizer))
            Path(self.project_dir).mkdir(parents=True, exist_ok=True)
            source_run = Path(config.model.pretrained_joint_checkpoint).parents[2]
            shutil.copy2(source_run / "dino_training_contract.json", Path(self.project_dir) / "dino_training_contract.json")
            shutil.copytree(args.output / "source_at_launch", Path(self.project_dir) / "source_at_launch")
            shared.write_json(Path(self.project_dir) / "local_adaptation_provenance.json", preparation)
            with shared.isolated_rng(args.seed + 9000, self.device):
                self.test_batches, self.test_manifest = shared.fixed_test_batches(self, args, episode_count=1, source_frame_rule="ceil")
            torch.cuda.reset_peak_memory_stats(self.device)
            self.lifecycle.before(lambda: shared.evaluate_fixed_test(self, self.test_batches, self.test_manifest, "before", args))
            result["before_comparison_complete"] = True
            self.record_peak()
            shared.write_json(args.output / "training_result.json", result)
            return loader

        def step(self, batch_input, global_step, local_step):
            if not self.lifecycle.before_done or self.lifecycle.after_done or global_step != self.lifecycle.completed:
                raise RuntimeError("Actual updates must remain within the fixed before/after protocol")
            used_lrs = self.group_lrs()
            synced, metrics = super().step(batch_input, global_step, local_step)
            self.lifecycle.update(global_step, synced)
            self.completed_updates = self.lifecycle.completed
            if synced:
                joint.validate_update_metrics(metrics, self._grad_norms)
                if self.cpu_master.completed_steps != self.completed_updates:
                    raise RuntimeError("CPU optimizer step count disagrees with actual synchronized updates")
                result["completed_updates"] = self.completed_updates
                result["updates"].append({"completed_step": self.completed_updates, "loss_last_microbatch": float(metrics["loss"]),
                    "used_group_lrs": used_lrs, "next_group_lrs": self.group_lrs(),
                    "pre_clip_grad_norms": {k: float(v) for k, v in self._grad_norms.items()}})
                result["last_training_metrics"] = {k: float(v) for k, v in metrics.items() if isinstance(v, (int,float)) and math.isfinite(float(v))}
                self.record_peak()
                shared.write_json(args.output / "training_result.json", result)
            return synced, metrics

        def record_peak(self):
            result["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(self.device)
            result["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(self.device)
            result["cuda_peak_scope"] = "Since prepare: fixed before TEST, training, after TEST; excludes cold model construction"

        def save_checkpoint(self, global_step):
            if global_step != args.max_steps:
                return None
            if self.completed_updates != args.max_steps:
                raise RuntimeError("Final model may only be saved after every actual update")
            path = save_final_model_only(self, global_step, args.max_steps)
            result["final_checkpoint"] = str(Path(path) / "model.safetensors")
            return path

        def finalize(self):
            self.lifecycle.after(lambda: shared.evaluate_fixed_test(self, self.test_batches, self.test_manifest, "after", args))
            result["after_group_sha256"] = joint.group_parameter_hashes(self.unwrap_model(), self.optimizer)
            result["all_five_groups_changed"] = all(result["before_group_sha256"][name] != result["after_group_sha256"][name] for name in joint.GROUPS)
            if not result["all_five_groups_changed"]:
                raise RuntimeError("Final VLM/Motion/Action joint update evidence is incomplete")
            if {path: shared.sha256(path) for path in preparation["dino_file_sha256"]} != preparation["dino_file_sha256"]:
                raise RuntimeError("Frozen DINO encoder identity changed")
            result["dino_files_unchanged"] = True
            result["final_optimizer"] = joint.cpu_state_receipt(self.cpu_master)
            final = Path(self.project_dir) / "checkpoints" / f"ckpt_{args.max_steps}" / "model.safetensors"
            if not final.is_file() or sorted(final.parent.iterdir()) != [final]:
                raise RuntimeError("Require one final model-only safetensors checkpoint")
            result["final_checkpoint_sha256"] = shared.sha256(final)
            result["final_checkpoint_size_bytes"] = final.stat().st_size
            result["comparison_complete"] = True
            self.record_peak()
            super().finalize()

    class LocalFactory:
        @staticmethod
        def instantiate(cfg, device_id):
            nonlocal instance
            instance = JointPilotTrainer(cfg, torch.device(f"cuda:{device_id}"))
            return instance

    saved_argv = sys.argv[:]
    saved_environment_info = os.environ.pop("_ENV_INFO_JSON", None)
    try:
        spec = importlib.util.spec_from_file_location("liftbox_joint_pilot_local_entrypoint", shared.REPO / "VLMExpert/scripts/train.py")
        entrypoint = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entrypoint)
        entrypoint.Trainer = LocalFactory
        sys.argv = argv
        entrypoint.train(config)
        if instance is None or not instance.lifecycle.after_done or not result["comparison_complete"] or instance.completed_updates != args.max_steps:
            raise RuntimeError("Training did not complete every configured update and both fixed TEST passes")
        before = json.loads((args.output / "test_before.json").read_text())
        after = json.loads((args.output / "test_after.json").read_text())
        result["metric_change_after_minus_before"] = {key: after["mean"][key] - before["mean"][key] for key in before["mean"]}
        result["status"] = "completed"
        result["interpretation"] = "Local pilot offline comparison only; no claim of physical task success or A100 training-budget equivalence"
    except BaseException as error:
        result.update(status="failed", exception={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        sys.argv = saved_argv
        if saved_environment_info is not None:
            os.environ["_ENV_INFO_JSON"] = saved_environment_info
        result["wall_time_s"] = time.monotonic() - started
        shared.write_json(args.output / "training_result.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared-run", "alignment", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--base-run", type=Path, help="Optional newer prepared local weight-only run; accepted data quality anchor stays explicit")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261021)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--train", action="store_true")
    args = parser.parse_args()
    args.test_windows = 20
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    sys.path[:0] = [str(shared.REPO / "VLMExpert/src"), str(shared.REPO)]
    config, argv, preparation = prepare(args)
    if args.train:
        train_joint(config, argv, args, preparation)
    print(json.dumps({"status": "completed" if args.train else "cpu_config_prepared", "output": str(args.output)}))


if __name__ == "__main__":
    main()
