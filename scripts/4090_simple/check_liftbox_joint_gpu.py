"""Two b1 joint updates through the production trainer, with local CPU AdamW masters.

--prepare-only is CPU only. --run-gpu must be launched separately after review,
using a fresh external output. This is not the A100 batch256/DDP training or a
task-success evaluation. No model/optimizer checkpoint is saved, and no remote
optimizer, scheduler or RNG is read. The existing Action-only launcher is unchanged.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import finetune_carrybox as shared

GROUPS = ("action_header", "lang_backbone", "vision_tower", "mm_projector", "motion_expert")
LR_FIELDS = dict(zip(GROUPS, ("train.learning_rate", "model.lang_backbone_lr", "model.vision_tower_lr",
                            "model.mm_projector_lr", "model.motion_expert_lr")))


def alignment_groups(alignment):
    model = alignment["configuration"]["model"]
    if (model.get("action_architecture") != "dual_source_v1" or model.get("tune_vlm") is not True
            or model.get("tune_motion") is not True
            or any(model.get(name) is not False for name in ("tune_mm_llm", "tune_mm_vision", "tune_mm_mlp"))):
        raise ValueError("Alignment must describe the actual full VLM/Motion/Action joint run")
    rows = alignment["optimizer_groups"]
    groups = {row["name"]: row for row in rows}
    if len(rows) != 5 or set(groups) != set(GROUPS):
        raise ValueError("Require exactly the five actual A100 optimizer groups")
    for name, key in LR_FIELDS.items():
        row = groups[name]
        lr = float(row["peak_lr"])
        section, field = key.split(".")
        if (not math.isfinite(lr) or lr <= 0 or type(row["trainable_parameters"]) is not int
                or row["trainable_parameters"] <= 0
                or alignment["configuration"][section][field] != lr):
            raise ValueError(f"Invalid or inconsistent A100 group {name}")
    return groups


def joint_overrides(alignment, output, timestamp):
    expected = alignment_groups(alignment)
    overrides = {key: expected[name]["peak_lr"] for name, key in LR_FIELDS.items()}
    overrides.update({
        "exp": "liftbox_joint_two_update_check", "timestamp": timestamp,
        "train.output_dir": str(output / "training"), "train.resume_from_checkpoint": None,
        "train.train_batch_size": 1, "train.gradient_accumulation_steps": 1,
        "train.max_training_steps": 2, "train.num_train_epochs": None,
        "train.lr_scheduler_type": "constant", "train.warmup_steps": 0, "train.warmup_ratio": None,
        "train.validation_steps": 0, "train.checkpointing_steps": 3,
        "train.early_checkpoint_steps": [], "train.training_safety_checks": True,
        "train.data_parallel": "ddp", "train.mixed_precision": "bf16",
        "train.optimizer_foreach": False, "train.keep_train_batch_on_cpu": True,
        "train.overfit_single_batch": False, "train.max_grad_norm": 1.0,
        "model.tune_vlm": True, "model.tune_motion": True,
        "model.tune_mm_llm": False, "model.tune_mm_vision": False, "model.tune_mm_mlp": False,
        "model.gradient_checkpointing": True,
    })
    return overrides


def dino_file_receipt(contract_path):
    contract = json.loads(Path(contract_path).read_text())
    directory = Path(contract["online_checkpoint"])
    if not directory.is_absolute():
        raise ValueError("DINO must be an absolute local directory")
    directory = directory.resolve(strict=True)
    receipt = {}
    for relative, expected in contract["encoder"]["local_file_sha256"].items():
        path = (directory / relative).resolve(strict=True)
        if not path.is_relative_to(directory) or shared.sha256(path) != expected:
            raise ValueError("Actual local DINO identity differs from the inherited cache encoder")
        receipt[str(path)] = expected
    return receipt


def verify_payload_bytes(root, checksums):
    """Bind existing numerical acceptance to actual payload bytes without recomputation."""
    root = Path(root).resolve(strict=True)
    if not checksums:
        raise ValueError("Accepted payload identities must be nonempty")
    for relative, expected in checksums.items():
        path = (root / relative).resolve(strict=True)
        if Path(relative).is_absolute() or not path.is_relative_to(root) or shared.sha256(path) != expected:
            raise ValueError(f"Accepted payload bytes changed: {relative}")
    return {"root": str(root), "checked_files": len(checksums), "sha256": checksums,
            "scope": "file bytes only; accepted physics/encoder numerical checks not repeated"}


def prepare(args, *, override_factory=joint_overrides, report_name="joint_check_preparation.json",
            scope="single GPU, two updates, b1 accumulation1",
            scheduler_scope="constant peak LRs for both nonzero local check updates; no A100 scheduler state restored"):
    args.prepared_run = args.prepared_run.resolve(strict=True)
    args.alignment = args.alignment.resolve(strict=True)
    if not args.output.is_absolute() or args.output.resolve().is_relative_to(shared.REPO):
        raise ValueError("Use an absolute output outside the repository")
    args.output = args.output.resolve()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use a fresh output; existing run artifacts are preserved")
    source_path = args.prepared_run / "run_config.json"
    source = json.loads(source_path.read_text())
    preparation_path = args.prepared_run / "preparation.json"
    previous = json.loads(preparation_path.read_text())
    quality = previous["quality_gate"]
    if (previous["task_id"] != "simple/G1WholebodyLiftBoxHoldSonic-v0"
            or quality["status"] != "actual_new_task_evidence_recomputed"
            or quality["source_frame_rule"] != "ceil"
            or quality["split_episode_counts"] != {"train": 3, "val": 1, "test": 1}
            or previous["optimizer_scheduler_rng_restored"] is not False
            or shared.sha256(source_path) != previous["launch_config_sha256"]):
        raise ValueError("Input must be the existing legally prepared independent Lift pilot")
    root = Path(source["data"]["root_dir"])
    contract_path = root / "LIFT_COMMANDED_ROLLOUT_READY.json"
    if shared.sha256(contract_path) != previous["conversion_ready_sha256"]:
        raise ValueError("Prepared physical dataset identity changed")
    if shared.sha256(root / "STATE_ACTION_CONTRACT.json") != previous["state_action_contract_sha256"]:
        raise ValueError("Prepared named state/action identity changed")
    pack = json.loads(contract_path.read_text())
    payload = verify_payload_bytes(root, pack["packed_files"])
    dino_features = {}
    for split, rows in previous["dino_checksums"].items():
        for row in rows:
            relative = f"{split}/episodes/{row['episode']:08d}/features.npy"
            dino_features[relative] = row["feature_identity"]["sha256"]
    feature_payload = verify_payload_bytes(Path(source["data"]["dino_cache_root"]), dino_features)
    for split, identity in previous["splits"].items():
        for path, key in ((root / split / "meta/info.json", "info_sha256"),
                          (root / split / "meta/episodes.jsonl", "episodes_sha256"),
                          (Path(source["data"]["dino_cache_root"]) / split / "manifest.json", "dino_manifest_sha256")):
            if shared.sha256(path) != identity[key]:
                raise ValueError(f"Prepared {split} metadata/cache identity changed")
    alignment = json.loads(args.alignment.read_text())
    expected = alignment_groups(alignment)
    timestamp = datetime.now().strftime("%y%m%d%H%M%S%f")
    overrides = override_factory(alignment, args.output, timestamp)
    target = copy.deepcopy(source)
    for key, value in overrides.items():
        shared.set_nested(target, key, value)
    args.output.mkdir(parents=True, exist_ok=True)
    original = (args.prepared_run / "argv.txt").read_text().split()
    original_config = shared.parse_training_argv(args.prepared_run / "argv.txt")
    if original_config.model_dump(mode="json") != source:
        raise ValueError("Prepared original argv and JSON disagree")
    argv = shared.rewrite_argv(original, overrides)
    (args.output / "argv.txt").write_text("\n".join(argv) + "\n")
    parsed = shared.parse_training_argv(args.output / "argv.txt")
    config = type(parsed).model_validate(target)
    if parsed.model_dump(mode="json") != config.model_dump(mode="json"):
        raise ValueError("Joint check argv/config is not replayable")
    normalization = shared.normalization_receipt(config.data.transform.field)
    if normalization != previous["normalization"]:
        raise ValueError("Source bounds must remain byte-for-byte inherited, never refitted")
    checkpoint = Path(config.model.pretrained_joint_checkpoint)
    if str(checkpoint) != previous["source_checkpoint"]:
        raise ValueError("Use the original weight-only A100 snapshot, not Action300 weights")
    dino_contract = checkpoint.parents[2] / "dino_training_contract.json"
    if shared.sha256(dino_contract) != previous["source_dino_contract_sha256"]:
        raise ValueError("Inherited DINO contract changed")
    dino = dino_file_receipt(dino_contract)
    shared.write_json(args.output / "run_config.json", config.model_dump(mode="json"))
    paths = [Path(__file__), Path(shared.__file__), shared.REPO / "VLMExpert/scripts/train.py"]
    paths += [shared.REPO / "VLMExpert/src/psi/trainers" / name
              for name in ("posttrain.py", "trainer.py", "training_safety.py")]
    paths += [shared.REPO / "VLMExpert/src/psi/models" / name
              for name in ("psi0.py", "frozen_motion.py", "dual_source_action.py")]
    paths += [args.alignment, preparation_path, source_path, contract_path, dino_contract]
    wrapper = Path(__file__).with_name("cpu_master_adamw.py")
    if wrapper.is_file():
        paths.append(wrapper)
    for path in paths:
        if path.is_relative_to(shared.REPO):
            destination = args.output / "source_at_launch" / path.relative_to(shared.REPO)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    report = {
        "status": "cpu_config_prepared", "gpu_training": "not_run", "scope": scope,
        "alignment_scope": "A100 five trainable groups/counts/peak LRs, joint flags and production safety; not A100 batch256/DDP/schedule equivalence",
        "scheduler_scope": scheduler_scope,
        "expected_groups": expected, "source_joint_sha256": previous["source_joint_sha256"],
        "source_checkpoint": str(checkpoint), "prepared_run": str(args.prepared_run),
        "normalization": normalization, "dataset_root": str(root), "dino_file_sha256": dino,
        "packed_payload_bytes": payload, "dino_feature_payload_bytes": feature_payload,
        "cpu_threads_requested": args.cpu_threads,
        "source_frame_rule": "ceil", "history_frames": 6, "history_hz": 5, "future_frames": 30, "action_hz": 30,
        "optimizer_scheduler_rng_restored": False, "model_and_optimizer_checkpoints_saved": False,
        "configuration_sha256": shared.sha256(args.output / "run_config.json"),
        "source_sha256": {str(path): shared.sha256(path) for path in paths},
    }
    shared.write_json(args.output / report_name, report)
    return config, argv, report


def check_optimizer_groups(model, optimizer, expected, *, device_type="cuda"):
    import torch
    groups = optimizer.param_groups
    names = [group.get("group_name") for group in groups]
    if names != list(GROUPS):
        raise ValueError(f"Expected original production five-group order, got {names}")
    parameters = [p for group in groups for p in group["params"]]
    ids = [id(p) for p in parameters]
    if len(set(ids)) != len(ids) or set(ids) != {id(p) for p in model.parameters() if p.requires_grad}:
        raise ValueError("Every trainable parameter must occur exactly once in the five optimizer groups")
    rows = []
    for group in groups:
        name = group["group_name"]
        count = sum(p.numel() for p in group["params"])
        if count != expected[name]["trainable_parameters"] or not math.isclose(group["lr"], expected[name]["peak_lr"], rel_tol=1e-12):
            raise ValueError(f"Actual group {name} counts/LR disagree with A100")
        if any(not p.requires_grad or p.dtype != torch.float32 or p.device.type != device_type for p in group["params"]):
            raise ValueError("Production model parameters must be trainable FP32 on the requested device")
        rows.append({"name": name, "trainable_parameters": count, "lr": float(group["lr"])})
    return rows


def group_parameter_hashes(model, optimizer):
    """Hash all five groups, with bounded CPU copies and finite final parameters."""
    import torch
    names = {id(p): name for name, p in model.named_parameters()}
    hashes = {}
    for group in optimizer.param_groups:
        digest = hashlib.sha256()
        for parameter in group["params"]:
            digest.update(names[id(parameter)].encode())
            digest.update(str((tuple(parameter.shape), str(parameter.dtype))).encode())
            flat = parameter.detach().reshape(-1)
            for start in range(0, flat.numel(), 4 << 20):
                chunk = flat[start:start + (4 << 20)].to("cpu").contiguous()
                if not torch.isfinite(chunk).all():
                    raise FloatingPointError("Non-finite joint model parameter")
                digest.update(chunk.view(torch.uint8).numpy().tobytes())
        hashes[group["group_name"]] = digest.hexdigest()
    return hashes


def cpu_state_receipt(optimizer):
    """Inspect actual tensors, not merely a claimed offload configuration."""
    masters = [p for group in optimizer.cpu_optimizer.param_groups for p in group["params"]]
    if any(p.device.type != "cpu" for p in masters):
        raise ValueError("CPU AdamW master parameters left CPU")
    state_devices = {str(value.device) for state in optimizer.cpu_optimizer.state.values()
                     for value in state.values() if hasattr(value, "device")}
    if state_devices - {"cpu"}:
        raise ValueError("AdamW moment or step tensor left CPU")
    return {"completed_steps": int(optimizer.completed_steps), "master_elements": sum(p.numel() for p in masters),
            "master_device_types": sorted({p.device.type for p in masters}),
            "native_adamw_state_entries": len(optimizer.cpu_optimizer.state), "actual_state_devices": sorted(state_devices),
            "wrapper_receipt": optimizer.receipt()}


def validate_update_metrics(metrics, norms):
    if not math.isfinite(float(metrics["loss"])):
        raise FloatingPointError("Non-finite training loss")
    if set(norms) != set(GROUPS) or any(not math.isfinite(float(v)) or float(v) <= 0 for v in norms.values()):
        raise FloatingPointError("Require finite nonzero pre-clip gradient norms in all five groups")


def run_gpu(config, argv, args, preparation):
    import torch
    torch.set_num_threads(args.cpu_threads)
    from psi.trainers.posttrain import PosttrainTrainer
    from cpu_master_adamw import CpuMasterAdamW
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This check is exactly one GPU/process")
    result = {"status": "running", "scope": preparation["scope"], "completed_updates": 0, "updates": [],
              "checkpoint_save_calls_suppressed": [], "model_and_optimizer_checkpoints_saved": False}
    gpu = torch.cuda.get_device_properties(0)
    result["runtime"] = {"python": sys.version, "torch": torch.__version__, "torch_cuda_build": torch.version.cuda,
                         "gpu_name": gpu.name, "gpu_total_memory_bytes": gpu.total_memory,
                         "torch_cpu_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads()}
    instance = None
    started = time.monotonic()

    class JointCheckTrainer(PosttrainTrainer):
        def create_datasets(self):
            train, val = super().create_datasets()
            candidates, columns, history = shared.eligible_dataset_rows(train, 3, source_frame_rule="ceil")
            indices = [i for episode in sorted(candidates) for i in candidates[episode]]
            result["train_complete_window_filter"] = {
                "real_history_reader_rows": len(train), "retained_full_future_windows": len(indices),
                "sources": [int(columns["source_episode_index"][history.source_episode_ranges[0][ep][0]]) for ep in sorted(candidates)],
                "retained_per_episode": {str(ep): len(rows) for ep, rows in candidates.items()},
                "source_frame_rule": "ceil", "history": "production six real 5 Hz frames", "future": "30 real 30 Hz labels; no tail copies",
            }
            accepted = json.loads((args.prepared_run / "preparation.json").read_text())
            if result["train_complete_window_filter"]["sources"] != shared.training_source_contract(accepted):
                raise ValueError("Joint check may only train on the explicitly accepted TRAIN source episodes")
            self.train_dataset = torch.utils.data.Subset(train, indices)
            return self.train_dataset, val

        def create_optimizers(self):
            super().create_optimizers()
            # Keep production grouping/kwargs; no changes to A100's shared trainer.
            check_optimizer_groups(self.model, self.optimizer, preparation["expected_groups"], device_type="cpu")
            self.cpu_master = CpuMasterAdamW.from_optimizer(self.optimizer)
            self.optimizer = self.cpu_master

        def prepare(self, accelerator):
            loader = super().prepare(accelerator)
            if accelerator.num_processes != 1 or self.resume_from_checkpoint()[0] != 0:
                raise ValueError("Require a fresh optimizer/scheduler/global step on one GPU")
            result["groups"] = check_optimizer_groups(self.unwrap_model(), self.optimizer, preparation["expected_groups"])
            initial = cpu_state_receipt(self.cpu_master)
            if initial["completed_steps"] != 0 or initial["native_adamw_state_entries"] != 0:
                raise ValueError("CPU master optimizer must start with no restored state or steps")
            if int(self.lr_scheduler.state_dict().get("last_epoch", -1)) not in (-1, 0):
                raise ValueError("Scheduler is not fresh")
            if shared.normalization_receipt(self.cfg.data.transform.field) != preparation["normalization"]:
                raise ValueError("Normalization changed before training")
            if self._get_training_safety() is None:
                raise ValueError("Production finite safety guard must remain enabled")
            result.update(initial_global_step=0, initial_optimizer=initial, project_dir=str(self.project_dir),
                          before_group_sha256=group_parameter_hashes(self.unwrap_model(), self.optimizer))
            torch.cuda.reset_peak_memory_stats(self.device)
            shared.write_json(args.output / "joint_check_result.json", result)
            return loader

        def step(self, batch_input, global_step, local_step):
            if global_step != result["completed_updates"] or global_step >= 2:
                raise ValueError("Unexpected update count in bounded joint check")
            input_sha = shared.tensor_signature(batch_input)
            torch.cuda.synchronize(self.device)
            step_started = time.monotonic()
            synced, metrics = super().step(batch_input, global_step, local_step)
            torch.cuda.synchronize(self.device)
            if not synced:
                raise ValueError("b1 accumulation1 must complete each actual update")
            validate_update_metrics(metrics, self._grad_norms)
            guard = self._get_training_safety()
            if guard.context.get("global_step") != global_step or not guard.snapshots:
                raise ValueError("Production guard did not capture the current real batch")
            state = cpu_state_receipt(self.cpu_master)
            if state["completed_steps"] != global_step + 1 or state["native_adamw_state_entries"] <= 0:
                raise ValueError("CPU native AdamW did not perform the expected update")
            result["completed_updates"] += 1
            result["updates"].append({"completed_step": global_step + 1, "input_sha256": input_sha,
                "loss": float(metrics["loss"]), "pre_clip_grad_norms": {k: float(v) for k, v in self._grad_norms.items()},
                "group_lrs": self.group_lrs(), "wall_time_s": time.monotonic() - step_started,
                "production_safety_capture": dict(guard.context), "cpu_optimizer": state})
            result["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(self.device)
            result["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(self.device)
            shared.write_json(args.output / "joint_check_result.json", result)
            return synced, metrics

        def save_checkpoint(self, global_step):
            result["checkpoint_save_calls_suppressed"].append(int(global_step))
            return None

        def finalize(self):
            if result["completed_updates"] != 2:
                raise ValueError("Need two complete actual updates")
            result["after_group_sha256"] = group_parameter_hashes(self.unwrap_model(), self.optimizer)
            result["all_five_groups_changed"] = all(result["before_group_sha256"][name] != result["after_group_sha256"][name] for name in GROUPS)
            if not result["all_five_groups_changed"]:
                raise ValueError("A joint trainable group did not change")
            actual = {path: shared.sha256(path) for path in preparation["dino_file_sha256"]}
            if actual != preparation["dino_file_sha256"]:
                raise ValueError("Frozen offline DINO weights changed")
            result["dino_files_unchanged"] = True
            result["dino_scope"] = "Frozen offline cache encoder; no DINO module in these five optimizer groups"
            result["final_optimizer"] = cpu_state_receipt(self.cpu_master)
            result["production_guard_class"] = type(self._get_training_safety()).__name__
            result["status"] = "completed_two_joint_updates"
            super().finalize()

    class LocalFactory:
        @staticmethod
        def instantiate(cfg, device_id):
            nonlocal instance
            instance = JointCheckTrainer(cfg, torch.device(f"cuda:{device_id}"))
            return instance

    saved_argv = sys.argv[:]
    saved_environment_info = os.environ.pop("_ENV_INFO_JSON", None)
    try:
        spec = importlib.util.spec_from_file_location("liftbox_joint_check_local_entrypoint", shared.REPO / "VLMExpert/scripts/train.py")
        entrypoint = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entrypoint)
        entrypoint.Trainer = LocalFactory
        sys.argv = argv
        entrypoint.train(config)
        if result["status"] != "completed_two_joint_updates":
            raise ValueError("Production entrypoint did not finish the complete joint check")
    except BaseException as error:
        result.update(status="failed", exception={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        sys.argv = saved_argv
        if saved_environment_info is not None:
            os.environ["_ENV_INFO_JSON"] = saved_environment_info
        result["wall_time_s"] = time.monotonic() - started
        result["provenance"] = str(args.output / "joint_check_preparation.json")
        shared.write_json(args.output / "joint_check_result.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-run", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-threads", type=int, default=4)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--run-gpu", action="store_true")
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    sys.path[:0] = [str(shared.REPO / "VLMExpert/src"), str(shared.REPO)]
    config, argv, preparation = prepare(args)
    if args.run_gpu:
        run_gpu(config, argv, args, preparation)
    print(json.dumps({"status": "joint_check_complete" if args.run_gpu else "cpu_config_prepared", "output": str(args.output)}))


if __name__ == "__main__":
    main()
