"""Prepare and supervise an eight-GPU, weight-only joint task adaptation.

The original joint run and task release are read-only. The local trainer factory
adds complete-future filtering and a fixed validation subset without changing
the shared UnifoLM trainer or its optimizer/finite-gradient checks.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import finetune_carrybox as shared

REPO = Path(__file__).resolve().parents[2]
for location in (REPO, REPO / "VLMExpert/src", REPO / "VLMExpert"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))
TRAIN_ID = "Handover-sonic11-train"
VAL_ID = "Handover-sonic11-val"
PEAK_LRS = {"action_header": 1e-5, "lang_backbone": 1e-7,
            "vision_tower": 1e-6, "mm_projector": 1e-5, "motion_expert": 1e-6}


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def overrides(source_run, dataset_root, dino_root, output, *, max_steps=2000, warmup=100,
              timestamp=None, source_step=40000, checkpoint_steps=200,
              validation_steps=200, early_checkpoint_steps=None):
    if max_steps < 2 or not 0 < warmup < max_steps:
        raise ValueError("Use a positive warmup below the optimizer-update budget")
    if (type(source_step) is not int or source_step <= 0
            or type(checkpoint_steps) is not int or checkpoint_steps <= 0
            or type(validation_steps) is not int or validation_steps <= 0):
        raise ValueError("Source step and checkpoint/validation intervals must be positive integers")
    early_checkpoint_steps = [20] if early_checkpoint_steps is None else list(early_checkpoint_steps)
    if any(type(step) is not int or not 0 < step <= max_steps for step in early_checkpoint_steps):
        raise ValueError("Early checkpoints must be positive updates within the task budget")
    source_run, dataset_root, dino_root, output = map(Path, (source_run, dataset_root, dino_root, output))
    return {
        "seed": 20261022, "exp": ("handover_task50_joint40k" if source_step == 40000
                                    else f"handover_task50_joint{source_step}"), "timestamp": timestamp,
        "auto_tag_run": False, "eval": False, "debug": False,
        "log.report_to": "wandb", "wandb.project": "cvpr-simple-task",
        "wandb.id": None, "wandb.name": None, "wandb.group": None,
        "wandb.entity": None, "wandb.resume": "never",
        "train.name": "posttrain", "train.output_dir": str(output / "training"),
        "train.resume_from_checkpoint": None, "train.skip_resumed_steps": False,
        "train.lora": False, "train.data_parallel": "ddp", "train.mixed_precision": "bf16",
        "train.deepspeed_config": str(REPO / "VLMExpert/scripts/deepspeed/zero3.json"),
        "train.train_batch_size": 4, "train.val_batch_size": 1,
        "train.gradient_accumulation_steps": 1, "train.num_workers": 4,
        "train.dataloader_start_method": "spawn", "train.training_safety_checks": True,
        "train.optimizer_foreach": False, "train.keep_train_batch_on_cpu": True,
        "train.sync_metrics_on_update_only": True, "train.overfit_single_batch": False,
        "train.max_grad_norm": 1.0, "train.max_training_steps": max_steps,
        "train.num_train_epochs": None, "train.learning_rate": PEAK_LRS["action_header"],
        "train.warmup_steps": warmup, "train.warmup_ratio": None,
        "train.lr_scheduler_type": "cosine", "train.checkpointing_steps": checkpoint_steps,
        "train.early_checkpoint_steps": early_checkpoint_steps, "train.max_checkpoints_to_keep": 12,
        "train.preserve_checkpoint_every": None,
        "train.validation_steps": validation_steps, "train.val_num_batches": -1,
        "data.root_dir": str(dataset_root), "data.train_repo_ids": [TRAIN_ID],
        "data.val_repo_ids": [VAL_ID], "data.dino_cache_root": str(dino_root),
        "data.transform.field.stat_path": str(source_run / "dataset_statistics.json"),
        "model.pretrained_joint_checkpoint": str(source_run / f"checkpoints/ckpt_{source_step}/model.safetensors"),
        "model.pretrained_action_header_path": None,
        "model.tune_vlm": True, "model.tune_motion": True,
        "model.tune_mm_llm": False, "model.tune_mm_vision": False, "model.tune_mm_mlp": False,
        "model.gradient_checkpointing": True, "model.vlm_skip_unused_logits": True,
        "model.lang_backbone_lr": PEAK_LRS["lang_backbone"],
        "model.vision_tower_lr": PEAK_LRS["vision_tower"],
        "model.mm_projector_lr": PEAK_LRS["mm_projector"],
        "model.motion_expert_lr": PEAK_LRS["motion_expert"],
    }


def check_config(config, source):
    model, data, train = config.model, config.data, config.train
    if (model.action_architecture != "dual_source_v1" or model.num_blocks != 6
            or (model.action_chunk_size, model.action_dim, model.odim) != (30, 80, 45)
            or not model.tune_vlm or not model.tune_motion
            or train.resume_from_checkpoint is not None
            or model.pretrained_action_header_path is not None
            or (train.train_batch_size, train.gradient_accumulation_steps) != (4, 1)):
        raise ValueError("Require the three-component architecture and fresh batch32 task adaptation")
    expected_data = copy.deepcopy(source["data"])
    actual_data = data.model_dump(mode="json")
    for key in ("root_dir", "train_repo_ids", "val_repo_ids", "dino_cache_root"):
        expected_data[key] = actual_data[key]
    expected_data["transform"]["field"]["stat_path"] = actual_data["transform"]["field"]["stat_path"]
    if expected_data != actual_data:
        raise ValueError("Task adaptation must inherit the original RGB/repack/history/normalization contract")


def prepare(args):
    source_run = args.source_run.resolve(strict=True)
    output = args.output.absolute()
    if not output.is_absolute() or output.resolve().is_relative_to(REPO) or output.exists():
        raise ValueError("Use a fresh absolute output outside the repository")
    source = read_json(source_run / "run_config.json")
    if source["model"]["action_architecture"] != "dual_source_v1":
        raise ValueError("Source must be our complete dual-source run")
    checkpoint = source_run / f"checkpoints/ckpt_{args.source_step}/model.safetensors"
    if not checkpoint.is_file():
        raise ValueError(f"Complete saved source model at step {args.source_step} is required")
    audit = read_json(args.weight_audit)
    if (audit.get("status") != "passed" or audit.get("all_stored_tensors_exactly_equal") is not True
            or audit.get("strict_load_all_three_components") is not True
            or Path(audit["source_checkpoint"]).resolve() != checkpoint.resolve()
            or shared.sha256(checkpoint) != audit["source_sha256"]):
        raise ValueError("Require real, successful strict three-component tensor equality evidence")
    timestamp = datetime.now(timezone.utc).strftime("%y%m%d%H%M%S")
    values = overrides(source_run, args.dataset_root, args.dino_root, output,
                       max_steps=args.max_steps, warmup=args.warmup_steps, timestamp=timestamp,
                       source_step=args.source_step, checkpoint_steps=args.checkpoint_steps,
                       validation_steps=args.validation_steps, early_checkpoint_steps=args.early_checkpoint_steps)
    original = (source_run / "argv.txt").read_text().split()
    argv = shared.rewrite_argv(original, values)
    output.mkdir(parents=True)
    (output / "argv.txt").write_text("\n".join(argv) + "\n")
    config = shared.parse_training_argv(output / "argv.txt")
    check_config(config, source)
    shared.write_json(output / "run_config.json", config.model_dump(mode="json"))
    identities = {str(path): shared.sha256(path) for path in
                  (source_run / "run_config.json", source_run / "argv.txt", source_run / "dataset_statistics.json")}
    state_contract = Path(source["data"]["root_dir"]) / source["data"]["train_repo_ids"][0] / "meta/info.json"
    identities[str(state_contract)] = shared.sha256(state_contract)
    source_dino_manifest = Path(source["data"]["dino_cache_root"]) / source["data"]["train_repo_ids"][0] / "manifest.json"
    identities[str(source_dino_manifest)] = shared.sha256(source_dino_manifest)
    plan = {"status": "prepared_pending_data_and_gpu_check", "created_utc": now(),
            "source_run": str(source_run), "source_step": args.source_step,
            "source_checkpoint": str(checkpoint), "source_checkpoint_bytes": checkpoint.stat().st_size,
            "source_checkpoint_mtime_ns": checkpoint.stat().st_mtime_ns,
            "source_checkpoint_sha256": audit["source_sha256"],
            "weight_audit": str(args.weight_audit.resolve(strict=True)),
            "weight_audit_sha256": shared.sha256(args.weight_audit),
            "source_metadata_sha256": identities, "config_sha256": shared.sha256(output / "run_config.json"),
            "argv_sha256": shared.sha256(output / "argv.txt"), "optimizer_scheduler_rng_restored": False,
            "gpu_count": 8, "per_gpu_batch": 4, "gradient_accumulation_steps": 1,
            "effective_global_batch": 32, "max_optimizer_steps": args.max_steps,
            "warmup_steps": args.warmup_steps, "peak_learning_rates": PEAK_LRS,
            "checkpointing_steps": args.checkpoint_steps,
            "early_checkpoint_steps": args.early_checkpoint_steps,
            "validation_steps": args.validation_steps,
            "train_episodes": 45, "val_episodes": 5, "test_used_for_training": False,
            "normalization": shared.normalization_receipt(config.data.transform.field),
            "dataset_root": str(args.dataset_root), "dino_root": str(args.dino_root),
            "state_contract": str(state_contract),
            "source_dino_manifest": str(source_dino_manifest),
            "timestamp": timestamp, "formal_physical_success": "not_measured"}
    shared.write_json(output / "preparation.json", plan)
    command = [str(args.python), str(Path(__file__).resolve()), "run", "--prepared", str(output)]
    (output / "launch.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\nexec " + shlex.join(command) + "\n")
    print(json.dumps(plan, ensure_ascii=False, indent=2))


def load_prepared(root):
    root = Path(root).resolve(strict=True)
    plan = read_json(root / "preparation.json")
    if (shared.sha256(root / "run_config.json") != plan["config_sha256"]
            or shared.sha256(root / "argv.txt") != plan["argv_sha256"]):
        raise ValueError("Prepared configuration changed")
    for path, expected in plan["source_metadata_sha256"].items():
        if shared.sha256(path) != expected:
            raise ValueError("Protected source metadata changed")
    source_stat = Path(plan["source_checkpoint"]).stat()
    if ((source_stat.st_size, source_stat.st_mtime_ns)
            != (plan["source_checkpoint_bytes"], plan["source_checkpoint_mtime_ns"])
            or shared.sha256(plan["weight_audit"]) != plan["weight_audit_sha256"]):
        raise ValueError("Source model or accepted loading evidence changed")
    config = shared.parse_training_argv(root / "argv.txt")
    if config.model_dump(mode="json") != read_json(root / "run_config.json"):
        raise ValueError("Launch argv and configuration disagree")
    check_config(config, read_json(Path(plan["source_run"]) / "run_config.json"))
    if shared.normalization_receipt(config.data.transform.field) != plan["normalization"]:
        raise ValueError("Inherited source normalization changed")
    return config, plan


def select_windows(dataset, episodes):
    candidates, columns, history = shared.eligible_dataset_rows(dataset, episodes, source_frame_rule="ceil")
    rows = [int(i) for values in candidates.values() for i in values]
    return rows, candidates


def fixed_validation(candidates, per_episode=8):
    import numpy as np
    result = []
    for episode, values in sorted(candidates.items()):
        if len(values) < per_episode:
            raise ValueError(f"Validation episode {episode} has too few complete windows")
        result.extend(values[int(i)] for i in np.linspace(0, len(values) - 1, per_episode, dtype=np.int64))
    return result


def datasets(config, processor):
    import torch
    transform_kwargs = {"vlm_processor": processor}
    train = config.data(split="train", transform_kwargs=transform_kwargs)
    val = config.data(split="val", transform_kwargs=transform_kwargs)
    train_rows, train_candidates = select_windows(train, 45)
    _, val_candidates = select_windows(val, 5)
    val_rows = fixed_validation(val_candidates)
    return (torch.utils.data.Subset(train, train_rows), torch.utils.data.Subset(val, val_rows),
            {"train_windows": len(train_rows), "validation_windows": len(val_rows),
             "train_windows_per_episode": {str(k): len(v) for k, v in train_candidates.items()},
             "validation_rows": list(map(int, val_rows)),
             "policy": "real past one second and complete real future30; no tail padding or repeated history"})


def check(root):
    from transformers import AutoProcessor
    config, plan = load_prepared(root)
    dataset_root = Path(plan["dataset_root"])
    ready = read_json(dataset_root / "ADAPTATION_READY.json")
    # The adapter's accepted manifest remains the authoritative release receipt;
    # model input reads additionally validate every DINO/source metadata binding.
    if (ready.get("contract") != "cvpr-handover-task50-30hz-selection-v1"
            or ready.get("status") != "cpu_30hz_adaptation_complete_dino_and_joint_preflight_pending"):
        raise ValueError("30Hz task adaptation is not complete")
    artifact_bindings = {}
    payloads = ready.get("payload_sha256", {})
    if not payloads:
        raise ValueError("30Hz adaptation lacks sealed payloads")
    for relative, expected in payloads.items():
        path = (dataset_root / relative).resolve(strict=True)
        if not path.is_relative_to(dataset_root) or shared.sha256(path) != expected:
            raise ValueError(f"30Hz payload changed: {relative}")
        artifact_bindings[str(path)] = {"bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
    original_info = read_json(plan["state_contract"])
    original_encoder = read_json(plan["source_dino_manifest"])["encoder"]
    target_state = original_info["features"]["observation.state"]
    target_hands = original_info["features"]["action"]["names"][:14]
    for split in (TRAIN_ID, VAL_ID):
        info = read_json(dataset_root / split / "meta/info.json")
        if info.get("fps") != 30 or info.get("total_episodes") != (45 if split == TRAIN_ID else 5):
            raise ValueError("Preserve the accepted 45/5 split at explicit 30Hz")
        if (info["features"]["observation.state"]["names"] != target_state["names"]
                or info["features"]["observation.state"]["shape"] != [43]
                or info["features"]["action"]["names"] != target_hands):
            raise ValueError("Task joint names/order differ from the source state43/hand14 contract")
        manifest_path = Path(plan["dino_root"]) / split / "manifest.json"
        manifest = read_json(manifest_path)
        if manifest.get("state") != "complete":
            raise ValueError("Frozen DINO extraction is incomplete")
        if manifest["encoder"]["local_file_sha256"] != original_encoder["local_file_sha256"]:
            raise ValueError("Task DINO weights differ from the source training encoder")
        for path in (manifest_path, *sorted(manifest_path.parent.glob("episodes/*/receipt.json"))):
            artifact_bindings[str(path)] = {"bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
    processor = AutoProcessor.from_pretrained(config.model.model_name_or_path, local_files_only=True)
    train, val, counts = datasets(config, processor)
    samples = [train[0], train[len(train) - 1], val[0], val[len(val) - 1]]
    for sample in samples:
        import numpy as np
        if np.asarray(sample["motion_img_embs"]).shape != (6, 5, 1024):
            raise ValueError("Expected six genuine DINO history observations")
        if tuple(sample["actions"].shape) != (30, 80) or tuple(sample["states"].shape) != (1, 45):
            raise ValueError("Task sample differs from the source model dimensions")
    report = {"status": "cpu_data_check_passed", "checked_utc": now(), **counts,
              "effective_epoch_budget": config.train.max_training_steps * 32 / len(train),
              "adaptation_receipt_sha256": shared.sha256(dataset_root / "ADAPTATION_READY.json"),
              "artifact_bindings": artifact_bindings,
              "bounds_coverage": bounds_coverage(config),
              "scope": "actual reader, history, complete future, normalization, four decoded samples; not GPU acceptance"}
    shared.write_json(Path(root) / "data_check.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def bounds_coverage(config):
    """Report clipping under inherited40k bounds, never refitting on held-outs."""
    import numpy as np
    import pyarrow.parquet as pq
    report = {}
    field = config.data.transform.field
    for split in (TRAIN_ID, VAL_ID):
        states, actions = [], []
        for path in sorted((Path(config.data.root_dir) / split / "data").glob("**/*.parquet")):
            table = pq.read_table(path, columns=["observation.state", "action.body_token_v1_1", "action"])
            states.append(np.asarray(table["observation.state"].to_pylist(), dtype=np.float64))
            actions.append(np.concatenate([np.asarray(table[key].to_pylist(), dtype=np.float64)
                                           for key in ("action.body_token_v1_1", "action")], axis=1))
        values = {"state43": np.concatenate(states), "action78": np.concatenate(actions)}
        report[split] = {}
        for key, array in values.items():
            low, high = ((field.state_min, field.state_max) if key == "state43"
                         else (field.action_min, field.action_max))
            low, high = np.asarray(low)[:array.shape[1]], np.asarray(high)[:array.shape[1]]
            if not np.isfinite(array).all() or low.shape != (array.shape[1],):
                raise ValueError("Task values or inherited bounds are invalid")
            outside = (array < low) | (array > high)
            report[split][key] = {"elements": array.size, "outside_elements": int(outside.sum()),
                "outside_fraction": float(outside.mean()),
                "maximum_overflow": float(np.maximum(np.maximum(low - array, array - high), 0).max())}
    return report


def verify_data_check(root, plan, *, require_launch_gate=True):
    root = Path(root)
    report = read_json(root / "data_check.json")
    ready = Path(plan["dataset_root"]) / "ADAPTATION_READY.json"
    if (report.get("status") != "cpu_data_check_passed"
            or shared.sha256(ready) != report["adaptation_receipt_sha256"]):
        raise ValueError("Accepted task adaptation changed")
    for path, identity in report["artifact_bindings"].items():
        stat = Path(path).stat()
        if (stat.st_size, stat.st_mtime_ns) != (identity["bytes"], identity["mtime_ns"]):
            raise ValueError(f"CPU-accepted artifact changed: {path}")
    if require_launch_gate:
        gate = read_json(root / "launch_gate.json")
        if gate["data_check_sha256"] != shared.sha256(root / "data_check.json"):
            raise ValueError("CPU data acceptance changed after launch")
    return report


def train(root):
    import torch
    from psi.trainers.posttrain import PosttrainTrainer
    config, plan = load_prepared(root)
    if not (Path(root) / "data_check.json").is_file():
        raise ValueError("Run the actual CPU data check before launching workers")
    verify_data_check(root, plan)
    if int(os.environ.get("WORLD_SIZE", "0")) != 8:
        raise ValueError("This configuration requires exactly eight GPU ranks")

    class TaskTrainer(PosttrainTrainer):
        def create_datasets(self):
            self.train_dataset, self.val_dataset, counts = datasets(self.cfg, self.vlm_processor)
            if int(os.environ.get("RANK", "0")) == 0:
                shared.write_json(Path(self.project_dir) / "task_window_selection.json", counts)
            return self.train_dataset, self.val_dataset

        def create_optimizers(self):
            super().create_optimizers()
            groups = {g["group_name"]: float(g["lr"]) for g in self.optimizer.param_groups}
            if groups != PEAK_LRS or len(self.optimizer.state):
                raise ValueError("Require all five fresh optimizer groups and the agreed peak learning rates")

        def prepare(self, accelerator):
            loader = super().prepare(accelerator)
            if accelerator.num_processes != 8 or self.resume_from_checkpoint()[0] != 0:
                raise ValueError("New task must start at step zero on eight GPUs")
            if accelerator.is_main_process:
                shared.write_json(Path(self.project_dir) / "task_initialization.json", {
                    "source_checkpoint": plan["source_checkpoint"], "source_step": plan["source_step"],
                    "strict_joint_loader": True, "fresh_optimizer_scheduler_rng": True,
                    "global_batch": 32, "five_peak_learning_rates": PEAK_LRS,
                    "scope": "strict production load; independent CPU tensor equality is recorded separately"})
            return loader

        def step(self, batch_input, global_step, local_step):
            synced, metrics = super().step(batch_input, global_step, local_step)
            completed = global_step + 1
            if synced and completed in (2, 20):
                # Small explicit cross-rank probes supplement DDP/finite guards.
                # This checks four values from every trainable tensor, not all
                # bytes of all parameters; record that scope honestly.
                model = self.unwrap_model()
                probe = torch.cat([p.detach().reshape(-1)[:4].float() for p in model.parameters()
                                   if p.requires_grad])
                gathered = self.accelerator.gather(probe).reshape(8, -1)
                if not torch.isfinite(gathered).all() or not torch.equal(gathered, gathered[0].expand_as(gathered)):
                    raise RuntimeError("Cross-rank parameter probes disagree")
                if self.accelerator.is_main_process:
                    shared.write_json(Path(self.project_dir) / f"gpu_check_update{completed}.json", {
                        "status": "passed", "completed_updates": completed,
                        "gpu_count": 8, "per_gpu_batch": 4, "accumulation": 1,
                        "all_rank_probe_values_equal": True, "probe_values_per_rank": probe.numel(),
                        "metrics": {k: float(v) for k, v in metrics.items()},
                        "scope": "production all-rank finite loss/gradient checks and four-value probes per trainable tensor; not full parameter equality",
                    })
            return synced, metrics

        def create_dataloaders(self, train_dataset, val_dataset):
            from torch.utils.data import DataLoader
            super().create_dataloaders(train_dataset, val_dataset)
            # Forty fixed windows, eight per held-out episode, divide exactly
            # across the eight Accelerator ranks without duplicate padding.
            self.val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                num_workers=0, collate_fn=self.train_dataloader.collate_fn)
            return self.train_dataloader, self.val_dataloader

        def evaluate(self):
            # Keep evaluation noise comparable without consuming training RNG.
            with shared.isolated_rng(self.cfg.seed + 100000 + self.accelerator.process_index, self.device):
                return super().evaluate()

    class Factory:
        @staticmethod
        def instantiate(cfg, device_id):
            return TaskTrainer(cfg, torch.device(f"cuda:{device_id}"))

    spec = importlib.util.spec_from_file_location("task_finetune_production", REPO / "VLMExpert/scripts/train.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    entry.Trainer = Factory
    sys.argv = (Path(root) / "argv.txt").read_text().split()
    entry.train(config)


def run(root):
    import fcntl
    root = Path(root).resolve(strict=True)
    with (root / "launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "training_state.json").exists():
            raise ValueError("A run was already launched here; preserve it and prepare a new directory")
        config, plan = load_prepared(root)
        if (root / "data_check.json").exists():
            # Rebind accepted bytes; do not decode or numerically revalidate
            # already accepted histories, future windows, or DINO features.
            verify_data_check(root, plan, require_launch_gate=False)
        else:
            check(root)
        if shared.sha256(plan["source_checkpoint"]) != plan["source_checkpoint_sha256"]:
            raise ValueError("Source model bytes changed before launch")
        shared.write_json(root / "launch_gate.json", {"data_check_sha256": shared.sha256(root / "data_check.json"),
                          "config_sha256": plan["config_sha256"], "verified_utc": now()})
        # GPU ownership is checked immediately before launch. Never evict another
        # job to satisfy this recipe, even if the launcher was prepared earlier.
        occupied = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
        if occupied.strip():
            raise RuntimeError("GPUs are occupied; no process was stopped")
        env = dict(os.environ, WANDB_MODE="offline", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   PYTHONNOUSERSITE="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                   CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7", NCCL_IB_DISABLE="1",
                   NCCL_P2P_DISABLE="0", NCCL_SHM_DISABLE="0")
        env["PYTHONPATH"] = os.pathsep.join(map(str, (REPO / "VLMExpert/src", REPO, REPO / "VLMExpert")))
        state = {"state": "starting", "started_utc": now(), "timestamp": plan["timestamp"],
                 "run_directory": str(root / "training/posttrain"), "log": str(root / "train.log"),
                 "max_optimizer_steps": plan["max_optimizer_steps"], "warmup_steps": plan["warmup_steps"],
                 "source_step": plan["source_step"], "preparation": str(root / "preparation.json")}
        # The production name includes global world size. Obtain it using the
        # saved configuration under a short-lived eight-rank naming environment.
        naming = dict(env, WORLD_SIZE="8", RANK="0", LOCAL_RANK="0")
        command = [sys.executable, str(Path(__file__).resolve()), "name", "--prepared", str(root)]
        naming_result = subprocess.run(command, env=naming, cwd=REPO / "VLMExpert", text=True,
                                       capture_output=True, check=False)
        (root / "run_name.log").write_text(naming_result.stdout + naming_result.stderr)
        if naming_result.returncode:
            raise RuntimeError("Task run naming failed; see run_name.log: " +
                               (naming_result.stdout + naming_result.stderr)[-2000:])
        named = naming_result.stdout
        state["run_directory"] = named.strip().splitlines()[-1]
        state_path = root / "training_state.json"
        with (root / "train.log").open("x") as log:
            process = subprocess.Popen([sys.executable, "-m", "torch.distributed.run", "--standalone",
                "--nproc_per_node=8", str(Path(__file__).resolve()), "train", "--prepared", str(root)],
                cwd=REPO / "VLMExpert", env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            state.update(pid=process.pid, state="running")
            shared.write_json(state_path, state)
            monitor_env = dict(env, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            monitor_log = (root / "monitor.log").open("a")
            monitor = subprocess.Popen([sys.executable, str(REPO / "VLMExpert/scripts/analysis/monitor_task_finetune.py"),
                "--state", str(state_path), "--output-dir", str(root / "progress")], env=monitor_env,
                stdin=subprocess.DEVNULL, stdout=monitor_log, stderr=subprocess.STDOUT)
            try:
                returncode = process.wait()
                state.update(state="complete" if returncode == 0 else "failed", returncode=returncode, finished_utc=now())
                shared.write_json(state_path, state)
                try:
                    monitor_returncode = monitor.wait(timeout=45)
                    state["monitor_returncode"] = monitor_returncode
                except subprocess.TimeoutExpired:
                    state["monitor_status"] = "final_refresh_pending"
                shared.write_json(state_path, state)
            finally:
                monitor_log.close()
            return returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("prepare")
    for name in ("source-run", "dataset-root", "dino-root", "output", "python", "weight-audit"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--source-step", type=int, default=40000)
    p.add_argument("--checkpoint-steps", type=int, default=200)
    p.add_argument("--validation-steps", type=int, default=200)
    p.add_argument("--early-checkpoint-steps", type=int, nargs="*", default=[20])
    for mode in ("check", "train", "run", "name"):
        sub.add_parser(mode).add_argument("--prepared", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args)
    elif args.mode == "check":
        check(args.prepared)
    elif args.mode == "train":
        train(args.prepared)
    elif args.mode == "run":
        return run(args.prepared)
    else:
        from psi.trainers.posttrain import PosttrainTrainer
        config, _ = load_prepared(args.prepared)
        print(PosttrainTrainer(config, "cpu").project_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
