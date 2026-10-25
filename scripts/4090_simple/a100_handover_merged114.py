"""Fresh eight-A100 adaptation on existing new14 + original100 demonstrations.

Reuse the accepted frozen model/training/validation implementation by explicit
path and SHA256. Preserve every 1000-update model and only the two newest full
recovery states, without touching prior datasets, runs, or checkpoints.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile

CONTRACT = "cvpr-handover-merged114-a100-fresh-v1"
PACK_CONTRACT = "cvpr-handover-original100-plus-new14-lerobot-pack-v1"
UPSTREAM_SHA = "57c7fad25ef18c40849a397d54b4f9e158b19b47205f80ab8da7844d26876923"
MIGRATION_SHA = "95bbb9f4414c7a133e03f253c2219ed9683406e9569bd1084961a6c17f6d8e5a"
SOURCE_SHA = "12a71e046e221b83f0bdb0efb50f9ba82442045aca0efabef3ae337f185320b7"
PEAK_LRS = {"action_header": 1e-5, "lang_backbone": 1e-7, "vision_tower": 1e-6,
            "mm_projector": 1e-5, "motion_expert": 1e-6}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix="."+path.name,
                                     suffix=".partial", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def now():
    return datetime.now(timezone.utc).isoformat()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def external(plan):
    for key in ("upstream_runner", "migration_helper"):
        if sha(plan[key]) != plan[key+"_sha256"]:
            raise ValueError(f"Accepted external helper changed: {key}")
    code = Path(plan["frozen_code"]).resolve(strict=True)
    locations = [str(p) for p in [code/"scripts/4090_simple", code/"VLMExpert/src", code, code/"VLMExpert"]]
    # PYTHONPATH already includes these in worker processes. Move them ahead of
    # this new launcher's directory unconditionally, rather than skipping them.
    sys.path[:] = locations+[p for p in sys.path if p not in locations]
    upstream = module("merged114_accepted_training", plan["upstream_runner"])
    if Path(upstream.CODE).resolve() != code:
        raise ValueError("The supplied frozen code root differs from the audited helper")
    if Path(upstream.shared.__file__).resolve() != code/"scripts/4090_simple/finetune_carrybox.py":
        raise ValueError("The external training helper imported a different data/config implementation")
    return upstream


def sealed_code(code):
    paths = [code/"VLMExpert/scripts/train.py"]
    for component in ("VLMExpert/src/psi", "MotionExpert", "ImageProcess", "scripts/4090_simple"):
        paths.extend(p for p in (code/component).rglob("*.py") if "__pycache__" not in p.parts)
    return {str(p): sha(p) for p in sorted(set(paths))}


def metadata_paths(release, val, val_dino):
    remote = read(release/"A100_DATASET_READY.json")
    return [release/"MERGED114_DATASET_READY.json", release/"A100_DATASET_READY.json",
            release/remote["production_report"], release/"packed/PACK_READY.json",
            release/"packed/train/meta/info.json", release/"packed/train/meta/episodes.jsonl",
            release/"packed/source_dataset_statistics.json", release/"dino/manifest.json",
            release/"dino/MERGED114_REINDEX.json", val/"meta/info.json",
            val/"meta/episodes.jsonl", val_dino/"manifest.json"]


def validate_remote_ready(release):
    ready = read(release/"A100_DATASET_READY.json")
    if (ready.get("contract") != "cvpr-handover-merged114-a100-ready-v1"
            or ready.get("status") != "ready_for_training"
            or ready.get("train_episodes") != 114 or ready.get("total_frames") != 49089
            or ready.get("train_windows") != 42363):
        raise ValueError("Actual A100 dataset/cache acceptance is incomplete")
    report = ready["production_report"]
    required = {"packed/PACK_READY.json", "dino/manifest.json", "dino/MERGED114_REINDEX.json", report}
    bindings = ready.get("payload_sha256", {})
    if not required.issubset(bindings):
        raise ValueError("Actual A100 readiness lacks pack/cache/reader bindings")
    for relative, expected in bindings.items():
        path = (release/relative).resolve(strict=True)
        if not path.is_relative_to(release) or sha(path) != expected:
            raise ValueError(f"Actual A100 accepted input changed: {relative}")
    if read(release/report).get("status") != "passed":
        raise ValueError("Actual remote production dataset reading has not passed")
    return ready


def validate_config(config, plan):
    m, t = config.model, config.train
    if (plan.get("fresh_run") is not True or plan.get("initial_global_step") != 0
            or plan.get("optimizer_scheduler_rng_restored") is not False
            or plan.get("gpu_count") != 8 or plan.get("gpus") != list(range(8))
            or plan.get("effective_global_batch") != 64
            or (not plan["smoke"] and (plan["max_optimizer_steps"], plan["warmup_steps"]) != (10000, 1000))):
        raise ValueError("Require an independent fresh eight-rank10k optimizer-update run")
    if (m.action_architecture != "dual_source_v1" or m.num_blocks != 6
            or m.state_drop_prob != 0 or not m.state_null_token
            or not m.tune_motion or not m.tune_vlm
            or (m.action_dim, m.action_chunk_size, m.odim) != (80, 30, 45)
            or str(m.pretrained_joint_checkpoint) != plan["source_checkpoint"]):
        raise ValueError("Require the accepted complete joint38k state-retention architecture")
    expected_steps = plan["max_optimizer_steps"]
    if ((t.train_batch_size, t.gradient_accumulation_steps) != (8, 1)
            or t.resume_from_checkpoint is not None or t.skip_resumed_steps
            or t.max_training_steps != expected_steps
            or t.warmup_steps != plan["warmup_steps"]
            or t.checkpointing_steps != (expected_steps if plan["smoke"] else 1000)
            or t.preserve_checkpoint_every != (expected_steps if plan["smoke"] else 1000)
            or list(t.early_checkpoint_steps) != []
            or t.mixed_precision != "bf16" or t.data_parallel != "ddp"
            or not t.training_safety_checks):
        raise ValueError("Require eight ranks, BS8/accum1, fresh optimization, and permanent 1k models")
    image = config.data.transform.model
    if image.shared_image_profile != "256x320" or image.motion_image_policy != "vlm_color_clean_motion":
        raise ValueError("Preserve actual clean DINO histories and VLM-only color augmentation")


def prepare(args):
    output = args.output.absolute()
    if output.exists():
        raise FileExistsError("Use a fresh prepared/output directory")
    release, parent, code, val, val_dino = (p.resolve(strict=True) for p in
        (args.release, args.parent_prepared, args.frozen_code, args.val, args.val_dino))
    validate_remote_ready(release)
    for path, expected in ((args.upstream_runner, args.upstream_sha256),
                           (args.migration_helper, args.migration_sha256)):
        if sha(path) != expected:
            raise ValueError(f"External source hash differs: {path}")
    ready, pack = read(release/"MERGED114_DATASET_READY.json"), read(release/"packed/PACK_READY.json")
    if (pack.get("contract") != PACK_CONTRACT or pack.get("train_episodes") != 114
            or pack.get("total_frames") != 49089 or pack.get("train_windows") != 42363):
        raise ValueError("Require the complete original100 + new14 package")
    if ready.get("status") != "ready_for_training":
        raise ValueError("Merged114 production input acceptance is not complete")
    prior = read(parent/"preparation.json")
    source = Path(prior["source_checkpoint"]).resolve(strict=True)
    if prior["source_step"] != 38000 or prior["source_checkpoint_sha256"] != SOURCE_SHA or sha(source) != SOURCE_SHA:
        raise ValueError("Fresh training requires the exact accepted joint38k weight source")
    train_meta = [json.loads(line) for line in (release/"packed/train/meta/episodes.jsonl").read_text().splitlines()]
    val_meta = [json.loads(line) for line in (val/"meta/episodes.jsonl").read_text().splitlines()]
    if (len(train_meta) != 114 or len(val_meta) != 5
            or {int(e["source_episode_index"]) for e in train_meta} &
               {int(e["source_episode_index"]) for e in val_meta}):
        raise ValueError("Preserve the existing independent five-episode VAL; never train on it")
    steps = args.smoke_updates or 10000
    if steps < 2 or steps > 10000:
        raise ValueError("Smoke checks need 2..10000 updates")
    source_stat = source.stat()
    plan = {"contract": CONTRACT, "kind": "experiment", "fresh_run": True, "initial_global_step": 0,
            "optimizer_scheduler_rng_restored": False, "created_utc": now(),
            "release": str(release), "parent_prepared": str(parent), "frozen_code": str(code),
            "upstream_runner": str(args.upstream_runner.resolve()), "upstream_runner_sha256": args.upstream_sha256,
            "migration_helper": str(args.migration_helper.resolve()), "migration_helper_sha256": args.migration_sha256,
            "source_checkpoint": str(source), "source_checkpoint_sha256": SOURCE_SHA, "source_step": 38000,
            "source_identity": {"bytes": source_stat.st_size, "mtime_ns": source_stat.st_mtime_ns},
            "gpus": list(range(8)), "gpu_count": 8, "per_gpu_batch": 8,
            "gradient_accumulation_steps": 1, "effective_global_batch": 64,
            "max_optimizer_steps": steps, "smoke": bool(args.smoke_updates),
            "warmup_steps": min(1000, max(1, steps//10)), "peak_learning_rates": PEAK_LRS,
            "checkpoint_policy": {"all_1000_update_model_weights": True, "full_recovery_states_to_keep": 2,
                "cleanup_scope": "Only older optimizer/scheduler/RNG files created in this new run; all model weights retained"},
            "val": str(val), "val_dino": str(val_dino), "train_episodes": 114,
            "train_windows": 42363, "validation_windows": 40,
            "execution_environment": {"CUDA_LAUNCH_BLOCKING": "1"},
            "runner_sha256": sha(__file__), "code_sha256": sealed_code(code)}
    up = external(plan)
    output.mkdir(parents=True)
    data = output/"data"
    for name, target in {"train": release/"packed/train", "val": val,
                         "dino/train": release/"dino", "dino/val": val_dino}.items():
        link = data/name
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target, target_is_directory=True)
    timestamp = datetime.now(timezone.utc).strftime("%y%m%d%H%M%S")
    plan["timestamp"] = timestamp
    interval = steps if plan["smoke"] else 1000
    changes = {"exp": "handover_merged114_joint38000_statekeep_8gpu", "timestamp": timestamp,
        "seed": 20261025, "train.output_dir": str(output/"training"),
        "train.resume_from_checkpoint": None, "train.skip_resumed_steps": False,
        "train.train_batch_size": 8, "train.gradient_accumulation_steps": 1,
        "train.max_training_steps": steps, "train.warmup_steps": plan["warmup_steps"],
        "train.checkpointing_steps": interval, "train.early_checkpoint_steps": [],
        "train.max_checkpoints_to_keep": 100, "train.preserve_checkpoint_every": interval,
        "train.validation_steps": interval, "data.root_dir": str(data),
        "data.train_repo_ids": ["train"], "data.val_repo_ids": ["val"],
        "data.dino_cache_root": str(data/"dino"),
        "data.transform.field.stat_path": str(release/"packed/source_dataset_statistics.json"),
        "wandb.project": "cvpr-handover-merged114", "wandb.id": None, "wandb.resume": "never",
        "wandb.name": None, "wandb.group": None, "wandb.entity": None}
    argv = up.shared.rewrite_argv((parent/"argv.txt").read_text().split(), changes)
    (output/"argv.txt").write_text("\n".join(argv)+"\n")
    config = up.shared.parse_training_argv(output/"argv.txt")
    validate_config(config, plan)
    # Inherit every model detail and optimizer LR from the latest accepted experiment.
    if config.model.model_dump(mode="json") != read(parent/"run_config.json")["model"]:
        raise ValueError("Fresh114 may not silently change the accepted fusion/state model")
    write(output/"run_config.json", config.model_dump(mode="json"))
    plan["bindings_sha256"] = {str(p): sha(p) for p in metadata_paths(release, val, val_dino)+
        [parent/"run_config.json", parent/"argv.txt", parent/"preparation.json"]}
    plan.update(config_sha256=sha(output/"run_config.json"), argv_sha256=sha(output/"argv.txt"))
    write(output/"preparation.json", plan)
    command = [sys.executable, str(Path(__file__).resolve()), "run", "--prepared", str(output)]
    (output/"launch.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\nexec "+shlex.join(command)+"\n")
    print(json.dumps({"prepared": str(output), "optimizer_updates": steps, "global_batch": 64,
                      "models_saved_every": interval, "full_recovery_states_kept": 2}))


def load(root):
    root = Path(root).resolve(strict=True)
    plan = read(root/"preparation.json")
    if plan.get("contract") != CONTRACT or plan["gpu_count"] != 8 or plan["effective_global_batch"] != 64:
        raise ValueError("Wrong eight-GPU merged114 training contract")
    if sha(__file__) != plan["runner_sha256"]:
        raise ValueError("Launcher changed after preparation")
    for path, expected in {**plan["code_sha256"], **plan["bindings_sha256"]}.items():
        if sha(path) != expected:
            raise ValueError(f"Bound code/data/normalization/validation changed: {path}")
    source_stat = Path(plan["source_checkpoint"]).stat()
    if {"bytes": source_stat.st_size, "mtime_ns": source_stat.st_mtime_ns} != plan["source_identity"]:
        raise ValueError("Source checkpoint changed")
    if sha(root/"argv.txt") != plan["argv_sha256"] or sha(root/"run_config.json") != plan["config_sha256"]:
        raise ValueError("Prepared configuration changed")
    up = external(plan)
    config = up.shared.parse_training_argv(root/"argv.txt")
    if config.model_dump(mode="json") != read(root/"run_config.json"):
        raise ValueError("Parsed argv differs from sealed configuration")
    validate_config(config, plan)
    return config, plan, up


def map_accepted_rows(history, episodes, expected_windows):
    if history is None or len(history.pack_info) != 1 or len(history.source_episode_ranges[0]) != len(episodes):
        raise ValueError("Require one real-history source pack with all114 episodes")
    eligible, counts = set(), {}
    for e in episodes:
        i = int(e["episode_index"])
        start, stop = history.source_episode_ranges[0][i]
        accepted = list(map(int, e["eligible_current_rows"]))
        if (e["split"] != "train" or stop-start != e["length"] or len(set(accepted)) != len(accepted)
                or not accepted or any(row < 30 or row+29 >= e["length"] for row in accepted)):
            raise ValueError("Invalid complete history/future admission window")
        eligible.update(start+row for row in accepted)
        counts[str(i)] = len(accepted)
    rows = [i for i, original in enumerate(history.original_indices) if int(original) in eligible]
    if len(rows) != expected_windows or len(eligible) != expected_windows:
        raise ValueError("Production history filter lost or duplicated admitted windows")
    return rows, counts


def datasets(config, processor, plan, shared):
    import numpy as np
    import torch
    train = config.data(split="train", transform_kwargs={"vlm_processor": processor})
    val = config.data(split="val", transform_kwargs={"vlm_processor": processor})
    episodes = [json.loads(line) for line in
        (Path(plan["release"])/"packed/train/meta/episodes.jsonl").read_text().splitlines()]
    rows, per_episode = map_accepted_rows(train.raw_dataset.motion_history, episodes, 42363)
    complete, _, _ = shared.eligible_dataset_rows(train, expected_episodes=114, source_frame_rule="ceil")
    if set(rows) != {int(row) for values in complete.values() for row in values}:
        raise ValueError("Admitted rows disagree with real source clock and complete future30")
    candidates, _, _ = shared.eligible_dataset_rows(val, expected_episodes=5, source_frame_rule="ceil")
    val_rows = []
    for _, values in sorted(candidates.items()):
        if len(values) < 8:
            raise ValueError("Incomplete independent validation episode")
        val_rows.extend(values[int(i)] for i in np.linspace(0, len(values)-1, 8, dtype=np.int64))
    counts = {"train_windows": len(rows), "validation_windows": len(val_rows),
              "train_windows_per_episode": per_episode, "validation_rows": list(map(int, val_rows)),
              "train_episode_indices": list(range(114)),
              "policy": "All new14 + original100 TRAIN, full actual history/future; old independent fixed40 VAL"}
    if len(val_rows) != 40:
        raise ValueError("Require the exact five-episode40-window validation selection")
    return torch.utils.data.Subset(train, rows), torch.utils.data.Subset(val, val_rows), counts


def check(root):
    import numpy as np
    from transformers import AutoProcessor
    config, plan, up = load(root)
    processor = AutoProcessor.from_pretrained(config.model.model_name_or_path, local_files_only=True)
    train, val, counts = datasets(config, processor, plan, up.shared)
    checked = []
    for split, data in (("train", train), ("val", val)):
        for i in (0, len(data)//2, len(data)-1):
            item = data[i]
            actions, mask = np.asarray(item["actions"]), np.asarray(item["actions_mask"])
            if (actions.shape != (30, 80) or mask.shape != (30, 80) or not np.isfinite(actions).all()
                    or not mask[:, :78].all() or mask[:, 78:].any()
                    or tuple(item["motion_img_embs"].shape) != (6, 5, 1024)):
                raise ValueError("Actual production action/mask/history sample invalid")
            checked.append({"split": split, "index": i, "actions_shape": [30, 80], "motion_shape": [6, 5, 1024]})
    write(Path(root)/"data_check.json", {"status": "passed", "checked_utc": now(),
        "config_sha256": plan["config_sha256"], "counts": counts, "samples": checked})
    print(json.dumps({"status": "passed", "train_windows": len(train), "validation_windows": len(val)}))


def retain_model_weights(checkpoint_root, keep_full=2):
    """Only discard old recovery payloads under this new run; keep all models."""
    root = Path(checkpoint_root).resolve(strict=True)
    if type(keep_full) is not int or keep_full < 1:
        raise ValueError("Keep at least one complete recovery state")
    checkpoints = sorted((p for p in root.iterdir() if re.fullmatch(r"ckpt_\d+", p.name)),
                         key=lambda p: int(p.name[5:]))
    removed = []
    for checkpoint in checkpoints[:-keep_full]:
        if (checkpoint.is_symlink() or not checkpoint.is_dir()
                or (checkpoint/"model.safetensors").is_symlink()
                or not (checkpoint/"model.safetensors").is_file()):
            raise ValueError("Refuse checkpoint cleanup without an independent saved model")
        for payload in checkpoint.iterdir():
            if payload.name in ("optimizer.bin", "scheduler.bin") or re.fullmatch(r"random_states_\d+\.pkl", payload.name):
                if payload.is_symlink() or not payload.is_file():
                    raise ValueError("Recovery cleanup only accepts ordinary files in the new run")
                removed.append({"path": str(payload), "bytes": payload.stat().st_size})
                payload.unlink()
    return {"all_saved_model_steps": [int(p.name[5:]) for p in checkpoints],
            "full_recovery_steps": [int(p.name[5:]) for p in checkpoints[-keep_full:]],
            "removed_old_recovery_payloads": removed, "all_models_preserved": True}


def trainer_class(plan, up):
    import torch
    up.datasets = lambda config, processor: datasets(config, processor, plan, up.shared)
    Base = up.trainer_class(plan)
    helper = module("merged114_state_dropout_migration", plan["migration_helper"])
    class Merged114Trainer(Base):
        def init_models(self):
            rank = int(os.environ.get("RANK", "0"))
            return helper.initialize_state_keep(self,
                strict_initializer=lambda: super(Merged114Trainer, self).init_models(),
                receipt_path=Path(self.project_dir)/f"state_policy_migration_rank_{rank}.json")

        def step(self, batch_input, global_step, local_step):
            model = self.unwrap_model()
            if (self.model_cfg.state_drop_prob != 0 or model.action_header.state_drop_prob != 0
                    or model.action_header.config.state_drop_prob != 0
                    or tuple(batch_input["states"].shape) != (8, 1, 45)):
                raise ValueError("Every rank must retain all eight measured proprioceptive states")
            synced, metrics = super().step(batch_input, global_step, local_step)
            if model.action_header.last_state_drop_frac != 0:
                raise RuntimeError("A training state was unexpectedly dropped")
            metrics["state_drop_frac"] = 0.
            return synced, metrics

        def save_checkpoint(self, global_step):
            saved = super().save_checkpoint(global_step)
            result = None
            if self.accelerator.is_main_process:
                try:
                    record = retain_model_weights(Path(self.project_dir)/"checkpoints", keep_full=2)
                    record.update(saved_update=global_step, checked_utc=now())
                    write(Path(self.project_dir)/"checkpoint_retention.json", record)
                    result = {"status": "passed"}
                except Exception as error:
                    result = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
            shared_result = [result]
            torch.distributed.broadcast_object_list(shared_result, src=0)
            if shared_result[0]["status"] != "passed":
                raise RuntimeError(shared_result[0]["error"])
            return saved
    return Merged114Trainer


def train(root):
    import torch
    config, plan, up = load(root)
    if int(os.environ.get("WORLD_SIZE", "0")) != 8:
        raise ValueError("Exactly eight ranks required")
    accepted = read(Path(root)/"data_check.json")
    if accepted.get("status") != "passed" or accepted["config_sha256"] != plan["config_sha256"]:
        raise ValueError("Run the real CPU production data check before training")
    cls = trainer_class(plan, up)
    class Factory:
        @staticmethod
        def instantiate(cfg, device_id):
            return cls(cfg, torch.device(f"cuda:{device_id}"))
    production = module("merged114_production_train", Path(plan["frozen_code"])/"VLMExpert/scripts/train.py")
    production.Trainer = Factory
    sys.argv = (Path(root)/"argv.txt").read_text().split()
    production.train(config)


def name(root):
    config, _, _ = load(root)
    from psi.trainers.posttrain import PosttrainTrainer
    print(PosttrainTrainer(config, "cpu").project_dir)


def run(root):
    root = Path(root).resolve(strict=True)
    with (root/"launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/"training_state.json").exists():
            raise FileExistsError("Already launched; do not overwrite or restart this run")
        _, plan, up = load(root)
        accepted = read(root/"data_check.json")
        if accepted.get("status") != "passed" or accepted["config_sha256"] != plan["config_sha256"]:
            raise ValueError("CPU production input acceptance missing")
        # Budget includes ten models, three optimizer states at save-before-prune peak,
        # eight RNG streams and20GiB margin. Prior jobs/files are never removed.
        minimum = 10*12066487572 + 3*24000000000 + 20*1024**3
        free = os.statvfs(root).f_bavail*os.statvfs(root).f_frsize
        if free < minimum:
            raise RuntimeError(f"Insufficient output space: need {minimum}, available {free} bytes")
        up.selected_gpu_idle(plan["gpus"])
        env = dict(os.environ, WANDB_MODE="offline", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
            PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
            CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7", NCCL_IB_DISABLE="1", NCCL_P2P_DISABLE="0",
            NCCL_SHM_DISABLE="0", CUDA_LAUNCH_BLOCKING="1")
        code = Path(plan["frozen_code"])
        env["PYTHONPATH"] = os.pathsep.join(map(str, [code/"scripts/4090_simple", code/"VLMExpert/src", code, code/"VLMExpert"]))
        diagnostic = Path("/dev/shm")/f"merged114-{plan['timestamp']}-{os.getpid()}"
        diagnostic.mkdir(mode=0o700)
        env["PSI0_ROLLING_DIAGNOSTICS_DIR"] = str(diagnostic)
        naming = dict(env, WORLD_SIZE="8", RANK="0", LOCAL_RANK="0")
        command = [sys.executable, str(Path(__file__).resolve())]
        result = subprocess.run(command+["name", "--prepared", str(root)], cwd=code/"VLMExpert",
                                env=naming, text=True, capture_output=True)
        (root/"run_name.log").write_text(result.stdout+result.stderr)
        if result.returncode:
            raise RuntimeError("Run naming failed; inspect run_name.log")
        state = {"state": "starting", "started_utc": now(), "run_directory": result.stdout.strip().splitlines()[-1],
                 "log": str(root/"train.log"), "preparation": str(root/"preparation.json"),
                 "max_optimizer_steps": plan["max_optimizer_steps"], "gpus": plan["gpus"],
                 "per_gpu_batch": 8, "gradient_accumulation_steps": 1, "effective_global_batch": 64,
                 "source_step": 38000, "fresh_run": True, "initial_global_step": 0,
                 "checkpoint_policy": plan["checkpoint_policy"], "output_free_bytes_at_launch": free}
        with (root/"train.log").open("x") as log:
            process = subprocess.Popen([sys.executable, "-m", "torch.distributed.run", "--standalone",
                "--nproc_per_node=8", str(Path(__file__).resolve()), "train", "--prepared", str(root)],
                cwd=code/"VLMExpert", env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            state.update(state="running", pid=process.pid)
            write(root/"training_state.json", state)
            status = process.wait()
            state.update(state="complete" if status == 0 else "failed", returncode=status, finished_utc=now())
            write(root/"training_state.json", state)
            return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("prepare")
    for key in ("release", "parent-prepared", "frozen-code", "upstream-runner", "migration-helper", "val", "val-dino", "output"):
        p.add_argument("--"+key, type=Path, required=True)
    p.add_argument("--upstream-sha256", default=UPSTREAM_SHA)
    p.add_argument("--migration-sha256", default=MIGRATION_SHA)
    p.add_argument("--smoke-updates", type=int, default=0)
    for mode in ("check", "name", "train", "run"):
        sub.add_parser(mode).add_argument("--prepared", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args)
    elif args.mode == "check":
        check(args.prepared)
    elif args.mode == "name":
        name(args.prepared)
    elif args.mode == "train":
        train(args.prepared)
    else:
        return run(args.prepared)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
