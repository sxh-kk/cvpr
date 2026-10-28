"""Fresh joint38000 -> Mobile95/EVAL5, 20k optimizer updates.

The original accepted100 archive remains immutable. A frozen whole-episode
split supplies disjoint views of that archive; previous Mobile100 weights must
never initialize this experiment. Prepare/check hide GPUs. Launch requires a
separate ordinary SSH context with full CPU affinity and nice=0.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import a100_mobile_pp100 as original

read, write, sha, require, now, module = (original.read, original.write, original.sha,
                                        original.require, original.now, original.module)
TASK_ID, RELEASE, SOURCE_SHA = original.TASK_ID, original.RELEASE, original.SOURCE_SHA
CONTRACT = "cvpr-mobile-pp95-eval5-a100-fresh20k-v1"
VALIDATION_ROLE = "heldout_mobile5_validation_not_independent_test"
SAVE_STEPS = tuple(range(4000, 20001, 500))
DIAGNOSTIC_STEPS = tuple(range(0, 20001, 1000))
RUNTIME_FILES = ("a100_mobile_pp95_eval5.py", "a100_mobile_pp100.py",
                 "mobile_pp_holdout_data.py", "mobile_pp_overfit_metrics.py")


def helper(plan, name):
    path = Path(plan["runtime_directory"]) / (name + ".py")
    require(sha(path) == plan["runtime_sha256"][path.name], "Prepared runtime helper changed")
    return module("mobile95_" + name, path)


def episodes(release):
    return [json.loads(line) for line in (Path(release) / "packed/train/meta/episodes.jsonl").read_text().splitlines()]


def config_overrides(output, release, timestamp):
    return {"exp": "mobile_pp95_eval5_joint38000_statekeep_8gpu", "timestamp": timestamp,
        "train.output_dir": str(output / "training"), "data.root_dir": str(output / "data"),
        "data.dino_cache_root": str(output / "data/dino"),
        "data.transform.field.stat_path": str(release / "packed/source_dataset_statistics.json"),
        "train.max_training_steps": 20000, "train.validation_steps": 0,
        "train.checkpointing_steps": 20000, "train.early_checkpoint_steps": list(SAVE_STEPS[:-1]),
        "train.preserve_checkpoint_every": 500}


def validate_config(value, plan, reference, root):
    config = value if isinstance(value, dict) else value.model_dump(mode="json")
    expected = copy.deepcopy(reference)
    for name, value in config_overrides(Path(root), Path(plan["release"]), plan["timestamp"]).items():
        cursor = expected
        parts = name.split(".")
        for part in parts[:-1]:
            cursor = cursor[part]
        cursor[parts[-1]] = value
    require(not original.changed_fields(expected, config), "Unexpected training configuration drift")
    require(plan.get("contract") == CONTRACT and plan.get("task_id") == TASK_ID
        and plan.get("fresh_run") is True and plan.get("initial_global_step") == 0
        and plan.get("optimizer_scheduler_rng_restored") is False
        and plan.get("source_step") == 38000 and plan.get("source_checkpoint_sha256") == SOURCE_SHA
        and config["model"]["pretrained_joint_checkpoint"] == plan["source_checkpoint"]
        and plan.get("gpus") == list(range(8)) and plan.get("gpu_count") == 8
        and plan.get("effective_global_batch") == 64 and plan.get("per_gpu_batch") == 8
        and plan.get("gradient_accumulation_steps") == 1
        and plan.get("peak_learning_rates") == original.PEAK_LRS
        and plan.get("max_optimizer_steps") == 20000 and plan.get("warmup_steps") == 1000
        and plan.get("train_episodes") == 95 and plan.get("validation_episodes") == 5
        and plan.get("validation_role") == VALIDATION_ROLE
        and plan.get("diagnostic_steps") == list(DIAGNOSTIC_STEPS)
        and plan.get("checkpoint_policy", {}).get("model_steps") == list(SAVE_STEPS),
        "Require the independent fresh95/5 Mobile20k experiment")
    train = config["train"]
    require(train["resume_from_checkpoint"] is None and train["skip_resumed_steps"] is False
        and train["max_training_steps"] == 20000 and train["warmup_steps"] == 1000
        and train["train_batch_size"] == 8 and train["gradient_accumulation_steps"] == 1
        and train["validation_steps"] == 0 and train["max_checkpoints_to_keep"] == 100,
        "Fresh optimizer state, update budget or custom exact-update diagnostics drifted")


def split_check(plan):
    splitter = helper(plan, "mobile_pp_holdout_data")
    require(sha(plan["split_file"]) == plan["split_sha256"], "Frozen episode split changed")
    metadata = Path(plan["release"]) / "packed/train/meta/episodes.jsonl"
    return splitter.validate_split(episodes(plan["release"]), read(plan["split_file"]), sha(metadata))


def normalization_binding(prior, release):
    path = Path(release) / "packed/source_dataset_statistics.json"
    expected = next(value for name, value in prior["bindings_sha256"].items()
                    if name.endswith("/packed/source_dataset_statistics.json"))
    require(sha(path) == expected, "Inherited joint38000 normalization identity changed")
    return {str(path): expected}


def prepare(args):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Prepare with GPUs hidden")
    release = args.release.resolve(strict=True)
    ready = original.validate_ready(release, args.expected_ready_sha256, rehash=False)
    _, prior, reference, files = original.load_reference(args.reference)
    source = Path(prior["source_checkpoint"])
    require(sha(source) == SOURCE_SHA, "Full original joint38000 weight SHA differs")
    output = args.output.absolute()
    protected = [release, args.reference.resolve(), source.parent, Path(prior["frozen_code"]),
                 Path(reference["train"]["output_dir"]).parent]
    require(not output.exists() and not output.is_symlink()
        and not any(output.resolve().is_relative_to(p.resolve()) or p.resolve().is_relative_to(output.resolve())
                    for p in protected), "Use a fresh independent95/5 run directory")
    runtime = Path(__file__).resolve().parent
    splitter = module("mobile95_prepare_split", runtime / "mobile_pp_holdout_data.py")
    split = read(args.split)
    splitter.validate_split(episodes(release), split, sha(release / "packed/train/meta/episodes.jsonl"))
    bindings = normalization_binding(prior, release)
    cpus = original.cpu_list(args.training_cpus)
    output.mkdir(parents=True, mode=0o700)
    archived = output / "reference"
    archived.mkdir(mode=0o700)
    for file in [Path(args.reference) / "REFERENCE_SUMMARY.json", *files.values()]:
        shutil.copy2(file, archived / file.name)
    shutil.copy2(args.split, output / "episode_split.json")
    timestamp = datetime.now(timezone.utc).strftime("%y%m%d%H%M%S")
    source_stat = source.stat()
    plan = {"contract": CONTRACT, "task_id": TASK_ID, "kind": "experiment", "fresh_run": True,
        "initial_global_step": 0, "optimizer_scheduler_rng_restored": False, "source_step": 38000,
        "source_checkpoint": str(source), "source_checkpoint_sha256": SOURCE_SHA,
        "source_identity": {"bytes": source_stat.st_size, "mtime_ns": source_stat.st_mtime_ns},
        "release": str(release), "expected_ready_sha256": args.expected_ready_sha256,
        "accepted_data_audit_reused": True, "new_full_payload_scan_performed": False,
        "reference_directory": str(archived), "reference_summary_sha256": original.SUMMARY_SHA,
        "frozen_code": prior["frozen_code"], "code_sha256": prior["code_sha256"],
        "legacy_runner": str(archived / "a100_handover_merged114.py"),
        "upstream_runner": str(archived / "quality100_bs8_sync_retry.py"),
        "upstream_runner_sha256": original.UPSTREAM_SHA,
        "migration_helper": str(archived / "state_dropout_migration.py"),
        "migration_helper_sha256": original.MIGRATION_SHA,
        "gpus": list(range(8)), "gpu_count": 8, "per_gpu_batch": 8,
        "gradient_accumulation_steps": 1, "effective_global_batch": 64,
        "max_optimizer_steps": 20000, "warmup_steps": 1000, "smoke": False,
        "peak_learning_rates": original.PEAK_LRS, "train_episodes": 95, "validation_episodes": 5,
        "source_total_frames": 61806, "source_total_windows": 55906,
        "validation_role": VALIDATION_ROLE, "source_role_override": split["source_role_override"],
        "split_file": str(output / "episode_split.json"), "split_sha256": sha(output / "episode_split.json"),
        "diagnostic_steps": list(DIAGNOSTIC_STEPS),
        "checkpoint_policy": {"model_steps": list(SAVE_STEPS), "full_recovery_states_to_keep": 2,
            "cleanup_scope": "Only older optimizer/scheduler/RNG inside this independent run; retain all33 models"},
        "training_cpus": cpus, "training_nice": 0, "execution_environment": original.TRAIN_ENV,
        "bindings_sha256": bindings, "timestamp": timestamp, "created_utc": now(),
        "runtime_directory": str(runtime), "runtime_sha256": {name: sha(runtime / name) for name in RUNTIME_FILES},
        "runner_sha256": sha(__file__), "accepted_payload_file_count": len(ready["payload_files"])}
    legacy, up = original.external(plan)
    require(legacy.sealed_code(Path(plan["frozen_code"])) == plan["code_sha256"], "Frozen production code changed")
    for name, target in {"train": release / "packed/train", "val": release / "packed/train",
                         "dino/train": release / "dino", "dino/val": release / "dino"}.items():
        link = output / "data" / name
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target, target_is_directory=True)
    argv = up.shared.rewrite_argv(files["argv.txt"].read_text().split(), config_overrides(output, release, timestamp))
    (output / "argv.txt").write_text("\n".join(argv) + "\n")
    config = up.shared.parse_training_argv(output / "argv.txt")
    validate_config(config, plan, reference, output)
    write(output / "run_config.json", config.model_dump(mode="json"))
    plan.update(config_sha256=sha(output / "run_config.json"), argv_sha256=sha(output / "argv.txt"))
    write(output / "preparation.json", plan)
    (output / "launch.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\nexec " +
        shlex.join([sys.executable, str(Path(__file__).resolve()), "run", "--prepared", str(output)]) + "\n")
    print(json.dumps({"prepared": str(output), "train_episodes": 95, "validation_episodes": 5,
                      "split_sha256": plan["split_sha256"], "training_started": False}))


def load(root):
    root = Path(root).resolve(strict=True)
    plan = read(root / "preparation.json")
    require(plan.get("contract") == CONTRACT and sha(__file__) == plan.get("runner_sha256"), "Run contract/source changed")
    for name in RUNTIME_FILES:
        require(sha(Path(plan["runtime_directory"]) / name) == plan["runtime_sha256"][name], "Bound runtime source changed")
    original.validate_ready(Path(plan["release"]), plan["expected_ready_sha256"], rehash=False)
    _, prior, reference, _ = original.load_reference(plan["reference_directory"])
    require(plan["source_checkpoint"] == prior["source_checkpoint"] and plan["frozen_code"] == prior["frozen_code"]
        and plan["code_sha256"] == prior["code_sha256"]
        and plan["bindings_sha256"] == normalization_binding(prior, plan["release"]), "Inherited source identity changed")
    for path, expected in {**plan["code_sha256"], **plan["bindings_sha256"]}.items():
        require(sha(path) == expected, "Immutable code/statistics binding differs")
    source = Path(plan["source_checkpoint"]).stat()
    require({"bytes": source.st_size, "mtime_ns": source.st_mtime_ns} == plan["source_identity"], "Verified joint38000 file changed")
    split_check(plan)
    require(sha(root / "run_config.json") == plan["config_sha256"] and sha(root / "argv.txt") == plan["argv_sha256"], "Config identity differs")
    for name, target in {"train": "packed/train", "val": "packed/train", "dino/train": "dino", "dino/val": "dino"}.items():
        link = root / "data" / name
        require(link.is_symlink() and link.resolve(strict=True) == (Path(plan["release"]) / target).resolve(strict=True), "Read-only data view changed")
    legacy, up = original.external(plan)
    config = up.shared.parse_training_argv(root / "argv.txt")
    require(config.model_dump(mode="json") == read(root / "run_config.json"), "Parsed config differs")
    validate_config(config, plan, reference, root)
    return config, plan, legacy, up


def datasets(config, processor, plan, legacy, shared):
    return helper(plan, "mobile_pp_holdout_data").datasets(config, processor, plan, legacy, shared)


def check(root):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Check with GPUs hidden")
    config, plan, legacy, up = load(root)
    import numpy as np
    import torch
    from transformers import AutoProcessor
    require(not torch.cuda.is_initialized(), "CPU check initialized CUDA")
    processor = AutoProcessor.from_pretrained(config.model.model_name_or_path, local_files_only=True)
    train, val, summary, bundle = datasets(config, processor, plan, legacy, up.shared)
    samples = []
    for role, data in (("train95", train), ("heldout_mobile5", val)):
        for index in (0, len(data) // 2, len(data) - 1):
            item = data[index]
            actions, mask = np.asarray(item["actions"]), np.asarray(item["actions_mask"])
            motion = np.asarray(item["motion_img_embs"])
            require(actions.shape == mask.shape == (30, 80) and np.isfinite(actions).all()
                and mask[:, :78].all() and not mask[:, 78:].any()
                and motion.shape == (6, 5, 1024) and np.isfinite(motion).all(), "New partition production read failed")
            samples.append({"role": role, "index": index, "actions_shape": [30, 80], "motion_shape": [6, 5, 1024]})
    require(not torch.cuda.is_initialized(), "CPU reader initialized CUDA")
    write(Path(root) / "data_check.json", {"contract": CONTRACT, "status": "passed", "checked_utc": now(),
        "config_sha256": plan["config_sha256"], "split_sha256": plan["split_sha256"],
        "expected_ready_sha256": plan["expected_ready_sha256"], "counts": summary,
        "samples": samples, "CUDA_initialized": False, "training_started": False,
        "scope": "New95/5 partition and representative production reads; original numerical audit reused"})
    print(json.dumps({"status": "passed", "counts": summary}))


def require_cpu_check(root, plan):
    value = read(Path(root) / "data_check.json")
    counts = value.get("counts", {})
    require(value.get("contract") == CONTRACT and value.get("status") == "passed"
        and all(value.get(key) == plan[key] for key in ("config_sha256", "split_sha256", "expected_ready_sha256"))
        and value.get("CUDA_initialized") is False
        and counts.get("train_episodes") == 95 and counts.get("validation_episodes") == 5
        and counts.get("source_total_windows") == 55906
        and counts.get("train_windows", 0) + counts.get("validation_windows", 0) == 55906
        and all(counts.get(key) is True for key in ("train_eval_uuid_disjoint", "train_eval_lineage_disjoint",
            "train_eval_original_rows_disjoint", "train_probe_subset_of_train", "counts_partition_sum_verified")),
        "Run the bound new95/5 CPU reader check first")


def trainer_class(plan, legacy, up):
    import torch
    bundle = {}
    def create(config, processor):
        train, val, summary, views = datasets(config, processor, plan, legacy, up.shared)
        bundle.update(views)
        return train, val, summary
    up.datasets = create
    Base = up.trainer_class(plan)
    migration = module("mobile95_statekeep", plan["migration_helper"])
    diagnostics = helper(plan, "mobile_pp_overfit_metrics")
    class Mobile95Trainer(Base):
        def init_models(self):
            return migration.initialize_state_keep(self,
                strict_initializer=lambda: super(Mobile95Trainer, self).init_models(),
                receipt_path=Path(self.project_dir) / f"state_policy_migration_rank_{os.environ.get('RANK', '0')}.json")

        def record_diagnostics(self, completed):
            result = diagnostics.evaluate_diagnostics(self, bundle, completed, plan["split_sha256"])
            status = [None]
            if self.accelerator.is_main_process:
                try:
                    directory = Path(self.project_dir) / "mobile_diagnostics"
                    target = directory / f"update_{completed:05d}.json"
                    require(not target.exists(), "Do not overwrite a previous diagnostic receipt")
                    result.update(validation_role=VALIDATION_ROLE, checked_utc=now())
                    write(target, result)
                    # Each exact-update JSON is authoritative; JSONL is a convenient curve index.
                    with (directory / "curve.jsonl").open("a") as stream:
                        stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                    status[0] = {"status": "passed"}
                except Exception as error:
                    status[0] = {"status": "failed", "error": repr(error)}
            torch.distributed.broadcast_object_list(status, src=0)
            require(status[0]["status"] == "passed", "Diagnostic receipt failed: " + str(status[0]))
            return result["metrics"]

        def prepare(self, accelerator):
            result = super().prepare(accelerator)
            # Production Trainer normally assigns this in its first step();
            # the fresh-model diagnostic happens before that first update.
            self.global_step = 0
            baseline = self.record_diagnostics(0)
            self.log({"train/mobile_eval/" + key: value for key, value in
                      {**baseline, "completed_update": 0}.items()})
            return result

        def step(self, batch_input, global_step, local_step):
            model = self.unwrap_model()
            require(self.model_cfg.state_drop_prob == model.action_header.state_drop_prob ==
                model.action_header.config.state_drop_prob == 0 and tuple(batch_input["states"].shape) == (8, 1, 45),
                "Retain measured state on every rank")
            synced, metrics = super().step(batch_input, global_step, local_step)
            require(model.action_header.last_state_drop_frac == 0, "Training state was dropped")
            metrics["state_drop_frac"] = 0.
            if synced and global_step + 1 in DIAGNOSTIC_STEPS:
                values = self.record_diagnostics(global_step + 1)
                metrics.update({"mobile_eval/" + key: value for key, value in
                                {**values, "completed_update": global_step + 1}.items()})
            return synced, metrics

        def save_checkpoint(self, global_step):
            require(global_step in SAVE_STEPS, "Unexpected checkpoint update")
            result = super().save_checkpoint(global_step)
            status = [None]
            if self.accelerator.is_main_process:
                try:
                    record = legacy.retain_model_weights(Path(self.project_dir) / "checkpoints", keep_full=2)
                    require(record["all_saved_model_steps"] == [step for step in SAVE_STEPS if step <= global_step], "Model inventory drifted")
                    record.update(saved_update=global_step, checked_utc=now(), task_id=TASK_ID)
                    write(Path(self.project_dir) / "checkpoint_retention.json", record)
                    status[0] = {"status": "passed"}
                except Exception as error:
                    status[0] = {"status": "failed", "error": repr(error)}
            torch.distributed.broadcast_object_list(status, src=0)
            require(status[0]["status"] == "passed", "Retention failed: " + str(status[0]))
            return result
    return Mobile95Trainer


def require_launch_acceptance(root, plan):
    path = Path(root) / "launch_acceptance.json"
    require(path.is_file() and not path.is_symlink()
        and sha(path) == os.environ.get("CVPR_MOBILE95_LAUNCH_RECEIPT_SHA256"), "Missing bound fresh launch receipt")
    accepted = read(path)
    require(accepted.get("contract") == CONTRACT and accepted.get("status") == "passed"
        and all(accepted.get(key) == plan[key] for key in ("config_sha256", "split_sha256", "expected_ready_sha256", "runner_sha256"))
        and accepted.get("accepted_data_audit_reused") is True
        and accepted.get("new_full_payload_scan_performed") is False
        and accepted.get("source_checkpoint_sha256_verified_this_launch") is True
        and accepted.get("source_checkpoint_sha256") == SOURCE_SHA, "Launch inputs differ")
    state = read(Path(root) / "training_state.json")
    require(state.get("state") in ("starting", "running") and state.get("launcher_pid") == accepted["launcher_pid"], "Do not restart this experiment")


def train(root):
    config, plan, legacy, up = load(root)
    require_cpu_check(root, plan)
    require(int(os.environ.get("WORLD_SIZE", "0")) == 8, "Exactly eight training ranks required")
    original.require_launch_runtime(plan)
    require_launch_acceptance(root, plan)
    import torch
    cls = trainer_class(plan, legacy, up)
    class Factory:
        @staticmethod
        def instantiate(cfg, device_id):
            return cls(cfg, torch.device(f"cuda:{device_id}"))
    production = module("mobile95_production_train", Path(plan["frozen_code"]) / "VLMExpert/scripts/train.py")
    production.Trainer = Factory
    sys.argv = (Path(root) / "argv.txt").read_text().split()
    production.train(config)


def name(root):
    config, _, _, _ = load(root)
    from psi.trainers.posttrain import PosttrainTrainer
    print(PosttrainTrainer(config, "cpu").project_dir)


def run(root):
    root = Path(root).resolve(strict=True)
    with (root / "launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not (root / "training_state.json").exists(), "Already launched; preserve this run")
        _, plan, _, up = load(root)
        require_cpu_check(root, plan)
        runtime = original.require_launch_runtime(plan)
        # Recheck the initialization model for this launch. This is separate
        # from the already accepted dataset numerical audit, which is reused.
        require(sha(plan["source_checkpoint"]) == SOURCE_SHA, "Original joint38000 bytes changed before launch")
        minimum = len(SAVE_STEPS) * plan["source_identity"]["bytes"] + 3 * 24000000000 + 20 * 1024 ** 3
        free = shutil.disk_usage(root).free
        require(free >= minimum, f"Need {minimum} output bytes; available {free}")
        up.selected_gpu_idle(plan["gpus"])
        code = Path(plan["frozen_code"])
        env = dict(os.environ, **original.TRAIN_ENV)
        env["PYTHONPATH"] = os.pathsep.join(map(str, [code / "scripts/4090_simple", code / "VLMExpert/src", code, code / "VLMExpert"]))
        diagnostic = Path("/dev/shm") / f"mobile95-{plan['timestamp']}-{os.getpid()}"
        diagnostic.mkdir(mode=0o700)
        env["PSI0_ROLLING_DIAGNOSTICS_DIR"] = str(diagnostic)
        command = [sys.executable, str(Path(__file__).resolve())]
        result = subprocess.run(command + ["name", "--prepared", str(root)], cwd=code / "VLMExpert",
            env=dict(env, WORLD_SIZE="8", RANK="0", LOCAL_RANK="0", CUDA_VISIBLE_DEVICES=""), text=True, capture_output=True)
        (root / "run_name.log").write_text(result.stdout + result.stderr)
        require(result.returncode == 0, "Run naming failed; inspect run_name.log")
        state = {"state": "starting", "started_utc": now(), "launcher_pid": os.getpid(),
            "run_directory": result.stdout.strip().splitlines()[-1], "max_optimizer_steps": 20000,
            "task_id": TASK_ID, "validation_role": VALIDATION_ROLE, "split_sha256": plan["split_sha256"],
            "config_sha256": plan["config_sha256"], "launch_runtime": runtime, "output_free_bytes_at_launch": free}
        acceptance = {"contract": CONTRACT, "status": "passed", "checked_utc": now(), "launcher_pid": os.getpid(),
            **{key: plan[key] for key in ("config_sha256", "split_sha256", "expected_ready_sha256", "runner_sha256")},
            "accepted_data_audit_reused": True, "new_full_payload_scan_performed": False,
            "source_checkpoint_sha256": SOURCE_SHA, "source_checkpoint_sha256_verified_this_launch": True,
            "launch_runtime": runtime, "training_started": False}
        receipt = root / "launch_acceptance.json"
        with receipt.open("x") as stream:
            stream.write(json.dumps(acceptance, indent=2, allow_nan=False) + "\n")
        env["CVPR_MOBILE95_LAUNCH_RECEIPT_SHA256"] = sha(receipt)
        write(root / "training_state.json", state)
        with (root / "train.log").open("x") as log:
            try:
                process = subprocess.Popen([sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=8",
                    str(Path(__file__).resolve()), "train", "--prepared", str(root)], cwd=code / "VLMExpert", env=env,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                state.update(state="running", pid=process.pid)
                write(root / "training_state.json", state)
                status = process.wait()
            except Exception as error:
                state.update(state="failed", error=repr(error), finished_utc=now())
                write(root / "training_state.json", state)
                raise
        state.update(state="complete" if status == 0 else "failed", returncode=status, finished_utc=now())
        write(root / "training_state.json", state)
        return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    prep = modes.add_parser("prepare")
    prep.add_argument("--release", type=Path, default=RELEASE)
    for key in ("reference", "output", "split"):
        prep.add_argument("--" + key, type=Path, required=True)
    prep.add_argument("--expected-ready-sha256", required=True)
    prep.add_argument("--training-cpus", required=True)
    for key in ("check", "name", "train", "run"):
        modes.add_parser(key).add_argument("--prepared", type=Path, required=True)
    args = parser.parse_args()
    return prepare(args) if args.mode == "prepare" else globals()[args.mode](args.prepared)


if __name__ == "__main__":
    raise SystemExit(main() or 0)
