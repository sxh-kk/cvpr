"""Fresh Mobile P&P100 training, bound to the actual Handover114 reference.

prepare/check are CPU operations. Run launch.sh from a separate ordinary SSH
session with the declared full CPU affinity and nice=0, never from a restricted
CPU inspection wrapper. No preparation or launch is allowed before actual A100
production-reader readiness. The five old Handover episodes monitor regression;
they are not a Mobile validation set. Mobile100 remains entirely train_only.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

CONTRACT = "cvpr-mobile-pp100-a100-fresh-v1"
TASK_ID = "simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0"
READY_CONTRACT = "cvpr-pico-mobile-pp-production-verification-v1"
PACK_CONTRACT = "cvpr-pico-mobile-pp-lerobot-pack-v1"
READY_NAME = "A100_PICO_DATASET_READY.json"
RELEASE = Path("/gaozt-test1/datasets/SIMPLE_sonic_v1_1/pico_mobile_pp_20261027_human100_01")
SUMMARY_SHA = "242f8b2c57bbe8bad283ddb82b18159b7b235216cb4e9c57915b08729a16ef82"
REFERENCE_CONTRACT = "cvpr-mobile-training-reference-handover114-readonly-v1"
SOURCE_SHA = "12a71e046e221b83f0bdb0efb50f9ba82442045aca0efabef3ae337f185320b7"
LEGACY_SHA = "8e48e55e2e822404920c9cc5304eb81c3d528cb3e12c4c3c56c7be5dcc0a0f68"
UPSTREAM_SHA = "57c7fad25ef18c40849a397d54b4f9e158b19b47205f80ab8da7844d26876923"
MIGRATION_SHA = "95bbb9f4414c7a133e03f253c2219ed9683406e9569bd1084961a6c17f6d8e5a"
PEAK_LRS = {"action_header": 1e-5, "lang_backbone": 1e-7, "vision_tower": 1e-6,
            "mm_projector": 1e-5, "motion_expert": 1e-6}
SAVE_STEPS = tuple(range(4000, 10001, 500))
TRAIN_ENV = {"WANDB_MODE": "offline", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4", "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
    "NCCL_IB_DISABLE": "1", "NCCL_P2P_DISABLE": "0", "NCCL_SHM_DISABLE": "0",
    "CUDA_LAUNCH_BLOCKING": "1"}
VALIDATION_ROLE = "historical_handover_regression_only_not_mobile_validation"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix="." + path.name,
                                     suffix=".partial", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def cpu_list(value):
    cpus = []
    for part in value.split(","):
        require(bool(re.fullmatch(r"\d+(?:-\d+)?", part)), "Invalid training CPU list")
        ends = list(map(int, part.split("-")))
        require(len(ends) == 1 or ends[0] <= ends[1], "Invalid CPU range")
        cpus.extend(range(ends[0], ends[-1] + 1))
    require(len(set(cpus)) == len(cpus) and len(cpus) >= 32,
            "Declare the full training CPU affinity, at least 32 distinct CPUs")
    return sorted(cpus)


def changed_fields(left, right, prefix=""):
    if isinstance(left, dict) and isinstance(right, dict):
        result = set()
        for key in left.keys() | right.keys():
            name = f"{prefix}.{key}" if prefix else key
            result.update(changed_fields(left[key], right[key], name)
                          if key in left and key in right else {name})
        return result
    return set() if left == right and type(left) is type(right) else {prefix}


def load_reference(directory):
    """Read archived original bytes; never call an old prepare/load/run method."""
    directory = Path(directory).resolve(strict=True)
    require(sha(directory / "REFERENCE_SUMMARY.json") == SUMMARY_SHA, "Reference summary SHA differs")
    summary = read(directory / "REFERENCE_SUMMARY.json")
    require(summary.get("contract") == REFERENCE_CONTRACT, "Wrong actual training reference")
    files = {}
    for entry in summary["reference_files"]:
        name = Path(entry["local_path"]).name
        require(name not in files, "Duplicate reference basename")
        path = directory / name
        require(not path.is_symlink() and path.is_file() and path.stat().st_size == entry["size_bytes"]
                and sha(path) == entry["sha256"], "Original reference file identity differs: " + name)
        files[name] = path
    prepared, actual = read(files["run_config.json"]), read(files["actual_run_config.json"])
    allowed_runtime = {"train.hf_token", "wandb.entity", "wandb.group", "wandb.id", "wandb.name"}
    require(changed_fields(prepared, actual) <= allowed_runtime,
            "Actual Handover configuration differs from the archived prepared configuration")
    prior = read(files["preparation.json"])
    require(prior["source_checkpoint_sha256"] == SOURCE_SHA and prior["source_step"] == 38000
            and prior["fresh_run"] is True and prior["optimizer_scheduler_rng_restored"] is False
            and prior["initial_global_step"] == 0 and prior["gpu_count"] == 8
            and prior["peak_learning_rates"] == PEAK_LRS, "Wrong actual fresh joint38000 reference")
    for name, expected in (("a100_handover_merged114.py", LEGACY_SHA),
                           ("quality100_bs8_sync_retry.py", UPSTREAM_SHA),
                           ("state_dropout_migration.py", MIGRATION_SHA)):
        require(sha(files[name]) == expected, "Accepted external source identity differs")
    return summary, prior, prepared, files


def owned(root, relative):
    require(isinstance(relative, str) and not Path(relative).is_absolute()
            and ".." not in Path(relative).parts, "Unsafe payload path")
    path = root
    for part in Path(relative).parts:
        path /= part
        require(not path.is_symlink(), "Payload symlink is not accepted: " + relative)
    require(path.resolve(strict=True).is_relative_to(root) and path.is_file(),
            "Missing or external payload: " + relative)
    return path


def validate_ready(release, expected_sha256, *, rehash=True):
    release = Path(release).resolve(strict=True)
    require(release == RELEASE, "Use only the explicitly authorized Mobile100 release")
    require(isinstance(expected_sha256, str) and re.fullmatch(r"[0-9a-f]{64}", expected_sha256),
            "The actual final A100 READY SHA must be supplied explicitly")
    receipt = owned(release, READY_NAME)
    require(sha(receipt) == expected_sha256, "Actual A100 READY SHA differs")
    ready = read(receipt)
    require(ready.get("contract") == READY_CONTRACT and ready.get("task_id") == TASK_ID
            and ready.get("selection_policy") == "native_only" and ready.get("status") == "passed"
            and ready.get("remote_production_ready") is True
            and ready.get("production_ready_for_bound_pack_and_cache") is True
            and ready.get("DINO_relocation_pending") is False
            and ready.get("training_started") is False and ready.get("CUDA_initialized") is False
            and ready.get("GPU_used") is False and ready.get("normalization_refitted") is False
            and ready.get("split_restriction") == "train_only"
            and (ready.get("train_episodes"), ready.get("total_frames"), ready.get("train_windows"))
                == (100, 61806, 55906), "Actual Mobile100 A100 production readiness is incomplete")
    require((ready.get("release_root"), ready.get("pack_root"), ready.get("dino_root")) ==
            (str(release), str(release / "packed"), str(release / "dino")), "READY paths are not rebound to A100")
    payload = ready.get("payload_files", {})
    required = {"packed/PACK_READY.json", "packed/train/meta/episodes.jsonl", "packed/train/meta/info.json",
                "packed/source_dataset_statistics.json", "dino/manifest.json"}
    require(required <= payload.keys(), "A100 READY lacks complete pack/cache identity bindings")
    for relative, identity in payload.items():
        path = owned(release, relative)
        stat = path.stat()
        require(stat.st_size == identity["size_bytes"] and stat.st_mtime_ns == identity["mtime_ns"]
                and (not rehash or sha(path) == identity["sha256"]), "Accepted A100 payload changed: " + relative)
    require(sha(release / "packed/PACK_READY.json") == ready["pack_ready_sha256"]
            and sha(release / "dino/manifest.json") == ready["cache_manifest_sha256"],
            "READY pack/cache digest differs")
    pack = read(release / "packed/PACK_READY.json")
    require(pack.get("contract") == PACK_CONTRACT and pack.get("task_id") == TASK_ID
            and pack.get("selection_policy") == "native_only"
            and (pack.get("train_episodes"), pack.get("total_frames"), pack.get("train_windows")) ==
                (100, 61806, 55906), "Wrong Mobile100 packed dataset")
    episodes = [json.loads(line) for line in
                (release / "packed/train/meta/episodes.jsonl").read_text().splitlines()]
    require(len(episodes) == 100 and {e["episode_index"] for e in episodes} == set(range(100))
            and sum(e["length"] for e in episodes) == 61806
            and sum(len(e["eligible_current_rows"]) for e in episodes) == 55906
            and all(e.get("task_id") == TASK_ID and e.get("split") == "train"
                    and e.get("split_restriction") == "train_only" for e in episodes),
            "Require all and only the frozen100 Mobile train_only episodes")
    return ready


def config_overrides(output, release, timestamp):
    return {"exp": "mobile_pp100_joint38000_statekeep_8gpu", "timestamp": timestamp,
        "train.output_dir": str(output / "training"), "data.root_dir": str(output / "data"),
        "data.dino_cache_root": str(output / "data/dino"),
        "data.transform.field.stat_path": str(release / "packed/source_dataset_statistics.json"),
        "train.checkpointing_steps": 10000, "train.early_checkpoint_steps": list(SAVE_STEPS[:-1]),
        "train.preserve_checkpoint_every": 500}


def expected_config(reference, plan, root):
    result = copy.deepcopy(reference)
    for name, value in config_overrides(Path(root), Path(plan["release"]), plan["timestamp"]).items():
        cursor = result
        parts = name.split(".")
        for part in parts[:-1]:
            cursor = cursor[part]
        cursor[parts[-1]] = value
    return result


def validate_config(value, plan, reference, root):
    config = value if isinstance(value, dict) else value.model_dump(mode="json")
    require(not changed_fields(expected_config(reference, plan, root), config),
            "Configuration drift outside the authorized dataset/output/save-policy changes")
    train = config["train"]
    require(plan.get("contract") == CONTRACT and plan.get("task_id") == TASK_ID
            and plan.get("fresh_run") is True and plan.get("initial_global_step") == 0
            and plan.get("optimizer_scheduler_rng_restored") is False
            and plan.get("gpus") == list(range(8)) and plan.get("gpu_count") == 8
            and plan.get("effective_global_batch") == 64 and plan.get("peak_learning_rates") == PEAK_LRS
            and plan.get("max_optimizer_steps") == 10000 and plan.get("warmup_steps") == 1000
            and plan.get("per_gpu_batch") == 8 and plan.get("gradient_accumulation_steps") == 1
            and (plan.get("train_episodes"), plan.get("total_frames"), plan.get("train_windows")) == (100, 61806, 55906)
            and plan.get("validation_role") == VALIDATION_ROLE
            and plan.get("source_step") == 38000 and plan.get("source_checkpoint_sha256") == SOURCE_SHA,
            "Require a fresh Mobile100 eight-rank joint38000 task, with historical Handover monitoring")
    require(train["resume_from_checkpoint"] is None and train["skip_resumed_steps"] is False
            and train["max_training_steps"] == 10000 and train["warmup_steps"] == 1000
            and train["train_batch_size"] == 8 and train["gradient_accumulation_steps"] == 1
            and train["validation_steps"] == 1000 and train["max_checkpoints_to_keep"] == 100
            and train["checkpointing_steps"] == 10000 and train["preserve_checkpoint_every"] == 500
            and train["early_checkpoint_steps"] == list(SAVE_STEPS[:-1])
            and str(config["model"]["pretrained_joint_checkpoint"]) == plan["source_checkpoint"],
            "Fresh state, optimizer-update budget, validation cadence or checkpoint policy differs")


def external(plan):
    # The archived launcher exposes pure helpers. Its old prepare/load/datasets
    # contracts remain untouched and are deliberately never called here.
    require(sha(plan["legacy_runner"]) == LEGACY_SHA, "Legacy helper SHA differs")
    legacy = module("mobile100_verified_legacy_helpers", plan["legacy_runner"])
    return legacy, legacy.external(plan)


def validate_reference_inputs(prior, release, files):
    bindings = prior["bindings_sha256"]
    val, val_dino = Path(prior["val"]), Path(prior["val_dino"])
    paths = [val / "meta/info.json", val / "meta/episodes.jsonl", val_dino / "manifest.json"]
    for path in paths:
        require(sha(path) == bindings[str(path)], "Historical Handover VAL input changed")
    stat_path = release / "packed/source_dataset_statistics.json"
    old_stat = next(p for p in bindings if p.endswith("/packed/source_dataset_statistics.json"))
    require(sha(stat_path) == bindings[old_stat], "Keep the exact inherited joint38000 normalization bytes")
    require(sha(val / "meta/episodes.jsonl") == sha(files["handover_val_episodes.jsonl"]),
            "Historical VAL episode order differs from the actual previous run")
    return {str(p): sha(p) for p in paths + [stat_path]}


def prepare(args):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Prepare must run with GPUs hidden")
    release = args.release.resolve(strict=True)
    validate_ready(release, args.expected_ready_sha256)
    _, prior, reference, files = load_reference(args.reference)
    source = Path(prior["source_checkpoint"])
    require(sha(source) == SOURCE_SHA, "The complete accepted joint38000 model SHA differs")
    bindings = validate_reference_inputs(prior, release, files)
    output = args.output.absolute()
    protected = [release, args.reference.resolve(), Path(prior["parent_prepared"]),
                 Path(reference["train"]["output_dir"]).parent, Path(prior["frozen_code"]), source.parent]
    require(not output.exists() and not output.is_symlink()
            and not any(output.resolve().is_relative_to(p.resolve()) or p.resolve().is_relative_to(output.resolve())
                        for p in protected), "Use a fresh independent Mobile prepared/output directory")
    training_cpus = cpu_list(args.training_cpus)
    # Identity checks precede creation; do not leave a plausible prepared run on
    # readiness failure. Failed partial preparations remain explicit evidence.
    output.mkdir(parents=True, mode=0o700)
    archived = output / "reference"
    archived.mkdir(mode=0o700)
    for file in [Path(args.reference) / "REFERENCE_SUMMARY.json", *files.values()]:
        shutil.copy2(file, archived / file.name)
    timestamp = datetime.now(timezone.utc).strftime("%y%m%d%H%M%S")
    source_stat = source.stat()
    plan = {"contract": CONTRACT, "task_id": TASK_ID, "kind": "experiment", "fresh_run": True,
        "initial_global_step": 0, "optimizer_scheduler_rng_restored": False, "source_step": 38000,
        "source_checkpoint": str(source), "source_checkpoint_sha256": SOURCE_SHA,
        "source_identity": {"bytes": source_stat.st_size, "mtime_ns": source_stat.st_mtime_ns},
        "release": str(release), "expected_ready_sha256": args.expected_ready_sha256,
        "reference_directory": str(archived), "reference_summary_sha256": SUMMARY_SHA,
        "frozen_code": prior["frozen_code"], "code_sha256": prior["code_sha256"],
        "legacy_runner": str(archived / "a100_handover_merged114.py"),
        "upstream_runner": str(archived / "quality100_bs8_sync_retry.py"), "upstream_runner_sha256": UPSTREAM_SHA,
        "migration_helper": str(archived / "state_dropout_migration.py"), "migration_helper_sha256": MIGRATION_SHA,
        "gpus": list(range(8)), "gpu_count": 8, "per_gpu_batch": 8, "gradient_accumulation_steps": 1,
        "effective_global_batch": 64, "max_optimizer_steps": 10000, "warmup_steps": 1000,
        "smoke": False, "peak_learning_rates": PEAK_LRS, "train_episodes": 100, "total_frames": 61806,
        "train_windows": 55906, "validation_windows": 40, "val": prior["val"], "val_dino": prior["val_dino"],
        "validation_role": VALIDATION_ROLE, "Mobile_validation_selected": False,
        "expected_validation_rows": read(files["task_window_selection.json"])["validation_rows"],
        "checkpoint_policy": {"model_steps": list(SAVE_STEPS), "full_recovery_states_to_keep": 2,
            "cleanup_scope": "Only older optimizer/scheduler/RNG in this new run; all13 models retained"},
        "training_cpus": training_cpus, "training_nice": 0, "execution_environment": TRAIN_ENV,
        "bindings_sha256": bindings, "timestamp": timestamp, "created_utc": now(), "runner_sha256": sha(__file__)}
    for path, expected in plan["code_sha256"].items():
        require(sha(path) == expected, "Actual frozen production source changed")
    legacy, up = external(plan)
    require(legacy.sealed_code(Path(plan["frozen_code"])) == plan["code_sha256"],
            "Frozen source inventory differs from the previous run")
    for name, target in {"train": release / "packed/train", "val": Path(plan["val"]),
                         "dino/train": release / "dino", "dino/val": Path(plan["val_dino"])}.items():
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
    command = [sys.executable, str(Path(__file__).resolve()), "run", "--prepared", str(output)]
    (output / "launch.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n"
        "# Use a new ordinary SSH session, with the declared full CPU affinity and nice=0.\n"
        "exec " + shlex.join(command) + "\n")
    print(json.dumps({"prepared": str(output), "train_episodes": 100, "train_windows": 55906,
        "validation_role": VALIDATION_ROLE, "checkpoint_steps": SAVE_STEPS, "training_started": False}))


def load(root):
    root = Path(root).resolve(strict=True)
    plan = read(root / "preparation.json")
    require(plan.get("contract") == CONTRACT and sha(__file__) == plan.get("runner_sha256"),
            "Prepared launcher contract or source changed")
    validate_ready(Path(plan["release"]), plan["expected_ready_sha256"], rehash=False)
    _, prior, reference, files = load_reference(plan["reference_directory"])
    require(plan["code_sha256"] == prior["code_sha256"] and plan["frozen_code"] == prior["frozen_code"],
            "Frozen source authority differs from the actual previous run")
    require(plan["expected_validation_rows"] == read(files["task_window_selection.json"])["validation_rows"]
            and plan["val"] == prior["val"] and plan["val_dino"] == prior["val_dino"]
            and plan["source_checkpoint"] == prior["source_checkpoint"],
            "Historical validation/source initialization binding differs")
    require(plan["bindings_sha256"] == validate_reference_inputs(prior, Path(plan["release"]), files),
            "Reference/normalization binding inventory differs")
    for path, expected in {**plan["code_sha256"], **plan["bindings_sha256"]}.items():
        require(sha(path) == expected, "Bound source/normalization/Handover VAL changed")
    stat = Path(plan["source_checkpoint"]).stat()
    require({"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns} == plan["source_identity"],
            "Source checkpoint identity changed after verified preparation")
    require(sha(root / "argv.txt") == plan["argv_sha256"] and sha(root / "run_config.json") == plan["config_sha256"],
            "Prepared configuration identity changed")
    expected_links = {"train": Path(plan["release"]) / "packed/train", "dino/train": Path(plan["release"]) / "dino",
                      "val": Path(prior["val"]), "dino/val": Path(prior["val_dino"])}
    for name, target in expected_links.items():
        require((root / "data" / name).is_symlink() and (root / "data" / name).resolve(strict=True) == target.resolve(strict=True),
                "Prepared dataset binding changed")
    legacy, up = external(plan)
    config = up.shared.parse_training_argv(root / "argv.txt")
    require(config.model_dump(mode="json") == read(root / "run_config.json"), "Parsed argv/config differs")
    validate_config(config, plan, reference, root)
    return config, plan, legacy, up


def select_validation_rows(candidates, expected):
    import numpy as np
    require(len(candidates) == 5, "Keep the five original Handover VAL episodes")
    rows = []
    for _, values in sorted(candidates.items()):
        require(len(values) >= 8, "Incomplete historical Handover VAL episode")
        rows.extend(int(values[int(i)]) for i in np.linspace(0, len(values) - 1, 8, dtype=np.int64))
    require(rows == expected and len(rows) == 40 and len(set(rows)) == 40,
            "Historical Handover regression windows differ from the actual previous40 rows")
    return rows


def datasets(config, processor, plan, legacy, shared):
    import torch
    train = config.data(split="train", transform_kwargs={"vlm_processor": processor})
    val = config.data(split="val", transform_kwargs={"vlm_processor": processor})
    episodes = [json.loads(line) for line in
                (Path(plan["release"]) / "packed/train/meta/episodes.jsonl").read_text().splitlines()]
    # This mapper selects actual eligible_current_rows; it never applies a
    # Handover task-name filter to the new Mobile task.
    rows, counts = legacy.map_accepted_rows(train.raw_dataset.motion_history, episodes, 55906)
    complete, _, _ = shared.eligible_dataset_rows(train, expected_episodes=100, source_frame_rule="ceil")
    require(set(rows) == {int(row) for values in complete.values() for row in values},
            "Mobile admissions differ from production real-history/complete-future windows")
    candidates, _, _ = shared.eligible_dataset_rows(val, expected_episodes=5, source_frame_rule="ceil")
    val_rows = select_validation_rows(candidates, plan["expected_validation_rows"])
    summary = {"train_windows": len(rows), "validation_windows": len(val_rows),
        "train_windows_per_episode": counts, "validation_rows": val_rows,
        "train_episode_indices": list(range(100)), "train_task_id": TASK_ID,
        "validation_role": VALIDATION_ROLE, "Mobile_validation_selected": False,
        "policy": "All frozen100 Mobile TRAIN; unchanged historical Handover40 regression windows"}
    return torch.utils.data.Subset(train, rows), torch.utils.data.Subset(val, val_rows), summary


def check(root):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU check must run with GPUs hidden")
    config, plan, legacy, up = load(root)
    import numpy as np
    import torch
    from transformers import AutoProcessor
    require(not torch.cuda.is_initialized(), "CPU check may not initialize CUDA")
    processor = AutoProcessor.from_pretrained(config.model.model_name_or_path, local_files_only=True)
    train, val, counts = datasets(config, processor, plan, legacy, up.shared)
    queries = []
    for split, data in (("train", train), ("handover_regression", val)):
        for index in (0, len(data) // 2, len(data) - 1):
            item = data[index]
            actions, mask = np.asarray(item["actions"]), np.asarray(item["actions_mask"])
            require(actions.shape == mask.shape == (30, 80) and np.isfinite(actions).all()
                    and mask[:, :78].all() and not mask[:, 78:].any()
                    and tuple(item["motion_img_embs"].shape) == (6, 5, 1024)
                    and np.isfinite(np.asarray(item["motion_img_embs"])).all(),
                    "Actual production action/mask/clean DINO sample invalid")
            queries.append({"split": split, "index": index, "actions_shape": [30, 80], "motion_shape": [6, 5, 1024]})
    require(not torch.cuda.is_initialized(), "CPU reader initialized CUDA")
    write(Path(root) / "data_check.json", {"contract": CONTRACT, "task_id": TASK_ID, "status": "passed",
        "checked_utc": now(), "config_sha256": plan["config_sha256"], "expected_ready_sha256": plan["expected_ready_sha256"],
        "counts": counts, "samples": queries, "CUDA_initialized": False, "training_started": False})
    print(json.dumps({"status": "passed", "train_windows": len(train), "handover_regression_windows": len(val)}))


def require_cpu_check(root, plan):
    accepted = read(Path(root) / "data_check.json")
    require(accepted.get("contract") == CONTRACT and accepted.get("task_id") == TASK_ID
            and accepted.get("status") == "passed" and accepted.get("config_sha256") == plan["config_sha256"]
            and accepted.get("expected_ready_sha256") == plan["expected_ready_sha256"]
            and accepted.get("counts", {}).get("train_windows") == 55906
            and accepted.get("counts", {}).get("validation_rows") == plan["expected_validation_rows"]
            and accepted.get("counts", {}).get("validation_role") == VALIDATION_ROLE,
            "Run the bound new-task CPU production input check before training")


def trainer_class(plan, legacy, up):
    import torch
    up.datasets = lambda config, processor: datasets(config, processor, plan, legacy, up.shared)
    Base = up.trainer_class(plan)
    helper = module("mobile100_state_keep_migration", plan["migration_helper"])
    class Mobile100Trainer(Base):
        def init_models(self):
            return helper.initialize_state_keep(self,
                strict_initializer=lambda: super(Mobile100Trainer, self).init_models(),
                receipt_path=Path(self.project_dir) / f"state_policy_migration_rank_{os.environ.get('RANK', '0')}.json")

        def step(self, batch_input, global_step, local_step):
            model = self.unwrap_model()
            require(self.model_cfg.state_drop_prob == model.action_header.state_drop_prob ==
                    model.action_header.config.state_drop_prob == 0 and tuple(batch_input["states"].shape) == (8, 1, 45),
                    "Every rank must retain all measured proprioceptive states")
            synced, metrics = super().step(batch_input, global_step, local_step)
            require(model.action_header.last_state_drop_frac == 0, "A training state was unexpectedly dropped")
            metrics["state_drop_frac"] = 0.
            return synced, metrics

        def save_checkpoint(self, global_step):
            require(global_step in SAVE_STEPS, "Unexpected checkpoint before4000 or outside500-update cadence")
            saved = super().save_checkpoint(global_step)
            result = None
            if self.accelerator.is_main_process:
                try:
                    record = legacy.retain_model_weights(Path(self.project_dir) / "checkpoints", keep_full=2)
                    require(record["all_saved_model_steps"] == [step for step in SAVE_STEPS if step <= global_step],
                            "Saved model inventory differs from the explicit13-step policy")
                    record.update(saved_update=global_step, checked_utc=now(), task_id=TASK_ID)
                    write(Path(self.project_dir) / "checkpoint_retention.json", record)
                    result = {"status": "passed"}
                except Exception as error:
                    result = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
            message = [result]
            torch.distributed.broadcast_object_list(message, src=0)
            if message[0]["status"] != "passed":
                raise RuntimeError(message[0]["error"])
            return saved
    return Mobile100Trainer


def train(root):
    config, plan, legacy, up = load(root)
    require_cpu_check(root, plan)
    require(int(os.environ.get("WORLD_SIZE", "0")) == 8, "Exactly eight fresh training ranks required")
    require_launch_runtime(plan)
    require_launch_acceptance(root, plan)
    import torch
    cls = trainer_class(plan, legacy, up)
    class Factory:
        @staticmethod
        def instantiate(cfg, device_id):
            return cls(cfg, torch.device(f"cuda:{device_id}"))
    production = module("mobile100_production_train", Path(plan["frozen_code"]) / "VLMExpert/scripts/train.py")
    production.Trainer = Factory
    sys.argv = (Path(root) / "argv.txt").read_text().split()
    production.train(config)


def name(root):
    config, _, _, _ = load(root)
    from psi.trainers.posttrain import PosttrainTrainer
    print(PosttrainTrainer(config, "cpu").project_dir)


def require_launch_runtime(plan):
    cpus = sorted(os.sched_getaffinity(0))
    nice = os.getpriority(os.PRIO_PROCESS, 0)
    require(cpus == plan["training_cpus"] and len(cpus) >= 32 and nice == plan["training_nice"] == 0,
            "Launch from a separate ordinary SSH session with declared full CPU affinity and nice=0; restricted CPU inspection context cannot launch training")
    return {"cpus": cpus, "nice": nice, "other_processes_modified": False}


def require_launch_acceptance(root, plan):
    """Bind each rank to this launch's full byte check; not an authentication claim."""
    path = Path(root) / "launch_acceptance.json"
    require(not path.is_symlink() and path.is_file()
            and sha(path) == os.environ.get("CVPR_MOBILE100_LAUNCH_RECEIPT_SHA256"),
            "Start training through launch.sh after this launch's full byte verification")
    accepted = read(path)
    require(accepted.get("contract") == CONTRACT and accepted.get("task_id") == TASK_ID
            and accepted.get("status") == "passed"
            and accepted.get("config_sha256") == plan["config_sha256"]
            and accepted.get("expected_ready_sha256") == plan["expected_ready_sha256"]
            and accepted.get("runner_sha256") == plan["runner_sha256"]
            and accepted.get("full_payload_sha256_verified") is True
            and accepted.get("source_checkpoint_sha256") == SOURCE_SHA,
            "This launch's full byte/initialization receipt does not match the prepared task")
    state = read(Path(root) / "training_state.json")
    require(state.get("state") in ("starting", "running")
            and state.get("launcher_pid") == accepted.get("launcher_pid")
            and state.get("expected_ready_sha256") == plan["expected_ready_sha256"],
            "Fresh launcher state is missing or already finished; do not restart this run")


def run(root):
    root = Path(root).resolve(strict=True)
    with (root / "launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not (root / "training_state.json").exists(), "Already launched; never resume or overwrite this run")
        _, plan, _, up = load(root)
        require_cpu_check(root, plan)
        runtime = require_launch_runtime(plan)
        # Scan once in the launcher; ranks subsequently verify the exact receipt
        # and file identities without repeating a large scan eight times.
        accepted_ready = validate_ready(Path(plan["release"]), plan["expected_ready_sha256"], rehash=True)
        require(sha(plan["source_checkpoint"]) == SOURCE_SHA,
                "Joint38000 source model bytes changed before training launch")
        minimum = 13 * 12066487572 + 3 * 24000000000 + 20 * 1024 ** 3
        free = os.statvfs(root).f_bavail * os.statvfs(root).f_frsize
        require(free >= minimum, f"Insufficient independent output space: need {minimum}, available {free}")
        up.selected_gpu_idle(plan["gpus"])
        env = dict(os.environ, **TRAIN_ENV)
        code = Path(plan["frozen_code"])
        env["PYTHONPATH"] = os.pathsep.join(map(str, [code / "scripts/4090_simple", code / "VLMExpert/src", code, code / "VLMExpert"]))
        diagnostic = Path("/dev/shm") / f"mobile100-{plan['timestamp']}-{os.getpid()}"
        diagnostic.mkdir(mode=0o700)
        env["PSI0_ROLLING_DIAGNOSTICS_DIR"] = str(diagnostic)
        command = [sys.executable, str(Path(__file__).resolve())]
        naming = dict(env, WORLD_SIZE="8", RANK="0", LOCAL_RANK="0", CUDA_VISIBLE_DEVICES="")
        result = subprocess.run(command + ["name", "--prepared", str(root)], cwd=code / "VLMExpert",
                                env=naming, text=True, capture_output=True)
        (root / "run_name.log").write_text(result.stdout + result.stderr)
        require(result.returncode == 0, "Run naming failed; inspect run_name.log")
        state = {"state": "starting", "started_utc": now(), "run_directory": result.stdout.strip().splitlines()[-1],
            "launcher_pid": os.getpid(),
            "task_id": TASK_ID, "validation_role": VALIDATION_ROLE, "log": str(root / "train.log"),
            "preparation": str(root / "preparation.json"), "max_optimizer_steps": 10000,
            "gpus": list(range(8)), "per_gpu_batch": 8, "gradient_accumulation_steps": 1,
            "effective_global_batch": 64, "source_step": 38000, "fresh_run": True, "initial_global_step": 0,
            "optimizer_scheduler_rng_restored": False, "checkpoint_policy": plan["checkpoint_policy"],
            "expected_ready_sha256": plan["expected_ready_sha256"], "launch_runtime": runtime,
            "output_free_bytes_at_launch": free}
        receipt = root / "launch_acceptance.json"
        acceptance = {"contract": CONTRACT, "task_id": TASK_ID, "status": "passed", "checked_utc": now(),
            "launcher_pid": os.getpid(), "config_sha256": plan["config_sha256"],
            "expected_ready_sha256": plan["expected_ready_sha256"], "runner_sha256": plan["runner_sha256"],
            "full_payload_sha256_verified": True, "payload_file_count": len(accepted_ready["payload_files"]),
            "source_checkpoint_sha256": SOURCE_SHA, "launch_runtime": runtime, "training_started": False}
        with receipt.open("x") as stream:
            stream.write(json.dumps(acceptance, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        env["CVPR_MOBILE100_LAUNCH_RECEIPT_SHA256"] = sha(receipt)
        write(root / "training_state.json", state)
        with (root / "train.log").open("x") as log:
            try:
                process = subprocess.Popen([sys.executable, "-m", "torch.distributed.run", "--standalone",
                    "--nproc_per_node=8", str(Path(__file__).resolve()), "train", "--prepared", str(root)],
                    cwd=code / "VLMExpert", env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                state.update(state="running", pid=process.pid)
                write(root / "training_state.json", state)
                status = process.wait()
            except Exception as error:
                state.update(state="failed", error=f"{type(error).__name__}: {error}", finished_utc=now())
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
    for name in ("reference", "output"):
        prep.add_argument("--" + name, type=Path, required=True)
    prep.add_argument("--expected-ready-sha256", required=True)
    prep.add_argument("--training-cpus", required=True, help="Full ordinary SSH training affinity, e.g.0-127")
    for mode in ("check", "name", "train", "run"):
        modes.add_parser(mode).add_argument("--prepared", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args)
    else:
        return globals()[args.mode](args.prepared) or 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
