"""Prepare a local Handover task checkpoint initialized from UnifoLM 38k.

This deliberately has a narrower contract than prepare_joint_snapshot: it accepts
only the audited task50/30Hz adaptation, not arbitrary changed model metadata.
The complete task checkpoint supplies all three trained modules. Older local
resources supply immutable Qwen/DINO construction metadata only.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sys

from export_bundle import export_motion, sha256, write_json

SOURCE_STEP = 38000
TASK_CONTRACT = "cvpr-handover-task50-30hz-selection-v1"
TASK_LRS = {"lang_backbone_lr": 1e-7, "vision_tower_lr": 1e-6,
            "mm_projector_lr": 1e-5, "motion_expert_lr": 1e-6}
PRETRAIN_LRS = {"lang_backbone_lr": 1e-6, "vision_tower_lr": 1e-5,
                "mm_projector_lr": 1e-4, "motion_expert_lr": 1e-5}


def validate_contract(original, previous, statistics, previous_statistics, identity,
                      task_initialization, fused_initialization):
    step = identity.get("task_step")
    if (type(step) is not int or step <= 0 or identity.get("step") != step
            or identity.get("source_unifolm_step") != SOURCE_STEP
            or identity.get("initial_task_step") != 0
            or identity.get("save_completion_marker_verified") is not True
            or identity.get("training_modified") is not False):
        raise ValueError("Require a save-complete task checkpoint initialized at UnifoLM38k/task0")
    expected_dataset = {"contract": TASK_CONTRACT, "fps": 30, "train_episodes": 45,
                        "val_episodes": 5, "successful_teacher_episodes": 50}
    dataset = identity.get("dataset_contract")
    if (not isinstance(dataset, dict)
            or any(dataset.get(key) != value for key, value in expected_dataset.items())):
        raise ValueError("Require the accepted task50/30Hz/45-train/5-val dataset contract")
    if (task_initialization.get("source_step") != SOURCE_STEP
            or task_initialization.get("strict_joint_loader") is not True
            or task_initialization.get("fresh_optimizer_scheduler_rng") is not True
            or task_initialization.get("global_batch") != 32
            or fused_initialization.get("mode") != "fresh_fused_restart_from38000"
            or fused_initialization.get("source_step") != SOURCE_STEP
            or fused_initialization.get("initial_global_step") != 0
            or fused_initialization.get("optimizer_scheduler_rng_restored") is not False):
        raise ValueError("Task initialization evidence must distinguish source38k and fresh task0")
    model, baseline = copy.deepcopy(original["model"]), copy.deepcopy(previous["model"])
    checkpoint = model.pop("pretrained_joint_checkpoint", None)
    baseline.pop("pretrained_joint_checkpoint", None)
    expected_suffix = f"checkpoints/ckpt_{SOURCE_STEP}/model.safetensors"
    if (not isinstance(checkpoint, str) or not checkpoint.endswith("/" + expected_suffix)
            or task_initialization.get("source_checkpoint") != checkpoint):
        raise ValueError("Task initialization must load the complete UnifoLM38k checkpoint")
    for key, value in TASK_LRS.items():
        if model.get(key) != value or baseline.get(key) != PRETRAIN_LRS[key]:
            raise ValueError(f"Unexpected accepted task learning rate: {key}")
        model[key] = baseline[key]
    if model != baseline:
        raise ValueError("Source architecture/construction changed outside the accepted task LR fields")
    if ((model.get("action_architecture"), model.get("action_dim"), model.get("odim"),
         model.get("action_chunk_size")) != ("dual_source_v1", 80, 45, 30)
            or model.get("tune_vlm") is not True or model.get("tune_motion") is not True):
        raise ValueError("Require the complete VLM/Motion/action dual-source architecture")
    if statistics != previous_statistics:
        raise ValueError("Source normalization changed; task bounds must inherit UnifoLM exactly")
    data, expected = copy.deepcopy(original["data"]), copy.deepcopy(previous["data"])
    if (data.get("train_repo_ids") != ["Handover-sonic11-train"]
            or data.get("val_repo_ids") != ["Handover-sonic11-val"]
            or Path(data.get("root_dir", "")).name != "handover_task50_30hz"
            or Path(data.get("dino_cache_root", "")).name != "handover_task50_30hz"):
        raise ValueError("Only the audited Handover task50 data and DINO split are supported")
    for key in ("root_dir", "dino_cache_root", "train_repo_ids", "val_repo_ids"):
        expected[key] = data[key]
    expected["transform"]["field"]["stat_path"] = data["transform"]["field"]["stat_path"]
    if data != expected:
        raise ValueError("Task RGB/repack/history/state/action/bounds contract changed")
    repack = data["transform"]["repack"]
    if (repack.get("action_keys") != ["action.body_token_v1_1", "action[:14]"]
            or repack.get("state_keys") != ["observation.state"]
            or (repack.get("pad_action_dim"), repack.get("pad_state_dim")) != (80, 45)):
        raise ValueError("Require SONICv1.1 body tokens and the 45-dimensional state contract")
    return step


def remap_config(original, argv, replacements):
    config = copy.deepcopy(original)
    for key, value in replacements.items():
        if value is not None and any(c.isspace() for c in str(value)):
            raise ValueError("Psi0 argv paths cannot contain whitespace")
        node = config
        parts = key.split(".")
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = value
    tokens, rewritten, found, i = argv.split(), [], set(), 0
    while i < len(tokens):
        token = tokens[i]
        key = token.split("=", 1)[0].removeprefix("--").replace("-", "_")
        if token.startswith("--") and key in replacements:
            value = replacements[key]
            rewritten.append(f"--{key}={'None' if value is None else value}")
            found.add(key)
            if "=" not in token:
                i += 1
                if i >= len(tokens) or tokens[i].startswith("--"):
                    raise ValueError(f"Missing value for {token}")
        else:
            rewritten.append(token)
        i += 1
    for key, value in replacements.items():
        if key not in found:
            rewritten.append(f"--{key}={'None' if value is None else value}")
    return config, "\n".join(rewritten) + "\n"


def validate_config(config, argv_path, repo):
    sys.path[:0] = [str(repo), str(repo / "VLMExpert/src"), str(repo / "VLMExpert")]
    from psi.utils import parse_args_to_tyro_config, apply_legacy_model_config_defaults
    schema = parse_args_to_tyro_config(argv_path)
    parsed = schema.model_validate(apply_legacy_model_config_defaults(config))
    if ((parsed.model.action_architecture, parsed.model.action_dim, parsed.model.odim)
            != ("dual_source_v1", 80, 45)):
        raise ValueError("Parsed deployment architecture changed")
    for key in ("action_min", "action_max", "state_min", "state_max"):
        if getattr(parsed.data.transform.field, key) != config["data"]["transform"]["field"][key]:
            raise ValueError(f"Parsed deployment normalization changed: {key}")


def prepare(source: Path, resources: Path, output: Path, repo: Path):
    source, resources, repo = (p.resolve(strict=True) for p in (source, resources, repo))
    output = output.absolute()
    if output.exists() or output.is_symlink() or output.resolve().is_relative_to(repo):
        raise ValueError("Use a fresh output directory outside the repository")
    base = resources / "joint20k"
    read = lambda path: json.loads(path.read_text())
    identity = read(source / "SNAPSHOT_READY.json")
    original = read(source / "run_config.json")
    statistics = read(source / "dataset_statistics.json")
    task_initialization = read(source / "task_initialization.json")
    fused_initialization = read(source / "task_fused_restart_initialization.json")
    step = validate_contract(original, read(base / "provenance/run_config.json"), statistics,
                             read(base / "dataset_statistics.json"), identity,
                             task_initialization, fused_initialization)
    weights = source / "model.safetensors"
    if (weights.is_symlink() or weights.stat().st_size != identity.get("model_bytes")
            or sha256(weights) != identity.get("model_sha256")):
        raise ValueError("Transferred task model bytes differ from the protected source snapshot")
    target = output / "checkpoints" / f"ckpt_{step}" / weights.name
    replacements = {
        "model.model_name_or_path": str(base / "qwen"),
        "model.motion_checkpoint": str(output / "motion/joint_motion.ckpt"),
        # This is training initialization only; inference loads the task checkpoint.
        "model.pretrained_joint_checkpoint": None,
        "data.root_dir": str(resources / "simple_local_usd_v1"),
        "data.dino_cache_root": None,
        "data.transform.field.stat_path": str(output / "dataset_statistics.json"),
        "train.output_dir": str(output / "unused_training"),
        "train.resume_from_checkpoint": None,
        "train.deepspeed_config": str(repo / "VLMExpert/scripts/deepspeed/zero3.json"),
    }
    config, argv = remap_config(original, (source / "argv.txt").read_text(), replacements)
    contract = read(base / "dino_training_contract.json")
    # Keep the encoder's source-identity metadata; only its active load path changes.
    contract["online_checkpoint"] = str(resources / "dinov2-with-registers-large")
    for path in (base / "qwen", base / "motion/joint_motion.ckpt",
                 resources / "dinov2-with-registers-large", resources / "simple_local_usd_v1"):
        if not path.exists():
            raise ValueError(f"Required local construction resource missing: {path}")
    output.mkdir(parents=True)
    provenance = output / "provenance"
    provenance.mkdir()
    for name in ("run_config.json", "argv.txt", "dataset_statistics.json", "SNAPSHOT_READY.json",
                 "task_initialization.json", "task_fused_restart_initialization.json"):
        shutil.copy2(source / name, provenance / name)
    write_json(output / "run_config.json", config)
    (output / "argv.txt").write_text(argv)
    shutil.copy2(source / "dataset_statistics.json", output / "dataset_statistics.json")
    target.parent.mkdir(parents=True)
    # Require a local hardlink; callers choose a target on the source filesystem.
    os.link(weights, target)
    write_json(output / "dino_training_contract.json", contract)
    sys.path[:0] = [str(repo), str(repo / "VLMExpert/src"), str(repo / "VLMExpert")]
    label = f"{SOURCE_STEP}_then_task_finetune_{step}"
    motion_report = export_motion(base / "motion/joint_motion.ckpt", target,
                                  output / "motion/joint_motion.ckpt", label, contract["encoder"])
    motion_report.update(source_unifolm_step=SOURCE_STEP, task_step=step, initial_task_step=0,
                         parameter_source="new_complete_task_checkpoint")
    write_json(output / "motion/export_validation.json", motion_report)
    validate_config(config, output / "argv.txt", repo)
    report = {"status": "local_task_run_prepared", "step": step, "task_step": step,
              "step_semantics": "task_finetune_optimizer_updates",
              "source_unifolm_step": SOURCE_STEP, "initial_task_step": 0,
              "joint_sha256": identity["model_sha256"], "run_dir": str(output),
              "source_snapshot": str(source), "dataset_contract": identity["dataset_contract"],
              "motion_weights": motion_report["weights"],
              "strict_motion_load": motion_report["strict_meta_load"],
              "optimizer_rng_restored": False, "gpu_validation": "not_run",
              "physical_task_success": "not_measured", "deployment_path_changes": replacements,
              "source_metadata_sha256": {p.name: sha256(p) for p in provenance.iterdir()}}
    write_json(output / "LOCAL_RUN_READY.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.resources, args.output, args.repo), indent=2))


if __name__ == "__main__":
    main()
