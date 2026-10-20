"""CPU-only, save-completion-gated export for the independent 4090 evaluation.

Run on the training host. Only resources/ is transferred; protected_checkpoint/
retains the full source save locally and must not be sent to the inference host.
MODEL_RESOURCES_READY does not imply SONIC/SIMPLE or rollout acceptance.
"""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from datetime import datetime, timezone


REPO = Path(__file__).resolve().parents[2]
MOTION_PREFIX = "motion_conditioner.model."


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_bundle(root):
    """Verify transferred bytes; this never substitutes for GPU/rollout checks."""
    root = root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["format"] != "cvpr-independent-4090-resources-v1" or manifest["step"] != 20000:
        raise ValueError("Unexpected resource bundle identity")
    expected_marker = (f"step=20000\njoint_sha256={manifest['joint_sha256']}\n"
                       "evaluation_ready=false\n")
    if (root / "MODEL_RESOURCES_READY").read_text() != expected_marker:
        raise ValueError("READY marker and manifest disagree")
    paths = set()
    for item in manifest["files"]:
        relative = Path(item["path"])
        path = root / relative
        if relative.is_absolute() or ".." in relative.parts or item["path"] in paths:
            raise ValueError(f"Unsafe or duplicate manifest path: {relative}")
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Nonportable resource path: {relative}")
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"Resource content mismatch: {relative}")
        paths.add(item["path"])
    model = "joint20k/checkpoints/ckpt_20000/model.safetensors"
    joint = next((item for item in manifest["files"] if item["path"] == model), None)
    if joint is None or joint["sha256"] != manifest["joint_sha256"]:
        raise ValueError("Joint model identity missing or inconsistent")
    return {"state": "byte_verification_passed", "step": 20000,
            "joint_sha256": manifest["joint_sha256"], "files": len(paths),
            "gpu_model_validation": "not_run", "evaluation_ready": False}


def link_or_copy(source, target):
    """Link immutable weights, without changing source contents or permissions."""
    # HF snapshots can contain symlinks into an A100-only blob directory.
    source = source.resolve(strict=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def canonical_visual_identity(contract):
    """UnifoLM adds two redundant labels absent from the older Motion schema.

    Derive them from the original file hashes, never discard identity checks.
    The original metadata remains unchanged in the exported checkpoint.
    """
    value = copy.deepcopy(contract)
    encoder = value["encoder"]
    local_files = encoder["local_file_sha256"]
    digest = hashlib.sha256(json.dumps(local_files, sort_keys=True, separators=(",", ":"),
                                       allow_nan=False).encode()).hexdigest()
    expected = {"model_id": "facebook/dinov2-with-registers-large", "weights_sha256": digest}
    for key, item in expected.items():
        if key in encoder and encoder[key] != item:
            raise ValueError(f"Invalid redundant DINO identity: {key}")
        encoder[key] = item
    return value


def checkpoint_complete(run, log, step, ranks):
    checkpoint = run / "checkpoints" / f"ckpt_{step}"
    names = ["model.safetensors", "optimizer.bin", "scheduler.bin"]
    names += [f"random_states_{rank}.pkl" for rank in range(ranks)]
    if not all((checkpoint / name).is_file() and (checkpoint / name).stat().st_size
               for name in names):
        return False
    # The directory and model file appear BEFORE optimizer/RNG saving finishes.
    marker = f"Saved state to {checkpoint}"
    with log.open("r", errors="replace") as stream:
        return any(line.strip() == marker for line in stream)


def protect_checkpoint(source, destination):
    """Publish an independent directory after the full save completion marker."""
    partial = destination.with_name(destination.name + ".partial")
    if destination.exists() or partial.exists():
        raise FileExistsError(destination)
    partial.mkdir(parents=True)
    for path in sorted(source.iterdir()):
        if not path.is_file():
            raise ValueError(f"Unexpected nested checkpoint item: {path}")
        link_or_copy(path, partial / path.name)
    partial.rename(destination)


def deployment_config(original, original_argv, target, target_repo):
    """Preserve all model/normalization values and remap deployment paths only."""
    config = copy.deepcopy(original)
    replacements = {
        "model.model_name_or_path": str(target / "joint20k/qwen"),
        "model.motion_checkpoint": str(target / "joint20k/motion/joint_motion.ckpt"),
        "data.root_dir": str(target / "simple"),
        "data.dino_cache_root": None,
        "data.transform.field.stat_path": str(target / "joint20k/dataset_statistics.json"),
        "train.output_dir": str(target.parent / "reports"),
        "train.resume_from_checkpoint": None,
        "train.deepspeed_config": str(target_repo / "VLMExpert/scripts/deepspeed/zero3.json"),
    }
    changes = []
    for dotted, value in replacements.items():
        if value is not None and any(c.isspace() for c in str(value)):
            raise ValueError("Current Psi0 argv parser cannot support paths containing whitespace")
        node = config
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node[part]
        before = node[parts[-1]]
        node[parts[-1]] = value
        changes.append({"field": dotted, "source": before, "deployment": value})
    tokens = original_argv.split()
    rewritten = []
    found = set()
    i = 0
    while i < len(tokens):
        token = tokens[i]
        key = token.split("=", 1)[0].removeprefix("--").replace("-", "_")
        if token.startswith("--") and key in replacements:
            value = replacements[key]
            rendered = "None" if value is None else str(value)
            if any(c.isspace() for c in rendered):
                raise ValueError("Current Psi0 argv parser cannot support paths containing whitespace")
            rewritten.append(f"--{key}={rendered}")
            found.add(key)
            if "=" not in token:
                i += 1
                if i >= len(tokens) or tokens[i].startswith("--"):
                    raise ValueError(f"Missing path argument for {token}")
        else:
            rewritten.append(token)
        i += 1
    for key in replacements:
        if key in found:
            continue
        value = replacements[key]
        rewritten.append(f"--{key}={'None' if value is None else value}")
    return config, "\n".join(rewritten) + "\n", changes


def prepare_static(args, resources):
    run = args.run_dir
    original = json.loads((run / "run_config.json").read_text())
    joint = resources / "joint20k"
    joint.mkdir(parents=True)
    provenance = joint / "provenance"
    provenance.mkdir()
    for name in ("run_config.json", "argv.txt", "dataset_statistics.json"):
        shutil.copy2(run / name, provenance / name)
    shutil.copy2(__file__, provenance / "export_bundle.py")
    statistics = run / "dataset_statistics.json"
    raw_stats = Path(original["data"]["transform"]["field"]["stat_path"])
    if json.loads(statistics.read_text()) != json.loads(raw_stats.read_text()):
        raise ValueError("Run statistics and training normalization source differ")
    shutil.copy2(statistics, joint / "dataset_statistics.json")
    config, argv, changes = deployment_config(
        original, (run / "argv.txt").read_text(), args.target_resources, args.target_repo)
    write_json(joint / "run_config.json", config)
    (joint / "argv.txt").write_text(argv)
    write_json(provenance / "deployment_path_changes.json", changes)
    qwen = Path(original["model"]["model_name_or_path"])
    qwen_target = joint / "qwen"
    qwen_target.mkdir()
    for path in sorted(qwen.iterdir()):
        if path.is_file() and path.suffix in {".json", ".txt", ".jinja", ".model"}:
            shutil.copy2(path, qwen_target / path.name)
    for required in ("config.json", "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json"):
        if not (qwen_target / required).is_file():
            raise ValueError(f"Missing Qwen construction resource: {required}")
    cache = Path(original["data"]["dino_cache_root"])
    train_id = original["data"]["train_repo_ids"][0]
    cache_manifest = json.loads((cache / train_id / "manifest.json").read_text())
    encoder = cache_manifest["encoder"]
    dino_target = resources / "dinov2-with-registers-large"
    dino_target.mkdir()
    for relative, expected_hash in encoder["local_file_sha256"].items():
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Unsafe DINO manifest path: {relative}")
        source = args.dino_dir / relative_path
        if sha256(source) != expected_hash:
            raise ValueError(f"DINO identity differs from training cache: {source}")
        link_or_copy(source, dino_target / relative_path)
    write_json(joint / "dino_training_contract.json", {
        "encoder": encoder,
        "online_checkpoint": str(args.target_resources / "dinov2-with-registers-large"),
        "shared_image_profile": original["data"]["shared_image_profile"],
        "history": {"fps": 5, "frames": 6, "shape": [6, 5, 1024],
                    "augmentation": "none", "timestamps": "simulation_time"},
        "processor_overrides_required": {
            "use_fast": False, "do_resize": False, "do_center_crop": False},
        "boundary": "preserve training encoder identity; online DINO/rollout acceptance pending",
    })
    return original


def export_motion(source_metadata, joint_weights, output, step, actual_encoder):
    import torch
    from safetensors import safe_open
    from MotionExpert.checkpoints import read_checkpoint, checkpoint_config, validate_checkpoint_statistics
    from MotionExpert.config import as_dict, architecture_metadata
    from MotionExpert.dataset.normalization import MotionNormalizer
    from MotionExpert.model import UniEgoMotion
    from MotionExpert.visual_contract import checkpoint_visual_contract, compare_visual_contracts
    from psi.models.checkpoint_aliases import restore_shared_tensor_aliases

    torch.set_num_threads(2)
    checkpoint = read_checkpoint(source_metadata)
    cfg = checkpoint_config(checkpoint)
    with torch.device("meta"):
        model = UniEgoMotion(cfg)
    with safe_open(joint_weights, framework="pt", device="cpu") as joint:
        state = {name.removeprefix(MOTION_PREFIX): joint.get_tensor(name)
                 for name in joint.keys() if name.startswith(MOTION_PREFIX)}
    if not state:
        raise ValueError("No Motion weights in the joint checkpoint")
    original_keys = set(state)
    state = restore_shared_tensor_aliases(model, state)
    model.load_state_dict(state, strict=True, assign=True)
    if any(not torch.isfinite(tensor).all().item() for tensor in state.values()):
        raise ValueError("Nonfinite Motion weight")
    stats = dict(checkpoint["motionexpert_normalization"])
    stats["normalization_clip"] = checkpoint.get(
        "motionexpert_normalization_clip", stats.get("normalization_clip"))
    normalizer = MotionNormalizer(stats)
    validate_checkpoint_statistics(cfg, checkpoint, normalizer)
    trained_visual = checkpoint_visual_contract(checkpoint)
    actual_visual = {
        "format": "motionexpert-visual-v1", "mode": "shared_rgb",
        "image_process": actual_encoder["image_process"], "encoder": actual_encoder,
    }
    trained_comparison = canonical_visual_identity(trained_visual)
    actual_comparison = canonical_visual_identity(actual_visual)
    runtime_transition = {}
    # The joint model has already trained on the actual post-training runtime.
    # Record this separately: library versions can affect pixels, so this is
    # broader than a size-only adaptation of the original Motion checkpoint.
    for key in ("numpy_version", "pillow_version"):
        runtime_transition[key] = {"motion_pretrain": trained_visual["image_process"][key],
                                   "joint_posttrain": actual_visual["image_process"][key]}
        for contract in (trained_comparison, actual_comparison):
            contract["image_process"].pop(key)
            contract["encoder"]["image_process"].pop(key, None)
    visual_adaptation = compare_visual_contracts(trained_comparison, actual_comparison,
                                                allow_visual_adaptation=True)
    visual_adaptation.update(scope="geometry_and_recorded_preprocessing_runtime_transition",
                             preprocessing_runtime_transition=runtime_transition,
                             deployment_runtime="match joint_posttrain versions, not original pretrain")
    visual_adaptation["redundant_identity_fields_derived_from_file_hashes"] = ["model_id", "weights_sha256"]
    payload = {
        "format": "motionexpert_weights_smplh_v1", "state_dict": state,
        "hyper_parameters": as_dict(cfg), "motionexpert_architecture": architecture_metadata(cfg),
        "motionexpert_normalization": normalizer.stats,
        "motionexpert_normalization_clip": normalizer.clip,
        "motionexpert_visual_conditioning": trained_visual,
        "weights": f"joint_posttrain_step_{step}",
    }
    metadata_json = json.dumps({key: value for key, value in payload.items() if key != "state_dict"},
                               sort_keys=True, default=lambda value: value.tolist())
    output.parent.mkdir(parents=True)
    torch.save(payload, output)
    exported = read_checkpoint(output)
    checkpoint_config(exported)
    checkpoint_visual_contract(exported)
    validate_checkpoint_statistics(cfg, exported, normalizer)
    # Verify every serialized tensor, including restored shared aliases.
    for name, value in state.items():
        restored = exported["state_dict"][name]
        if value.dtype != restored.dtype or not torch.equal(value, restored):
            raise ValueError(f"Motion serialization changed tensor: {name}")
    return {"joint_tensor_count": len(original_keys), "strict_tensor_count": len(state),
            "restored_shared_aliases": sorted(set(state) - original_keys),
            "strict_meta_load": "passed", "exact_serialization_comparison": "passed",
            "normalization_and_visual_contract": "passed",
            "joint_posttrain_visual_adaptation": visual_adaptation,
            "weights": payload["weights"], "ema_optimizer_included": False,
            "metadata_source": str(source_metadata),
            "metadata_source_bytes": source_metadata.stat().st_size,
            "metadata_source_mtime_ns": source_metadata.stat().st_mtime_ns,
            "exported_metadata_sha256": hashlib.sha256(metadata_json.encode()).hexdigest()}


def validate_config_cpu(joint):
    from psi.config.train.posttrain_sonic_psi0_config import DynamicLaunchConfig
    from psi.utils.utils import parse_args_to_tyro_config
    config = json.loads((joint / "run_config.json").read_text())
    parsed = parse_args_to_tyro_config(joint / "argv.txt")
    result = parsed.model_validate(config)
    direct = DynamicLaunchConfig.model_validate(config)
    # The deployment stat path is on the other host; serialized bounds must survive.
    for key in ("action_min", "action_max", "state_min", "state_max"):
        if getattr(result.data.transform.field, key) != config["data"]["transform"]["field"][key]:
            raise ValueError(f"Normalization changed while loading deployment config: {key}")
    if result.model.model_dump() != direct.model.model_dump():
        raise ValueError("argv and JSON config load paths disagree")


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--verify":
        print(json.dumps(verify_bundle(Path(sys.argv[2])), indent=2))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--training-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-resources", type=Path, required=True)
    parser.add_argument("--target-repo", type=Path, required=True)
    parser.add_argument("--dino-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, default=20000, choices=[20000])
    parser.add_argument("--ranks", type=int, default=8)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=20)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Use a fresh output directory: {args.output}")
    args.output.mkdir(parents=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path[:0] = [str(REPO), str(REPO / "VLMExpert/src")]
    status = {"state": "preparing_static_resources", "step": args.step,
              "source_run": str(args.run_dir), "pid": os.getpid(),
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "gpu_used": False, "evaluation_ready": False}
    status_path = args.output / "status.json"
    write_json(status_path, status)
    partial = args.output / "resources.partial"
    try:
        original = prepare_static(args, partial)
        validate_config_cpu(partial / "joint20k")
        status["state"] = "waiting_for_complete_checkpoint"
        write_json(status_path, status)
        while not checkpoint_complete(args.run_dir, args.training_log, args.step, args.ranks):
            if not args.wait:
                raise RuntimeError("Requested checkpoint has not finished saving")
            time.sleep(args.poll_seconds)
        source = args.run_dir / "checkpoints" / f"ckpt_{args.step}"
        protected = args.output / "protected_checkpoint" / f"ckpt_{args.step}"
        protect_checkpoint(source, protected)
        status.update(state="exporting_protected_checkpoint", protected_checkpoint=str(protected))
        write_json(status_path, status)
        joint = partial / "joint20k"
        weights = protected / "model.safetensors"
        target_weights = joint / "checkpoints" / f"ckpt_{args.step}" / "model.safetensors"
        link_or_copy(weights, target_weights)
        actual_encoder = json.loads((joint / "dino_training_contract.json").read_text())["encoder"]
        motion_report = export_motion(Path(original["model"]["motion_checkpoint"]), weights,
                                      joint / "motion/joint_motion.ckpt", args.step, actual_encoder)
        write_json(joint / "motion/export_validation.json", motion_report)
        inventory = []
        for path in sorted(partial.rglob("*")):
            if path.is_file():
                inventory.append({"path": str(path.relative_to(partial)),
                                  "bytes": path.stat().st_size, "sha256": sha256(path)})
        source_entry = next(item for item in inventory
                            if item["path"] == str(target_weights.relative_to(partial)))
        manifest = {"format": "cvpr-independent-4090-resources-v1", "step": args.step,
                    "source_run": str(args.run_dir), "joint_sha256": source_entry["sha256"],
                    "target_resources": str(args.target_resources), "files": inventory,
                    "motion_validation": motion_report, "cpu_config_validation": "passed",
                    "gpu_model_validation": "pending_on_4090", "evaluation_ready": False,
                    "missing_resources": ["matched SONIC body_token_v1_1 controller/decoder identity",
                                          "SIMPLE task/USD/texture/initial-condition assets"],
                    "pending_integration": ["online real 6-frame history and session reset",
                                            "43-state mapping and 64-latent + 14-hand controller mapping",
                                            "strict GPU load, closed-loop, memory and latency acceptance"]}
        write_json(partial / "manifest.json", manifest)
        (partial / "MODEL_RESOURCES_READY").write_text(
            f"step={args.step}\njoint_sha256={source_entry['sha256']}\nevaluation_ready=false\n")
        partial.rename(args.output / "resources")
        status.update(state="model_resources_ready", evaluation_ready=False,
                      resources=str(args.output / "resources"), joint_sha256=source_entry["sha256"],
                      finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(status_path, status)
        print(json.dumps(status, ensure_ascii=False), flush=True)
    except Exception as error:
        status.update(state="failed", error=f"{type(error).__name__}: {error}")
        write_json(status_path, status)
        raise


if __name__ == "__main__":
    main()
