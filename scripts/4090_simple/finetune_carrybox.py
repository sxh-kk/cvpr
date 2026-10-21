"""Local Action-only joint warmstart, with a fixed, independent TEST comparison.

Preparation is CPU-only. Training calls the unchanged VLMExpert train(config)
entrypoint through a launcher-local Trainer factory. TEST is observed twice:
after model/optimizer preparation before update zero, and after the final update.
It never supplies gradients, checkpoint selection, or stopping criteria.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
from datetime import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[2]
SCHEMA = "posttrain_sonic_psi0_config"
TRAIN_QUALITY_SOURCE_EPISODES = (0, 1, 2)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalization_receipt(field):
    statistics = Path(field.stat_path)
    if not statistics.is_absolute():
        raise ValueError("Normalization statistics must use an absolute local path")
    return {"stat_path": str(statistics), "stat_file_sha256": sha256(statistics),
            "bounds_sha256": {name: hashlib.sha256(json.dumps(getattr(field, name), separators=(",", ":"),
                                                              allow_nan=False).encode()).hexdigest()
                              for name in ("action_min", "action_max", "state_min", "state_max")}}


def set_nested(mapping, key, value):
    parts = key.split(".")
    for part in parts[:-1]:
        mapping = mapping[part]
    mapping[parts[-1]] = value


def flag_key(token):
    parts = token.split("=", 1)[0].removeprefix("--").replace("-", "_").split(".")
    if parts[-1].startswith("no_"):
        parts[-1] = parts[-1][3:]
    return ".".join(parts)


def rewrite_argv(original, overrides):
    """Replace entire flag groups, including multi-value and negative booleans."""
    if len(original) < 2 or original[1] != SCHEMA:
        raise ValueError(f"Expected the {SCHEMA} config schema")
    result = original[:2]
    index = 2
    while index < len(original):
        token = original[index]
        if not token.startswith("--"):
            raise ValueError(f"Unexpected positional argument in source argv: {token!r}")
        stop = index + 1
        while stop < len(original) and not original[stop].startswith("--"):
            stop += 1
        if flag_key(token) not in overrides:
            result.extend(original[index:stop])
        index = stop
    for key, value in overrides.items():
        flag = key.replace("_", "-")
        if isinstance(value, bool) and key == "train.optimizer_foreach":
            # Optional[bool] is a value option in Tyro, rather than a flag.
            result.append(f"--{flag}={value}")
        elif isinstance(value, bool):
            if not value:
                pieces = flag.split(".")
                pieces[-1] = "no-" + pieces[-1]
                flag = ".".join(pieces)
            result.append("--" + flag)
        elif isinstance(value, (list, tuple)):
            result.append("--" + flag)
            result.extend(str(item) for item in value)
        else:
            result.append(f"--{flag}={'None' if value is None else value}")
    return result


def uniform_windows(candidates, count=20, expected_episodes=5):
    """Equal deterministic coverage; candidate IDs already satisfy real history/future."""
    import numpy as np
    if len(candidates) != expected_episodes or count % expected_episodes:
        raise ValueError(f"TEST must have exactly {expected_episodes} episodes with equal window counts")
    each = count // expected_episodes
    if not 20 <= count <= 40:
        raise ValueError("TEST window count must be between 20 and 40")
    selected = []
    for episode, rows in sorted(candidates.items()):
        if len(rows) < each:
            raise ValueError(f"TEST episode {episode} has only {len(rows)} eligible windows")
        offsets = np.linspace(0, len(rows) - 1, each, dtype=np.int64)
        selected.extend((episode, int(rows[int(offset)])) for offset in offsets)
    return selected


def future_is_complete(current, stop, frames, timestamps, source_frames, source_episodes, horizon=30, *, source_frame_rule="floor"):
    """Reject LeRobot's repeated tail padding and every gap in future labels."""
    import numpy as np
    if current + horizon > stop:
        return False
    window = slice(current, current + horizon)
    expected_time = frames[window] / 30
    # Converted timestamps are float32. Adjacent subtraction doubles their
    # quantization error; compare absolute 30Hz times with one float32 ULP.
    time_tolerance = np.maximum(2e-6, np.abs(np.spacing(expected_time.astype(np.float32))))
    if source_frame_rule not in ("floor", "ceil"):
        raise ValueError("Unknown physical source-frame selection rule")
    expected_source = (frames[window] * 50 + (29 if source_frame_rule == "ceil" else 0)) // 30
    return bool(np.all(np.diff(frames[window]) == 1)
                # Original demonstrations are native50Hz; conversion selects
                # floor(n * 50 / 30). These source IDs are not DINO row ordinals.
                and np.array_equal(source_frames[window], expected_source)
                and np.all(source_episodes[window] == source_episodes[current])
                and np.isfinite(timestamps[window]).all()
                and np.all(np.abs(timestamps[window] - expected_time) <= time_tolerance))


def action_metrics(prediction, normalized_target, raw_target, denormalize):
    import numpy as np
    prediction = np.asarray(prediction, dtype=np.float32)
    normalized_target = np.asarray(normalized_target, dtype=np.float32)
    raw_target = np.asarray(raw_target, dtype=np.float32)
    if any(value.shape != (30, 80) or not np.isfinite(value).all()
           for value in (prediction, normalized_target, raw_target)):
        raise ValueError("TEST actions must be finite 30x80 tensors")
    denormalized = np.asarray(denormalize(prediction), dtype=np.float32)
    if denormalized.shape != prediction.shape or not np.isfinite(denormalized).all():
        raise ValueError("Action denormalization returned invalid values")
    result = {}
    for name, sl in (("body", slice(0, 64)), ("hand", slice(64, 78))):
        result[f"{name}_normalized_action_l1"] = float(np.abs(prediction[:, sl] - normalized_target[:, sl]).mean())
        # Compare with original labels, including any values clipped by normalization.
        result[f"{name}_denormalized_action_l1"] = float(np.abs(denormalized[:, sl] - raw_target[:, sl]).mean())
    return result


def parse_training_argv(path):
    from psi.utils import parse_args_to_tyro_config
    saved = sys.argv[:]
    try:
        return parse_args_to_tyro_config(path)
    finally:
        sys.argv = saved


def make_overrides(args, checkpoint, timestamp):
    return {
        "seed": args.seed, "exp": "carrybox_v11_action_only", "timestamp": timestamp,
        "auto_tag_run": False, "eval": False, "debug": False,
        "log.report_to": None, "wandb.id": None, "wandb.name": None, "wandb.group": None,
        "wandb.entity": None, "wandb.resume": "never", "train.hf_token": ".hf_token",
        "train.name": "posttrain", "train.output_dir": str(args.output / "training"),
        "train.resume_from_checkpoint": None, "train.lora": False,
        "train.data_parallel": "ddp", "train.mixed_precision": "bf16",
        "train.train_batch_size": args.batch_size, "train.val_batch_size": 1,
        "train.gradient_accumulation_steps": args.grad_accum,
        "train.max_training_steps": args.max_steps, "train.num_train_epochs": None,
        "train.learning_rate": args.lr, "train.warmup_ratio": None,
        "train.warmup_steps": min(args.warmup_steps, args.max_steps - 1),
        "train.num_workers": 2, "train.dataloader_start_method": "spawn",
        "train.training_safety_checks": True, "train.optimizer_foreach": False,
        "train.keep_train_batch_on_cpu": True, "train.sync_metrics_on_update_only": True,
        "train.overfit_single_batch": False, "train.max_grad_norm": 1.0,
        "train.checkpointing_steps": args.max_steps + 1, "train.early_checkpoint_steps": [],
        "train.max_checkpoints_to_keep": 1, "train.preserve_checkpoint_every": None,
        "train.validation_steps": 0,
        "data.root_dir": str(args.dataset_root), "data.train_repo_ids": ["train"],
        "data.val_repo_ids": ["val"], "data.dino_cache_root": str(args.dino_cache_root),
        "data.task_sample_weights": [],
        "model.pretrained_joint_checkpoint": str(checkpoint),
        "model.pretrained_action_header_path": None,
        "model.tune_vlm": False, "model.tune_mm_llm": False,
        "model.tune_mm_vision": False, "model.tune_mm_mlp": False, "model.tune_motion": False,
    }


def prepare(args, *, dataset_contract_path=None, episode_counts=None, experiment="carrybox_v11_action_only"):
    from psi.utils import apply_legacy_model_config_defaults
    from psi.data.lerobot.dino_cache import load_cache_manifest
    args.base_run = args.base_run.resolve(strict=True)
    for name in ("dataset_root", "dino_cache_root"):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    if not args.output.is_absolute() or args.output.resolve().is_relative_to(REPO):
        raise ValueError("--output must be an absolute directory outside the repository")
    args.output = args.output.resolve()
    if any(char.isspace() for value in (args.base_run, args.dataset_root, args.dino_cache_root, args.output)
           for char in str(value)):
        raise ValueError("Replay argv requires paths without whitespace")
    if args.max_steps < 1 or args.batch_size < 1 or args.grad_accum < 1 or args.warmup_steps < 0 or args.lr <= 0:
        raise ValueError("Steps, batches, accumulation and LR must be positive; warmup nonnegative")
    counts = episode_counts or {"train": 40, "val": 5, "test": 5}
    uniform_windows({episode: list(range(args.test_windows)) for episode in range(counts["test"])}, args.test_windows,
                    expected_episodes=counts["test"])
    ready_path = args.base_run / "LOCAL_RUN_READY.json"
    ready = json.loads(ready_path.read_text())
    if ready.get("status") != "local_joint_run_prepared" or ready.get("optimizer_rng_restored") is not False:
        raise ValueError("Source must be a verified weight-only local joint snapshot")
    checkpoint = args.base_run / "checkpoints" / f"ckpt_{ready['step']}" / "model.safetensors"
    if sha256(checkpoint) != ready["joint_sha256"]:
        raise ValueError("Source joint checkpoint SHA does not match LOCAL_RUN_READY")
    source = apply_legacy_model_config_defaults(json.loads((args.base_run / "run_config.json").read_text()))
    if (source["model"]["action_architecture"] != "dual_source_v1"
            or (source["model"]["action_chunk_size"], source["model"]["action_dim"], source["model"]["odim"]) != (30, 80, 45)):
        raise ValueError("Source joint checkpoint must use the audited 30x80 action / 45 state layout")
    if (source["data"]["transform"]["model"]["motion_image_policy"] != "vlm_color_clean_motion"
            or source["data"]["shared_image_profile"] != "256x320"):
        raise ValueError("Source must use the audited clean Motion / VLM color policy and 256x320 geometry")
    conversion_path = dataset_contract_path or args.dataset_root / "CONVERSION_READY.json"
    conversion = json.loads(conversion_path.read_text())
    if dataset_contract_path is None and (conversion.get("status") != "cpu_conversion_complete"
            or conversion.get("complete_50_episode_conversion") is not True
            or conversion.get("legacy_token_used") is not False
            or conversion.get("all_numeric_finite") is not True
            or conversion.get("all_output_rgb_verified_lossless") is not True):
        raise ValueError("Require the complete, finite, lossless fifty-episode v1.1 conversion")
    source_encoder = json.loads((args.base_run / "dino_training_contract.json").read_text())["encoder"]
    identities = {}
    split_sources = {}
    for split in ("train", "val", "test"):
        episode_file = args.dataset_root / split / "meta/episodes.jsonl"
        info = json.loads((args.dataset_root / split / "meta/info.json").read_text())
        episodes = [json.loads(line) for line in episode_file.read_text().splitlines() if line.strip()]
        expected_count = counts[split]
        if info["fps"] != 30 or len(episodes) != expected_count:
            raise ValueError("Require native30Hz train/val packs and an independent five-episode TEST pack")
        split_sources[split] = {int(item["source_episode_index"]) for item in episodes}
        if len(split_sources[split]) != len(episodes):
            raise ValueError("Every split episode must correspond to a different original demonstration")
        manifest = args.dino_cache_root / split / "manifest.json"
        if not manifest.is_file():
            raise ValueError(f"DINO manifest is missing: {manifest}")
        cache = load_cache_manifest(manifest.parent, args.dataset_root / split, expected_profile="256x320")
        for key in ("model_id", "weights_sha256", "local_file_sha256", "preprocessing", "image_process"):
            if cache["encoder"].get(key) != source_encoder.get(key):
                raise ValueError(f"DINO cache differs from the joint source encoder: {split}.{key}")
        if set(cache["episode_indices"]) != {int(item["episode_index"]) for item in episodes}:
            raise ValueError("DINO cache must cover every episode of each independent source pack")
        identities[split] = {"episodes_sha256": sha256(episode_file), "info_sha256": sha256(args.dataset_root / split / "meta/info.json"),
                             "dino_manifest_sha256": sha256(manifest), "source_episode_ids": sorted(split_sources[split]),
                             "episode_count": len(episodes)}
    if any(split_sources[a] & split_sources[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Train, val and TEST must have disjoint original source episode IDs")
    previous = args.output / "preparation.json"
    launch_args = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
                   if key not in ("train", "prepare_only")}
    if previous.is_file():
        prior = json.loads(previous.read_text())
        if prior["launcher_arguments"] != launch_args:
            raise ValueError("Existing preparation has different launcher arguments; use a fresh output")
        timestamp = prior["timestamp"]
    else:
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError("Use a fresh output directory, or reuse an identical prepare-only output")
        timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%y%m%d%H%M%S%f")
    if (args.output / "training_result.json").exists() or (args.output / "training").exists():
        raise ValueError("This output already contains a training attempt; use a fresh output")
    overrides = make_overrides(args, checkpoint, timestamp)
    overrides["exp"] = experiment
    target = copy.deepcopy(source)
    for key, value in overrides.items():
        set_nested(target, key, value)
    args.output.mkdir(parents=True, exist_ok=True)
    argv = rewrite_argv((args.base_run / "argv.txt").read_text().split(), overrides)
    argv_path = args.output / "argv.txt"
    argv_path.write_text("\n".join(argv) + "\n")
    parsed = parse_training_argv(argv_path)
    config = type(parsed).model_validate(target)
    if parsed.model_dump(mode="json") != config.model_dump(mode="json"):
        raise ValueError("Source argv / full JSON differ; refuse a non-replayable launch")
    base_config = type(parsed).model_validate(source)
    normalization = normalization_receipt(config.data.transform.field)
    if normalization != normalization_receipt(base_config.data.transform.field):
        raise ValueError("Adaptation must inherit the source's exact normalization statistics and bounds")
    write_json(args.output / "run_config.json", config.model_dump(mode="json"))
    write_json(args.output / "launcher_arguments.json", launch_args)
    provenance = args.output / "provenance"
    provenance.mkdir(exist_ok=True)
    for name in ("argv.txt", "run_config.json", "LOCAL_RUN_READY.json", "dataset_statistics.json", "dino_training_contract.json"):
        shutil.copy2(args.base_run / name, provenance / name)
    report = {"status": "cpu_config_prepared", "timestamp": timestamp, "timezone": "Asia/Shanghai",
              "launcher_arguments": launch_args, "source_checkpoint": str(checkpoint), "source_joint_sha256": ready["joint_sha256"],
              "source_step": ready["step"], "source_run_config_sha256": sha256(args.base_run / "run_config.json"),
              "source_statistics_sha256": sha256(args.base_run / "dataset_statistics.json"),
              "source_dino_contract_sha256": sha256(args.base_run / "dino_training_contract.json"),
              "conversion_ready_sha256": sha256(conversion_path),
              "state_action_contract_sha256": sha256(args.dataset_root / "STATE_ACTION_CONTRACT.json"),
              "launch_config_sha256": sha256(args.output / "run_config.json"), "splits": identities,
              "initial_global_step": 0, "optimizer_scheduler_rng_restored": False,
              "normalization": normalization,
              "test_policy": "Fixed real TEST windows before update zero and after final step only; never for selection, gradients or stopping",
              "gpu_training": "not_run", "launcher_source_sha256": sha256(__file__)}
    report["quality_gate"] = {"status": "blocked", "pack_status": "deprecated_measured_mode0_unaccepted",
                              "required_train_quality_source_episodes": list(TRAIN_QUALITY_SOURCE_EPISODES),
                              "test_episodes_used_for_quality_selection": False,
                              "all_training_episodes_require_quality_records": True,
                              "reason": "Command semantics and genuine physical task completion remain unaccepted; CPU/DINO/file checks cannot authorize updates"}
    if dataset_contract_path is not None:
        # The new task entrypoint recomputes raw evidence before calling this
        # shared config writer. This function itself never authorizes training.
        report["quality_gate"] = {"status": "task_entrypoint_must_recompute_before_training",
                                  "dataset_contract_sha256": sha256(dataset_contract_path),
                                  "training_authorized_by_config_preparation": False}
    write_json(previous, report)
    return config, argv, report


def tensor_signature(value):
    """Hash fixed input tensors without serializing images or large feature arrays."""
    import numpy as np
    import torch
    digest = hashlib.sha256()
    def add(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(str((str(tensor.dtype), tuple(tensor.shape))).encode())
            # dtype reinterpretation requires a non-scalar final dimension.
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(str((str(array.dtype), array.shape)).encode())
            digest.update(array.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item):
                digest.update(key.encode())
                add(item[key])
        elif isinstance(item, (list, tuple)):
            for part in item:
                add(part)
        else:
            digest.update(json.dumps(item, sort_keys=True, default=str).encode())
    add(value)
    return digest.hexdigest()


def component_hashes(model):
    return {name: tensor_signature(getattr(model, name).state_dict())
            for name in ("vlm_model", "motion_conditioner", "action_header")}


def seed_local_rng(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)


@contextlib.contextmanager
def isolated_rng(seed=None, device="cpu"):
    """TEST construction/evaluation cannot advance any training RNG, even on error."""
    import numpy as np
    import torch
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.device(device).type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            if seed is not None:
                seed_local_rng(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def optimizer_contract(model, optimizer):
    expected = {id(param) for param in model.action_header.parameters() if param.requires_grad}
    trainable = {id(param) for param in model.parameters() if param.requires_grad}
    grouped = [param for group in optimizer.param_groups for param in group["params"]]
    if not expected or trainable != expected or {id(param) for param in grouped} != expected or len(grouped) != len(expected):
        raise ValueError("Optimizer and all trainable parameters must be exactly ActionExpert, once each")
    if any(param.requires_grad for module in (model.vlm_model, model.motion_conditioner) for param in module.parameters()):
        raise ValueError("VLM and MotionExpert must be frozen")
    if len(optimizer.state):
        raise ValueError("Before TEST requires a fresh optimizer with no restored state")
    return {"fresh_state_entries": len(optimizer.state), "trainable_parameter_count": sum(param.numel() for param in grouped),
            "groups": [{"name": group.get("group_name", str(index)), "parameters": len(group["params"]),
                        "elements": sum(param.numel() for param in group["params"]), "lr": float(group["lr"])}
                       for index, group in enumerate(optimizer.param_groups)]}


def fixed_test_batches(trainer, args, *, episode_count=5, source_frame_rule="floor"):
    import numpy as np
    from psi.trainers.qwen3vl_mixin import PaddedCollatorForTogether
    test_config = trainer.cfg.data.model_copy(deep=True)
    test_config.train_repo_ids = ["test"]
    test_config.val_repo_ids = ["test"]
    dataset = test_config(split="val", transform_kwargs={"vlm_processor": trainer.vlm_processor, "no_aug": True})
    candidates, columns, history = eligible_dataset_rows(dataset, expected_episodes=episode_count, source_frame_rule=source_frame_rule)
    selected = uniform_windows(candidates, args.test_windows, expected_episodes=episode_count)
    collator = PaddedCollatorForTogether(model_max_length=trainer.tokenizer.model_max_length,
                                        pad_token_id=trainer.tokenizer.pad_token_id, padding_side="right")
    batches, manifest = [], []
    for ordinal, (episode, filtered) in enumerate(selected):
        original = int(history.original_indices[filtered])
        sample = dataset[filtered]
        actions = np.asarray(sample["actions"])
        mask = np.asarray(sample["actions_mask"])
        if (actions.shape != (30, 80) or mask.shape != (30, 80)
                or not np.isfinite(actions).all() or not np.all(mask[:, :78]) or np.any(mask[:, 78:])):
            raise ValueError("TEST requires all thirty future body/hand labels valid and only two padded channels")
        if tuple(sample["motion_img_embs"].shape) != (6, 5, 1024):
            raise ValueError("TEST requires six actual clean DINO feature frames")
        batch = collator([sample])
        batches.append(batch)
        start, _ = history.source_episode_ranges[0][episode]
        history_original = start + history.history_indices[filtered]
        manifest.append({"ordinal": ordinal, "episode": episode, "filtered_row": filtered, "original_row": original,
                         "frame": int(columns["frame_index"][original]), "timestamp_s": float(columns["timestamp"][original]),
                         "source_episode": int(columns["source_episode_index"][original]),
                         "source_frame": int(columns["source_frame_index"][original]),
                         "history_frames": columns["frame_index"][history_original].astype(int).tolist(),
                         "history_source_frames": columns["source_frame_index"][history_original].astype(int).tolist(),
                         "history_timestamps_s": columns["timestamp"][history_original].tolist(),
                         "last_future_frame": int(columns["frame_index"][original + 29]),
                         "fm_seed": args.seed + 10000 + ordinal * 2,
                         "inference_seed": args.seed + 10001 + ordinal * 2,
                         "input_sha256": tensor_signature(batch)})
    write_json(args.output / "test_windows.json", {"split": "test", "no_aug": True, "windows": manifest,
               "eligible_windows_per_episode": {str(key): len(value) for key, value in candidates.items()},
               "selection": "equal episode coverage and uniform positions among real complete history/future windows"})
    return batches, manifest


def eligible_dataset_rows(dataset, expected_episodes, *, source_frame_rule="floor"):
    """Metadata-only filter after the production reader's genuine-history filter."""
    import numpy as np
    raw = dataset.raw_dataset
    history = raw.motion_history
    if history is None or len(history.pack_info) != 1 or len(history.source_episode_ranges[0]) != expected_episodes:
        raise ValueError(f"Expected one history-enabled pack with {expected_episodes} episodes")
    metadata = raw.base_dataset.hf_dataset.with_format("numpy")
    columns = {name: np.asarray(metadata[name]).reshape(-1) for name in
               ("frame_index", "timestamp", "source_frame_index", "source_episode_index")}
    candidates = {episode: [] for episode in history.source_episode_ranges[0]}
    for filtered, original in enumerate(history.original_indices):
        episode = int(history.episode_ids[filtered])
        _, stop = history.source_episode_ranges[0][episode]
        if future_is_complete(int(original), stop, columns["frame_index"], columns["timestamp"],
                              columns["source_frame_index"], columns["source_episode_index"], source_frame_rule=source_frame_rule):
            candidates[episode].append(filtered)
    if not all(candidates.values()):
        raise ValueError("Every source episode must retain real complete history/future windows")
    return candidates, columns, history


def evaluate_fixed_test(trainer, batches, manifest, stage, args):
    with isolated_rng(device=trainer.device):
        return _evaluate_fixed_test(trainer, batches, manifest, stage, args)


def _evaluate_fixed_test(trainer, batches, manifest, stage, args):
    import numpy as np
    import torch
    from psi.utils import batch_str_to_tensor, move_to_device
    started = time.monotonic()
    model = trainer.unwrap_model()
    was_training = model.training
    model.eval()
    rows, predictions = [], []
    # Inference mutates scheduler.timesteps: restore it so TEST cannot alter training state.
    saved_scheduler = copy.deepcopy(trainer.noise_scheduler)
    try:
        for cpu_batch, identity in zip(batches, manifest, strict=True):
            if tensor_signature(cpu_batch) != identity["input_sha256"]:
                raise RuntimeError("Fixed TEST inputs were mutated")
            batch = move_to_device(batch_str_to_tensor(copy.deepcopy(cpu_batch)), trainer.device)
            with torch.no_grad(), isolated_rng(identity["fm_seed"], trainer.device):
                with trainer.forward_autocast():
                    fm_loss = float(trainer.forward_and_loss(model, batch)["loss"].detach().float().cpu())
                seed_local_rng(identity["inference_seed"])
                with trainer.forward_autocast():
                    prediction = trainer.inference(model, batch).detach().float().cpu().numpy()[0]
            if not math.isfinite(fm_loss):
                raise RuntimeError("TEST flow-matching loss is not finite")
            target = cpu_batch["actions"][0].detach().cpu().numpy()
            raw_target = np.asarray(cpu_batch["raw_actions"][0])
            row = {"ordinal": identity["ordinal"], "episode": identity["episode"], "fm_loss": fm_loss,
                   **action_metrics(prediction, target, raw_target, trainer.maxmin.denormalize)}
            rows.append(row)
            predictions.append(prediction)
            del batch
    finally:
        trainer.noise_scheduler = saved_scheduler
        model.train(was_training)
    keys = ("fm_loss", "body_normalized_action_l1", "hand_normalized_action_l1",
            "body_denormalized_action_l1", "hand_denormalized_action_l1")
    report = {"stage": stage, "optimizer_updates_completed": trainer.completed_updates,
              "window_count": len(rows), "wall_time_s": time.monotonic() - started,
              "mean": {key: float(np.mean([row[key] for row in rows])) for key in keys},
              "per_episode": {str(episode): {key: float(np.mean([row[key] for row in rows if row["episode"] == episode]))
                                              for key in keys} for episode in sorted({row["episode"] for row in rows})},
              "windows": rows, "protocol": "same clean real inputs, private fixed RNG seeds, no_grad + AMP; offline action error, not physical task success"}
    np.savez_compressed(args.output / f"test_{stage}_actions.npz", predictions=np.stack(predictions),
                        normalized_targets=np.stack([batch["actions"][0].numpy() for batch in batches]),
                        raw_targets=np.stack([np.asarray(batch["raw_actions"][0]) for batch in batches]))
    write_json(args.output / f"test_{stage}.json", report)
    return report


class ComparisonLifecycle:
    """Small explicit state machine, independently covered by CPU tests."""
    def __init__(self, maximum):
        self.maximum = maximum
        self.completed = 0
        self.before_done = False
        self.after_done = False

    def before(self, callback):
        if self.before_done or self.completed:
            raise RuntimeError("Before TEST must run once before any optimizer update")
        result = callback()
        self.before_done = True
        return result

    def update(self, global_step, synced):
        if not self.before_done or self.after_done or global_step != self.completed:
            raise RuntimeError("Training step occurred outside the fixed before/after protocol")
        if synced:
            self.completed += 1

    def after(self, callback):
        if not self.before_done or self.after_done or self.completed != self.maximum:
            raise RuntimeError("After TEST requires all configured optimizer updates")
        result = callback()
        self.after_done = True
        return result


def train_local(config, argv, args, preparation):
    require_training_quality()
    return _train_with_comparison(config, argv, args, preparation)


def _train_with_comparison(config, argv, args, preparation, *, train_episode_count=40, test_episode_count=5,
                           source_frame_rule="floor"):
    """Shared local mechanics; task-specific entrypoints must verify their own raw data first.

    The original CarryBox CLI always calls its unconditional blocker before
    reaching this function. No production training entrypoint or registry changes.
    """
    import torch
    from psi.trainers.posttrain import PosttrainTrainer
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This local launcher requires exactly one GPU process")
    result = {"status": "running", "source_joint_sha256": preparation["source_joint_sha256"],
              "maximum_updates": args.max_steps, "completed_updates": 0, "comparison_complete": False}
    write_json(args.output / "training_result.json", result)
    started = time.monotonic()
    instance = None

    class CarryBoxTrainer(PosttrainTrainer):
        def __init__(self, cfg, device):
            super().__init__(cfg, device)
            self.lifecycle = ComparisonLifecycle(args.max_steps)
            self.completed_updates = 0

        def create_datasets(self):
            train_dataset, val_dataset = super().create_datasets()
            candidates, columns, history = eligible_dataset_rows(train_dataset, expected_episodes=train_episode_count,
                                                                 source_frame_rule=source_frame_rule)
            indices = [index for episode in sorted(candidates) for index in candidates[episode]]
            rows = []
            for episode, eligible in sorted(candidates.items()):
                start, stop = history.source_episode_ranges[0][episode]
                filtered_count = int((history.episode_ids == episode).sum())
                rows.append({"episode": episode, "source_episode": int(columns["source_episode_index"][start]),
                             "source_rows": stop - start, "with_real_history": filtered_count,
                             "retained_complete_windows": len(eligible),
                             "excluded_after_history_filter": filtered_count - len(eligible),
                             "first_frame": int(columns["frame_index"][history.original_indices[eligible[0]]]),
                             "last_frame": int(columns["frame_index"][history.original_indices[eligible[-1]]])})
            write_json(args.output / "train_window_filter.json", {"source_rows_after_history": len(train_dataset),
                       "retained_windows": len(indices), "excluded_after_history_filter": len(train_dataset) - len(indices),
                       "policy": "Production real one-second history plus complete thirty-frame real future; no LeRobot tail padding",
                       "episodes": rows})
            self.train_dataset = torch.utils.data.Subset(train_dataset, indices)
            return self.train_dataset, val_dataset

        def prepare(self, accelerator):
            loader = super().prepare(accelerator)
            if accelerator.num_processes != 1:
                raise ValueError("TEST comparison is single GPU only")
            model = self.unwrap_model()
            result["optimizer"] = optimizer_contract(model, self.optimizer)
            result["scheduler_initial_last_epoch"] = int(self.lr_scheduler.state_dict().get("last_epoch", -1))
            if result["scheduler_initial_last_epoch"] not in (-1, 0):
                raise RuntimeError("Weight-only warmstart requires a fresh scheduler")
            if self.resume_from_checkpoint()[0] != 0:
                raise RuntimeError("Weight-only warmstart must start at global step zero")
            result["initial_global_step"] = 0
            result["normalization"] = normalization_receipt(self.cfg.data.transform.field)
            if result["normalization"] != preparation["normalization"]:
                raise RuntimeError("Normalization statistics changed after preparation")
            result["project_dir"] = self.project_dir
            shutil.copy2(args.base_run / "dino_training_contract.json", Path(self.project_dir) / "dino_training_contract.json")
            write_json(Path(self.project_dir) / "local_adaptation_provenance.json", preparation)
            if (args.output / "source_at_launch").is_dir():
                shutil.copytree(args.output / "source_at_launch", Path(self.project_dir) / "source_at_launch")
            self.initial_hashes = component_hashes(model)
            result["before_component_sha256"] = self.initial_hashes
            with isolated_rng(args.seed + 9000, self.device):
                self.test_batches, self.test_manifest = fixed_test_batches(self, args, episode_count=test_episode_count,
                                                                          source_frame_rule=source_frame_rule)
            torch.cuda.reset_peak_memory_stats(self.device)
            self.lifecycle.before(lambda: evaluate_fixed_test(self, self.test_batches, self.test_manifest, "before", args))
            result["before_comparison_complete"] = True
            self.record_peak()
            write_json(args.output / "training_result.json", result)
            return loader

        def step(self, batch_input, global_step, local_step):
            if not self.lifecycle.before_done or self.lifecycle.after_done or global_step != self.lifecycle.completed:
                raise RuntimeError("Before TEST must precede the first microbatch")
            synced, metrics = super().step(batch_input, global_step, local_step)
            self.lifecycle.update(global_step, synced)
            self.completed_updates = self.lifecycle.completed
            if synced:
                result["completed_updates"] = self.completed_updates
                result["last_training_metrics"] = {key: float(value) for key, value in metrics.items()
                                                   if isinstance(value, (int, float)) and math.isfinite(float(value))}
                self.record_peak()
            return synced, metrics

        def record_peak(self):
            result["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(self.device)
            result["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(self.device)
            result["cuda_peak_scope"] = "Since prepare: before TEST, training, after TEST; last successful API read"

        def finalize(self):
            self.lifecycle.after(lambda: evaluate_fixed_test(self, self.test_batches, self.test_manifest, "after", args))
            final_hashes = component_hashes(self.unwrap_model())
            result["after_component_sha256"] = final_hashes
            result["frozen_unchanged"] = all(self.initial_hashes[key] == final_hashes[key]
                                             for key in ("vlm_model", "motion_conditioner"))
            result["action_changed"] = self.initial_hashes["action_header"] != final_hashes["action_header"]
            if not result["frozen_unchanged"] or not result["action_changed"]:
                raise RuntimeError("Frozen component equality / ActionExpert update evidence failed")
            checkpoint_root = Path(self.project_dir) / "checkpoints"
            checkpoints = sorted(checkpoint_root.glob("ckpt_*"))
            checkpoint = checkpoint_root / f"ckpt_{args.max_steps}" / "model.safetensors"
            if checkpoints != [checkpoint.parent] or not checkpoint.is_file():
                raise RuntimeError("Require exactly one saved final checkpoint")
            result["final_checkpoint"] = str(checkpoint)
            result["final_checkpoint_sha256"] = sha256(checkpoint)
            self.record_peak()
            result["comparison_complete"] = True
            super().finalize()

    class LocalTrainerFactory:
        @staticmethod
        def instantiate(cfg, device_id):
            nonlocal instance
            instance = CarryBoxTrainer(cfg, torch.device(f"cuda:{device_id}"))
            return instance

    saved_argv = sys.argv[:]
    saved_environment_info = os.environ.pop("_ENV_INFO_JSON", None)
    try:
        spec = importlib.util.spec_from_file_location("carrybox_local_training_entrypoint", REPO / "VLMExpert/scripts/train.py")
        entrypoint = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entrypoint)
        # Only this module's factory changes. A100's entrypoint/registry stay unchanged.
        entrypoint.Trainer = LocalTrainerFactory
        sys.argv = argv
        entrypoint.train(config)
        if (instance is None or not instance.lifecycle.after_done or instance.completed_updates != args.max_steps
                or not result["comparison_complete"]):
            raise RuntimeError("Training returned without complete update/checkpoint/TEST evidence")
        before = json.loads((args.output / "test_before.json").read_text())
        after = json.loads((args.output / "test_after.json").read_text())
        result["metric_change_after_minus_before"] = {key: after["mean"][key] - before["mean"][key] for key in before["mean"]}
        result["status"] = "completed"
        result["interpretation"] = "Fixed TEST offline before/after only; no claim of SIMPLE closed-loop task success"
    except BaseException as error:
        result["status"] = "failed"
        result["exception"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        sys.argv = saved_argv
        if saved_environment_info is not None:
            os.environ["_ENV_INFO_JSON"] = saved_environment_info
        result["wall_time_s"] = time.monotonic() - started
        if instance is not None:
            result["completed_updates"] = instance.completed_updates
        write_json(args.output / "training_result.json", result)


def require_training_quality():
    """Fail closed until the command semantics and physical replay are accepted.

    The current measured-trajectory mode0 conversion is not established as the
    original teleoperation command, and its oracle replay has not succeeded.
    A hand-written success receipt or an artifact/shape check cannot unlock SFT.
    Replace this blocker only with an independently reviewed verifier that
    recomputes source/controller bindings and genuine physical task completion.
    """
    raise RuntimeError("CarryBox training is blocked: the deprecated measured-mode0 pack has no accepted command semantics or successful physical task replay. "
                       "Use --prepare-only for CPU checks; no status receipt or force override can authorize updates.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--dino-cache-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--test-windows", type=int, default=20)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--train", action="store_true")
    args = parser.parse_args()
    if args.train:
        require_training_quality()
    sys.path[:0] = [str(REPO / "VLMExpert/src"), str(REPO)]
    config, argv, preparation = prepare(args)
    if args.train:
        train_local(config, argv, args, preparation)
    print(json.dumps({"status": "completed" if args.train else "cpu_config_prepared", "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
