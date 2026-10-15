"""Check only the new window-cache path against the existing accepted reader."""
import argparse
from copy import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import as_dict, load_config, validate_architecture
from .dataset.data_module import cache_training_collate
from .dataset.smplh_motion_dataset import SMPLHMotionDataset
from .watch_dino import atomic_json, now, process_alive, process_identity

FORMAT = "motionexpert-window-cache-acceptance-v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_from_config(cfg, split, *, cache=True, episode_indices=None, verified_archive=None):
    d = cfg.DATA
    if verified_archive is not None and getattr(verified_archive, "model_sha256", None) != d.MODEL_SHA256:
        raise ValueError("Verified archive model differs from the frozen configuration")
    return SMPLHMotionDataset(
        archive_root=d.ARCHIVE_ROOT, split=split, split_manifest=d.SPLIT_MANIFEST,
        stats_path=d.STATS_PATH, do_normalization=True,
        cond_img_feat=d.COND_IMG_FEAT, cond_traj=d.COND_TRAJ, cond_betas=d.COND_BETAS,
        window=d.WINDOW, fps=d.FPS, forecast_prefix=d.FORECAST_PREFIX, segment_stride=d.SEGMENT_STRIDE,
        sparse_joint_indices=cfg.SPARSE_JOINTS.INDICES, repre_type=d.REPRE_TYPE, img_feat_type=d.IMG_FEAT_TYPE,
        row_group_cache_mb=d.ROW_GROUP_CACHE_MB, parquet_file_cache=d.PARQUET_FILE_CACHE,
        rotation_basis_path=d.ROTATION_BASIS_PATH, rotation_basis_mode=d.ROTATION_BASIS_MODE,
        target_joint_basis=d.TARGET_JOINT_BASIS, require_verified_rotations=d.REQUIRE_VERIFIED_ROTATIONS,
        normalization_clip=d.NORMALIZATION_CLIP, contact_speed_m_s=d.CONTACT_SPEED_M_S,
        contact_height_m=d.CONTACT_HEIGHT_M, min_valid_image_fraction=d.MIN_VALID_IMAGE_FRACTION,
        train_task=d.TRAIN_TASK, require_full_window=d.REQUIRE_FULL_WINDOW,
        motion_supervision=d.MOTION_SUPERVISION, vision_profile_root=d.VISION_PROFILE_ROOT,
        image_profile=d.IMAGE_PROFILE, window_cache_root=d.WINDOW_CACHE_ROOT if cache else None,
        episode_indices=episode_indices, verified_archive=verified_archive,
    )


def validate_config_change(source_cfg, resume_cfg):
    before, after = as_dict(source_cfg), as_dict(resume_cfg)
    before["DATA"].pop("WINDOW_CACHE_ROOT", None)
    after["DATA"].pop("WINDOW_CACHE_ROOT", None)
    if before != after:
        raise ValueError("The cache transition may only change DATA.WINDOW_CACHE_ROOT")
    if not resume_cfg.DATA.WINDOW_CACHE_ROOT:
        raise ValueError("A completed window cache must be explicitly configured")


def compare_sample(cached, original):
    """Masks/images and normalized targets must be exactly the same."""
    info = original["misc"]
    context = (f"episode={info.get('episode_index')} start={info.get('start_idx')} "
               f"source={info.get('source')}")
    def compare_tensor(actual, expected, field):
        try:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        except AssertionError as error:
            raise AssertionError(f"Cache {field} differs ({context}): {error}") from error

    for key in ("motion", "traj", "feature_valid_mask", "trajectory_valid",
                "world_from_window", "floor_transform"):
        compare_tensor(cached["misc"][key], original["misc"][key], f"misc.{key}")
    for key, value in original["y"].items():
        actual = cached["y"][key]
        if isinstance(value, torch.Tensor):
            compare_tensor(actual, value, f"y.{key}")
        elif actual != value:
            raise AssertionError(f"Different conditioning field: {key}")
    for key in ("source", "seq_name", "start_idx", "end_idx", "episode_index", "fps",
                "num_real_frames", "floor_height", "ground_fallback_used"):
        if cached["misc"][key] != original["misc"][key]:
            raise AssertionError(f"Different decoding/provenance field: {key}")
    if cached["y"]["img_embs"][6:].any() or cached["y"]["valid_img_embs"][6:].any():
        raise AssertionError("Future images entered the cache conditions")


def representative_indices(dataset, count, required_episodes=()):
    """Cover both sources, ground policies and positions inside episodes."""
    keys = {}
    positions_by_sequence = {}
    for index, (sequence, _) in enumerate(dataset.idx_to_sidx_fidx):
        positions_by_sequence.setdefault(sequence, []).append(index)
    for sequence, positions in positions_by_sequence.items():
        record = dataset.records[sequence]
        key = (record["source"], bool(record["metadata"].get("ground", {}).get("valid", False)))
        keys.setdefault(key, sequence)
    indices = set()
    selected_sequences = set(keys.values())
    required = set(required_episodes)
    selected_sequences.update(sequence for sequence in positions_by_sequence
                              if dataset.records[sequence].get("episode_index") in required)
    for sequence in selected_sequences:
        positions = positions_by_sequence[sequence]
        indices.update((positions[0], positions[len(positions) // 2], positions[-1]))
    indices.update(np.linspace(0, len(dataset) - 1, min(count, len(dataset)), dtype=int).tolist())
    return sorted(indices)


def validate_selected_indices(dataset, selected_sequences):
    """Replay the existing index rule only for sampled episodes, sharing source validation."""
    reference = copy(dataset)
    reference.idx_to_sidx_fidx, reference.rejected_windows = [], {}
    for sequence in sorted(selected_sequences):
        reference._index_record(sequence, dataset.records[sequence])
    expected = [item for item in dataset.idx_to_sidx_fidx if item[0] in selected_sequences]
    if reference.idx_to_sidx_fidx != expected:
        raise AssertionError("Cached sampled-episode window index differs from the original reader")


class OriginalWindowPath(Dataset):
    """Use the unchanged source audit path without another full metadata scan."""
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset.get_sample(index)


def gpu_equivalence(cfg, original_samples, cached_samples, device):
    """One real 400M model, fixed input randomness, BF16 loss and gradient check."""
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("GPU acceptance requires an available CUDA device")
    with torch.cuda.device(device):
        if not torch.cuda.is_bf16_supported():
            raise ValueError("GPU acceptance requires BF16 support on the selected device")
    from .module.uem_module import UEM_Module
    torch.set_num_threads(2)
    torch.manual_seed(int(cfg.TRAIN.SEED))
    module = UEM_Module(cfg).to(device).train()
    batches = [cache_training_collate(samples) for samples in (original_samples, cached_samples)]

    def move(value):
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, dict):
            return {key: move(v) for key, v in value.items()}
        return value

    batches = [move(batch) for batch in batches]
    generator = torch.Generator(device=device).manual_seed(183)
    noise = torch.randn(batches[0]["misc"]["motion"].shape, generator=generator, device=device)
    times = torch.linspace(.25, .9, noise.shape[0], device=device)
    old_grads = None
    losses = []
    maximum_difference = 0.
    checked_gradients = 0
    for batch in batches:
        module.zero_grad(set_to_none=True)
        torch.manual_seed(731)
        y = module._task_conditioning(batch["y"], "train")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            terms = module.flow.training_losses(module.model, batch["misc"]["motion"],
                                                model_kwargs={"y": y}, noise=noise, t=times)
            loss = terms["loss"].mean()
            aux = module.model.get_moe_auxiliary_loss()
            if aux is not None:
                loss = loss + float(cfg.MODEL.MOTION_EXPERT.LOAD_BALANCE_WEIGHT) * aux
        if not torch.isfinite(loss):
            raise AssertionError("Nonfinite GPU cache validation loss")
        loss.backward()
        losses.append(float(loss.detach()))
        grads = {name: None if p.grad is None else p.grad.detach().cpu().clone()
                 for name, p in module.model.named_parameters()}
        if any(value is not None and not torch.isfinite(value).all() for value in grads.values()):
            raise AssertionError("Nonfinite GPU validation gradient")
        if old_grads is None:
            old_grads = grads
        else:
            for name, current in grads.items():
                previous = old_grads[name]
                if (previous is None) != (current is None):
                    raise AssertionError(f"Changed gradient availability: {name}")
                if current is not None:
                    if not torch.isfinite(current).all():
                        raise AssertionError(f"Nonfinite gradient: {name}")
                    torch.testing.assert_close(current, previous, rtol=2e-3, atol=2e-5)
                    maximum_difference = max(maximum_difference, float((current - previous).abs().max()))
                    checked_gradients += 1
    torch.testing.assert_close(torch.tensor(losses[0]), torch.tensor(losses[1]), rtol=2e-4, atol=2e-5)
    if not checked_gradients:
        raise AssertionError("No trainable gradients checked")
    result = {"passed": True, "device": str(device), "gpu_count": 1,
              "precision": "bf16-mixed", "batch": len(cached_samples),
              "parameter_count": sum(p.numel() for p in module.model.parameters()), "losses": losses,
              "gradient_tensors": checked_gradients, "max_gradient_difference": maximum_difference}
    del old_grads, grads, batches, module
    torch.cuda.empty_cache()
    return result


def loader_benchmark(dataset, indices, *, batch_size, workers, batches):
    """IO-only measurement, reported separately from eight-GPU training throughput."""
    chosen = list(indices)
    if not chosen:
        raise ValueError("Cannot benchmark an empty dataset")
    size = batch_size * batches
    chosen = (chosen * ((size + len(chosen) - 1) // len(chosen)))[:size]
    options = {"prefetch_factor": 2} if workers else {}
    loader = DataLoader(dataset, sampler=chosen, batch_size=batch_size, num_workers=workers,
                        collate_fn=cache_training_collate, pin_memory=False, **options)
    start = time.monotonic()
    intervals = []
    last = start
    samples = 0
    for batch in loader:
        current = time.monotonic()
        intervals.append(current - last)
        last = current
        samples += len(batch["misc"]["motion"])
    seconds = time.monotonic() - start
    return {"samples": samples, "seconds": seconds, "samples_per_second": samples / seconds,
            "batch_size": batch_size, "workers": workers, "batch_seconds": intervals,
            "scope": "IO-only; repeated representative windows; not full eight-GPU throughput"}


def wait_for_cache(root, builder_pid=None):
    identity = process_identity(builder_pid) if builder_pid is not None else None
    path = Path(root) / "manifest.json"
    while True:
        manifest = json.loads(path.read_text()) if path.is_file() else {}
        if manifest.get("state") == "complete":
            return manifest
        if manifest.get("state") == "failed":
            raise RuntimeError(f"Cache generation failed: {manifest.get('error')}")
        if builder_pid is not None and not process_alive(identity):
            # A writer may publish its complete manifest just before exiting.
            manifest = json.loads(path.read_text()) if path.is_file() else {}
            if manifest.get("state") == "complete":
                return manifest
            raise RuntimeError("Cache builder exited before publishing a complete cache")
        print(f"[{now()}] Waiting for complete window cache: {root}", flush=True)
        time.sleep(60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--samples-per-split", type=int, default=8)
    parser.add_argument("--episode-indices", nargs="+", type=int, help="Explicit diagnostic cache selection")
    parser.add_argument("--required-episode-indices", nargs="+", type=int, default=[],
                        help="Also compare first/middle/last windows of these episodes, preserving full scope")
    parser.add_argument("--gpu-check", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--benchmark-batches", type=int, default=0)
    parser.add_argument("--benchmark-workers", type=int, default=2)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--builder-pid", type=int)
    args = parser.parse_args()
    if args.samples_per_split < 1 or args.benchmark_batches < 0 or args.benchmark_workers < 0:
        parser.error("Sample count must be positive; benchmark counts cannot be negative")
    cfg, source = load_config(args.config), load_config(args.source_config)
    validate_architecture(cfg)
    validate_config_change(source, cfg)
    root = Path(cfg.DATA.WINDOW_CACHE_ROOT).resolve()
    report = {"format": FORMAT, "passed": False, "cache_root": str(root), "started_at": now(),
              "state": "waiting_cache" if args.wait else "validating",
              "source_config_sha256": sha256(args.source_config), "resume_config_sha256": sha256(args.config),
              "previous_physical_acceptance_repeated": False}
    atomic_json(args.report, report)
    try:
        manifest = wait_for_cache(root, args.builder_pid) if args.wait else json.loads((root / "manifest.json").read_text())
        report["state"] = "validating"
        report["cache_manifest_sha256"] = sha256(root / "manifest.json")
        required = set(args.required_episode_indices)
        available = {item["episode_index"] for item in manifest["episodes"] if item["windows"] > 0}
        if not required <= available:
            raise ValueError(f"Required cache episodes are absent or have no windows: {sorted(required - available)}")
        report["required_episode_indices"] = sorted(required)
        atomic_json(args.report, report)
        comparisons = []
        report["comparisons"] = comparisons
        gpu_cases = {}
        verified_archive = None
        for split in manifest.get("splits", ("train", "val", "test")):
            if manifest.get("indices", {}).get(split, {}).get("windows") == 0:
                continue
            print(f"[{now()}] Validating new {split} cache path", flush=True)
            cached = dataset_from_config(cfg, split, episode_indices=args.episode_indices,
                                         verified_archive=verified_archive)
            verified_archive = cached.archive
            indices = representative_indices(cached, args.samples_per_split, required)
            selected_sequences = {cached.idx_to_sidx_fidx[i][0] for i in indices}
            validate_selected_indices(cached, selected_sequences)
            original = OriginalWindowPath(cached)
            split_checks = []
            split_report = {"split": split, "windows": len(cached), "checked": split_checks, "exact_equal": False}
            comparisons.append(split_report)
            for index in indices:
                sequence, start = cached.idx_to_sidx_fidx[index]
                key = cached.records[sequence]["source_key"]
                report["current_sample"] = {"split": split, "episode": cached.records[sequence]["episode_index"],
                                            "start": int(start), "index": int(index)}
                atomic_json(args.report, report)
                print(f"[{now()}] Comparing {report['current_sample']}", flush=True)
                expected = cached.get_from_seq_and_st(key, int(start), idx=index)
                actual = cached[index]
                compare_sample(actual, expected)
                split_checks.append({"episode": cached.records[sequence]["episode_index"], "start": int(start)})
                atomic_json(args.report, report)
                if split == "train":
                    case = (actual["misc"]["source"], bool(actual["misc"]["ground_fallback_used"]))
                    gpu_cases.setdefault(case, (expected, actual))
            split_report["exact_equal"] = True
            if args.benchmark_batches:
                report.setdefault("loader_benchmarks", {})[split] = {
                    "original": loader_benchmark(original, indices, batch_size=int(cfg.DATA.BATCH_SIZE),
                                                 workers=args.benchmark_workers, batches=args.benchmark_batches),
                    "cache": loader_benchmark(cached, indices, batch_size=int(cfg.DATA.BATCH_SIZE),
                                              workers=args.benchmark_workers, batches=args.benchmark_batches)}
            del original, cached
        report["comparisons"] = comparisons
        report.pop("current_sample", None)
        atomic_json(args.report, report)
        if args.gpu_check:
            if not gpu_cases:
                raise ValueError("GPU acceptance requires nonempty training windows")
            cases = list(gpu_cases.values())[:4]
            gpu_original, gpu_cached = map(list, zip(*cases))
            print(f"[{now()}] Checking BF16 loss and gradients on {args.device}", flush=True)
            report["gpu"] = gpu_equivalence(cfg, gpu_original, gpu_cached, args.device)
            report["gpu"]["cases"] = [list(key) for key in list(gpu_cases)[:4]]
        diagnostic = bool(manifest.get("diagnostic", False) or manifest.get("scope") != "all")
        report.update(passed=not diagnostic and bool(args.gpu_check), diagnostic=diagnostic,
                      checks_passed=True, state="complete", finished_at=now())
        if sha256(root / "manifest.json") != report["cache_manifest_sha256"]:
            raise ValueError("Cache manifest changed during validation")
        for path, key in ((args.source_config, "source_config_sha256"),
                          (args.config, "resume_config_sha256")):
            if sha256(path) != report[key]:
                raise ValueError("Frozen training configuration changed during cache validation")
        atomic_json(args.report, report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0
    except Exception as error:
        if report.get("current_sample"):
            report["failed_sample"] = report["current_sample"]
        report.update(error=f"{type(error).__name__}: {error}", finished_at=now(), passed=False, state="failed")
        atomic_json(args.report, report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
