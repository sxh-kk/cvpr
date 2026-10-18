"""Extract immutable epoch-100 t=1 motion tokens for token-subset probes.

This diagnostic never constructs a Trainer, an optimizer or a source archive.
It reuses the accepted 20/100 probe population and inputs and writes only to a
new job outside the repository. Logical shards stay identical to the original
eight-GPU job, even when fewer physical GPUs process them in sequential waves.
"""
from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time
import traceback

import numpy as np

from .probe_prior import (atomic_json, condition_inputs, event, file_stamp,
                          forward_hidden, make_noise, make_shards,
                          _expert_from_snapshot)


FORMAT = "motionexpert-token-ablation-v1"
PAIRED_RTOL = 2e-5
PAIRED_ATOL = 2e-5


def _function_ast(source, name):
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.dump(node, include_attributes=False)
    raise ValueError(f"Missing paired extraction function: {name}")


def assert_bindings(plan):
    """Metadata hashes and payload stamps only; do not hash cached payloads."""
    base = Path(plan["reuse_job"]["job_dir"])
    for name, expected in plan["reuse_job"]["artifact_stamps"].items():
        if file_stamp(base / name) != expected:
            raise ValueError(f"Read-only baseline artifact changed: {name}")
    for name, expected in plan["reuse_job"]["metadata_sha256"].items():
        if hashlib.sha256((base / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Read-only baseline metadata changed: {name}")
    original = json.loads((base / "plan.json").read_text())
    if plan["batch_size"] != original["batch_size"] or plan["logical_shards"] != original["gpus"]:
        raise ValueError("Paired token extraction must retain original baseline batch shapes and logical shards")
    if file_stamp(plan["checkpoint"]) != plan["checkpoint_stamp"]:
        raise ValueError("Read-only epoch100 checkpoint changed")
    repo = Path(__file__).resolve().parents[1]
    for name, expected in plan["source_files"].items():
        if hashlib.sha256((repo / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Token extraction source changed after preparation: {name}")


def prepare(base_dir, cache, job_dir, *, gpus, batch_size=128, sleep_seconds=12.0,
            memory_fraction=None):
    """Reuse accepted rows/labels/DINO, binding only small metadata and headers."""
    base, job = Path(base_dir).resolve(strict=True), Path(job_dir).resolve()
    repo = Path(__file__).resolve().parents[1]
    if (job == base or job.is_relative_to(base) or base.is_relative_to(job)
            or job.is_relative_to(repo)):
        raise ValueError("Token output must be a new independent job outside Git and its baseline")
    if (not gpus or len(set(gpus)) != len(gpus) or any(type(g) is not int or g < 0 for g in gpus)
            or type(batch_size) is not int or batch_size < 1
            or not np.isfinite(sleep_seconds) or not 0 <= sleep_seconds <= 30
            or (memory_fraction is not None and not 0 < memory_fraction <= 1)):
        raise ValueError("Invalid GPU, batch, throttle or memory limit")
    if job.exists() and any(job.iterdir()):
        raise FileExistsError("Token output already contains artifacts")
    names = ("plan.json", "selection.json", "review.json", "extraction-complete.json")
    initial = {name: file_stamp(base / name) for name in names}
    raw = {name: (base / name).read_bytes() for name in names}
    old = json.loads(raw["plan.json"])
    if (json.loads(raw["review.json"]).get("passed") is not True
            or json.loads(raw["extraction-complete.json"]).get("passed") is not True
            or old.get("format") != "motionexpert-frozen-prior-probe-v1"
            or old.get("reuse_job") or set(old.get("checkpoints", {})) != {"20", "100"}
            or old.get("precision") != "float32" or old.get("weights") != "ema"):
        raise ValueError("Token extraction requires the accepted original FP32 EMA baseline")
    if batch_size != old.get("batch_size"):
        raise ValueError("Paired token extraction must keep the original baseline batch_size; "
                         "changing FP32 batch shapes can flip hard top-k MoE routes. "
                         "Use the diagnostic sleep control to limit GPU contention instead.")
    if (str(cache.root) != old["cache_root"] or str(cache.vision_root) != old["vision_root"]
            or cache.manifest_sha256 != old["cache_manifest_sha256"]
            or any(cache.binding[key] != old[key] for key in ("archive_fingerprint", "split_fingerprint"))):
        raise ValueError("Accepted window/vision identities differ from the baseline")
    for name in ("MotionExpert/dataset/probe_cache.py", "MotionExpert/prior_probe_math.py"):
        if hashlib.sha256((repo / name).read_bytes()).hexdigest() != old["source_files"].get(name):
            raise ValueError(f"Baseline cache/target semantics changed: {name}")
    snapshot = base / "sources/probe_prior-extraction.py"
    original_source = snapshot.read_text()
    if hashlib.sha256(snapshot.read_bytes()).hexdigest() != old["source_files"]["MotionExpert/probe_prior.py"]:
        raise ValueError("Original extraction source snapshot identity differs")
    current_source = (repo / "MotionExpert/probe_prior.py").read_text()
    for name in ("noise_seed", "make_noise", "make_shards", "condition_inputs", "forward_hidden", "_expert_from_snapshot"):
        if _function_ast(original_source, name) != _function_ast(current_source, name):
            raise ValueError(f"Paired extraction semantics changed: {name}")
    entries = json.loads(raw["selection.json"])
    if (not isinstance(entries, list) or len(entries) != old["windows"]
            or any(row.get("row") != i for i, row in enumerate(entries))):
        raise ValueError("Original selection shape/order differs")
    for row in entries:
        split, index = row["split"], row["index"]
        indices = cache.indices(split)
        if type(index) is not int or not 0 <= index < len(indices):
            raise ValueError("Original selection contains an invalid packed row")
        sequence, start = map(int, indices[index])
        record = cache.records[split][sequence]
        if start != row["start_frame"] or any(row[key] != record[key] for key in
                ("episode_index", "source", "source_key", "split_group_ids")):
            raise ValueError("Original selection identity differs from accepted cache")
    artifacts = [*names, "targets.npz", "dino.npy", "epoch100.npy", "dino-linear-probe.pt",
                 "epoch100-linear-probe.pt", "sources/probe_prior-extraction.py"]
    for logical_gpu in old["gpus"]:
        name = f"done-100-gpu{logical_gpu}.json"
        if json.loads((base / name).read_text()).get("passed") is not True:
            raise ValueError("Original per-shard acceptance is missing")
        artifacts.append(name)
    for name, shape in (("epoch100.npy", (len(entries), 768)), ("dino.npy", (len(entries), 30720))):
        array = np.load(base / name, mmap_mode="r", allow_pickle=False)
        if array.shape != shape or array.dtype != np.float32:
            raise ValueError(f"Original feature header differs: {name}")
        del array
    with np.load(base / "targets.npz", allow_pickle=False) as labels:
        if set(labels.files) != set(old["label_specs"]):
            raise ValueError("Original target field membership differs")
        for name, spec in old["label_specs"].items():
            if list(labels[name].shape) != spec["shape"] or str(labels[name].dtype) != spec["dtype"]:
                raise ValueError(f"Original target header differs: {name}")
    checkpoint = old["checkpoints"]["100"]
    if file_stamp(checkpoint) != old["checkpoint_stamps"]["100"]:
        raise ValueError("Original epoch100 checkpoint identity changed")
    source_names = ("MotionExpert/probe_tokens.py", "MotionExpert/probe_prior.py",
                    "MotionExpert/dataset/probe_cache.py", "MotionExpert/prior_probe_math.py")
    plan = {"format": FORMAT, "job_dir": str(job), "epoch": 100,
        "checkpoint": checkpoint, "checkpoint_stamp": file_stamp(checkpoint),
        "cache_root": old["cache_root"], "vision_root": old["vision_root"],
        "cache_manifest_sha256": old["cache_manifest_sha256"],
        "archive_fingerprint": old["archive_fingerprint"], "split_fingerprint": old["split_fingerprint"],
        "seed": old["seed"], "gpus": gpus, "logical_shards": old["gpus"],
        "batch_size": batch_size, "sleep_seconds": float(sleep_seconds),
        "memory_fraction": memory_fraction, "precision": "float32", "weights": "ema",
        "windows": len(entries), "counts": old.get("counts"), "label_specs": old["label_specs"],
        "token_shape": [11, 768], "input": old["input"],
        "paired_rtol": PAIRED_RTOL, "paired_atol": PAIRED_ATOL,
        "source_files": {name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in source_names},
        "reuse_job": {"job_dir": str(base),
            "artifact_stamps": {name: file_stamp(base / name) for name in artifacts},
            "metadata_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in raw.items()}},
        "motionexpert_training_started": False, "training_configuration_changed": False}
    if any(file_stamp(base / name) != value for name, value in initial.items()):
        raise ValueError("Baseline changed while binding the token job")
    assert_bindings(plan)
    job.mkdir(parents=True, exist_ok=True)
    for name in ("selection.json", "targets.npz", "dino.npy", "epoch100.npy", "dino-linear-probe.pt",
                 "epoch100-linear-probe.pt"):
        (job / name).symlink_to(base / name)
    atomic_json(job / "plan.json", plan)
    atomic_json(job / "control.json", {"paused": False, "sleep_seconds": float(sleep_seconds)})
    event(job, "prepared", windows=len(entries), gpu_count=len(gpus), logical_shards=len(old["gpus"]))
    return plan, entries


def read_control(job, default_sleep):
    """Read this diagnostic's own live throttle; it never controls training."""
    path = Path(job) / "control.json"
    control = json.loads(path.read_text()) if path.exists() else {}
    paused, delay = control.get("paused", False), control.get("sleep_seconds", default_sleep)
    if (type(paused) is not bool or isinstance(delay, bool)
            or not isinstance(delay, (int, float)) or not np.isfinite(delay) or not 0 <= delay <= 30):
        raise ValueError("Diagnostic control requires boolean paused and sleep_seconds between 0 and 30")
    return paused, float(delay)


def _gpu_acceptance(expert, conditions, plan):
    import torch
    small = {key: value[:2] for key, value in conditions.items()}
    with torch.inference_mode():
        _, official_noise = expert._sampling_inputs(small, seed=plan["seed"], task="fore",
            forecast_prefix=6, normalized=True, output_frames=11)
        pooled, hidden, prediction = forward_hidden(expert, small, official_noise.cpu())
        official = expert.forward_features(small, seed=plan["seed"])
    torch.testing.assert_close(official["one_step_hidden"], hidden, rtol=PAIRED_RTOL, atol=PAIRED_ATOL)
    if (not torch.isfinite(hidden).all() or not torch.isfinite(prediction).all()
            or expert.model.training or any(p.requires_grad or p.grad is not None for p in expert.parameters())):
        raise AssertionError("Frozen expert GPU acceptance failed")
    return {"passed": True, "batch": len(official_noise), "hidden_shape": list(hidden.shape),
        "public_hidden_max_abs_error": float((official["one_step_hidden"] - hidden).abs().max()),
        "global_equals_mean": bool(torch.equal(pooled, hidden.mean(1))),
        "expert_trainable_parameters": 0, "expert_gradients": False,
        "trajectory_future_labels_hidden": True, "full_gaussian_t1": True}


def _run_shard(gpu, logical_gpu, shard, plan, cfg, normalizer, visual, checkpoint, cache):
    job = Path(plan["job_dir"])
    with (job / f"extract-shard{logical_gpu}-gpu{gpu}.log").open("w", buffering=1) as handle:
        sys.stdout = sys.stderr = handle
        try:
            import torch
            torch.set_num_threads(1)
            torch.cuda.set_device(gpu)
            if plan["memory_fraction"] is not None:
                torch.cuda.set_per_process_memory_fraction(plan["memory_fraction"], gpu)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.set_float32_matmul_precision("highest")
            device = torch.device(f"cuda:{gpu}")
            expert = _expert_from_snapshot(cfg, normalizer, visual, plan["checkpoint"], checkpoint, device)
            tokens = np.load(job / "tokens.npy", mmap_mode="r+", allow_pickle=False)
            image_valid = np.load(job / "image-valid.npy", mmap_mode="r+", allow_pickle=False)
            baseline = np.load(job / "epoch100.npy", mmap_mode="r", allow_pickle=False)
            dino = np.load(job / "dino.npy", mmap_mode="r", allow_pickle=False)
            noise_digest, image_digest = hashlib.sha256(), hashlib.sha256()
            started, completed, maximum_error, acceptance = time.monotonic(), 0, 0.0, None
            total_batch_seconds = 0.0
            for start in range(0, len(shard), plan["batch_size"]):
                paused, delay = read_control(job, plan["sleep_seconds"])
                while paused:
                    time.sleep(1)
                    paused, delay = read_control(job, plan["sleep_seconds"])
                batch_started = time.monotonic()
                rows = shard[start:start + plan["batch_size"]]
                conditions = condition_inputs([cache.read(row["split"], row["index"], include_images=True) for row in rows])
                indices = [row["row"] for row in rows]
                if not np.array_equal(conditions["img_embs"].numpy().reshape(len(rows), -1), dino[indices]):
                    raise AssertionError("Historical image features differ from accepted baseline")
                noise = make_noise(rows, plan["seed"], plan["archive_fingerprint"])
                noise_digest.update(noise.numpy().tobytes())
                image_digest.update(conditions["img_embs"].numpy().tobytes())
                if acceptance is None:
                    acceptance = _gpu_acceptance(expert, conditions, plan)
                    atomic_json(job / f"gpu-check-shard{logical_gpu}.json", acceptance)
                with torch.inference_mode():
                    pooled, hidden, prediction = forward_hidden(expert, conditions, noise)
                if not torch.isfinite(hidden).all() or not torch.isfinite(prediction).all():
                    raise AssertionError("Full token or prediction contains nonfinite values")
                value = hidden.cpu().numpy()
                mean = pooled.cpu().numpy()
                expected = baseline[indices]
                np.testing.assert_allclose(mean, expected, rtol=PAIRED_RTOL, atol=PAIRED_ATOL)
                maximum_error = max(maximum_error, float(np.max(np.abs(mean - expected))))
                tokens[indices] = value
                image_valid[indices] = conditions["valid_img_embs"].numpy()
                batch_seconds = time.monotonic() - batch_started
                total_batch_seconds += batch_seconds
                completed += len(rows)
                if start == 0 or completed == len(shard) or (start // plan["batch_size"]) % 16 == 0:
                    atomic_json(job / f"extract-shard{logical_gpu}.json", {"gpu": gpu,
                        "logical_shard": logical_gpu, "completed": completed, "total": len(shard),
                        "elapsed_seconds": time.monotonic() - started,
                        "last_batch_seconds_excluding_throttle": batch_seconds,
                        "total_batch_seconds_excluding_throttle": total_batch_seconds,
                        "sleep_seconds": delay,
                        "peak_gpu_bytes": torch.cuda.max_memory_allocated(gpu),
                        "paired_mean_max_abs_error": maximum_error})
                    print(f"shard={logical_gpu} gpu={gpu} windows={completed}/{len(shard)}", flush=True)
                if delay:
                    time.sleep(delay)
            tokens.flush()
            image_valid.flush()
            if any(p.requires_grad or p.grad is not None for p in expert.parameters()):
                raise AssertionError("Expert acquired gradients")
            guard = cache.assert_current()
            assert_bindings(plan)
            result = {"passed": True, "epoch": 100, "gpu": gpu, "logical_shard": logical_gpu,
                "windows": completed, "hidden_shape": [completed, 11, 768],
                "rows_sha256": hashlib.sha256(np.asarray([row["row"] for row in shard], dtype=np.int64).tobytes()).hexdigest(),
                "noise_sha256": noise_digest.hexdigest(), "dino_sha256": image_digest.hexdigest(),
                "paired_mean_max_abs_error": maximum_error, "paired_rtol": PAIRED_RTOL, "paired_atol": PAIRED_ATOL,
                "peak_gpu_bytes": torch.cuda.max_memory_allocated(gpu), "guard": guard, "acceptance": acceptance,
                "total_batch_seconds_excluding_throttle": total_batch_seconds,
                "elapsed_seconds": time.monotonic() - started, "expert_gradients": False,
                "expert_trainable_parameters": 0}
            old = json.loads((Path(plan["reuse_job"]["job_dir"]) / f"done-100-gpu{logical_gpu}.json").read_text())
            for key in ("rows_sha256", "noise_sha256", "dino_sha256", "windows"):
                if result[key] != old[key]:
                    raise AssertionError(f"Paired baseline shard differs: {key}")
            atomic_json(job / f"done-shard{logical_gpu}.json", result)
        except BaseException as error:
            traceback.print_exc()
            atomic_json(job / f"failed-shard{logical_gpu}.json", {"error": repr(error), "traceback": traceback.format_exc()})
            raise


def extract(plan, cache, entries):
    import torch
    from .checkpoints import read_checkpoint, checkpoint_config, validate_checkpoint_statistics
    from .dataset.normalization import MotionNormalizer
    from .visual_contract import checkpoint_visual_contract, compare_visual_contracts
    job = Path(plan["job_dir"])
    assert_bindings(plan)
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU parent cannot initialize CUDA before diagnostic forks")
    path = job / "tokens.npy"
    if path.exists() or (job / "image-valid.npy").exists():
        raise FileExistsError("Refuse to overwrite existing tokens")
    checkpoint = read_checkpoint(plan["checkpoint"])
    if (checkpoint.get("epoch") != 99
            or checkpoint.get("ema_state_format") != "original_model_with_ema_optimizer_v1"):
        raise ValueError("Token extraction requires original epoch100 EMA checkpoint")
    cfg = checkpoint_config(checkpoint)
    normalizer = MotionNormalizer(checkpoint["motionexpert_normalization"])
    validate_checkpoint_statistics(cfg, checkpoint, normalizer)
    for key in ("archive_fingerprint", "split_fingerprint"):
        if normalizer.stats[key] != cache.binding[key]:
            raise ValueError(f"Checkpoint/cache identity differs: {key}")
    visual = checkpoint_visual_contract(checkpoint)
    compare_visual_contracts(visual, {"format": "motionexpert-visual-v1", "mode": "shared_rgb",
        "image_process": cache.vision_manifest["image_process"], "encoder": cache.vision_manifest["encoder"]})
    array = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(entries), 11, 768))
    array.flush()
    del array
    array = np.lib.format.open_memmap(job / "image-valid.npy", mode="w+", dtype=np.bool_, shape=(len(entries), 6))
    array.flush()
    del array
    shards = make_shards(entries, plan["logical_shards"])
    context = mp.get_context("fork")
    started = time.monotonic()
    for offset in range(0, len(shards), len(plan["gpus"])):
        children = []
        for gpu, shard_index in zip(plan["gpus"], range(offset, min(offset + len(plan["gpus"]), len(shards)))):
            if not shards[shard_index]:
                raise ValueError("Original logical shard is empty")
            logical_gpu = plan["logical_shards"][shard_index]
            children.append(context.Process(target=_run_shard, args=(gpu, logical_gpu, shards[shard_index],
                plan, cfg, normalizer, visual, checkpoint, cache)))
        event(job, "extracting", epoch=100, windows=len(entries), wave=offset // len(plan["gpus"]),
            gpu_count=len(children), checkpoint_parameters_updated=False)
        for child in children:
            child.start()
        try:
            while any(child.is_alive() for child in children):
                if any(child.exitcode not in (None, 0) for child in children):
                    raise RuntimeError("Full-token diagnostic child failed")
                time.sleep(1)
            if any(child.exitcode != 0 for child in children):
                raise RuntimeError("Full-token diagnostic child failed")
        except BaseException:
            for child in children:
                if child.is_alive():
                    child.terminate()
            for child in children:
                child.join(timeout=20)
            raise
        for child in children:
            child.join()
    results = [json.loads((job / f"done-shard{logical}.json").read_text()) for logical in plan["logical_shards"]]
    if not all(result.get("passed") is True for result in results) or sum(result["windows"] for result in results) != len(entries):
        raise ValueError("Completed logical shards do not cover the immutable selection")
    assert_bindings(plan)
    guard = cache.assert_current()
    result = {"passed": True, "epoch": 100, "windows": len(entries), "token_shape": [len(entries), 11, 768],
        "logical_shards": len(shards), "physical_gpus": plan["gpus"],
        "paired_mean_max_abs_error": max(value["paired_mean_max_abs_error"] for value in results),
        "peak_gpu_bytes": max(value["peak_gpu_bytes"] for value in results),
        "all_tokens_finite": True, "expert_gradients": False, "expert_trainable_parameters": 0,
        "historical_image_valid_shape": [len(entries), 6],
        "unchanged_baseline": True, "guard": guard, "elapsed_seconds": time.monotonic() - started}
    atomic_json(job / "extraction-complete.json", result)
    event(job, "features_extracted", **result)
    del checkpoint
    gc.collect()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-job")
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sleep-seconds", type=float, default=12.0)
    parser.add_argument("--memory-fraction", type=float)
    parser.add_argument("--stage", choices=("prepare", "extract", "all"), default="all")
    args = parser.parse_args(argv)
    job = Path(args.job_dir).resolve()
    # Refuse foreign/unsafe paths before writing failure state.
    repo = Path(__file__).resolve().parents[1]
    if job.is_relative_to(repo):
        raise ValueError("Diagnostic artifacts must be outside Git")
    owned = False
    try:
        from .dataset.probe_cache import AcceptedProbeCache
        if args.stage == "extract":
            plan = json.loads((job / "plan.json").read_text())
            if plan.get("format") != FORMAT or Path(plan["job_dir"]).resolve() != job:
                raise ValueError("Token plan belongs to another protocol or job")
            owned = True
            assert_bindings(plan)
            entries = json.loads((job / "selection.json").read_text())
            cache = AcceptedProbeCache(plan["cache_root"], plan["vision_root"])
        else:
            if not args.reuse_job:
                raise ValueError("Preparation requires --reuse-job")
            if job.exists() and any(job.iterdir()):
                raise FileExistsError("Token output already contains artifacts")
            base = Path(args.reuse_job).resolve(strict=True)
            if job == base or job.is_relative_to(base) or base.is_relative_to(job):
                raise ValueError("Token job must be independent of its baseline")
            old = json.loads((base / "plan.json").read_text())
            cache = AcceptedProbeCache(old["cache_root"], old["vision_root"])
            owned = True
            plan, entries = prepare(base, cache, job, gpus=[int(value) for value in args.gpus.split(",")],
                batch_size=args.batch_size, sleep_seconds=args.sleep_seconds, memory_fraction=args.memory_fraction)
        if args.stage != "prepare":
            extract(plan, cache, entries)
    except BaseException as error:
        if owned:
            try:
                job.mkdir(parents=True, exist_ok=True)
                event(job, "failed", error=repr(error), traceback=traceback.format_exc())
            except BaseException:
                pass
        raise


if __name__ == "__main__":
    main()
