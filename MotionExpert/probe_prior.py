"""Frozen t=1 mean-pooled motion-prior diagnostics against historical DINO.

Only the independent linear readouts are fitted. MotionExpert and DINO remain
frozen; this entry point never imports a Trainer or resumes motion training.
All artifacts are written to an explicitly supplied directory outside Git.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def file_stamp(path):
    value = Path(path).stat()
    return {"bytes": value.st_size, "mtime_ns": value.st_mtime_ns,
            "inode": value.st_ino, "device": value.st_dev}


def event(job, stage, **values):
    value = {"stage": stage, "pid": os.getpid(),
             "updated_at": datetime.now(timezone.utc).isoformat(), **values}
    atomic_json(Path(job) / "progress.json", value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def record_failure(job, error, status_path=None):
    """Best-effort failure evidence; never hide the operation's exception.

    Preparation may fail before a plan can identify the shared status file.
    Even with a plan, update that file only while it still names this job.
    """
    recording_errors = []
    try:
        Path(job).mkdir(parents=True, exist_ok=True)
        event(job, "failed", error=repr(error), traceback=traceback.format_exc())
    except BaseException as recording_error:
        recording_errors.append(recording_error)
    if status_path is not None:
        try:
            current = json.loads(Path(status_path).read_text())
            owner = current.get("job_dir")
            if isinstance(owner, str) and Path(owner).resolve() == Path(job).resolve():
                atomic_json(status_path, {"state": "failed", "job_dir": str(job), "error": repr(error),
                                          "motionexpert_training_started": False})
        except FileNotFoundError:
            pass
        except BaseException as recording_error:
            recording_errors.append(recording_error)
    for recording_error in recording_errors:
        try:
            print(f"Unable to record probe failure: {recording_error!r}", file=sys.stderr, flush=True)
        except BaseException:
            pass


def noise_seed(seed, archive_fingerprint, episode, start, draw=0):
    payload = json.dumps([int(seed), archive_fingerprint, int(episode), int(start), int(draw)],
                         separators=(",", ":")).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") % (2**63 - 1)


def make_noise(entries, seed, fingerprint, draw=0):
    import torch
    return torch.stack([torch.randn((11, 245), generator=torch.Generator(device="cpu").manual_seed(
        noise_seed(seed, fingerprint, row["episode_index"], row["start_frame"], draw)), dtype=torch.float32)
        for row in entries])


def select_rows(cache, seed, train_episodes=1024, val_episodes=256, windows_per_episode=16):
    selected = cache.select_balanced(max_per_episode=windows_per_episode,
        max_episodes_per_source={"train": train_episodes, "val": val_episodes},
        min_start_gap=60, seed=seed)
    entries = []
    for split in ("train", "val"):
        for index in selected[split]:
            sequence, start = map(int, cache.indices(split)[index])
            record = cache.records[split][sequence]
            entries.append({"row": len(entries), "split": split, "index": int(index),
                "source": record["source"], "episode_index": int(record["episode_index"]),
                "start_frame": start, "source_key": record["source_key"],
                "split_group_ids": record["split_group_ids"]})
    if not entries or {row["split"] for row in entries} != {"train", "val"}:
        raise ValueError("Probe requires nonempty immutable train and validation selections")
    return entries


def prepare_labels(cache, entries, job):
    from .prior_probe_math import build_probe_targets
    collected = defaultdict(list)
    for start in range(0, len(entries), 512):
        samples = [cache.read(row["split"], row["index"], include_images=False)
                   for row in entries[start:start + 512]]
        motion = np.stack([sample["motion"] for sample in samples])
        valid = np.stack([sample["motion_valid"] for sample in samples])
        floor = np.stack([sample["floor_transform"] for sample in samples])
        labels = build_probe_targets(motion, valid, floor)
        labels.update(current_contact=motion[:, 5, 243:245],
                      current_contact_valid=valid[:, 5, 243:245])
        for key, value in labels.items():
            collected[key].append(value)
    arrays = {key: np.concatenate(value) for key, value in collected.items()}
    np.savez(Path(job) / "targets.npz", **arrays)
    return {key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in arrays.items()}


def make_shards(entries, gpus):
    episodes = defaultdict(list)
    for row in entries:
        episodes[row["episode_index"]].append(row)
    shards, loads = [[] for _ in gpus], [0] * len(gpus)
    for episode, rows in sorted(episodes.items(), key=lambda item: (-len(item[1]), item[0])):
        rank = min(range(len(gpus)), key=lambda item: (loads[item], item))
        shards[rank].extend(rows)
        loads[rank] += len(rows)
    for shard in shards:
        shard.sort(key=lambda item: (item["split"], item["index"]))
    return shards


def assert_reuse_current(plan):
    """Only check recorded identities; never rescan/revalidate source payloads."""
    reuse = plan.get("reuse_job")
    if not reuse:
        return
    base = Path(reuse["job_dir"])
    for name, expected in reuse["artifact_stamps"].items():
        if file_stamp(base / name) != expected:
            raise ValueError(f"Reused probe artifact changed: {name}")
    for name, expected in reuse["metadata_sha256"].items():
        if hashlib.sha256((base / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Reused probe metadata changed: {name}")


def prepare_reuse(base_dir, cache, checkpoint160, job, gpus, batch_size):
    """Bind an incremental evaluation to an accepted, immutable 20/100 job.

    Existing payloads are linked for read-only consumers. The extraction and
    fitting paths are explicitly limited to epoch160, so those links are never
    opened for writing. No selection, labels or DINO PCA are regenerated.
    """
    base, job = Path(base_dir).resolve(strict=True), Path(job).resolve()
    if base == job or job.is_relative_to(base):
        raise ValueError("Incremental probe output must be outside its baseline job")
    metadata = ("plan.json", "selection.json", "metrics.json", "review.json", "extraction-complete.json")
    initial = {name: file_stamp(base / name) for name in metadata}
    raw = {name: (base / name).read_bytes() for name in metadata}
    old = json.loads(raw["plan.json"])
    metrics = json.loads(raw["metrics.json"])
    if (json.loads(raw["review.json"]).get("passed") is not True
            or json.loads(raw["extraction-complete.json"]).get("passed") is not True):
        raise ValueError("Reuse requires a completed, independently accepted baseline job")
    if (old.get("format") != "motionexpert-frozen-prior-probe-v1"
            or old.get("reuse_job") or set(old.get("checkpoints", {})) != {"20", "100"}
            or old.get("precision") != "float32" or old.get("weights") != "ema"):
        raise ValueError("Baseline job does not implement the original frozen 20/100 protocol")
    if gpus != old["gpus"]:
        raise ValueError("Reuse requires the same GPU layout for exact paired-input shard digests")
    if (str(cache.root) != old["cache_root"] or str(cache.vision_root) != old["vision_root"]
            or cache.manifest_sha256 != old["cache_manifest_sha256"]
            or any(cache.binding[key] != old[key] for key in ("archive_fingerprint", "split_fingerprint"))):
        raise ValueError("Baseline and current accepted cache/vision identities differ")
    repo = Path(__file__).resolve().parents[1]
    for name in ("MotionExpert/dataset/probe_cache.py", "MotionExpert/prior_probe_math.py"):
        if hashlib.sha256((repo / name).read_bytes()).hexdigest() != old["source_files"].get(name):
            raise ValueError(f"Baseline cache/label semantics differ: {name}")
    entries = json.loads(raw["selection.json"])
    if (not isinstance(entries, list) or len(entries) != old["windows"]
            or any(row.get("row") != index for index, row in enumerate(entries))):
        raise ValueError("Baseline selection rows differ from its declared population")
    for row in entries:
        split, index = row["split"], row["index"]
        indices = cache.indices(split)
        if type(index) is not int or not 0 <= index < len(indices):
            raise ValueError("Reused selection contains an invalid packed index")
        sequence, start = map(int, indices[index])
        record = cache.records[split][sequence]
        if (start != row["start_frame"] or any(row[key] != record[key] for key in
                ("episode_index", "source", "source_key", "split_group_ids"))):
            raise ValueError("Baseline selection identity/order differs from the accepted cache")
    noise_rows = old.get("noise_rows", [])
    if (not noise_rows or len(set(noise_rows)) != len(noise_rows)
            or any(type(index) is not int or not 0 <= index < len(entries)
                   or entries[index]["split"] != "val" for index in noise_rows)):
        raise ValueError("Baseline noise rows do not identify its validation subset")
    artifacts = ["selection.json", "targets.npz", "dino.npy", "dino-linear-probe.pt"]
    for epoch in (20, 100):
        artifacts += [f"epoch{epoch}.npy", f"noise-epoch{epoch}.npy", f"epoch{epoch}-linear-probe.pt"]
        for gpu in gpus:
            name = f"done-{epoch}-gpu{gpu}.json"
            if json.loads((base / name).read_text()).get("passed") is not True:
                raise ValueError("Baseline per-GPU extraction evidence is not accepted")
            artifacts.append(name)
    for name, width in (("epoch20", 768), ("epoch100", 768), ("dino", 30720)):
        array = np.load(base / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        if array.shape != (len(entries), width) or array.dtype != np.float32:
            raise ValueError(f"Baseline feature header differs: {name}")
        del array
    for epoch in (20, 100):
        array = np.load(base / f"noise-epoch{epoch}.npy", mmap_mode="r", allow_pickle=False)
        if array.shape != (len(noise_rows), 3, 768) or array.dtype != np.float32:
            raise ValueError("Baseline noise array header differs")
        del array
    with np.load(base / "targets.npz", allow_pickle=False) as labels:
        if set(labels.files) != set(old["label_specs"]):
            raise ValueError("Baseline target field membership differs")
        for name, spec in old["label_specs"].items():
            if list(labels[name].shape) != spec["shape"] or str(labels[name].dtype) != spec["dtype"]:
                raise ValueError(f"Baseline target shape/dtype differs: {name}")
    if not {"epoch20", "epoch100", "dino"} <= metrics.get("models", {}).keys():
        raise ValueError("Baseline has no completed frozen readouts")
    checkpoints = dict(old["checkpoints"])
    stamps = dict(old["checkpoint_stamps"])
    for epoch, path in checkpoints.items():
        if file_stamp(path) != stamps[epoch]:
            raise ValueError("Baseline frozen checkpoint identity changed")
    checkpoints["160"] = str(Path(checkpoint160).resolve(strict=True))
    stamps["160"] = file_stamp(checkpoints["160"])
    plan = {**old, "job_dir": str(job), "checkpoints": checkpoints,
            "checkpoint_stamps": stamps, "gpus": gpus, "batch_size": batch_size,
            "extract_epochs": [160], "fit_models": ["epoch160"],
            "reuse_job": {"job_dir": str(base),
                "metadata_sha256": {name: hashlib.sha256(value).hexdigest() for name, value in raw.items()},
                "artifact_stamps": {name: file_stamp(base / name) for name in set(artifacts) | set(metadata)},
                "readout_models": ["epoch20", "epoch100", "dino"]},
            "baseline_source_files": old["source_files"],
            "source_files": {str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (Path(__file__), repo / "MotionExpert/dataset/probe_cache.py",
                             repo / "MotionExpert/prior_probe_math.py", repo / "MotionExpert/prior_probe_fit.py")}}
    if any(file_stamp(base / name) != stamp for name, stamp in initial.items()):
        raise ValueError("Baseline metadata changed while binding the incremental probe")
    assert_reuse_current(plan)
    # Only payloads consumed by read-only paths are inherited into the new job.
    for name in artifacts:
        if not name.startswith("done-"):
            (job / name).symlink_to(base / name)
    return plan, entries


def condition_inputs(samples):
    """A strict whitelist: six historical images and their observed validity."""
    import torch
    image = np.stack([sample["image_features"] for sample in samples])
    valid = np.stack([sample["historical_image_valid"] for sample in samples])
    if image.shape[1:] != (6, 5, 1024) or valid.dtype != np.bool_:
        raise ValueError("Expected six chronological CLS+register DINO frames")
    if not np.isfinite(image).all() or np.any(image[~valid] != 0):
        raise ValueError("Observed DINO values violate their accepted mask")
    return {"img_embs": torch.from_numpy(image), "valid_img_embs": torch.from_numpy(valid),
            "valid_frames": torch.ones((len(samples), 6), dtype=torch.bool)}


def forward_hidden(expert, conditions, noise):
    import torch
    from .utils.task_conditioning import apply_task_conditioning
    y, _, frames, observed = expert._prepare(conditions, normalized=True, task="fore", output_frames=11, forecast_prefix=6)
    y = apply_task_conditioning(y, "fore", forecast_prefix=6, trajectory_condition=False)
    y["observed_prefix"] = 6
    if (frames, observed) != (11, 6) or not y["valid_frames"].all():
        raise AssertionError("Unexpected generated-state time domain")
    if y["trajectory_valid"].any() or y["traj"].any() or not y["traj_mask"].bool().all():
        raise AssertionError("Camera trajectory entered image-only prior")
    if y["valid_img_embs"][:, 6:].any() or y["img_embs"][:, 6:].any():
        raise AssertionError("Future images entered frozen prior")
    if {"motion", "motion_valid", "feature_valid_mask"} & set(y):
        raise AssertionError("Ground-truth motion/masks entered prior input")
    t = torch.ones(noise.shape[0], device=expert.device, dtype=torch.float32)
    prediction, hidden, global_hidden = expert.model(noise.to(expert.device), t, y,
                                                    cond_scale=None, return_features=True)
    if hidden.shape[1:] != (11, 768) or global_hidden.shape[1:] != (768,):
        raise AssertionError("Frozen prior feature shape differs")
    if not torch.equal(global_hidden, hidden.mean(1)) or not torch.isfinite(global_hidden).all():
        raise AssertionError("Invalid mean-pooled prior")
    return global_hidden, hidden, prediction


def _expert_from_snapshot(cfg, normalizer, visual, checkpoint_path, checkpoint, device):
    import torch
    from .motion_expert import MotionExpert
    from .model import UniEgoMotion
    from .checkpoints import model_state
    expert = MotionExpert.__new__(MotionExpert)
    expert.cfg, expert.normalizer = cfg, normalizer
    expert.device, expert.dtype = device, torch.float32
    expert.visual_conditioning, expert.weights_path, expert.weights_kind = visual, str(checkpoint_path), "ema"
    expert.model = UniEgoMotion(cfg).to(device=device, dtype=torch.float32).eval()
    # Keep requires_grad=True until the original EMA ordering has been resolved.
    expert.model.load_state_dict(model_state(expert.model, checkpoint, use_ema=True), strict=True)
    expert.model.requires_grad_(False)
    return expert


def gpu_acceptance(expert, conditions, entries, plan):
    import torch
    small = {key: value[:4] for key, value in conditions.items()}
    entries = entries[:4]
    noise = make_noise(entries, plan["seed"], plan["archive_fingerprint"])
    first_calls = []
    def check_input(module, args):
        first_calls.append(bool(torch.equal(args[0], noise.to(expert.device))) and bool((args[1] == 1).all()))
    hook = expert.model.register_forward_pre_hook(check_input)
    try:
        with torch.inference_mode():
            global_hidden, hidden, prediction = forward_hidden(expert, small, noise)
    finally:
        hook.remove()
    # Compare the explicit paired-noise path to the public feature interface,
    # using that interface's own deterministic CUDA generator.
    y, official_noise = expert._sampling_inputs(small, seed=plan["seed"], task="fore", forecast_prefix=6,
                                                normalized=True, output_frames=11)
    with torch.inference_mode():
        expected, _, _ = forward_hidden(expert, small, official_noise.cpu())
        official = expert.forward_features(small, seed=plan["seed"])["one_step_global"]
    torch.testing.assert_close(official, expected, rtol=2e-5, atol=2e-5)
    if first_calls != [True] or expert.model.training or any(p.requires_grad for p in expert.parameters()):
        raise AssertionError("Frozen extraction mode/noise violated")
    head = torch.nn.Linear(768, 2, device=expert.device)
    old = head.weight.detach().clone()
    optimizer = torch.optim.SGD(head.parameters(), lr=.01)
    # Clone outside inference mode: readout autograd must not touch the expert.
    inputs = global_hidden.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    head(inputs).square().mean().backward()
    optimizer.step()
    if torch.equal(old, head.weight) or any(p.grad is not None for p in expert.parameters()):
        raise AssertionError("Probe head update changed or attached to frozen expert")
    return {"passed": True, "batch": len(entries), "hidden_shape": list(hidden.shape),
            "global_equals_hidden_mean": True, "pure_gaussian_t1": True,
            "public_interface_max_abs_error": float((official - expected).abs().max()),
            "head_parameters_updated": True, "expert_requires_grad": False, "expert_gradients": False,
            "trajectory_future_labels_hidden": True, "prediction_finite": bool(torch.isfinite(prediction).all())}


def run_rank(rank, gpu, shard, plan, cfg, normalizer, visual, checkpoint, cache, epoch):
    """Forked from a CPU-only parent; all writes belong to this diagnostic."""
    job = Path(plan["job_dir"])
    with (job / f"extract-{epoch}-gpu{gpu}.log").open("w", buffering=1) as handle:
        sys.stdout = sys.stderr = handle
        try:
            import torch
            torch.set_num_threads(2)
            torch.cuda.set_device(gpu)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.set_float32_matmul_precision("highest")
            device = torch.device(f"cuda:{gpu}")
            expert = _expert_from_snapshot(cfg, normalizer, visual, plan["checkpoints"][str(epoch)], checkpoint, device)
            features = np.load(job / f"epoch{epoch}.npy", mmap_mode="r+")
            dino = np.load(job / "dino.npy", mmap_mode="r+") if epoch == 20 and not plan.get("reuse_job") else None
            paired_dino = np.load(job / "dino.npy", mmap_mode="r") if plan.get("reuse_job") else None
            stability = np.load(job / f"noise-epoch{epoch}.npy", mmap_mode="r+")
            stable_lookup = {row: i for i, row in enumerate(plan["noise_rows"])}
            noise_digest, image_digest = hashlib.sha256(), hashlib.sha256()
            started, completed, acceptance = time.monotonic(), 0, None
            for start in range(0, len(shard), plan["batch_size"]):
                rows = shard[start:start + plan["batch_size"]]
                samples = [cache.read(row["split"], row["index"], include_images=True) for row in rows]
                conditions = condition_inputs(samples)
                indices = [row["row"] for row in rows]
                if paired_dino is not None and not np.array_equal(
                        conditions["img_embs"].numpy().reshape(len(rows), -1), paired_dino[indices]):
                    raise AssertionError("Incremental checkpoint historical images differ from the accepted baseline")
                noise = make_noise(rows, plan["seed"], plan["archive_fingerprint"])
                noise_digest.update(noise.numpy().tobytes())
                image_digest.update(conditions["img_embs"].numpy().tobytes())
                if acceptance is None:
                    acceptance = gpu_acceptance(expert, conditions, rows, plan)
                    atomic_json(job / f"gpu-check-{epoch}-{gpu}.json", acceptance)
                with torch.inference_mode():
                    pooled, _, _ = forward_hidden(expert, conditions, noise)
                features[indices] = pooled.cpu().numpy()
                if dino is not None:
                    dino[indices] = conditions["img_embs"].numpy().reshape(len(rows), -1)
                subset = [i for i, row in enumerate(rows) if row["row"] in stable_lookup]
                if subset:
                    chosen = [rows[i] for i in subset]
                    ids = [stable_lookup[row["row"]] for row in chosen]
                    stability[ids, 0] = pooled[subset].cpu().numpy()
                    subconditions = {key: value[subset] for key, value in conditions.items()}
                    for draw in (1, 2):
                        with torch.inference_mode():
                            value, _, _ = forward_hidden(expert, subconditions,
                                make_noise(chosen, plan["seed"], plan["archive_fingerprint"], draw))
                        stability[ids, draw] = value.cpu().numpy()
                completed += len(rows)
                elapsed = time.monotonic() - started
                if start == 0 or (start // plan["batch_size"]) % 8 == 0 or completed == len(shard):
                    atomic_json(job / f"extract-{epoch}-gpu{gpu}.json", {"epoch": epoch, "gpu": gpu,
                        "completed": completed, "total": len(shard), "elapsed_seconds": elapsed,
                        "windows_per_second": completed / elapsed, "peak_gpu_bytes": torch.cuda.max_memory_allocated(gpu)})
                    print(f"epoch={epoch} gpu={gpu} windows={completed}/{len(shard)} rate={completed/elapsed:.1f}/s", flush=True)
            features.flush()
            stability.flush()
            if dino is not None:
                dino.flush()
            guard = cache.assert_current()
            assert_reuse_current(plan)
            if any(p.grad is not None or p.requires_grad for p in expert.parameters()):
                raise AssertionError("Frozen expert acquired gradients")
            result = {"passed": True, "epoch": epoch, "gpu": gpu, "windows": completed,
                "rows_sha256": hashlib.sha256(np.asarray([row["row"] for row in shard], dtype=np.int64).tobytes()).hexdigest(),
                "noise_sha256": noise_digest.hexdigest(), "dino_sha256": image_digest.hexdigest(),
                "expert_trainable_parameters": 0, "expert_gradients": False, "acceptance": acceptance,
                "guard": guard, "elapsed_seconds": time.monotonic() - started}
            if epoch != 20:
                comparison_job = Path(plan["reuse_job"]["job_dir"]) if plan.get("reuse_job") else job
                previous = json.loads((comparison_job / f"done-20-gpu{gpu}.json").read_text())
                for name in ("rows_sha256", "noise_sha256", "dino_sha256", "windows"):
                    if result[name] != previous[name]:
                        raise AssertionError(f"Checkpoint comparisons differ in {name}")
            atomic_json(job / f"done-{epoch}-gpu{gpu}.json", result)
        except BaseException:
            traceback.print_exc()
            raise


def extract(plan, cache, entries):
    import torch
    from .checkpoints import read_checkpoint, checkpoint_config, validate_checkpoint_statistics
    from .dataset.normalization import MotionNormalizer
    from .visual_contract import checkpoint_visual_contract, compare_visual_contracts
    job = Path(plan["job_dir"])
    count = len(entries)
    epochs = plan.get("extract_epochs", [20, 100])
    arrays = [(f"epoch{epoch}", (count, 768)) for epoch in epochs]
    arrays += [(f"noise-epoch{epoch}", (len(plan["noise_rows"]), 3, 768)) for epoch in epochs]
    if not plan.get("reuse_job"):
        arrays.append(("dino", (count, 30720)))
    assert_reuse_current(plan)
    for name, shape in arrays:
        path = job / f"{name}.npy"
        if path.exists():
            raise FileExistsError(f"Refuse to overwrite prior probe artifacts: {path}")
        array = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
        array.flush()
        del array
    shards = make_shards(entries, plan["gpus"])
    context = mp.get_context("fork")
    normalizer, visual = None, None
    for epoch in epochs:
        event(job, "extracting", epoch=epoch, windows=count, gpu_count=len(plan["gpus"]))
        checkpoint = read_checkpoint(plan["checkpoints"][str(epoch)])
        if checkpoint.get("epoch") != epoch - 1:
            raise ValueError("Checkpoint completed epoch identity differs")
        if checkpoint.get("ema_state_format") != "original_model_with_ema_optimizer_v1":
            raise ValueError("Probe comparison requires original training EMA checkpoints")
        cfg = checkpoint_config(checkpoint)
        candidate = MotionNormalizer(checkpoint["motionexpert_normalization"])
        validate_checkpoint_statistics(cfg, checkpoint, candidate)
        candidate_visual = checkpoint_visual_contract(checkpoint)
        if plan.get("reuse_job"):
            # Statistics and architecture from an inherited checkpoint are
            # authoritative; the new training checkpoint must share both.
            baseline = read_checkpoint(plan["checkpoints"]["20"])
            baseline_cfg = checkpoint_config(baseline)
            baseline_normalizer = MotionNormalizer(baseline["motionexpert_normalization"])
            validate_checkpoint_statistics(cfg, checkpoint, baseline_normalizer)
            compare_visual_contracts(checkpoint_visual_contract(baseline), candidate_visual)
            from .config import architecture_metadata
            if architecture_metadata(cfg) != architecture_metadata(baseline_cfg):
                raise ValueError("Incremental checkpoint model architecture differs from the baseline")
            del baseline, baseline_cfg, baseline_normalizer
            gc.collect()
        if normalizer is not None:
            validate_checkpoint_statistics(cfg, checkpoint, normalizer)
            compare_visual_contracts(visual, candidate_visual)
        else:
            normalizer, visual = candidate, candidate_visual
        for key in ("archive_fingerprint", "split_fingerprint"):
            if normalizer.stats[key] != cache.binding[key]:
                raise ValueError(f"Frozen weights and cache differ: {key}")
        actual_visual = {"format": "motionexpert-visual-v1", "mode": "shared_rgb",
                         "image_process": cache.vision_manifest["image_process"],
                         "encoder": cache.vision_manifest["encoder"]}
        compare_visual_contracts(candidate_visual, actual_visual)
        if torch.cuda.is_initialized():
            raise RuntimeError("CPU parent must not initialize CUDA before extraction forks")
        children = [context.Process(target=run_rank, args=(rank, gpu, shards[rank], plan, cfg,
                    normalizer, visual, checkpoint, cache, epoch)) for rank, gpu in enumerate(plan["gpus"])]
        for child in children:
            child.start()
        try:
            while any(child.is_alive() for child in children):
                for child in children:
                    if child.exitcode not in (None, 0):
                        raise RuntimeError(f"Extraction child {child.pid} failed with exit {child.exitcode}")
                time.sleep(2)
            if any(child.exitcode != 0 for child in children):
                raise RuntimeError("Frozen feature extraction did not complete")
        except BaseException:
            for child in children:
                if child.is_alive():
                    child.terminate()
            for child in children:
                child.join(timeout=20)
            raise
        for child in children:
            child.join()
        for gpu in plan["gpus"]:
            if not json.loads((job / f"done-{epoch}-gpu{gpu}.json").read_text())["passed"]:
                raise RuntimeError("Missing passed GPU extraction evidence")
        if file_stamp(plan["checkpoints"][str(epoch)]) != plan["checkpoint_stamps"][str(epoch)]:
            raise RuntimeError("Read-only checkpoint identity changed")
        del checkpoint, candidate
        gc.collect()
    assert_reuse_current(plan)
    return {"passed": True, "windows": count, "gpu_count": len(plan["gpus"]), "epochs": epochs,
            "reused_epochs": [20, 100] if plan.get("reuse_job") else []}


def fit_and_record(job, device):
    job = Path(job).resolve()
    status_path = None
    try:
        from .prior_probe_fit import fit_job
        plan = json.loads((job / "plan.json").read_text())
        if Path(plan["job_dir"]).resolve() != job:
            raise ValueError("Probe plan belongs to another job")
        status_path = Path(plan["cache_root"]).parent.parent / ".pipeline/current_motionexpert_prior_probe.json"
        completed = json.loads((job / "extraction-complete.json").read_text())
        if completed.get("passed") is not True:
            raise ValueError("Readout fitting requires completed GPU extraction")
        for epoch, path in plan["checkpoints"].items():
            if file_stamp(path) != plan["checkpoint_stamps"][epoch]:
                raise ValueError("Frozen checkpoint changed since extraction")
        assert_reuse_current(plan)
        event(job, "fitting_frozen_readouts", device=device)
        atomic_json(status_path, {"state": "fitting_frozen_readouts", "pid": os.getpid(),
                                  "job_dir": str(job), "motionexpert_training_started": False})
        result = fit_job(job, device=device)
        assert_reuse_current(plan)
        event(job, "complete", report=str(job / "report.md"), metrics=str(job / "metrics.json"))
        atomic_json(status_path, {"state": "complete", "job_dir": str(job),
            "report": str(job / "report.md"), "metrics": str(job / "metrics.json"),
            "motionexpert_training_started": False})
    except BaseException as error:
        record_failure(job, error, status_path)
        raise
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="MotionExpert/configs/smplh_k12_psi0_pretrain.yaml")
    parser.add_argument("--checkpoint20")
    parser.add_argument("--checkpoint100")
    parser.add_argument("--checkpoint160")
    parser.add_argument("--reuse-job", help="Accepted original 20/100 probe; inherit all old inputs/readouts read-only")
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-episodes-per-source", type=int, default=1024)
    parser.add_argument("--val-episodes-per-source", type=int, default=256)
    parser.add_argument("--windows-per-episode", type=int, default=16)
    parser.add_argument("--seed", type=int, default=620017)
    parser.add_argument("--stage", choices=("extract", "fit", "all"), default="all")
    args = parser.parse_args()
    job = Path(args.output).resolve()
    repo = Path(__file__).resolve().parents[1]
    if job.is_relative_to(repo):
        raise ValueError("Probe artifacts must remain outside the repository")
    if args.stage == "fit":
        fit_and_record(job, device=f"cuda:{args.gpus.split(',')[0]}")
        return
    if args.reuse_job:
        if not args.checkpoint160 or args.checkpoint20 or args.checkpoint100:
            parser.error("--reuse-job requires --checkpoint160; original checkpoints come from the baseline plan")
    elif not args.checkpoint20 or not args.checkpoint100 or args.checkpoint160:
        parser.error("Use --checkpoint20 and --checkpoint100, or --reuse-job with --checkpoint160")
    import torch
    from .config import load_config
    from .dataset.probe_cache import AcceptedProbeCache
    torch.set_num_threads(1)
    cfg = load_config(args.config)
    gpus = [int(gpu) for gpu in args.gpus.split(",")]
    if not gpus or len(set(gpus)) != len(gpus) or args.batch_size < 4:
        raise ValueError("Use distinct GPU indices and batches of at least four windows")
    job.mkdir(parents=True, exist_ok=False)
    status_path = None
    try:
        event(job, "opening_accepted_cache")
        cache = AcceptedProbeCache(cfg.DATA.WINDOW_CACHE_ROOT, cfg.DATA.VISION_PROFILE_ROOT)
        if args.reuse_job:
            plan, entries = prepare_reuse(args.reuse_job, cache, args.checkpoint160, job, gpus, args.batch_size)
        else:
            entries = select_rows(cache, args.seed, args.train_episodes_per_source,
                                  args.val_episodes_per_source, args.windows_per_episode)
            atomic_json(job / "selection.json", entries)
            label_specs = prepare_labels(cache, entries, job)
            noise_rows = []
            for source in sorted({row["source"] for row in entries}):
                candidates = [row for row in entries if row["source"] == source and row["split"] == "val"]
                # One window per episode avoids a short contiguous block of one scene.
                by_episode = {}
                for row in candidates:
                    by_episode.setdefault(row["episode_index"], row["row"])
                noise_rows.extend(list(by_episode.values())[:128])
            checkpoints = {"20": str(Path(args.checkpoint20).resolve(strict=True)),
                           "100": str(Path(args.checkpoint100).resolve(strict=True))}
            counts = Counter((row["split"], row["source"]) for row in entries)
            plan = {"format": "motionexpert-frozen-prior-probe-v1", "job_dir": str(job), "seed": args.seed,
                "checkpoints": checkpoints, "checkpoint_stamps": {key: file_stamp(path) for key, path in checkpoints.items()},
                "cache_root": str(cache.root), "vision_root": str(cache.vision_root),
                "cache_manifest_sha256": cache.manifest_sha256,
                "archive_fingerprint": cache.binding["archive_fingerprint"],
                "split_fingerprint": cache.binding["split_fingerprint"],
                "gpus": gpus, "batch_size": args.batch_size, "precision": "float32", "weights": "ema",
                "windows": len(entries), "counts": {f"{split}/{source}": count for (split, source), count in counts.items()},
                "label_specs": label_specs, "noise_rows": sorted(noise_rows), "noise_draws": 3,
                "input": "six historical DINO frames only; all245 Gaussian channels; t=1; no trajectory",
                "readout": "all eleven motion hidden states mean; no motion parameter optimization",
                "extract_epochs": [20, 100], "fit_models": ["epoch20", "epoch100", "dino"],
                "sampling": {"train_episodes_per_source": args.train_episodes_per_source,
                             "val_episodes_per_source": args.val_episodes_per_source,
                             "max_windows_per_episode": args.windows_per_episode, "min_native_start_gap": 60},
                "source_files": {str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in (Path(__file__), repo / "MotionExpert/dataset/probe_cache.py",
                                              repo / "MotionExpert/prior_probe_math.py",
                                              repo / "MotionExpert/prior_probe_fit.py")}}
        atomic_json(job / "plan.json", plan)
        status_path = Path(cfg.DATA.ARCHIVE_ROOT).parent / ".pipeline/current_motionexpert_prior_probe.json"
        atomic_json(status_path, {"state": "extracting", "pid": os.getpid(), "job_dir": str(job),
                    "log_path": str(job.parent / (job.name + ".log")), "motionexpert_training_started": False})
        acceptance = extract(plan, cache, entries)
        atomic_json(job / "extraction-complete.json", acceptance)
        event(job, "features_extracted", **acceptance)
        if args.stage == "all":
            fit_and_record(job, device=f"cuda:{gpus[0]}")
        else:
            atomic_json(status_path, {"state": "features_extracted", "job_dir": str(job),
                                      "motionexpert_training_started": False})
    except BaseException as error:
        record_failure(job, error, status_path)
        raise


if __name__ == "__main__":
    main()
