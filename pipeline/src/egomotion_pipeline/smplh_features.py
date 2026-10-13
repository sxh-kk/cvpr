"""Versioned DINO caches from original cameras, without rewriting accepted motion.

Native30Hz frame IDs preserve every original frame. Offline extraction uses
deterministic pretraining geometry. Random window augmentation is online.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
from importlib.metadata import version, PackageNotFoundError
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil

import numpy as np
import pyarrow.parquet as pq

from .io import atomic_json, digest, file_sha256, read_json
from .dino_optimization import optimization_options, prepared_batches
from .features import DinoEncoder, extraction_options
from .contracts import DINO_SHAPE, PARAMETER_SCHEMA_VERSION
from .video import image_preprocessor, indexed_video_frames, nearest_mcap_video_frames

CONTRACT = "smplh-dino-shared-rgb-cache-v1"
JOIN_COLUMNS = ("episode_index", "frame_index", "index", "timestamp",
                "observation.source_timestamp", "source.video_frame_index", "observation.valid.dino")
_ENCODER = None


def _now():
    return datetime.now(timezone.utc).isoformat()


def _identity(path):
    stat = Path(path).stat()
    return {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _owned(root, name):
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("SMPL-H inputs must use owned relative paths")
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("SMPL-H input escapes its archive")
    return path


def _video(record):
    source = record.get("dino", {}).get("source_extraction", {})
    video = dict(source.get("video", {}))
    if record["source"] == "egosuite":
        if video.get("kind") != "mcap" or not video.get("topic") or "clock_origin_ns" not in video:
            raise ValueError("EgoSuite re-extraction requires original MCAP capture-clock provenance")
    elif record["source"] == "egoexo4d_hm":
        if not record.get("metadata", {}).get("frame_alignment", {}).get("verified"):
            raise ValueError("HM video mapping must retain its verified release frame contract")
        video["path"] = video.get("video_path", video.get("path"))
    else:
        raise ValueError("Unsupported SMPL-H visual source")
    if not video.get("path"):
        raise ValueError("Original camera video provenance is missing")
    video["path"] = str(Path(video["path"]).resolve(strict=True))
    expected = (source.get("video_fingerprint") if record["source"] == "egosuite"
                else video.get("video_fingerprint"))
    if expected is None or _identity(video["path"]) != expected:
        raise ValueError("Original video identity differs from its certified frame/clock provenance")
    video.setdefault("preprocessing", {"mode": "raw", "rotate_ccw_quarters": 0})
    return video


def plan_extraction(dataset, config, *, fps=30, episode_indices=None):
    from ImageProcess import SharedImageProcessor, load_profile
    if type(fps) is not int or fps != 30:
        raise ValueError("The visual cache preserves native30Hz; MotionExpert samples 5Hz when reading")
    extraction_options(config)
    optimization_options(config)
    profile = SharedImageProcessor(load_profile(config.get("image_profile", "256x480")))
    try:
        environment = {name: version(name) for name in ("torch", "transformers")}
    except PackageNotFoundError as error:
        raise ValueError("Install the pipeline DINO dependencies before extraction") from error
    checkpoint_identity = None
    if config.get("revision") == "local":
        checkpoint = Path(config.get("checkpoint", "")).resolve(strict=True)
        if not checkpoint.is_dir():
            raise ValueError("Local DINO checkpoint must be a directory")
        paths = sorted(p for p in checkpoint.rglob("*") if p.is_file() and p.suffix in {".json", ".bin", ".safetensors"})
        if not paths or not any(p.suffix in {".bin", ".safetensors"} for p in paths):
            raise ValueError("Local DINO checkpoint has no model weights")
        checkpoint_identity = {str(p.relative_to(checkpoint)): _identity(p) for p in paths}
    root = Path(dataset).resolve(strict=True)
    info = read_json(root / "meta/info.json")
    if info.get("schema_version") != PARAMETER_SCHEMA_VERSION or info.get("fps") != 30:
        raise ValueError("Expected the native30Hz SMPL-H parameter archive")
    state = read_json(root / ".pipeline/dataset-state.json")
    if (state.get("state") != "accepted" or state.get("independent_acceptance_pending") is not False):
        raise ValueError("Re-extraction requires an accepted SMPL-H source archive")
    records = [read_json(p) for p in sorted((root / ".pipeline/episodes").glob("*.json"))]
    if not records or any(r.get("episode_index") != i for i, r in enumerate(records)):
        raise ValueError("SMPL-H episode identities must be contiguous")
    selected = list(range(len(records))) if episode_indices is None else list(episode_indices)
    if (not selected or any(type(i) is not int or not 0 <= i < len(records) for i in selected)
            or len(set(selected)) != len(selected)):
        raise ValueError("Select unique existing episode IDs")
    items = []
    for i in selected:
        record = records[i]
        path = _owned(root, record["data_path"])
        receipt = read_json(_owned(root, f"episodes/{i:08d}/receipt.json"))
        if receipt.get("record") != record or receipt.get("schema_version") != PARAMETER_SCHEMA_VERSION:
            raise ValueError("SMPL-H record does not match its committed receipt")
        original = receipt.get("output_identities", {}).get("motion", {})
        if _identity(path) != {"size_bytes": original.get("bytes"), "mtime_ns": original.get("mtime_ns")}:
            raise ValueError("SMPL-H motion identity differs from its committed receipt")
        video = _video(record)
        items.append({"episode_index": i, "record": record, "motion_path": str(path),
                      "motion_identity": _identity(path), "video": video,
                      "video_identity": _identity(video["path"]),
                      "frames": len(range(0, record["length"], 30 // fps))})
    binding = {"info_sha256": file_sha256(root / "meta/info.json"),
               "episodes_metadata_sha256": file_sha256(root / "meta/episodes.parquet"),
               "split_metadata_sha256": {str(p.relative_to(root)): file_sha256(p)
                                         for p in sorted((root / "meta").glob("splits*.json"))},
               "records_sha256": digest(records)}
    return {"contract": CONTRACT, "source_root": str(root), "source_binding": binding,
            "image_process": profile.metadata, "fps": fps, "frame_phase": 0,
            "native_fps": 30, "sampling_step": 30 // fps, "shape": list(DINO_SHAPE),
            "storage_dtype": "float32", "checkpoint_identity": checkpoint_identity,
            "encoder_environment": environment,
            "scope": "all" if episode_indices is None else "subset",
            "episode_indices": selected, "source_episode_count": len(records), "items": items,
            "total_frames": sum(item["frames"] for item in items)}


def _read_join(item, step):
    table = pq.read_table(item["motion_path"], columns=list(JOIN_COLUMNS))
    if len(table) != item["record"]["length"]:
        raise ValueError("Motion length changed before visual extraction")
    selected = np.arange(0, len(table), step, dtype=np.int64)
    arrays = {k: table[k].combine_chunks().to_numpy(zero_copy_only=False)[selected].copy() for k in JOIN_COLUMNS}
    if (not np.array_equal(arrays["frame_index"], selected)
            or not np.all(arrays["episode_index"] == item["episode_index"])
            or not np.array_equal(arrays["index"], selected + item["record"]["dataset_from_index"])):
        raise ValueError("SMPL-H frame identity changed")
    return arrays


def _episode_signature(item, plan, encoder):
    return digest({"source_binding": plan["source_binding"], "record": item["record"],
                   "motion_identity": item["motion_identity"], "video": item["video"],
                   "video_identity": item["video_identity"], "fps": plan["fps"],
                   "frame_phase": plan["frame_phase"], "max_video_error_s": plan["max_video_error_s"],
                   "encoder": encoder.metadata})


def _resume(directory, signature):
    receipt_path = directory / "receipt.json"
    if not receipt_path.exists():
        return None
    receipt = read_json(receipt_path)
    if receipt.get("state") != "complete" or receipt.get("signature") != signature:
        raise ValueError("Existing visual cache uses a different source/encoder/profile; choose a new output")
    for name, identity in receipt["output_identities"].items():
        if _identity(directory / name) != identity:
            raise ValueError("Committed visual cache file changed")
    value = np.load(directory / "features.npy", mmap_mode="r", allow_pickle=False)
    if value.shape != (receipt["frames"], *DINO_SHAPE) or value.dtype != np.float32:
        raise ValueError("Committed visual cache shape/dtype differs")
    return receipt


def extract_episode(item, plan, output, config, encoder):
    """Commit one projected-frame cache; original motion/vision are read only."""
    if _identity(item["motion_path"]) != item["motion_identity"] or _identity(item["video"]["path"]) != item["video_identity"]:
        raise ValueError("Original extraction input changed")
    directory = Path(output) / "episodes" / f"{item['episode_index']:08d}"
    if not directory.resolve().is_relative_to(Path(output).resolve()):
        raise ValueError("Visual output episode directory escapes the profile cache")
    signature = _episode_signature(item, plan, encoder)
    saved = _resume(directory, signature)
    if saved is not None:
        return saved
    directory.mkdir(parents=True, exist_ok=True)
    join = _read_join(item, plan["sampling_step"])
    count = len(join["frame_index"])
    original_valid = join["observation.valid.dino"].astype(bool).copy()
    join["source.valid.dino"] = original_valid.copy()
    join["observation.valid.dino"][:] = False
    join["observation.video_timestamp"] = np.zeros(count, np.float64)
    join["observation.video_time_error"] = np.zeros(count, np.float64)
    batch_size, tolerance = extraction_options(config)
    options = optimization_options(config)
    video = item["video"]
    if item["record"]["source"] == "egosuite":
        targets = (join["observation.source_timestamp"] - float(video.get("timestamp_origin", 0))
                   + float(video["from_timestamp"]))
        frames = nearest_mcap_video_frames(video, targets, tolerance, decoder_threads=options["decoder_threads"])
    else:
        ids = join["source.video_frame_index"]
        if ids.dtype.kind not in "iu" or np.any(ids < 0):
            raise ValueError("HM exact video frame IDs are invalid")
        targets = ids / 30
        frames = indexed_video_frames(video["path"], ids, decoder_threads=options["decoder_threads"])
    geometry = image_preprocessor(video["preprocessing"])
    def batches():
        images, indices, pts_values = [], [], []
        try:
            for index, (image, pts) in enumerate(frames):
                if image is None or not original_valid[index] or pts is None or abs(pts - targets[index]) > tolerance:
                    continue
                if video.get("kind") == "mcap" and not float(video["from_timestamp"])-1e-6 <= pts <= float(video["to_timestamp"])+1e-6:
                    continue
                images.append(geometry(image)); indices.append(index); pts_values.append(pts)
                if len(images) == batch_size:
                    yield images, indices, pts_values
                    images, indices, pts_values = [], [], []
            if images:
                yield images, indices, pts_values
        finally:
            frames.close()
    feature_path = directory / "features.npy.tmp"
    join_path = directory / "join.npz.tmp"
    if feature_path.is_symlink() or join_path.is_symlink():
        raise ValueError("Visual output temporary files must not be symlinks")
    features = np.lib.format.open_memmap(feature_path, mode="w+", dtype=np.float32, shape=(count, *DINO_SHAPE))
    features[:] = 0
    prepared = prepared_batches(batches(), encoder.prepare, options["prefetch_batches"])
    try:
        for inputs, indices, pts_values in prepared:
            values = np.asarray(encoder.encode_prepared(inputs))
            if values.dtype != np.float32 or values.shape != (len(indices), *DINO_SHAPE) or not np.isfinite(values).all():
                raise ValueError("DINO must return finite float32 CLS + registers [N,5,1024]")
            features[indices] = values
            join["observation.valid.dino"][indices] = True
            join["observation.video_timestamp"][indices] = pts_values
            join["observation.video_time_error"][indices] = np.asarray(pts_values) - targets[indices]
        features.flush()
        del features
        with join_path.open("wb") as stream:
            np.savez(stream, **join)
        if _identity(item["motion_path"]) != item["motion_identity"] or _identity(video["path"]) != item["video_identity"]:
            raise ValueError("Original extraction input changed during encoding")
        if not np.array_equal(join["observation.valid.dino"], original_valid):
            raise ValueError("Re-extracted DINO validity differs from source; no episode was committed")
        os.replace(feature_path, directory / "features.npy")
        os.replace(join_path, directory / "join.npz")
        receipt = {"contract": CONTRACT, "state": "complete", "signature": signature,
                   "episode_index": item["episode_index"], "source_key": item["record"]["source_key"],
                   "source_record_sha256": digest(item["record"]), "frames": count,
                   "valid_count": int(join["observation.valid.dino"].sum()),
                   "validity_equal_source": bool(np.array_equal(join["observation.valid.dino"], original_valid)),
                   "encoder": encoder.metadata, "max_video_error_s": tolerance,
                   "output_identities": {name: _identity(directory / name) for name in ("features.npy", "join.npz")},
                   "completed_at": _now()}
        atomic_json(directory / "receipt.json", receipt)
        return receipt
    finally:
        prepared.close()
        for path in (feature_path, join_path):
            if path.exists():
                path.unlink()


def _initialize_worker(config, devices, diagnostics=None):
    global _ENCODER
    gpu = devices.get()
    try:
        import torch
        torch.set_num_threads(optimization_options(config)["cpu_threads"])
        torch.cuda.set_device(gpu)
        _ENCODER = DinoEncoder(dict(config, device=f"cuda:{gpu}"))
        print(f"DINO worker ready: pid={os.getpid()} gpu={gpu}", flush=True)
    except BaseException as error:
        if diagnostics is not None:
            import traceback
            atomic_json(Path(diagnostics) / f"{os.getpid()}.json", {
                "gpu": gpu, "pid": os.getpid(), "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(), "at": _now()})
        raise


def _worker_episode(item, plan, output, config):
    return extract_episode(item, plan, output, config, _ENCODER)


@contextmanager
def _output_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".extraction.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def run_extraction(dataset, output, config, *, fps=30, episode_indices=None, gpu_ids=None,
                   workers_per_gpu=1, dry_run=False, encoder=None):
    plan = plan_extraction(dataset, config, fps=fps, episode_indices=episode_indices)
    output = Path(output).resolve()
    source = Path(plan["source_root"])
    if output == source or output == source / "episodes" or source.is_relative_to(output):
        raise ValueError("Use a new dedicated visual-profile directory, not the source archive")
    if any(output == Path(item["motion_path"]).parent for item in plan["items"]):
        raise ValueError("Cannot write visual profiles into committed source episode directories")
    if gpu_ids is not None and (not gpu_ids or any(type(i) is not int or i < 0 for i in gpu_ids) or len(set(gpu_ids)) != len(gpu_ids)):
        raise ValueError("GPU IDs must be unique nonnegative integers")
    if type(workers_per_gpu) is not int or workers_per_gpu < 1 or (gpu_ids is None and workers_per_gpu != 1):
        raise ValueError("workers_per_gpu must be a positive integer and multiple workers require --gpus")
    public = {k: v for k, v in plan.items() if k != "items"}
    public["output_root"] = str(output)
    public["config"] = config
    public["max_video_error_s"] = extraction_options(config)[1]
    plan["max_video_error_s"] = public["max_video_error_s"]
    if dry_run:
        return dict(public, state="ready", estimated_feature_bytes=plan["total_frames"] * 5 * 1024 * 4)
    manifest_path = output / "manifest.json"
    # Physical GPU selection may change on resume. Other policies remain explicit.
    public["encoder_config"] = {k: v for k, v in config.items() if k != "device"}
    binding = digest({k: public[k] for k in ("contract", "source_binding", "image_process", "fps", "frame_phase",
                                           "episode_indices", "max_video_error_s", "encoder_config", "checkpoint_identity", "encoder_environment")})
    state = dict(public, binding=binding, state="extracting", completed_episodes=0, completed_frames=0,
                 valid_frames=0, validity_equal_source=True, updated_at=_now())
    state["execution"] = {"gpu_ids": gpu_ids, "workers_per_gpu": workers_per_gpu}
    with _output_lock(output):
        if not manifest_path.exists() and any(p.name != ".extraction.lock" for p in output.iterdir()):
            raise ValueError("Output is not an empty directory or a recognized visual-profile cache")
        old = read_json(manifest_path) if manifest_path.exists() else None
        if old is not None and old.get("binding") != binding:
            raise ValueError("Existing visual profile has a different source/configuration; choose a new output")
        if shutil.disk_usage(output).free < plan["total_frames"] * 5 * 1024 * 4 and old is None:
            raise OSError("Insufficient disk space for the float32 DINO cache")
        atomic_json(manifest_path, state)
        def commit(receipt):
            if state.get("encoder") is None:
                state["encoder"] = receipt["encoder"]
            elif state["encoder"] != receipt["encoder"]:
                raise ValueError("DINO workers returned different encoders/preprocessing")
            state["completed_episodes"] += 1
            state["completed_frames"] += receipt["frames"]
            state["valid_frames"] += receipt["valid_count"]
            state["validity_equal_source"] &= receipt["validity_equal_source"]
            state["updated_at"] = _now()
            atomic_json(manifest_path, state)
            print(f"DINO profile {receipt['source_key']}: {receipt['valid_count']}/{receipt['frames']} "
                  f"episodes={state['completed_episodes']}/{len(plan['items'])}", flush=True)
        try:
            if gpu_ids is not None:
                if encoder is not None:
                    raise ValueError("Injected encoders are single-process only")
                context = mp.get_context("spawn")
                devices = context.Queue()
                for _ in range(workers_per_gpu):
                    for gpu in gpu_ids:
                        devices.put(gpu)
                worker_plan = {k: v for k, v in plan.items() if k != "items"}
                # Bound outstanding work to one episode per GPU; errors cannot queue the entire dataset.
                worker_count = len(gpu_ids) * workers_per_gpu
                pool = ProcessPoolExecutor(max_workers=worker_count, mp_context=context,
                                           initializer=_initialize_worker,
                                           initargs=(config, devices, output / "worker_errors"))
                # Python3.10's broken-pool shutdown can join a feeder blocked
                # on a full pipe after all consumers died. Completed futures
                # already acknowledge all calls; cancellation avoids that join.
                if getattr(pool, "_call_queue", None) is not None:
                    pool._call_queue.cancel_join_thread()
                pending = set()
                try:
                    remaining = iter(plan["items"])
                    pending = {pool.submit(_worker_episode, item, worker_plan, output, config)
                               for item in [next(remaining, None) for _ in range(worker_count)] if item is not None}
                    while pending:
                        ready, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for future in ready:
                            commit(future.result())
                            item = next(remaining, None)
                            if item is not None:
                                pending.add(pool.submit(_worker_episode, item, worker_plan, output, config))
                except BaseException:
                    for future in pending:
                        future.cancel()
                    # Python3.10 has no terminate_workers(). Capture only this pool's children.
                    processes = list((getattr(pool, "_processes", None) or {}).values())
                    for process in processes:
                        if process.is_alive():
                            process.terminate()
                    for process in processes:
                        process.join(timeout=5)
                        if process.is_alive():
                            process.kill()
                            process.join(timeout=5)
                    pool.shutdown(wait=True, cancel_futures=True)
                    raise
                else:
                    pool.shutdown(wait=True)
                finally:
                    devices.close()
                    devices.join_thread()
            else:
                owns = encoder is None
                encoder = encoder or DinoEncoder(config)
                try:
                    for item in plan["items"]:
                        commit(extract_episode(item, plan, output, config, encoder))
                finally:
                    if owns:
                        encoder.close()
            state.update(state="complete", updated_at=_now())
            atomic_json(manifest_path, state)
            return state
        except BaseException as error:
            state.update(state="failed", error=f"{type(error).__name__}: {error}", updated_at=_now())
            atomic_json(manifest_path, state)
            raise
