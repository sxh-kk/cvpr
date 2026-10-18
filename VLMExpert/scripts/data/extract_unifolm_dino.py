#!/usr/bin/env python3
"""Extract native-frame UnifoLM DINO caches, using only explicit local weights.

The output root belongs to one LeRobot pack. Completed episodes are resumed;
interrupted episodes are recomputed and never exposed to the training reader.
No dataset, model weights or dependencies are downloaded by this command.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import sys

if __name__ == "__main__":
    print(json.dumps({"stage": "imports", "module": "numpy", "at": datetime.now(timezone.utc).isoformat()}), flush=True)
import numpy as np

# The extraction runtime needs numpy/PIL/PyAV/pyarrow/transformers/torch, not
# LeRobot and the training stack imported by psi.data's package initializers.
REPOSITORY = Path(__file__).resolve().parents[3]
for location in (REPOSITORY, REPOSITORY / "pipeline/src"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))
_SPEC = importlib.util.spec_from_file_location(
    "_unifolm_dino_cache", REPOSITORY / "VLMExpert/src/psi/data/lerobot/dino_cache.py")
cache = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = cache
_SPEC.loader.exec_module(cache)


def now():
    return datetime.now(timezone.utc).isoformat()


def stage(name, **details):
    print(json.dumps({"stage": name, "at": now(), **details}), flush=True)


def plan_source(dataset_root, *, video_key="observation.images.egocentric", episodes=None,
                num_shards=1, shard_index=0):
    root = Path(dataset_root).resolve(strict=True)
    info = cache.read_json(cache.owned_file(root, "meta/info.json"))
    if info.get("fps") != 30:
        raise ValueError("UnifoLM DINO extraction preserves the native30Hz source; resampling is forbidden")
    feature = info.get("features", {}).get(video_key, {})
    if feature.get("dtype") != "video" or feature.get("video_info", {}).get("video.fps") != 30:
        raise ValueError("Select an explicit native30Hz video feature from meta/info.json")
    records = []
    with cache.owned_file(root, "meta/episodes.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                records.append(json.loads(line))
    indexed = {record["episode_index"]: record for record in records}
    if len(indexed) != len(records) or not records or len(records) != info.get("total_episodes"):
        raise ValueError("LeRobot episode metadata contains duplicate/missing episode identities")
    selected = sorted(indexed) if episodes is None else list(episodes)
    if not selected or len(set(selected)) != len(selected) or any(index not in indexed for index in selected):
        raise ValueError("Select unique existing episode indices")
    if (type(num_shards) is not int or num_shards < 1 or type(shard_index) is not int
            or not 0 <= shard_index < num_shards):
        raise ValueError("Require num_shards >= 1 and 0 <= shard_index < num_shards")
    source = {"dataset_root": str(root), "fps": info["fps"], "video_key": video_key,
              "episode_indices": selected,
              "total_episodes": len(records), "total_frames": info["total_frames"],
              "meta_files": {str(path.relative_to(root)): cache.file_identity(path)
                             for path in sorted((root / "meta").iterdir())
                             if path.is_file() and path.suffix in (".json", ".jsonl")}}
    chunks_size = info.get("chunks_size", 1000)
    if type(chunks_size) is not int or chunks_size < 1:
        raise ValueError("LeRobot chunks_size must be positive")
    items = []
    for position, index in enumerate(selected):
        if position % num_shards != shard_index:
            continue
        record = indexed[index]
        values = {"episode_chunk": index // chunks_size, "episode_index": index, "video_key": video_key}
        paths = {"parquet": info["data_path"].format(**values), "video": info["video_path"].format(**values)}
        item = {"episode_index": index, "record": record, "record_sha256": cache.digest(record)}
        for kind, relative in paths.items():
            path = cache.owned_file(root, relative)
            item[kind] = {"path": str(path.relative_to(root)), "identity": cache.file_identity(path)}
        items.append(item)
    return source, items


def build_manifest(source, items, encoder_metadata, *, shared_image_profile="256x320", storage_dtype="float16"):
    from ImageProcess import SharedImageProcessor

    if storage_dtype not in ("float16", "float32"):
        raise ValueError("storage_dtype must be float16 or float32")
    metadata = dict(encoder_metadata)
    local_files = metadata.get("local_file_sha256", {})
    if metadata.get("revision") != "local" or not local_files:
        raise ValueError("Extraction requires fingerprinted explicit local DINO weights")
    metadata.update(model_id=cache.DINO_MODEL_ID, weights_sha256=cache.digest(local_files))
    image_process = SharedImageProcessor(shared_image_profile).metadata
    if metadata.get("image_process") != image_process:
        raise ValueError("DINO encoder and extraction must share the exact same ImageProcess profile")
    manifest = {"contract": cache.CONTRACT, "state": "extracting", "source": source,
                "image_process": image_process, "encoder": metadata,
                "extraction": {"stage": "validation", "training": False, "sampling_step": 1},
                "storage_dtype": storage_dtype, "shape": list(cache.DINO_SHAPE),
                "episode_indices": source["episode_indices"]}
    manifest["signature"] = cache.manifest_signature(manifest)
    return manifest


def read_episode_rows(source_root, item):
    import pyarrow.parquet as pq

    table = pq.read_table(cache.owned_file(source_root, item["parquet"]["path"]),
                          columns=["episode_index", "frame_index", "timestamp"])
    arrays = {name: np.asarray(table[name].combine_chunks().to_pylist()).reshape(-1)
              for name in table.column_names}
    frames = arrays["frame_index"]
    timestamp = arrays["timestamp"].astype(np.float64)
    if (not len(frames) or frames.dtype.kind not in "iu" or np.any(frames < 0)
            or np.any(np.diff(frames.astype(np.int64)) <= 0)
            or not np.all(arrays["episode_index"] == item["episode_index"])
            or not np.isfinite(timestamp).all() or np.any(np.diff(timestamp) <= 0)
            or len(frames) != item["record"]["length"]):
        raise ValueError("LeRobot episode frame/timestamp identities or metadata length differ")
    return frames.astype(np.int64), timestamp


def native_video_frames(path, *, decoder_threads=1):
    """Decode every original video frame once, retaining its ordinal and PTS."""
    import av

    with av.open(str(path)) as container:
        if len(container.streams.video) != 1:
            raise ValueError("Expected exactly one camera stream per LeRobot episode video")
        stream = container.streams.video[0]
        if stream.average_rate is None or abs(float(stream.average_rate) - 30) > 1e-6:
            raise ValueError("Episode video does not declare native30Hz")
        stream.codec_context.thread_count = decoder_threads
        for ordinal, frame in enumerate(container.decode(video=0)):
            pts = np.nan if frame.pts is None else float(frame.pts * frame.time_base)
            yield ordinal, pts, frame.to_ndarray(format="rgb24")


@contextmanager
def output_lock(output, *, blocking=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".extraction.lock"
    if lock.is_symlink():
        raise ValueError("Cache lock must not be a symlink")
    with lock.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _check_inputs(source_root, item):
    for kind in ("video", "parquet"):
        value = item[kind]
        cache.check_file_identity(cache.owned_file(source_root, value["path"]), value["identity"])


def _extract_episode(item, manifest, output, encoder, *, batch_size=32, workers=4,
                     decoder_threads=1, frame_iterator=None, row_reader=None):
    """Commit an episode, with injectable small CPU fixtures for regression tests."""
    from ImageProcess import SharedImageProcessor

    if type(batch_size) is not int or batch_size < 1 or type(workers) is not int or workers < 1:
        raise ValueError("batch_size and workers must be positive integers")
    source_root = Path(manifest["source"]["dataset_root"])
    directory = cache.episode_directory(output, item["episode_index"])
    _check_inputs(source_root, item)
    if (directory / "receipt.json").exists():
        # The top-level manifest may still be extracting during a resumed run.
        saved = cache.load_episode_cache(output, item["episode_index"], manifest=manifest, _allow_incomplete=True)
        if saved.receipt["source"] != item:
            raise ValueError("Committed episode differs from current source fingerprints")
        return saved.receipt
    directory.mkdir(parents=True, exist_ok=True)
    read_rows = row_reader or read_episode_rows
    frame_index, timestamp = read_rows(source_root, item)
    count = len(frame_index)
    sidecar = {"frame_index": frame_index, "timestamp": timestamp,
               "source_frame_index": np.arange(count, dtype=np.int64),
               "video_timestamp": np.full(count, np.nan, dtype=np.float64),
               "valid": np.zeros(count, dtype=np.bool_)}
    shared = SharedImageProcessor(manifest["image_process"]["profile"])
    feature_temporary = directory / "features.npy.partial"
    frames_temporary = directory / "frames.npz.partial"
    for path in (feature_temporary, frames_temporary, directory / "features.npy", directory / "frames.npz"):
        if path.is_symlink():
            raise ValueError("DINO output files must not be symlinks")
    features = np.lib.format.open_memmap(feature_temporary, mode="w+",
                                         dtype=manifest["storage_dtype"], shape=(count, *cache.DINO_SHAPE))
    frames = (frame_iterator if frame_iterator is not None else native_video_frames(
        cache.owned_file(source_root, item["video"]["path"]), decoder_threads=decoder_threads))
    tolerance = 0.25 / 30
    last_pts = -np.inf
    seen = 0
    images, indices = [], []

    def geometry(image):
        return shared.process_window([image], stage="validation", training=False).images[0]

    def encode(pool):
        processed = list(pool.map(geometry, images))
        inputs = encoder.prepare(processed, processed=True, image_processor=shared)
        values = np.asarray(encoder.encode_prepared(inputs))
        if values.shape != (len(indices), *cache.DINO_SHAPE) or not np.isfinite(values).all():
            raise ValueError("DINO must return finite CLS + 4 register features [B,5,1024]")
        values = values.astype(manifest["storage_dtype"])
        if not np.isfinite(values).all():
            raise ValueError("DINO features overflowed the selected storage dtype")
        features[indices] = values

    try:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="unifolm-shared-rgb") as pool:
            for ordinal, pts, image in frames:
                if ordinal != seen or ordinal >= count:
                    raise ValueError("Video and parquet do not have a verified one-frame-per-row mapping")
                seen += 1
                if not np.isfinite(pts) or pts <= last_pts:
                    raise ValueError("Native video PTS are missing or not strictly increasing")
                last_pts = pts
                sidecar["video_timestamp"][ordinal] = pts
                # Video and LeRobot timestamps must already share their clock;
                # no guessed offset, seeking or nearest-frame duplication.
                sidecar["valid"][ordinal] = abs(pts - timestamp[ordinal]) <= tolerance
                images.append(image)
                indices.append(ordinal)
                if len(images) == batch_size:
                    encode(pool)
                    images, indices = [], []
            if seen != count:
                raise ValueError(f"Video decoded {seen} frames but parquet has {count}; no episode was committed")
            if images:
                encode(pool)
        if not sidecar["valid"].all():
            raise ValueError("Video PTS do not align with the original LeRobot timestamps; no guessed mapping was committed")
        features.flush()
        del features
        with feature_temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        with frames_temporary.open("wb") as stream:
            np.savez(stream, **sidecar)
            stream.flush()
            os.fsync(stream.fileno())
        _check_inputs(source_root, item)
        os.replace(feature_temporary, directory / "features.npy")
        os.replace(frames_temporary, directory / "frames.npz")
        receipt = {"contract": cache.CONTRACT, "state": "complete", "episode_index": item["episode_index"],
                   "manifest_signature": manifest["signature"], "source": item,
                   "signature": cache.digest({"manifest_signature": manifest["signature"], "source": item}),
                   "frames": count, "valid_frames": int(sidecar["valid"].sum()),
                   "outputs": {name: cache.file_identity(directory / name)
                               for name in ("features.npy", "frames.npz")}, "completed_at": now()}
        # This receipt is the episode's atomic completion marker.
        cache.atomic_json(directory / "receipt.json", receipt)
        return receipt
    finally:
        close = getattr(frames, "close", None)
        if close is not None:
            close()
        for path in (feature_temporary, frames_temporary):
            if path.exists():
                path.unlink()


def extract_episode(item, manifest, output, encoder, **kwargs):
    # Multiple independently launched GPU shards can safely share this cache.
    # Overlapping episode selections refuse a simultaneous writer immediately.
    with output_lock(cache.episode_directory(output, item["episode_index"]), blocking=False):
        return _extract_episode(item, manifest, output, encoder, **kwargs)


def extract_dataset(dataset_root, output, encoder, *, shared_image_profile="256x320",
                    storage_dtype="float16", video_key="observation.images.egocentric",
                    episodes=None, batch_size=32, workers=4, decoder_threads=1,
                    num_shards=1, shard_index=0):
    stage("planning", dataset_root=str(dataset_root), shard_index=shard_index, num_shards=num_shards)
    source, items = plan_source(dataset_root, video_key=video_key, episodes=episodes,
                                num_shards=num_shards, shard_index=shard_index)
    manifest = build_manifest(source, items, encoder.metadata, shared_image_profile=shared_image_profile,
                              storage_dtype=storage_dtype)
    stage("planning_ready", assigned_episodes=len(items), expected_episodes=len(manifest["episode_indices"]),
          assigned_frames=sum(item["record"]["length"] for item in items), signature=manifest["signature"])
    output = Path(output)
    if output.resolve().is_relative_to(Path(dataset_root).resolve()):
        raise ValueError("Keep DINO caches outside the read-only source LeRobot pack")
    with output_lock(output):
        path = output / "manifest.json"
        if path.exists():
            previous = cache.read_json(path)
            if previous.get("signature") != manifest["signature"] or cache.manifest_signature(previous) != manifest["signature"]:
                raise ValueError("Output already belongs to another source/profile/encoder; choose a new cache root")
        elif any(child.name != ".extraction.lock" for child in output.iterdir()):
            raise ValueError("Refusing a nonempty output without an owned extraction manifest")
        # Do not downgrade an already completed cache when rerunning a shard.
        if not path.exists() or previous.get("state") != "complete":
            cache.atomic_json(path, manifest)
    completed_frames = 0
    for item in items:
        receipt = extract_episode(item, manifest, output, encoder, batch_size=batch_size, workers=workers,
                                  decoder_threads=decoder_threads)
        completed_frames += receipt["frames"]
        print(json.dumps({"episode_index": item["episode_index"], "state": "complete", "frames": receipt["frames"],
                          "shard_index": shard_index, "shard_completed_frames": completed_frames}), flush=True)
    with output_lock(output):
        # Recheck the complete source metadata binding immediately before commit.
        for relative, identity in source["meta_files"].items():
            cache.check_file_identity(cache.owned_file(dataset_root, relative), identity, checksum=True)
        completed = []
        for index in manifest["episode_indices"]:
            marker = cache.episode_directory(output, index) / "receipt.json"
            if marker.exists():
                receipt = cache.read_json(marker)
                if receipt.get("state") != "complete" or receipt.get("manifest_signature") != manifest["signature"]:
                    raise ValueError("A committed episode belongs to a stale extraction")
                completed.append(index)
        if len(completed) == len(manifest["episode_indices"]):
            manifest.update(state="complete", completed_at=now())
            cache.atomic_json(path, manifest)
            cache.load_cache_manifest(output, dataset_root, shared_image_profile)
        print(json.dumps({"state": manifest["state"], "completed_episodes": len(completed),
                          "expected_episodes": len(manifest["episode_indices"]), "shard_index": shard_index}), flush=True)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Cache root for exactly one LeRobot pack")
    parser.add_argument("--model-path", required=True, type=Path, help="Explicit local dinov2-with-registers-large directory")
    parser.add_argument("--shared-image-profile", "--image-profile", default="256x320")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4, help="CPU workers applying canonical shared RGB geometry")
    parser.add_argument("--decoder-threads", type=int, default=1)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--storage-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--inference-precision", choices=("fp32", "tf32", "bf16"), default="fp32")
    parser.add_argument("--video-key", default="observation.images.egocentric")
    parser.add_argument("--episodes", type=int, nargs="+", help="Explicit subset; use a separate output from a full-pack cache")
    parser.add_argument("--num-shards", type=int, default=1, help="Independent GPU processes sharing one output root")
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args(argv)
    for name in ("batch_size", "workers", "decoder_threads", "cpu_threads"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("Require --num-shards >= 1 and 0 <= --shard-index < --num-shards")
    model_path = args.model_path.resolve(strict=True)
    if not model_path.is_dir():
        parser.error("--model-path must be an existing local directory")
    # The existing pipeline validates the actual ViT-L/14 model configuration,
    # freezes its weights, fingerprints every model/config file and explicitly
    # disables HF resize/center-crop after shared RGB processing.
    stage("imports", module="torch_and_pipeline")
    import torch
    from egomotion_pipeline.features import DinoEncoder

    torch.set_num_threads(args.cpu_threads)
    stage("encoder_loading", model_path=str(model_path), device=args.device,
          shared_image_profile=args.shared_image_profile, inference_precision=args.inference_precision)
    encoder = DinoEncoder({"checkpoint": str(model_path), "revision": "local", "device": args.device,
                           "image_profile": args.shared_image_profile, "inference_precision": args.inference_precision})
    stage("encoder_ready", image_profile=encoder.shared_processor.profile.profile_id,
          local_file_count=len(encoder.metadata["local_file_sha256"]), frozen=True)
    try:
        extract_dataset(args.dataset_root, args.output, encoder, shared_image_profile=args.shared_image_profile,
                        storage_dtype=args.storage_dtype, video_key=args.video_key, episodes=args.episodes,
                        batch_size=args.batch_size, workers=args.workers, decoder_threads=args.decoder_threads,
                        num_shards=args.num_shards, shard_index=args.shard_index)
    finally:
        encoder.close()


if __name__ == "__main__":
    main()
