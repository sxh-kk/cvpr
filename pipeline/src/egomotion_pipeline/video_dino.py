"""Resumable video-only DINO sidecars using the existing EgoSuite encoder.

The 30 Hz clock belongs to the frame-aligned take, not to merged HM NPZ rows.
No motion, camera-pose, or joint-axis certification is created by this module.
"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
import fcntl
import json
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path, PurePosixPath
import shutil
import time
import traceback

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .body_model import file_sha256
from .dino_optimization import optimization_options, prepared_batches
from .features import DinoEncoder, extraction_options
from .schema import DINO_FLAT_DIM, DINO_SHAPE, dino_format
from .storage import atomic_json, digest, read_json
from .video import image_preprocessor, nearest_video_frames

FEATURE = "observation.features.dino"
VALID = "observation.valid.dino"
PTS = "observation.video_timestamp"
ERROR = "observation.video_time_error"
CONTRACT = "egoexo-video-dino5-30hz-v1"
SCHEMA = pa.schema([
    pa.field("frame_index", pa.int64(), nullable=False),
    pa.field("timestamp", pa.float64(), nullable=False),
    pa.field("observation.source_timestamp", pa.float64(), nullable=False),
    pa.field(PTS, pa.float64(), nullable=False),
    pa.field(ERROR, pa.float64(), nullable=False),
    pa.field(VALID, pa.bool_(), nullable=False),
    pa.field(FEATURE, pa.list_(pa.float32(), DINO_FLAT_DIM), nullable=False),
])


def now():
    return datetime.now(timezone.utc).isoformat()


def input_fingerprint(path):
    stat = Path(path).stat()
    return {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _inspect_video(item):
    with av.open(item["video_path"]) as container:
        if not container.streams.video:
            raise ValueError("No video stream: " + item["video_path"])
        stream = container.streams.video[0]
        info = {"width": stream.width, "height": stream.height, "frames": stream.frames,
                "average_rate": str(stream.average_rate), "time_base": str(stream.time_base),
                "start_time_s": float(stream.start_time * stream.time_base) if stream.start_time is not None else None,
                "duration_s": float(stream.duration * stream.time_base) if stream.duration is not None else None,
                "codec": stream.codec_context.name, "rotation_tag": stream.metadata.get("rotate")}
        if info["rotation_tag"] not in {None, "0"}:
            raise ValueError("Video rotation needs an explicit geometry policy: " + item["video_path"])
    return {**item, "video_info": info, "video_fingerprint": input_fingerprint(item["video_path"])}


def plan_videos(root, output, workers=16):
    """Select exactly the ego RGB video files from the existing download plan."""
    root, output = Path(root).resolve(), Path(output).resolve()
    manifest = read_json(root / "download_plan.json")
    takes = {t["take_name"]: t for t in read_json(root / "EgoExo4D/takes.json")}
    items = []
    for entry in manifest["s3_files"]:
        if entry["part"] != "takes":
            continue
        relative = PurePosixPath(entry["relative_path"])
        if len(relative.parts) != 4 or relative.parts[0] != "takes" or relative.suffix != ".mp4" or ".." in relative.parts:
            raise ValueError("Unexpected ego video path: " + str(relative))
        take = takes[relative.parts[1]]
        matches = []
        for camera in take["capture"]["cameras"]:
            if not camera["is_ego"]:
                continue
            for stream in take["frame_aligned_videos"].get(camera["cam_id"], {}).values():
                if (not stream.get("is_collage") and stream.get("relative_path")
                        and str(PurePosixPath(take["root_dir"]) / stream["relative_path"]) == str(relative)):
                    matches.append(camera)
        if len(matches) != 1:
            raise ValueError("Video does not map to one ego camera: " + str(relative))
        path = root / "EgoExo4D" / relative
        if not path.is_file() or path.stat().st_size != entry["size"]:
            raise ValueError("Missing/incomplete source video: " + str(path))
        camera = matches[0]
        count = int(take["timesync_end_idx"] - take["timesync_start_idx"])
        if count < 1:
            raise ValueError("Empty take: " + take["take_name"])
        items.append({"video_id": f"{take['take_name']}/{relative.stem}",
                      "take_name": take["take_name"], "take_uid": take["take_uid"],
                      "camera": camera["cam_id"], "device_type": camera.get("device_type"),
                      "video_path": str(path), "relative_video_path": str(relative),
                      "output_path": str(output / "takes" / take["take_name"] / (relative.stem + ".parquet")),
                      "length": count, "fps": 30,
                      "clock": "frame_aligned_take_seconds",
                      "preprocessing": {"mode": "raw", "rotate_ccw_quarters": 0},
                      "hm_row_alignment": "not_established_use_timestamps_after_explicit_mapping"})
    if not items or len({item["video_id"] for item in items}) != len(items):
        raise ValueError("Empty/duplicate video selection")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        items = list(executor.map(_inspect_video, items))
    items.sort(key=lambda item: item["video_id"])
    return {"contract": CONTRACT, "dataset_root": str(root), "output_root": str(output),
            "hm_revision": manifest["hm_revision"], "created_at": now(),
            "video_count": len(items), "total_frames": sum(item["length"] for item in items),
            "geometry_policy": "source_raw_no_undistortion_no_rotation",
            "hm_row_alignment_established": False, "items": items}


def signature(item, config, encoder_metadata):
    if input_fingerprint(item["video_path"]) != item["video_fingerprint"]:
        raise ValueError("Source video changed since preflight: " + item["video_id"])
    return digest({"contract": CONTRACT, "item": item, "config": config,
                   "encoder": encoder_metadata, "sidecar_code_sha256": file_sha256(__file__)})


def _blocks(item, batch_size, tolerance, decoder_threads):
    targets = np.arange(item["length"], dtype=np.float64) / 30
    frames = nearest_video_frames(item["video_path"], targets, tolerance,
                                  decoder_threads=decoder_threads)
    transform = image_preprocessor(item["preprocessing"])
    images, indices, timestamps = [], [], []
    try:
        for index, (image, pts) in enumerate(frames):
            indices.append(index)
            usable = image is not None and 0 <= pts <= targets[-1] + 1e-6
            timestamps.append(pts if usable else float("nan"))
            if usable:
                images.append(transform(image))
            if len(indices) == batch_size:
                yield images, indices, timestamps
                images, indices, timestamps = [], [], []
        if indices:
            yield images, indices, timestamps
    finally:
        frames.close()


def extract_video(item, config, encoder, progress=None):
    batch_size, tolerance = extraction_options(config)
    options = optimization_options(config)
    identity = signature(item, config, encoder.metadata)
    path = Path(item["output_path"])
    meta_path = path.with_suffix(".json")
    if path.exists() and meta_path.exists():
        meta = read_json(meta_path)
        if meta.get("signature") != identity:
            raise ValueError("Existing sidecar uses different input/model/config: " + item["video_id"])
        validate_video(path, meta)
        return {"kind": "skipped", "record": meta}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.tmp")
    leaf = FEATURE + ".list.element"
    writer_options = ({"use_dictionary": False, "use_byte_stream_split": [leaf],
                       "compression_level": {leaf: 1}} if options["optimized_storage"] else {})
    blocks = _blocks(item, batch_size, tolerance, options["decoder_threads"])
    prepare = getattr(encoder, "prepare", lambda images: images)
    prepared = prepared_batches(blocks, lambda images: prepare(images) if images else None,
                                options["prefetch_batches"])
    encode = getattr(encoder, "encode_prepared", encoder)
    total, valid_count = 0, 0
    tables, buffered_rows = [], 0
    try:
        with pq.ParquetWriter(temporary, SCHEMA, compression="zstd", **writer_options) as writer:
            for inputs, indices, timestamps in prepared:
                indices = np.asarray(indices, np.int64)
                timestamps = np.asarray(timestamps, np.float64)
                valid = np.isfinite(timestamps)
                features = np.zeros((len(indices), DINO_FLAT_DIM), np.float32)
                if valid.any():
                    value = np.asarray(encode(inputs), np.float32)
                    if value.shape != (int(valid.sum()), *DINO_SHAPE) or not np.isfinite(value).all():
                        raise ValueError("Encoder did not return finite (batch,5,1024) features")
                    features[valid] = value.reshape(int(valid.sum()), DINO_FLAT_DIM)
                target = indices.astype(np.float64) / 30
                columns = [pa.array(indices), pa.array(target), pa.array(target),
                           pa.array(np.where(valid, timestamps, 0)),
                           pa.array(np.where(valid, timestamps - target, 0)), pa.array(valid),
                           pa.FixedSizeListArray.from_arrays(pa.array(features.reshape(-1)), DINO_FLAT_DIM)]
                tables.append(pa.Table.from_arrays(columns, schema=SCHEMA))
                buffered_rows += len(indices)
                total += len(indices)
                valid_count += int(valid.sum())
                if buffered_rows >= 3000:
                    writer.write_table(pa.concat_tables(tables), row_group_size=3000)
                    tables, buffered_rows = [], 0
                if progress:
                    progress(total, valid_count)
            if tables:
                writer.write_table(pa.concat_tables(tables), row_group_size=3000)
        if total != item["length"] or not valid_count:
            raise ValueError("No aligned video frames or incorrect target frame count: " + item["video_id"])
        # Check that the source stayed fixed throughout decoding.
        if signature(item, config, encoder.metadata) != identity:
            raise ValueError("Source/config changed during extraction")
        meta = {"contract": CONTRACT, "state": "complete", "signature": identity,
                "video": item, "encoder": encoder.metadata, "config": config,
                "length": total, "valid_count": valid_count, "missing_count": total - valid_count,
                "max_video_error_s": tolerance, "parquet_sha256": file_sha256(temporary),
                "completed_at": now()}
        validate_video(temporary, meta)
        temporary.replace(path)
        atomic_json(meta_path, meta)
        return {"kind": "committed", "record": meta}
    finally:
        prepared.close()
        temporary.unlink(missing_ok=True)


def validate_video(path, meta):
    """Independent disk read: schema, clock, masks, PTS, finite values, checksum."""
    path = Path(path)
    if file_sha256(path) != meta["parquet_sha256"]:
        raise ValueError("DINO sidecar checksum differs: " + str(path))
    parquet = pq.ParquetFile(path)
    if not parquet.schema_arrow.equals(SCHEMA, check_metadata=False):
        raise ValueError("DINO sidecar schema differs: " + str(path))
    offset = valid_count = 0
    for batch in parquet.iter_batches(batch_size=1024):
        if any(column.null_count for column in batch.columns):
            raise ValueError("Null values in DINO sidecar")
        columns = {name: batch.column(i) for i, name in enumerate(batch.schema.names)}
        feature = columns[FEATURE]
        if feature.values.null_count:
            raise ValueError("Null feature elements")
        x = feature.values.to_numpy().reshape(len(batch), DINO_FLAT_DIM)
        valid = columns[VALID].to_numpy(zero_copy_only=False)
        index = columns["frame_index"].to_numpy()
        target = columns["timestamp"].to_numpy()
        source = columns["observation.source_timestamp"].to_numpy()
        pts, errors = columns[PTS].to_numpy(), columns[ERROR].to_numpy()
        if (not np.array_equal(index, np.arange(offset, offset + len(batch)))
                or not np.array_equal(target, index / 30) or not np.array_equal(source, target)):
            raise ValueError("DINO sidecar frame clock differs")
        if not all(np.isfinite(value).all() for value in (x, pts, errors)):
            raise ValueError("Nonfinite DINO sidecar values")
        if np.any(x[~valid] != 0) or np.any(pts[~valid] != 0) or np.any(errors[~valid] != 0):
            raise ValueError("Invalid DINO rows must be zero")
        if (np.any(np.abs(errors[valid]) > meta["max_video_error_s"] + 1e-9)
                or not np.allclose(errors[valid], pts[valid] - target[valid], atol=1e-12, rtol=0)):
            raise ValueError("DINO sidecar PTS mismatch")
        valid_count += int(valid.sum())
        offset += len(batch)
    if offset != meta["length"] or valid_count != meta["valid_count"] or valid_count < 1:
        raise ValueError("DINO sidecar count mismatch")
    return {"frames": offset, "valid_frames": valid_count, "missing_frames": offset - valid_count}


def _worker(connection, config, gpu_id):
    encoder = None
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        options = optimization_options(config)
        os.environ["OMP_NUM_THREADS"] = str(options["cpu_threads"])
        os.environ["MKL_NUM_THREADS"] = str(options["cpu_threads"])
        import torch
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=options["cpu_threads"])
        pa.set_cpu_count(options["cpu_threads"])
        pa.set_io_thread_count(2)
        torch.set_num_threads(options["cpu_threads"])
        torch.set_num_interop_threads(1)
        encoder = DinoEncoder(config)
        connection.send({"kind": "ready", "gpu_id": gpu_id, "pid": os.getpid(), "encoder": encoder.metadata})
        while True:
            item = connection.recv()
            if item is None:
                return
            last = time.monotonic()

            def progress(count, valid):
                nonlocal last
                if time.monotonic() - last >= 15:
                    connection.send({"kind": "batch", "video_id": item["video_id"],
                                     "encoded_frames": count, "valid_frames": valid})
                    last = time.monotonic()

            for attempt in range(3):
                try:
                    event = extract_video(item, config, encoder, progress)
                    connection.send(event)
                    break
                except Exception as error:
                    if attempt < 2:
                        time.sleep(2 ** attempt)
                    else:
                        connection.send({"kind": "failed", "video_id": item["video_id"],
                                         "error": f"{type(error).__name__}: {error}",
                                         "traceback": traceback.format_exc()})
    except BaseException:
        connection.send({"kind": "fatal", "traceback": traceback.format_exc()})
    finally:
        if encoder is not None:
            encoder.close()
        connection.close()


@contextmanager
def output_lock(output):
    Path(output).mkdir(parents=True, exist_ok=True)
    with (Path(output) / ".dino.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another process is writing these video sidecars") from error
        yield


def run_videos(plan, config, job, gpu_ids, reference_encoder=None):
    job = Path(job)
    job.mkdir(parents=True, exist_ok=True)
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids) or any(g < 0 for g in gpu_ids):
        raise ValueError("GPU IDs must be distinct and nonnegative")
    state = {"stage": "starting", "pid": os.getpid(), "gpu_ids": gpu_ids,
             "dataset_root": plan["dataset_root"], "output_root": plan["output_root"],
             "total_videos": len(plan["items"]), "total_frames": sum(x["length"] for x in plan["items"]),
             "completed_videos": 0, "completed_frames": 0, "valid_frames": 0,
             "failed_videos": 0, "workers": {}, "started_at": now(), "format": dino_format()}
    started = time.monotonic()
    processes, assignments, encoders = {}, {}, {}
    items = iter(sorted(plan["items"], key=lambda item: item["length"], reverse=True))
    failures = []

    def update():
        elapsed = time.monotonic() - started
        rate = state["completed_frames"] / max(elapsed, 1e-6)
        state.update(updated_at=now(), elapsed_seconds=elapsed, committed_frames_per_second=rate,
                     estimated_remaining_seconds=(state["total_frames"] - state["completed_frames"]) / rate if rate else None)
        atomic_json(job / "status.json", state)

    def assign(connection):
        item = next(items, None)
        connection.send(item)
        gpu = processes[connection][1]
        if item is None:
            state["workers"][str(gpu)]["stage"] = "finished"
            assignments.pop(connection, None)
        else:
            assignments[connection] = item
            state["workers"][str(gpu)].update(stage="extracting", video_id=item["video_id"],
                                              length=item["length"], encoded_frames=0)
        update()

    def cleanup():
        for connection, (process, _) in processes.items():
            if process.is_alive() and state["stage"] not in {"completed", "completed_with_errors"}:
                process.terminate()
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join()
            connection.close()

    update()
    try:
        with ExitStack() as scope:
            scope.enter_context(output_lock(plan["output_root"]))
            # Stop our workers before releasing the shared output write lock.
            scope.callback(cleanup)
            if shutil.disk_usage(plan["output_root"]).free < state["total_frames"] * DINO_FLAT_DIM * 4:
                raise RuntimeError("Insufficient free space for DINO output")
            context = mp.get_context("spawn")
            for gpu_id in gpu_ids:
                parent, child = context.Pipe()
                process = context.Process(target=_worker, args=(child, config, gpu_id), name=f"video-dino-gpu-{gpu_id}")
                process.start()
                child.close()
                processes[parent] = (process, gpu_id)
            waiting = set(processes)
            while waiting:
                for connection in wait(list(waiting), timeout=10):
                    event = connection.recv()
                    process, gpu = processes[connection]
                    if event["kind"] != "ready":
                        raise RuntimeError("Encoder startup failed: " + str(event))
                    metadata = event["encoder"]
                    if reference_encoder is not None and metadata != reference_encoder:
                        raise ValueError("Encoder differs from completed EgoSuite extraction")
                    encoders[gpu] = metadata
                    if any(value != metadata for value in encoders.values()):
                        raise ValueError("GPU workers use different encoders")
                    state["workers"][str(gpu)] = {"pid": process.pid, "stage": "ready"}
                    print(f"GPU {gpu}: encoder ready pid={process.pid}", flush=True)
                    waiting.remove(connection)
                    update()
                for connection in waiting:
                    if not processes[connection][0].is_alive():
                        raise RuntimeError("GPU worker exited during startup")
            state["stage"] = "extracting"
            atomic_json(job / "encoder.json", next(iter(encoders.values())))
            for connection in processes:
                assign(connection)
            with (job / "events.jsonl").open("a") as events:
                while assignments:
                    ready = wait(list(assignments), timeout=10)
                    if not ready:
                        update()
                    for connection in ready:
                        event = connection.recv()
                        process, gpu = processes[connection]
                        event.update(gpu_id=gpu, timestamp=now())
                        kind = event["kind"]
                        if kind == "batch":
                            state["workers"][str(gpu)].update(encoded_frames=event["encoded_frames"],
                                                              valid_frames=event["valid_frames"])
                            update()
                            continue
                        if kind == "fatal":
                            raise RuntimeError(str(event))
                        item = assignments[connection]
                        if kind == "failed":
                            failures.append(event)
                            state["failed_videos"] += 1
                            print(f"FAILED {item['video_id']}: {event['error']}", flush=True)
                        elif kind in {"committed", "skipped"}:
                            record = event.pop("record")
                            event.update(video_id=item["video_id"], length=record["length"], valid_count=record["valid_count"])
                            state["completed_videos"] += 1
                            state["completed_frames"] += record["length"]
                            state["valid_frames"] += record["valid_count"]
                            print(f"[{state['completed_videos']}/{state['total_videos']}] {kind} {item['video_id']} "
                                  f"{record['valid_count']}/{record['length']}", flush=True)
                        else:
                            raise RuntimeError("Unknown worker event: " + str(event))
                        events.write(json.dumps(event, ensure_ascii=False) + "\n")
                        events.flush()
                        assign(connection)
                    for connection in assignments:
                        if not processes[connection][0].is_alive() and not connection.poll():
                            raise RuntimeError("GPU worker exited during extraction")
            # Independent full read after all workers have finished committing.
            state["stage"] = "validating"
            update()
            report = validate_plan(plan, workers=min(8, len(gpu_ids)))
            atomic_json(job / "validation.json", report)
            state.update(stage="completed" if report["passed"] else "completed_with_errors",
                         validation_passed=report["passed"], missing_frames=state["total_frames"] - state["valid_frames"],
                         finished_at=now(), failures=failures)
            update()
            atomic_json(Path(plan["output_root"]) / "manifest.json", {"plan": plan, "config": config,
                        "encoder": next(iter(encoders.values())), "job_dir": str(job), "validation": report})
            return report
    except BaseException as error:
        state.update(stage="failed", fatal_error=f"{type(error).__name__}: {error}", finished_at=now())
        update()
        raise
def validate_plan(plan, workers=8):
    def check(item):
        try:
            meta = read_json(Path(item["output_path"]).with_suffix(".json"))
            if meta["video"] != item or meta["state"] != "complete" or meta["contract"] != CONTRACT:
                raise ValueError("Sidecar identity differs")
            if meta["signature"] != signature(item, meta["config"], meta["encoder"]):
                raise ValueError("Sidecar signature differs")
            report = validate_video(item["output_path"], meta)
            return {"video_id": item["video_id"], "passed": True, **report}
        except Exception as error:
            return {"video_id": item["video_id"], "passed": False, "error": f"{type(error).__name__}: {error}"}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(check, plan["items"]))
    good = [result for result in results if result["passed"]]
    return {"passed": len(good) == len(results), "checked_at": now(), "total_videos": len(results),
            "completed_videos": len(good), "frames": sum(result["frames"] for result in good),
            "valid_frames": sum(result["valid_frames"] for result in good),
            "missing_frames": sum(result["missing_frames"] for result in good),
            "failed_videos": [result for result in results if not result["passed"]],
            "geometry_policy": plan["geometry_policy"], "hm_row_alignment_established": False}
