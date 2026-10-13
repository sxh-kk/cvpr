"""One DINO process per GPU, with bounded results and one archive writer."""

import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import time
import traceback

from .features import (DinoEncoder, episode_signature, extract_episode_arrays,
                       extraction_options)
from .schema import SCHEMA_VERSION, V4_SCHEMA_VERSION
from .dino_optimization import optimization_options
from .storage import atomic_json, digest, read_arrays, write_arrays


def _worker(connection, root, specs, config, gpu_id, encoder_factory):
    record = None
    try:
        # Spawned workers bind the physical GPU before importing Torch.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        options = optimization_options(config)
        os.environ["OMP_NUM_THREADS"] = str(options["cpu_threads"])
        os.environ["MKL_NUM_THREADS"] = str(options["cpu_threads"])
        if encoder_factory is DinoEncoder:
            import torch
            import pyarrow as pa
            from threadpoolctl import threadpool_limits
            threadpool_limits(limits=options["cpu_threads"])
            pa.set_cpu_count(options["cpu_threads"])
            pa.set_io_thread_count(2)
            torch.set_num_threads(options["cpu_threads"])
            torch.set_num_interop_threads(1)
            if not torch.cuda.is_available():
                raise RuntimeError(f"GPU {gpu_id}: CUDA is unavailable")
        encoder = encoder_factory(config)
        connection.send({"kind": "worker_ready", "gpu_id": gpu_id,
                         "pid": os.getpid(), "encoder": encoder.metadata})
        while True:
            record = connection.recv()
            if record is None:
                return
            index = record["episode_index"]
            path = Path(root)/f"data/chunk-{index//1000:03d}/file-{index%1000:03d}.parquet"
            arrays = read_arrays(path, specs)
            class ProgressEncoder:
                metadata = encoder.metadata
                def __init__(self):
                    self.frames = 0
                    self.last = time.monotonic()
                def __call__(self, images):
                    features = encoder(images)
                    self.report(len(features))
                    return features
                def prepare(self, images):
                    return encoder.prepare(images) if hasattr(encoder, "prepare") else images
                def encode_prepared(self, inputs):
                    features = (encoder.encode_prepared(inputs) if hasattr(encoder, "encode_prepared")
                                else encoder(inputs))
                    self.report(len(features))
                    return features
                def report(self, count):
                    self.frames += count
                    if time.monotonic()-self.last >= 15:
                        connection.send({"kind": "batch", "gpu_id": gpu_id,
                                         "episode_index": index, "encoded_frames": self.frames})
                        self.last = time.monotonic()
            extract_episode_arrays(record, arrays, config, ProgressEncoder())
            connection.send({"kind": "result", "gpu_id": gpu_id,
                             "episode_index": index, "record": record, "arrays": arrays})
            del arrays
    except BaseException as error:
        try:
            connection.send({"kind": "error", "gpu_id": gpu_id,
                             "episode_index": record["episode_index"] if record else None,
                             "error": f"{type(error).__name__}: {error}",
                             "traceback": traceback.format_exc()})
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if "encoder" in locals() and hasattr(encoder, "close"):
            encoder.close()
        connection.close()


def extract_features_parallel(archive, config, gpu_ids, *, progress=None,
                              encoder_factory=DinoEncoder):
    """Extract episodes in parallel. The caller must hold ``archive.lock()``.

    Workers only read the archive and return results through bounded pipes.
    Only this process commits Parquet, provenance, and shared metadata.
    GPU assignments are absent from feature signatures, so resume can use a
    different number of GPUs with the same encoder/configuration.
    """
    archive.assert_ready()
    if archive.schema_version not in {V4_SCHEMA_VERSION, SCHEMA_VERSION}:
        raise ValueError("Legacy CLS-only archive: run upgrade-dino before extracting 5x1024 features")
    extraction_options(config)
    options = optimization_options(config)
    gpu_ids = list(gpu_ids)
    if (not gpu_ids or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in gpu_ids)
            or len(set(gpu_ids)) != len(gpu_ids)):
        raise ValueError("GPU IDs must be distinct nonnegative integers")
    if config.get("device", "cuda") not in {"cuda", "cuda:0"}:
        raise ValueError("Parallel DINO requires device='cuda'; GPU IDs select physical devices")
    def log(event):
        if event["kind"] == "worker_ready":
            print(f"DINO GPU {event['gpu_id']} ready pid={event['pid']}", flush=True)
        elif event["kind"] == "batch":
            print(f"DINO GPU {event['gpu_id']} episode={event['episode_index']} "
                  f"frames={event['encoded_frames']}", flush=True)
    emit = progress or log
    context = mp.get_context("spawn")
    processes, connections = {}, []

    def receive(connection):
        try:
            message = connection.recv()
        except EOFError as error:
            process = processes[connection]
            raise RuntimeError(f"DINO worker {process.pid} exited before returning its episode") from error
        if message["kind"] == "error":
            raise RuntimeError(f"GPU {message['gpu_id']}, episode {message['episode_index']}: "
                               f"{message['error']}\n{message['traceback']}")
        return message

    def ready(items):
        while True:
            available = wait(items, timeout=10)
            if available:
                return available
            for connection in items:
                if not processes[connection].is_alive():
                    raise RuntimeError(f"DINO worker {processes[connection].pid} exited unexpectedly")
            emit({"kind": "heartbeat"})

    try:
        for gpu_id in gpu_ids:
            parent, child = context.Pipe()
            process = context.Process(target=_worker,
                args=(child, str(archive.root), archive.specs, config, gpu_id, encoder_factory),
                name=f"dino-gpu-{gpu_id}", daemon=True)
            process.start()
            child.close()
            connections.append(parent)
            processes[parent] = process
        loading = set(connections)
        metadata = None
        while loading:
            for connection in ready(list(loading)):
                message = receive(connection)
                if message["kind"] != "worker_ready":
                    raise RuntimeError("Unexpected DINO worker initialization message")
                if metadata is not None and digest(metadata) != digest(message["encoder"]):
                    raise ValueError("GPU workers use different DINO encoders/preprocessing")
                metadata = message["encoder"]
                loading.remove(connection)
                emit(message)
        pending = []
        records = archive.records()
        # Check all completed signatures before writing any new episode.
        for record in records:
            signature, _ = episode_signature(record, config, metadata)
            if record["dino"]["state"] == "complete":
                if record["dino"].get("signature") != signature:
                    raise ValueError("Existing DINO features use a different model/preprocessing; create a new archive")
            else:
                pending.append(record)
        items = iter(pending)
        active = {}
        for connection in connections:
            record = next(items, None)
            if record is not None:
                connection.send(record)
                active[connection] = record["episode_index"]
                emit({"kind": "assigned", "gpu_id": gpu_ids[connections.index(connection)],
                      "episode_index": record["episode_index"], "source_key": record["source_key"],
                      "length": record["length"]})
        last_finalize = time.monotonic()
        committed = 0
        while active:
            for connection in ready(list(active)):
                message = receive(connection)
                if message.get("episode_index") != active[connection]:
                    raise RuntimeError("GPU worker returned a different episode")
                if message["kind"] == "batch":
                    emit(message)
                    continue
                if message["kind"] != "result":
                    raise RuntimeError("Unexpected DINO worker result")
                index = active.pop(connection)
                record, arrays = message["record"], message["arrays"]
                if record["source_key"] != records[index]["source_key"]:
                    raise RuntimeError("GPU worker returned a different source")
                write_arrays(archive.data_path(index), arrays, archive.specs,
                             optimized_dino=options["optimized_storage"])
                atomic_json(archive.record_path(index), record)
                records[index].update(record)
                del arrays, message
                committed += 1
                if committed % 25 == 0 or time.monotonic()-last_finalize >= 60:
                    archive.finalize()
                    last_finalize = time.monotonic()
                emit({"kind": "committed", "gpu_id": gpu_ids[connections.index(connection)],
                      "episode_index": index, "source_key": record["source_key"],
                      "length": record["length"], "valid_count": record["dino"]["valid_count"]})
                print(f"DINO {record['source_key']}: {record['dino']['valid_count']}/{record['length']}", flush=True)
                following = next(items, None)
                if following is not None:
                    connection.send(following)
                    active[connection] = following["episode_index"]
                    emit({"kind": "assigned", "gpu_id": gpu_ids[connections.index(connection)],
                          "episode_index": following["episode_index"],
                          "source_key": following["source_key"], "length": following["length"]})
    finally:
        for connection in connections:
            connection.close()
        for process in processes.values():
            if process.is_alive():
                process.terminate()
        for process in processes.values():
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join()
        archive.finalize()
