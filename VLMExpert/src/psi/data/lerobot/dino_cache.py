"""Versioned, episode-local native-frame DINO caches shared by writer and reader.

Only a completed manifest and an atomically committed episode receipt are
readable. Feature arrays remain mmap-backed; checksums can be requested for an
explicit audit, while ordinary reads check their committed size/mtime identity.
This module has no torch, LeRobot or video-decoder dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

CONTRACT = "unifolm-dino-native-frames-v1"
DINO_MODEL_ID = "facebook/dinov2-with-registers-large"
DINO_SHAPE = (5, 1024)
FRAME_COLUMNS = ("frame_index", "timestamp", "source_frame_index", "video_timestamp", "valid")


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def file_identity(path: str | Path, *, checksum: bool = True) -> dict:
    path = Path(path)
    stat = path.stat()
    if not path.is_file():
        raise ValueError(f"Expected a regular file: {path}")
    result = {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if checksum:
        result["sha256"] = file_sha256(path)
    # A concurrent rewrite must never acquire a valid receipt.
    after = path.stat()
    if (stat.st_size, stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"File changed while fingerprinting: {path}")
    return result


def check_file_identity(path: str | Path, expected: dict, *, checksum: bool = False) -> None:
    current = file_identity(path, checksum=checksum)
    keys = ("size_bytes", "mtime_ns", "sha256") if checksum else ("size_bytes", "mtime_ns")
    if any(current.get(key) != expected.get(key) for key in keys):
        raise ValueError(f"Stale or modified DINO cache/source file: {path}")


def owned_file(root: str | Path, relative: str) -> Path:
    root = Path(root).resolve(strict=True)
    name = Path(relative)
    if name.is_absolute() or ".." in name.parts:
        raise ValueError("Cache/source paths must be relative and stay within their root")
    path = (root / name).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Cache/source path escapes its root: {relative}")
    return path


def read_json(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def atomic_json(path: str | Path, value: dict) -> None:
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    if path.is_symlink() or temporary.is_symlink():
        raise ValueError("Cache metadata files must not be symlinks")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def episode_directory(cache_root: str | Path, episode_index: int) -> Path:
    if type(episode_index) is not int or episode_index < 0:
        raise ValueError("episode_index must be a nonnegative integer")
    root = Path(cache_root).resolve()
    path = root / "episodes" / f"{episode_index:08d}"
    if not path.resolve().is_relative_to(root):
        raise ValueError("Episode directory escapes cache root")
    return path


def manifest_signature(manifest: dict) -> str:
    return digest({key: manifest[key] for key in
                   ("contract", "source", "image_process", "encoder", "extraction",
                    "storage_dtype", "shape", "episode_indices")})


def load_cache_manifest(cache_root: str | Path, dataset_root: str | Path | None = None,
                        expected_profile="256x320") -> dict:
    """Validate metadata/source binding once before opening episode mmap files.

    ``expected_profile`` is the very same named/JSON ImageProcess profile used
    in Psi0's data configuration. Version mismatches deliberately require a new
    cache; changing a size in a second independent configuration is unsupported.
    """
    from ImageProcess import SharedImageProcessor

    root = Path(cache_root).resolve(strict=True)
    manifest = read_json(owned_file(root, "manifest.json"))
    if manifest.get("contract") != CONTRACT or manifest.get("state") != "complete":
        raise ValueError("DINO cache is incomplete or uses an unsupported contract")
    if manifest.get("shape") != list(DINO_SHAPE) or manifest.get("storage_dtype") not in ("float16", "float32"):
        raise ValueError("DINO cache must store float16/float32 CLS + 4 registers [T,5,1024]")
    if manifest.get("extraction") != {"stage": "validation", "training": False, "sampling_step": 1}:
        raise ValueError("DINO cache must preserve every native frame without random augmentation")
    if manifest.get("image_process") != SharedImageProcessor(expected_profile).metadata:
        raise ValueError("DINO cache image profile/implementation differs from the configured Psi0 shared RGB profile")
    encoder = manifest.get("encoder", {})
    local_files = encoder.get("local_file_sha256")
    if (encoder.get("model_id") != DINO_MODEL_ID or encoder.get("revision") != "local"
            or not isinstance(local_files, dict) or not local_files
            or not any(name.endswith((".bin", ".safetensors")) for name in local_files)
            or encoder.get("weights_sha256") != digest(local_files)
            or encoder.get("image_process") != manifest["image_process"]):
        raise ValueError("DINO cache has no pinned local dinov2-with-registers-large weights/profile identity")
    preprocessing = encoder.get("preprocessing", {})
    if (preprocessing.get("do_resize") is not False or preprocessing.get("do_center_crop") is not False
            or preprocessing.get("do_rescale") is not True or preprocessing.get("do_normalize") is not True
            or preprocessing.get("image_mean") != [0.485, 0.456, 0.406]
            or preprocessing.get("image_std") != [0.229, 0.224, 0.225]
            or preprocessing.get("rescale_factor") != 1 / 255):
        raise ValueError("DINO cache normalization must follow shared RGB without a second resize/crop")
    indices = manifest.get("episode_indices")
    if (not isinstance(indices, list) or not indices or len(set(indices)) != len(indices)
            or any(type(index) is not int or index < 0 for index in indices)):
        raise ValueError("DINO cache episode index is malformed")
    try:
        signature = manifest_signature(manifest)
    except KeyError as error:
        raise ValueError("DINO cache manifest is missing its identity fields") from error
    if signature != manifest.get("signature"):
        raise ValueError("DINO cache manifest signature differs")
    source = manifest.get("source", {})
    if source.get("fps") != 30 or not source.get("video_key") or not source.get("meta_files"):
        raise ValueError("DINO cache requires the original native30Hz LeRobot source binding")
    source_root = Path(dataset_root or source.get("dataset_root", "")).resolve(strict=True)
    if dataset_root is not None and source_root != Path(source["dataset_root"]).resolve():
        raise ValueError("DINO cache belongs to a different source dataset root")
    for relative, identity in source["meta_files"].items():
        check_file_identity(owned_file(source_root, relative), identity, checksum=True)
    return manifest


@dataclass(frozen=True)
class EpisodeDinoCache:
    features: np.ndarray
    frame_index: np.ndarray
    timestamp: np.ndarray
    source_frame_index: np.ndarray
    video_timestamp: np.ndarray
    valid: np.ndarray
    receipt: dict


def load_episode_cache(cache_root: str | Path, episode_index: int, *, manifest: dict | None = None,
                       dataset_root: str | Path | None = None, expected_profile="256x320",
                       verify_checksums: bool = False, _allow_incomplete: bool = False) -> EpisodeDinoCache:
    """Open one committed episode with bounded metadata and mmap-only features.

    ``verify_checksums=True`` additionally streams large video/parquet/feature
    files for an explicit audit. Normal training does not scan an entire video
    or feature array every time its bounded worker LRU opens an episode.
    """
    manifest = manifest or load_cache_manifest(cache_root, dataset_root, expected_profile)
    if (manifest.get("contract") != CONTRACT or manifest.get("signature") != manifest_signature(manifest)
            or (manifest.get("state") != "complete" and not _allow_incomplete)):
        raise ValueError("DINO manifest is incomplete or its signature differs")
    if episode_index not in manifest["episode_indices"]:
        raise ValueError(f"Episode {episode_index} is absent from this DINO extraction")
    directory = episode_directory(cache_root, episode_index)
    receipt = read_json(owned_file(directory, "receipt.json"))
    if (receipt.get("contract") != CONTRACT or receipt.get("state") != "complete"
            or receipt.get("manifest_signature") != manifest["signature"]
            or receipt.get("episode_index") != episode_index):
        raise ValueError("DINO episode is partial or belongs to a stale extraction")
    source = receipt.get("source", {})
    if receipt.get("signature") != digest({"manifest_signature": manifest["signature"], "source": source}):
        raise ValueError("DINO episode source signature differs")
    source_root = Path(dataset_root or manifest["source"]["dataset_root"]).resolve(strict=True)
    for name in ("video", "parquet"):
        item = source.get(name, {})
        check_file_identity(owned_file(source_root, item["path"]), item["identity"], checksum=verify_checksums)
    outputs = receipt.get("outputs", {})
    if set(outputs) != {"features.npy", "frames.npz"}:
        raise ValueError("DINO episode has incomplete committed output identities")
    for name, identity in outputs.items():
        check_file_identity(owned_file(directory, name), identity,
                            checksum=verify_checksums or name == "frames.npz")
    features = np.load(owned_file(directory, "features.npy"), mmap_mode="r", allow_pickle=False)
    frames = receipt.get("frames")
    if features.shape != (frames, *DINO_SHAPE) or features.dtype != np.dtype(manifest["storage_dtype"]):
        raise ValueError("DINO episode feature shape/dtype differs from its receipt")
    with np.load(owned_file(directory, "frames.npz"), allow_pickle=False) as sidecar:
        if set(sidecar.files) != set(FRAME_COLUMNS):
            raise ValueError("DINO episode frame sidecar has unsupported columns")
        arrays = {name: np.array(sidecar[name], copy=True) for name in FRAME_COLUMNS}
    if (type(frames) is not int or frames < 1 or any(value.shape != (frames,) for value in arrays.values())
            or arrays["frame_index"].dtype != np.int64 or arrays["source_frame_index"].dtype != np.int64
            or arrays["timestamp"].dtype != np.float64 or arrays["video_timestamp"].dtype != np.float64
            or arrays["valid"].dtype != np.bool_):
        raise ValueError("DINO episode frame sidecar shape/dtypes differ")
    if (np.any(arrays["frame_index"] < 0) or np.any(np.diff(arrays["frame_index"]) <= 0)
            or not np.array_equal(arrays["source_frame_index"], np.arange(frames, dtype=np.int64))
            or not np.isfinite(arrays["timestamp"]).all() or np.any(np.diff(arrays["timestamp"]) <= 0)
            or not np.isfinite(arrays["video_timestamp"]).all() or np.any(np.diff(arrays["video_timestamp"]) <= 0)):
        raise ValueError("DINO episode has invalid native frame/timestamp identities")
    return EpisodeDinoCache(features=features, receipt=receipt, **arrays)
