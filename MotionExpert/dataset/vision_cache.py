"""Read-only native30Hz DINO caches with 5Hz motion selection; no pipeline imports."""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import numpy as np


CONTRACT = "smplh-dino-shared-rgb-cache-v1"
JOIN_DTYPES = {
    "frame_index": "int64", "episode_index": "int64", "index": "int64",
    "timestamp": "float32", "observation.source_timestamp": "float64",
    "source.video_frame_index": "int64", "observation.valid.dino": "bool",
    "source.valid.dino": "bool", "observation.video_timestamp": "float64",
    "observation.video_time_error": "float64",
}
SOURCE_JOIN = ("frame_index", "episode_index", "index", "timestamp",
               "observation.source_timestamp", "source.video_frame_index")


def storage_digest(value):
    """Match pipeline.io.digest's pretty JSON, without a pipeline dependency."""
    digest = hashlib.sha256()
    encoder = json.JSONEncoder(ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    for chunk in encoder.iterencode(value):
        digest.update(chunk.encode("utf-8"))
    digest.update(b"\n")
    return digest.hexdigest()


def _stamp(path):
    stat = Path(path).stat()
    return stat.st_size, stat.st_mtime_ns


class VisionProfileCache:
    """Bound a completed visual version to the unchanged accepted motion archive.

    Features use NumPy mmap and only requested native frame IDs are materialized.
    Join arrays are small and retained in a bounded per-process episode LRU.
    The initial implementation requires unchanged validity so existing motion
    statistics and training-window populations can safely be reused.
    """

    def __init__(self, root, archive, *, image_profile=None, file_cache=8):
        if type(file_cache) is not int or file_cache < 1:
            raise ValueError("Vision cache file_cache must be a positive integer")
        from ImageProcess import load_profile

        self.root = Path(root).resolve(strict=True)
        self.manifest_path = self._owned("manifest.json")
        self.manifest = json.loads(self.manifest_path.read_text())
        self._manifest_stamp = _stamp(self.manifest_path)
        manifest = self.manifest
        if (manifest.get("contract") != CONTRACT or manifest.get("state") != "complete"
                or manifest.get("scope") != "all"):
            raise ValueError("Vision profile must be complete and cover all source episodes")
        if manifest.get("validity_equal_source") is not True:
            raise ValueError("Vision profile validity changed: existing motion statistics cannot be reused")
        cache_fps, cache_step = manifest.get("fps"), manifest.get("sampling_step")
        if (type(cache_fps) is not int or type(cache_step) is not int
                or (cache_fps, cache_step) != (30, 1) or manifest.get("native_fps") != 30
                or manifest.get("frame_phase") != 0
                or manifest.get("shape") != [5, 1024] or manifest.get("storage_dtype") != "float32"):
            raise ValueError("Vision profile requires native30Hz float32 DINO [5,1024]")
        self.fps, self.cache_sampling_step = cache_fps, cache_step
        records = archive.records()
        expected_ids = list(range(len(records)))
        total_frames = sum(len(range(0, record["length"], cache_step)) for record in records)
        if (manifest.get("episode_indices") != expected_ids or manifest.get("source_episode_count") != len(records)
                or manifest.get("completed_episodes") != len(records)
                or manifest.get("total_frames") != total_frames or manifest.get("completed_frames") != total_frames):
            raise ValueError("Vision profile population differs from the owned motion archive")
        expected_binding = {
            key: archive.binding[key]
            for key in ("info_sha256", "episodes_metadata_sha256", "split_metadata_sha256")
        }
        expected_binding["records_sha256"] = storage_digest(records)
        if manifest.get("source_binding") != expected_binding:
            raise ValueError("Vision profile source binding differs from the accepted motion archive")
        process = manifest.get("image_process", {})
        profile = load_profile(process.get("profile"))
        if (process.get("profile_sha256") != profile.fingerprint
                or process.get("profile_id") != profile.profile_id
                or (process.get("height"), process.get("width")) != profile.output_hw):
            raise ValueError("Vision profile image-processing hash/dimensions differ")
        if image_profile is not None and load_profile(image_profile).fingerprint != profile.fingerprint:
            raise ValueError("IMAGE_PROFILE differs from the extracted vision profile hash")
        self.image_profile = profile.profile_id
        self.profile_sha256 = profile.fingerprint
        self.metadata = {
            "contract": CONTRACT, "image_process": deepcopy(process),
            "manifest_sha256": hashlib.sha256(self.manifest_path.read_bytes()).hexdigest(),
            "encoder_sha256": storage_digest(manifest.get("encoder")),
            "fps": cache_fps, "native_fps": 30, "sampling_step": cache_step,
            "frame_phase": 0, "validity_equal_source": True,
        }
        if not isinstance(manifest.get("encoder"), dict) or not manifest["encoder"]:
            raise ValueError("Vision profile encoder identity is missing")
        if manifest["encoder"].get("image_process") != process:
            raise ValueError("Vision profile encoder image-processing identity differs from its manifest")
        self._episodes = {}
        valid_frames = 0
        for record in records:
            episode = record["episode_index"]
            prefix = f"episodes/{episode:08d}"
            receipt_path = self._owned(f"{prefix}/receipt.json")
            receipt = json.loads(receipt_path.read_text())
            count = len(range(0, record["length"], cache_step))
            if (receipt.get("contract") != CONTRACT or receipt.get("state") != "complete"
                    or receipt.get("episode_index") != episode or receipt.get("source_key") != record["source_key"]
                    or receipt.get("source_record_sha256") != storage_digest(record)
                    or receipt.get("frames") != count or receipt.get("validity_equal_source") is not True
                    or receipt.get("encoder") != manifest["encoder"]):
                raise ValueError("Vision profile episode receipt/source/encoder differs")
            valid_count = receipt.get("valid_count")
            if type(valid_count) is not int or not 0 <= valid_count <= count:
                raise ValueError("Vision profile episode validity count is invalid")
            valid_frames += valid_count
            identities = receipt.get("output_identities", {})
            if set(identities) != {"features.npy", "join.npz"}:
                raise ValueError("Vision profile episode artifacts are incomplete")
            artifacts = {}
            for name, identity in identities.items():
                path = self._owned(f"{prefix}/{name}")
                stamp = (identity.get("size_bytes"), identity.get("mtime_ns"))
                if _stamp(path) != stamp:
                    raise ValueError("Committed vision profile file identity changed")
                artifacts[name] = (path, stamp)
            self._episodes[episode] = {
                "frames": count, "length": record["length"], "offset": record["dataset_from_index"],
                "valid_count": valid_count, "receipt_path": receipt_path,
                "receipt_stamp": _stamp(receipt_path), "artifacts": artifacts,
            }
        if manifest.get("valid_frames") != valid_frames:
            raise ValueError("Vision profile manifest validity count differs from receipts")
        self.file_cache = file_cache
        self._cache = OrderedDict()
        self._pid = os.getpid()
        if _stamp(self.manifest_path) != self._manifest_stamp:
            raise ValueError("Vision profile manifest changed while opening")

    def _owned(self, relative):
        name = Path(relative)
        if name.is_absolute() or ".." in name.parts:
            raise ValueError("Vision profile artifacts must be owned relative paths")
        path = (self.root / name).resolve(strict=True)
        if not path.is_relative_to(self.root) or not path.is_file():
            raise ValueError("Vision profile artifact escapes its profile directory")
        return path

    def assert_current(self, episode_index=None):
        if _stamp(self.manifest_path) != self._manifest_stamp:
            raise ValueError("Vision profile manifest changed during reading")
        episodes = self._episodes if episode_index is None else (episode_index,)
        for episode in episodes:
            descriptor = self._episodes[episode]
            if _stamp(descriptor["receipt_path"]) != descriptor["receipt_stamp"]:
                raise ValueError("Vision profile episode receipt changed during reading")
            if any(_stamp(path) != stamp for path, stamp in descriptor["artifacts"].values()):
                raise ValueError("Committed vision profile file changed during reading")

    def _episode(self, episode_index, load_features):
        if self._pid != os.getpid():
            self._cache = OrderedDict()
            self._pid = os.getpid()
        self.assert_current(episode_index)
        descriptor = self._episodes[episode_index]
        if episode_index in self._cache:
            cached = self._cache.pop(episode_index)
        else:
            with np.load(descriptor["artifacts"]["join.npz"][0], allow_pickle=False) as saved:
                if set(saved.files) != set(JOIN_DTYPES):
                    raise ValueError("Vision profile join columns differ")
                join = {key: saved[key].copy() for key in JOIN_DTYPES}
            for key, dtype in JOIN_DTYPES.items():
                if join[key].shape != (descriptor["frames"],) or join[key].dtype != np.dtype(dtype):
                    raise ValueError(f"Vision profile {key} shape/dtype mismatch")
                if not np.isfinite(join[key]).all():
                    raise ValueError(f"Vision profile {key} contains nonfinite values")
            native = np.arange(0, descriptor["length"], self.cache_sampling_step, dtype=np.int64)
            if (not np.array_equal(join["frame_index"], native)
                    or not np.all(join["episode_index"] == episode_index)
                    or not np.array_equal(join["index"], native + descriptor["offset"])
                    or not np.array_equal(join["timestamp"], (native / 30).astype(np.float32))):
                raise ValueError("Vision profile native frame/global-index/time join differs")
            if (not np.array_equal(join["observation.valid.dino"], join["source.valid.dino"])
                    or int(join["observation.valid.dino"].sum()) != descriptor["valid_count"]):
                raise ValueError("Vision profile validity changed despite its equivalence declaration")
            cached = {"join": join, "features": None}
        if load_features and cached["features"] is None:
            features = np.load(descriptor["artifacts"]["features.npy"][0], mmap_mode="r", allow_pickle=False)
            if features.shape != (descriptor["frames"], 5, 1024) or features.dtype != np.float32:
                raise ValueError("Vision profile features must be float32 [Ncache,5,1024]")
            cached["features"] = features
        self._cache[episode_index] = cached
        while len(self._cache) > self.file_cache:
            _, removed = self._cache.popitem(last=False)
            if removed["features"] is not None:
                removed["features"]._mmap.close()
        return cached

    def read(self, record, selected, *, expected=None, load_features=False):
        """Read explicit native frame IDs; image values are materialized only on request."""
        selected = np.asarray(selected)
        if (selected.ndim != 1 or selected.dtype.kind not in "iu" or np.any(selected < 0)
                or np.any(selected % 6) or np.any(selected >= record["length"])
                or np.any(np.diff(selected.astype(np.int64)) <= 0)):
            raise ValueError("Vision profile selection must use increasing zero-phase native 5Hz frame IDs")
        episode = record["episode_index"]
        cached = self._episode(episode, load_features)
        rows = selected // self.cache_sampling_step
        output = {key: value[rows].copy() for key, value in cached["join"].items()}
        if expected is not None:
            for key in (*SOURCE_JOIN, "observation.valid.dino"):
                if key in expected and not np.array_equal(output[key], expected[key]):
                    raise ValueError(f"Vision profile/motion frame or validity join differs: {key}")
        if load_features:
            value = np.asarray(cached["features"][rows]).copy()
            if not np.isfinite(value).all():
                raise ValueError("Vision profile contains nonfinite selected features")
            valid = output["observation.valid.dino"]
            if np.any(value[~valid] != 0):
                raise ValueError("Vision profile invalid embeddings must be zero")
            output["observation.features.dino"] = value.reshape(len(rows), 5120)
        self.assert_current(episode)
        return output

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_cache"] = OrderedDict()
        return state
