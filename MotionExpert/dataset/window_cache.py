"""Immutable raw motion windows; mmap reads avoid source FK/Parquet/join work."""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import hashlib
import inspect
import json
import os
from pathlib import Path

import numpy as np

CONTRACT = "smplh245-window-cache-v1"
ARRAY_SPECS = {
    "start_frames": ("int64", ()),
    "motion": ("float32", (11, 245)),
    "motion_valid": ("bool", (11, 245)),
    "trajectory": ("float32", (11, 18)),
    "trajectory_valid": ("bool", (11, 18)),
    "world_from_window": ("float64", (4, 4)),
    "window_from_world": ("float64", (4, 4)),
    "floor_transform": ("float64", (11, 4, 4)),
    "floor_height": ("float64", ()),
    "ground_fallback_used": ("bool", ()),
    "ground_point_count": ("int32", ()),
    "historical_image_valid": ("bool", (6,)),
}


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def window_protocol(**values):
    from ..config import K12_JOINTS, SMPLH_REPRESENTATION_VERSION
    protocol = {
        "fps": 5, "window": 11, "forecast_prefix": 6, "frame_phase": 0,
        "sampling_step": 6, "segment_stride": 5, "train_task": "fore",
        "motion_supervision": "joint", "require_full_window": True,
        "contact_speed_m_s": .2, "contact_height_m": .1,
        "min_valid_image_fraction": .5,
        "representation": SMPLH_REPRESENTATION_VERSION,
        "joint_indices": list(K12_JOINTS), "rotation_basis_mode": "smplh_model_defined",
        "target_joint_basis": "smplh_body22_axes_v1", "storage": "raw_float32",
        "ground_estimation_scope": "observed_prefix", "image_validity_scope": "observed_prefix",
    }
    unknown = set(values) - set(protocol)
    if unknown:
        raise ValueError(f"Unknown window-cache protocol fields: {sorted(unknown)}")
    protocol.update(values)
    if (protocol["fps"], protocol["window"], protocol["forecast_prefix"],
            protocol["sampling_step"], protocol["frame_phase"], protocol["train_task"],
            protocol["motion_supervision"], protocol["require_full_window"], protocol["storage"]) != (
            5, 11, 6, 6, 0, "fore", "joint", True, "raw_float32"):
        raise ValueError("Window cache requires the fixed raw 5Hz/11-frame joint protocol")
    if type(protocol["segment_stride"]) is not int or protocol["segment_stride"] < 1:
        raise ValueError("Window-cache segment stride must be positive")
    for name in ("contact_speed_m_s", "contact_height_m", "min_valid_image_fraction"):
        value = protocol[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"Invalid window-cache {name}")
    if (min(protocol["contact_speed_m_s"], protocol["contact_height_m"]) <= 0
            or not 0 <= protocol["min_valid_image_fraction"] <= 1):
        raise ValueError("Invalid window-cache contact/image thresholds")
    return protocol


def protocol_from_config(cfg):
    data = cfg.DATA
    return window_protocol(**{key: getattr(data, key.upper()) for key in (
        "fps", "window", "forecast_prefix", "segment_stride", "frame_phase", "train_task",
        "motion_supervision", "require_full_window", "contact_speed_m_s", "contact_height_m",
        "min_valid_image_fraction", "rotation_basis_mode", "target_joint_basis")})


def source_code_binding():
    from . import representation, smplh_representation, smplh_archive
    from ..utils import rotation_conversions
    from .smplh_motion_dataset import SMPLHMotionDataset
    return {module.__name__: file_sha256(module.__file__) for module in (
        representation, smplh_representation, smplh_archive, rotation_conversions)} | {
        "window_index": hashlib.sha256(inspect.getsource(SMPLHMotionDataset._index_record).encode()).hexdigest()}


def cache_binding(archive, protocol):
    vision = archive.vision_cache
    if vision is None:
        raise ValueError("Window cache requires a validated native30Hz VISION_PROFILE_ROOT")
    return {
        "archive_binding": deepcopy(archive.binding), "archive_fingerprint": archive.archive_fingerprint,
        "split_fingerprint": archive.split_fingerprint, "model_sha256": archive.model_sha256,
        "protocol": window_protocol(**protocol), "representation_sha256": source_code_binding(),
        "vision_manifest_sha256": vision.metadata["manifest_sha256"],
        "image_profile_sha256": vision.profile_sha256,
    }


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(canonical_json(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def stamp(path):
    stat = Path(path).stat()
    return stat.st_size, stat.st_mtime_ns


class WindowCache:
    """Startup verifies identity once; sample reads touch only cached mmap pages."""
    def __init__(self, root, *, archive, split, protocol, allow_diagnostic=False, file_cache=8):
        if split not in ("train", "val", "test") or type(file_cache) is not int or file_cache < 1:
            raise ValueError("Invalid window-cache split or file cache")
        self.root, self.archive, self.split = Path(root).resolve(strict=True), archive, split
        self.manifest_path = self._owned("manifest.json")
        self.manifest = json.loads(self.manifest_path.read_text())
        self._manifest_stamp = stamp(self.manifest_path)
        manifest = self.manifest
        if manifest.get("contract") != CONTRACT or manifest.get("state") != "complete":
            raise ValueError("Window cache must have a complete committed manifest")
        if (manifest.get("diagnostic") or manifest.get("scope") != "all") and not allow_diagnostic:
            raise ValueError("Diagnostic/partial window cache cannot be used for formal training")
        self._packed_spec = manifest.get("packed", {}).get(split)
        self._packed = None
        if self._packed_spec is None and not manifest.get("diagnostic"):
            raise ValueError("Formal training requires packed split-wide window arrays")
        if manifest.get("packed") and set(manifest["packed"]) != {"train", "val", "test"}:
            raise ValueError("Packed window cache must declare all original splits")
        expected = cache_binding(archive, protocol)
        if manifest.get("binding") != expected or manifest.get("binding_sha256") != digest(expected):
            raise ValueError("Window-cache source/visual/protocol/representation binding differs")
        self.source_binding = deepcopy(expected["archive_binding"])
        self.archive_fingerprint, self.split_fingerprint = expected["archive_fingerprint"], expected["split_fingerprint"]
        records_path = self._owned(manifest["records_path"])
        if file_sha256(records_path) != manifest["records_sha256"]:
            raise ValueError("Window-cache record digest mismatch")
        cached_records = json.loads(records_path.read_text())
        originals = archive.records()
        if any(type(record.get("episode_index")) is not int
               or not 0 <= record["episode_index"] < len(originals)
               or record != originals[record["episode_index"]]
               for record in cached_records):
            raise ValueError("Window-cache source records differ from the committed archive")
        selection = manifest.get("selected_episode_indices")
        expected_records = [record for record in originals
            if (selection is None or record["episode_index"] in selection)
            and record["metadata"].get("training_eligible") is not False
            and record["metadata"].get("canonical_training_eligible") is not False
            and not record["metadata"].get("quarantine_reasons", [])]
        if cached_records != expected_records or (not manifest.get("diagnostic") and selection is not None):
            raise ValueError("Window-cache source population/order differs from its declared scope")
        indices = manifest["records_by_split"][split]
        if indices != [index for index, record in enumerate(cached_records) if record["split"] == split]:
            raise ValueError("Window-cache records leak or reorder the original split")
        self.records = [cached_records[index] for index in indices]
        self._descriptors = {}
        self._cache, self._image_cache, self._pid = OrderedDict(), OrderedDict(), os.getpid()
        self._verified, self._image_verified = set(), set()
        self.file_cache = file_cache
        descriptor_ids = [item["episode_index"] for item in manifest["episodes"]]
        if len(set(descriptor_ids)) != len(descriptor_ids):
            raise ValueError("Duplicate window-cache episode descriptor")
        if descriptor_ids != [record["episode_index"] for record in cached_records]:
            raise ValueError("Window-cache episode descriptors differ from the source record population/order")
        for descriptor, record in zip(manifest["episodes"], cached_records):
            if (descriptor.get("split") != record["split"]
                    or descriptor.get("source_record_sha256") != digest(record)
                    or type(descriptor.get("windows")) is not int or descriptor["windows"] < 0):
                raise ValueError("Window-cache descriptor source/split/population differs")
        descriptors = {item["episode_index"]: item for item in manifest["episodes"]}
        self._cache_stamps = {records_path: stamp(records_path)}
        for record in self.records:
            episode = record["episode_index"]
            descriptor = deepcopy(descriptors[episode])
            receipt_path = self._owned(descriptor["receipt_path"])
            if file_sha256(receipt_path) != descriptor["receipt_sha256"]:
                raise ValueError("Window-cache episode receipt digest mismatch")
            receipt = json.loads(receipt_path.read_text())
            if (receipt.get("contract") != CONTRACT or receipt.get("state") != "complete"
                    or receipt.get("binding_sha256") != manifest["binding_sha256"]
                    or receipt.get("episode_index") != episode or receipt.get("split") != split
                    or receipt.get("source_record_sha256") != digest(record)
                    or receipt.get("windows") != descriptor["windows"]
                    or set(receipt.get("arrays", {})) != set(ARRAY_SPECS)):
                raise ValueError("Window-cache episode identity/shape declaration differs")
            count = descriptor["windows"]
            for key, (dtype, tail) in ARRAY_SPECS.items():
                array = receipt["arrays"][key]
                if array.get("shape") != [count, *tail] or array.get("dtype") != dtype:
                    raise ValueError(f"Window-cache array declaration differs: {key}")
                if self._packed_spec is None:
                    path = self._owned(array["path"])
                    if stamp(path) != (array["size_bytes"], array["mtime_ns"]):
                        raise ValueError("Committed window-cache artifact identity changed")
            descriptor["receipt"] = receipt
            self._descriptors[episode] = descriptor
        index = manifest["indices"][split]
        index_path = self._owned(index["path"])
        if file_sha256(index_path) != index["sha256"]:
            raise ValueError("Window-cache split index digest mismatch")
        values = np.load(index_path, allow_pickle=False)
        if values.dtype != np.int64 or values.shape != (index["windows"], 2):
            raise ValueError("Window-cache split index shape/dtype differs")
        if len(values):
            if (np.any(values[:, 0] < 0) or np.any(values[:, 0] >= len(self.records))
                    or np.any(values[:, 1] < 0) or np.any(values[:, 1] % 6)
                    or np.any(np.diff(values[:, 0]) < 0)):
                raise ValueError("Window-cache index identities are invalid")
        self.idx_to_sidx_fidx = [tuple(map(int, row)) for row in values]
        if sum(item["windows"] for item in self._descriptors.values()) != len(values):
            raise ValueError("Window-cache split population differs from its shards")
        if self._packed_spec is not None:
            packed = self._packed_spec
            source_descriptors = [item for item in manifest["episodes"] if item["split"] == split]
            sources = [{"episode_index": item["episode_index"], "receipt_sha256": item["receipt_sha256"]}
                       for item in source_descriptors]
            if (packed.get("contract") != CONTRACT or packed.get("state") != "complete"
                    or packed.get("split") != split or packed.get("windows") != len(values)
                    or packed.get("binding_sha256") != manifest["binding_sha256"]
                    or packed.get("index_sha256") != index["sha256"]
                    or packed.get("source_receipts_sha256") != digest(sources)
                    or set(packed.get("arrays", {})) != set(ARRAY_SPECS)):
                raise ValueError("Packed window cache identity/source/index differs")
            for key, (dtype, tail) in ARRAY_SPECS.items():
                identity = packed["arrays"][key]
                if identity.get("shape") != [len(values), *tail] or identity.get("dtype") != dtype:
                    raise ValueError("Packed window array declaration differs")
                path = self._owned(identity["path"])
                if stamp(path) != (identity["size_bytes"], identity["mtime_ns"]):
                    raise ValueError("Packed window array identity changed")
                self._cache_stamps[path] = stamp(path)
            self._open_packed()
        offset = 0
        native_stride = protocol["segment_stride"] * 6
        for sequence, record in enumerate(self.records):
            descriptor = self._descriptors[record["episode_index"]]
            count = descriptor["windows"]
            if self._packed_spec is not None:
                if descriptor.get("row_offset") != offset:
                    raise ValueError("Packed episode row offsets differ from the original order")
                starts = self._packed["start_frames"][offset:offset + count]
            else:
                spec = descriptor["receipt"]["arrays"]["start_frames"]
                starts = np.load(self.root / spec["path"], allow_pickle=False)
            if (starts.shape != (count,) or starts.dtype != np.int64
                    or np.any(starts < 0) or np.any(starts % native_stride)
                    or np.any(starts + 60 >= record["length"])
                    or np.any(np.diff(starts) <= 0)
                    or np.any(np.diff(starts) % native_stride)
                    or digest(starts.tolist()) != descriptor["receipt"]["start_frames_sha256"]):
                raise ValueError("Window-cache episode start-frame order/stride/bounds/digest differs")
            rows = values[offset:offset + count]
            if not np.all(rows[:, 0] == sequence) or not np.array_equal(rows[:, 1], starts):
                raise ValueError("Window-cache split index keys differ from the episode receipt")
            offset += count
        self._source_stamps = {path: original for path, original in archive._metadata_stamps.items()}
        self._source_stamps.update(archive._asset_stamps)
        self._source_stamps[archive.state_path] = stamp(archive.state_path)
        self._source_stamps[archive.vision_cache.manifest_path] = archive.vision_cache._manifest_stamp
        self._cache_stamps[index_path] = stamp(index_path)
        self.assert_current()

    def _owned(self, relative):
        value = Path(relative)
        if value.is_absolute() or ".." in value.parts:
            raise ValueError("Window-cache artifacts must be owned relative paths")
        result = (self.root / value).resolve(strict=True)
        if not result.is_relative_to(self.root) or not result.is_file():
            raise ValueError("Window-cache artifact escapes its directory")
        return result

    def assert_current(self):
        """Epoch-level publication/metadata check, never a per-sample source walk."""
        if stamp(self.manifest_path) != self._manifest_stamp:
            raise ValueError("Window-cache publication changed")
        for path, original in {**self._source_stamps, **self._cache_stamps}.items():
            if stamp(path) != original:
                raise ValueError("Window-cache source/index binding changed")

    def _reset_for_process(self):
        if self._pid != os.getpid():
            self._cache, self._image_cache = OrderedDict(), OrderedDict()
            self._verified, self._image_verified = set(), set()
            self._packed = None
            self._pid = os.getpid()

    def _open_packed(self):
        if self._packed_spec is None or self._packed is not None:
            return
        opened = {}
        for key, identity in self._packed_spec["arrays"].items():
            path = self.root / identity["path"]
            if stamp(path) != (identity["size_bytes"], identity["mtime_ns"]):
                raise ValueError("Packed window array changed before its first mapping")
            value = np.load(path, mmap_mode="r" if self._packed_spec["windows"] else None, allow_pickle=False)
            if list(value.shape) != identity["shape"] or str(value.dtype) != identity["dtype"]:
                raise ValueError("Packed window mmap header differs from its manifest")
            opened[key] = value
        self._packed = opened

    def _arrays(self, episode):
        self._reset_for_process()
        if self._packed_spec is not None:
            self._open_packed()
            descriptor = self._descriptors[episode]
            first, count = descriptor["row_offset"], descriptor["windows"]
            # Plain ndarray views intentionally lack a _mmap close method.
            # self._packed owns every mapping for the worker's lifetime.
            return {key: np.asarray(value[first:first + count]) for key, value in self._packed.items()}
        if episode in self._cache:
            arrays = self._cache.pop(episode)
        else:
            receipt = self._descriptors[episode]["receipt"]
            arrays = {}
            for key, spec in receipt["arrays"].items():
                # Paths and file identities were checked at startup. Do not
                # repeat GPFS stat/JSON reads in DataLoader workers.
                path = self.root / spec["path"]
                if episode not in self._verified and stamp(path) != (spec["size_bytes"], spec["mtime_ns"]):
                    raise ValueError("Window-cache artifact changed before its first mapping")
                value = np.load(path, mmap_mode="r", allow_pickle=False)
                if list(value.shape) != spec["shape"] or str(value.dtype) != spec["dtype"]:
                    raise ValueError("Window-cache mmap header differs from its receipt")
                arrays[key] = value
            self._verified.add(episode)
        self._cache[episode] = arrays
        while len(self._cache) > self.file_cache:
            _, removed = self._cache.popitem(last=False)
            for value in removed.values():
                if getattr(value, "_mmap", None) is not None:
                    value._mmap.close()
        return arrays

    def read_window(self, sequence, start_frame):
        if type(sequence) is not int or not 0 <= sequence < len(self.records):
            raise ValueError("Window-cache sequence is outside the selected split")
        if type(start_frame) is not int or start_frame < 0:
            raise ValueError("Window-cache start frame must be a non-negative native integer")
        record = self.records[sequence]
        episode = record["episode_index"]
        arrays = self._arrays(episode)
        row = int(np.searchsorted(arrays["start_frames"], start_frame))
        if row >= len(arrays["start_frames"]) or int(arrays["start_frames"][row]) != start_frame:
            raise ValueError("Requested window is absent from the frozen cache index")
        output = {key: np.asarray(value[row]).copy() for key, value in arrays.items() if key != "start_frames"}
        selected = start_frame + np.arange(11, dtype=np.int64) * 6
        metadata = deepcopy(self._descriptors[episode]["receipt"]["template_metadata"])
        ground, details = metadata["ground"], metadata["training_window"]
        ground["height_m"] = float(output["floor_height"])
        ground["window_fallback_used"] = bool(output["ground_fallback_used"])
        if "window_fallback_valid_point_count" in ground:
            ground["window_fallback_valid_point_count"] = int(output["ground_point_count"])
        details.update(archive_world_from_window=output["world_from_window"].tolist(),
                       window_from_archive_world=output["window_from_world"].tolist(),
                       floor_height_m=float(output["floor_height"]),
                       floor_rotation=output["floor_transform"][:, :3, :3].tolist(),
                       floor_position=output["floor_transform"][:, :3, 3].tolist())
        output.update(metadata=metadata, source_frame_indices=selected)
        return output

    def read_images(self, record, selected, valid=None):
        self._reset_for_process()
        selected = np.asarray(selected)
        if (selected.shape != (6,) or selected.dtype.kind not in "iu" or np.any(selected < 0)
                or np.any(selected % 6) or np.any(np.diff(selected) != 6)
                or selected[-1] >= record["length"]):
            raise ValueError("Window-cache visual input must select six native 5Hz historical frames")
        if valid is None or np.asarray(valid).shape != (6,):
            raise ValueError("Window-cache visual reads require the frozen historical validity")
        episode = record["episode_index"]
        if episode in self._image_cache:
            features = self._image_cache.pop(episode)
        else:
            descriptor = self.archive.vision_cache._episodes[episode]
            path, original = descriptor["artifacts"]["features.npy"]
            if episode not in self._image_verified and stamp(path) != original:
                raise ValueError("DINO cache changed before its first mapping")
            features = np.load(path, mmap_mode="r", allow_pickle=False)
            if features.dtype != np.float32 or features.shape != (record["length"], 5, 1024):
                raise ValueError("Window-cache DINO shape/dtype differs from native30Hz features")
            self._image_verified.add(episode)
        self._image_cache[episode] = features
        output = np.asarray(features[selected]).copy()
        while len(self._image_cache) > self.file_cache:
            _, removed = self._image_cache.popitem(last=False)
            removed._mmap.close()
        if not np.isfinite(output).all() or np.any(output[~np.asarray(valid).astype(bool)] != 0):
            raise ValueError("Window-cache visual embeddings violate frozen validity/finite values")
        return output

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_cache"], state["_image_cache"] = OrderedDict(), OrderedDict()
        state["_verified"], state["_image_verified"] = set(), set()
        state["_packed"] = None
        return state
