"""Direct reads of accepted packed windows for frozen-feature diagnostics.

This deliberately does not construct a source archive or repeat its numerical
acceptance. Only publication metadata, the requested split indices/headers and
the DINO episodes actually consumed by the probe are opened.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .window_cache import CONTRACT, digest, stamp, window_protocol
from .vision_cache import CONTRACT as VISION_CONTRACT, storage_digest


PROBE_ARRAYS = {
    "start_frames": ("int64", ()),
    "motion": ("float32", (11, 245)),
    "motion_valid": ("bool", (11, 245)),
    "world_from_window": ("float64", (4, 4)),
    "floor_transform": ("float64", (11, 4, 4)),
    "historical_image_valid": ("bool", (6,)),
}


def _owned(root, relative):
    name = Path(relative)
    if name.is_absolute() or ".." in name.parts:
        raise ValueError("Accepted probe artifact must use an owned relative path")
    path = (root / name).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Accepted probe artifact escapes its publication")
    return path


class AcceptedProbeCache:
    """Read raw labels and six native30Hz historical DINO frames on demand.

    ``records`` is keyed by split. ``indices(split)`` contains the complete
    accepted ``[split_record_index, native_start_frame]`` index. ``read`` takes
    a packed row number from this index; it never changes the original split.
    ``select_balanced`` chooses one deterministic set for all feature variants.
    """

    def __init__(self, cache_root, vision_root, splits=("train", "val"), *, file_cache=4):
        self.root = Path(cache_root).resolve(strict=True)
        self.vision_root = Path(vision_root).resolve(strict=True)
        self.splits = tuple(splits)
        if (not self.splits or len(set(self.splits)) != len(self.splits)
                or set(self.splits) - {"train", "val"}):
            raise ValueError("Frozen-prior probes may open only distinct train/val splits")
        if type(file_cache) is not int or not 1 <= file_cache <= 32:
            raise ValueError("DINO feature LRU must contain between one and 32 episodes")
        self.file_cache = file_cache
        self._stamps, self.open_paths = {}, []
        self.manifest_path = _owned(self.root, "manifest.json")
        manifest_bytes = self._read(self.manifest_path)
        self.manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        self.manifest = manifest = json.loads(manifest_bytes)
        if (manifest.get("contract") != CONTRACT or manifest.get("state") != "complete"
                or manifest.get("scope") != "all" or manifest.get("diagnostic") is not False
                or manifest.get("selected_episode_indices") is not None):
            raise ValueError("Probe requires a complete non-diagnostic accepted window publication")
        binding = manifest.get("binding")
        if not isinstance(binding, dict) or manifest.get("binding_sha256") != digest(binding):
            raise ValueError("Accepted window source binding is missing or changed")
        required = {"archive_binding", "archive_fingerprint", "split_fingerprint",
                    "vision_manifest_sha256", "protocol", "image_profile_sha256"}
        if not required <= binding.keys():
            raise ValueError("Accepted window binding is incomplete")
        if binding["protocol"] != window_protocol():
            raise ValueError("Accepted window protocol differs from the fixed 5Hz representation")
        self.binding = binding
        archive_root = Path(manifest.get("archive_root", "")).resolve(strict=True)
        self.state_path = _owned(archive_root, ".pipeline/dataset-state.json")
        state = self._json(self.state_path)
        if (state.get("state") != "accepted" or state.get("formal_training_ready") is not True
                or state.get("independent_acceptance_pending") is not False
                or state.get("window_cache_manifest_sha256") != self.manifest_sha256):
            raise ValueError("Dataset state does not accept this exact packed window publication")
        self.accepted_state = state
        records = self._json(_owned(self.root, manifest["records_path"]), manifest["records_sha256"])
        if (not isinstance(records, list)
                or len({r.get("episode_index") for r in records}) != len(records)):
            raise ValueError("Accepted cache records duplicate episodes")
        descriptors = manifest.get("episodes", [])
        if [d.get("episode_index") for d in descriptors] != [r["episode_index"] for r in records]:
            raise ValueError("Accepted episode descriptors differ from the bound record population")
        self.records, self._indices, self._packed = {}, {}, {}
        self._index_paths, self._packed_paths = {}, {}
        groups = {}
        for split in self.splits:
            selected = manifest.get("records_by_split", {}).get(split)
            expected = [i for i, record in enumerate(records) if record.get("split") == split]
            if selected != expected or not expected:
                raise ValueError(f"Accepted {split} record membership/order differs")
            self.records[split] = [records[i] for i in expected]
            groups[split] = {group for record in self.records[split]
                             for group in record.get("split_group_ids", [])}
            split_descriptors = [descriptors[i] for i in expected]
            if any(d.get("split") != split or d.get("source_record_sha256") != digest(record)
                   for d, record in zip(split_descriptors, self.records[split])):
                raise ValueError(f"Accepted {split} descriptor source/split differs")
            self._open_split(split, split_descriptors)
        if len(self.splits) == 2 and groups[self.splits[0]] & groups[self.splits[1]]:
            raise ValueError("Accepted train/val recording or subject groups overlap")
        del records
        self.vision_manifest_path = _owned(self.vision_root, "manifest.json")
        self.vision_manifest = vision = self._json(self.vision_manifest_path, binding["vision_manifest_sha256"])
        if (vision.get("contract") != VISION_CONTRACT or vision.get("state") != "complete"
                or vision.get("scope") != "all" or vision.get("validity_equal_source") is not True
                or (vision.get("fps"), vision.get("native_fps"), vision.get("sampling_step"),
                    vision.get("frame_phase")) != (30, 30, 1, 0)
                or vision.get("shape") != [5, 1024] or vision.get("storage_dtype") != "float32"):
            raise ValueError("DINO publication is not the accepted native30Hz shared profile")
        for key in ("info_sha256", "episodes_metadata_sha256", "split_metadata_sha256"):
            if vision.get("source_binding", {}).get(key) != binding["archive_binding"].get(key):
                raise ValueError(f"DINO/source binding differs: {key}")
        process = vision.get("image_process", {})
        if (process.get("profile_sha256") != binding["image_profile_sha256"]
                or not isinstance(vision.get("encoder"), dict) or not vision["encoder"]
                or vision["encoder"].get("image_process") != process):
            raise ValueError("DINO encoder or shared RGB profile binding differs")
        self._pid, self._images, self._consumed = os.getpid(), OrderedDict(), {}
        self.constructor_open_paths = list(self.open_paths)
        self.assert_current()

    def _read(self, path, expected_sha=None):
        original = stamp(path)
        self.open_paths.append(str(path))
        data = path.read_bytes()
        if expected_sha is not None and hashlib.sha256(data).hexdigest() != expected_sha:
            raise ValueError(f"Accepted probe metadata SHA256 changed: {path}")
        if stamp(path) != original:
            raise ValueError(f"Accepted metadata changed while reading: {path}")
        self._stamps[path] = original
        return data

    def _json(self, path, expected_sha=None):
        return json.loads(self._read(path, expected_sha))

    def _map(self, path, expected_stamp=None):
        original = stamp(path) if expected_stamp is None else expected_stamp
        if stamp(path) != original:
            raise ValueError(f"Accepted probe payload stamp changed: {path}")
        self.open_paths.append(str(path))
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        if stamp(path) != original:
            value._mmap.close()
            raise ValueError(f"Accepted payload changed while mapping: {path}")
        self._stamps[path] = original
        return value

    def _open_split(self, split, descriptors):
        manifest = self.manifest
        index_spec = manifest.get("indices", {}).get(split, {})
        path = _owned(self.root, index_spec["path"])
        self._read(path, index_spec["sha256"])
        values = self._map(path)
        count = index_spec.get("windows")
        if type(count) is not int or values.dtype != np.int64 or values.shape != (count, 2):
            raise ValueError(f"Accepted {split} index shape/dtype differs")
        seq, starts = values[:, 0], values[:, 1]
        records = self.records[split]
        stride = self.binding["protocol"]["segment_stride"] * 6
        if (np.any(seq < 0) or np.any(seq >= len(records)) or np.any(np.diff(seq) < 0)
                or np.any(starts < 0) or np.any(starts % stride)):
            raise ValueError(f"Accepted {split} index sequence/native stride is invalid")
        lengths = np.asarray([record["length"] for record in records])
        if np.any(starts + 60 >= lengths[seq]):
            raise ValueError(f"Accepted {split} index exceeds a complete eleven-frame window")
        same = seq[1:] == seq[:-1]
        if np.any(np.diff(starts)[same] <= 0) or np.any(np.diff(starts)[same] % stride):
            raise ValueError(f"Accepted {split} index duplicates or reorders windows")
        receipts = [{"episode_index": d["episode_index"], "receipt_sha256": d["receipt_sha256"]}
                    for d in descriptors]
        packed = manifest.get("packed", {}).get(split)
        if (not isinstance(packed, dict) or packed.get("contract") != CONTRACT
                or packed.get("state") != "complete" or packed.get("split") != split
                or packed.get("windows") != count or packed.get("binding_sha256") != manifest["binding_sha256"]
                or packed.get("index_sha256") != index_spec["sha256"]
                or packed.get("source_receipts_sha256") != digest(receipts)):
            raise ValueError(f"Accepted packed {split} identity/source/index differs")
        offset = 0
        for sequence, descriptor in enumerate(descriptors):
            windows = descriptor.get("windows")
            if (type(windows) is not int or windows < 0 or descriptor.get("row_offset") != offset
                    or int(np.searchsorted(seq, sequence, side="left")) != offset
                    or int(np.searchsorted(seq, sequence, side="right")) != offset + windows):
                raise ValueError(f"Accepted {split} descriptor offsets/counts differ from index")
            offset += windows
        if offset != count:
            raise ValueError(f"Accepted {split} total population differs")
        self._indices[split], self._index_paths[split] = values, path
        self._packed[split], self._packed_paths[split] = {}, {}
        for key, (dtype, tail) in PROBE_ARRAYS.items():
            spec = packed.get("arrays", {}).get(key, {})
            if spec.get("shape") != [count, *tail] or spec.get("dtype") != dtype:
                raise ValueError(f"Accepted packed {split} declaration differs: {key}")
            path = _owned(self.root, spec["path"])
            if path.parent != self.root / "packed" / split:
                raise ValueError("Probe cannot open another split's packed payload")
            value = self._map(path, (spec["size_bytes"], spec["mtime_ns"]))
            if value.shape != tuple(spec["shape"]) or str(value.dtype) != dtype:
                value._mmap.close()
                raise ValueError(f"Accepted packed {split} header differs: {key}")
            self._packed[split][key], self._packed_paths[split][key] = value, path

    def indices(self, split):
        """Read-only complete packed-row identities for an explicitly opened split."""
        if split not in self.splits:
            raise ValueError("Probe split was not explicitly opened")
        return self._indices[split]

    def select_balanced(self, *, max_per_episode=20, max_per_source=None,
                        max_episodes_per_source=None, min_start_gap=60, seed=20261017):
        """Cap both sources consistently while retaining broad episode coverage.

        ``max_episodes_per_source`` may be a per-split mapping or one integer.
        Window starts in an episode remain at least ``min_start_gap`` native
        frames apart. Counts are not padded: short episodes contribute less.
        Optional source window caps are allocated round-robin across episodes.
        Sampling is independent of labels and validation outcomes.
        """
        if type(max_per_episode) is not int or not 1 <= max_per_episode <= 20:
            raise ValueError("Probe selection allows between one and 20 windows per episode")
        if max_per_source is not None and (type(max_per_source) is not int or max_per_source < 1):
            raise ValueError("Probe source cap must be a positive integer")
        if type(min_start_gap) is not int or min_start_gap < 60 or min_start_gap % 6:
            raise ValueError("Probe windows require at least 60 native frames between starts")
        output = {}
        for split in self.splits:
            rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0 if split == "train" else 1]))
            index = self.indices(split)
            seq = index[:, 0]
            by_source = {}
            limit = (max_episodes_per_source.get(split) if isinstance(max_episodes_per_source, dict)
                     else max_episodes_per_source)
            if limit is not None and (type(limit) is not int or limit < 1):
                raise ValueError("Probe episode cap must be a positive integer for every split")
            for sequence, record in enumerate(self.records[split]):
                left, right = np.searchsorted(seq, sequence, side="left"), np.searchsorted(seq, sequence, side="right")
                if left == right:
                    continue
                starts = index[left:right, 1]
                candidates = []
                next_row = int(rng.integers(min(right - left, max(1, min_start_gap // 30))))
                while next_row < right - left:
                    candidates.append(int(left + next_row))
                    next_row = int(np.searchsorted(starts, starts[next_row] + min_start_gap))
                count = min(max_per_episode, len(candidates))
                # One candidate from each temporal bin, with deterministic jitter.
                edges = np.linspace(0, len(candidates), count + 1).astype(np.int64)
                rows = [candidates[int(rng.integers(edges[i], edges[i + 1]))] for i in range(count)]
                by_source.setdefault(record["source"], []).append(rows)
            if set(by_source) != {"egosuite", "egoexo4d_hm"}:
                raise ValueError(f"Probe {split} must cover exactly EgoSuite and EgoExo4D-HM")
            selected = []
            for source in sorted(by_source):
                episodes = by_source[source]
                rng.shuffle(episodes)
                if limit is not None:
                    episodes = episodes[:limit]
                quota = sum(map(len, episodes))
                if max_per_source is not None:
                    quota = min(quota, max_per_source)
                chosen = []
                for depth in range(max_per_episode):
                    for rows in episodes:
                        if depth < len(rows):
                            chosen.append(rows[depth])
                            if len(chosen) == quota:
                                break
                    if len(chosen) == quota:
                        break
                selected.extend(chosen)
            output[split] = np.asarray(sorted(selected), dtype=np.int64)
        return output

    def _reset_for_process(self):
        if self._pid != os.getpid():
            for value in self._images.values():
                value._mmap.close()
            self._images, self._consumed, self._pid = OrderedDict(), {}, os.getpid()

    def _features(self, record):
        self._reset_for_process()
        episode = record["episode_index"]
        if episode in self._images:
            value = self._images.pop(episode)
            self._images[episode] = value
            return value
        prefix = f"episodes/{episode:08d}"
        receipt_path = _owned(self.vision_root, f"{prefix}/receipt.json")
        receipt = self._json(receipt_path)
        if (receipt.get("contract") != VISION_CONTRACT or receipt.get("state") != "complete"
                or receipt.get("episode_index") != episode or receipt.get("source_key") != record["source_key"]
                or receipt.get("source_record_sha256") != storage_digest(record)
                or receipt.get("frames") != record["length"] or receipt.get("validity_equal_source") is not True
                or receipt.get("encoder") != self.vision_manifest["encoder"]):
            raise ValueError("Consumed DINO receipt/source/encoder differs from accepted publication")
        count = receipt.get("valid_count")
        identities = receipt.get("output_identities", {})
        if (type(count) is not int or not 0 <= count <= record["length"]
                or set(identities) != {"features.npy", "join.npz"}):
            raise ValueError("Consumed DINO validity/artifact declarations are incomplete")
        path = _owned(self.vision_root, f"{prefix}/features.npy")
        identity = identities["features.npy"]
        expected = identity.get("size_bytes"), identity.get("mtime_ns")
        value = self._map(path, expected)
        if value.shape != (record["length"], 5, 1024) or value.dtype != np.float32:
            value._mmap.close()
            raise ValueError("Consumed DINO feature header differs")
        consumed = {"receipt_path": receipt_path, "receipt_stamp": self._stamps[receipt_path],
                    "feature_path": path, "feature_stamp": expected}
        if episode in self._consumed and self._consumed[episode] != consumed:
            value._mmap.close()
            raise ValueError("Consumed DINO identity changed after LRU eviction")
        self._consumed[episode] = consumed
        self._images[episode] = value
        while len(self._images) > self.file_cache:
            _, removed = self._images.popitem(last=False)
            removed._mmap.close()
        return value

    def read(self, split, index, *, include_images=True):
        """Copy a single raw window; labels are never model conditioning."""
        values = self.indices(split)
        if (isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer))
                or not 0 <= index < len(values)):
            raise IndexError("Probe row outside its explicitly opened split")
        index = int(index)
        sequence, start = map(int, values[index])
        record = self.records[split][sequence]
        packed = self._packed[split]
        if int(packed["start_frames"][index]) != start:
            raise ValueError("Packed probe start differs from its SHA-bound index")
        result = {key: np.asarray(value[index]).copy() for key, value in packed.items()
                  if key != "start_frames"}
        result.update(split=split, index=index, source=record["source"], episode_index=record["episode_index"],
                      source_key=record["source_key"], start_frame=start,
                      split_group_ids=tuple(record.get("split_group_ids", ())))
        if include_images:
            native_frames = start + np.arange(6, dtype=np.int64) * 6
            features = np.asarray(self._features(record)[native_frames]).copy()
            if not np.isfinite(features).all() or np.any(features[~result["historical_image_valid"]] != 0):
                raise ValueError("Consumed historical DINO violates accepted finite/validity contract")
            result["image_features"] = features
        return result

    def assert_current(self):
        """Check opened publication stamps; never walk unseen episodes/payloads."""
        for path, original in self._stamps.items():
            if stamp(path) != original:
                raise ValueError(f"Accepted probe publication changed: {path}")
        return {"passed": True, "artifacts_checked": len(self._stamps),
                "consumed_dino_episodes": len(getattr(self, "_consumed", {})),
                "unseen_episode_scan": False, "payloads_hashed": False}

    def close(self):
        for value in self._images.values():
            value._mmap.close()
        self._images.clear()
        for arrays in self._packed.values():
            for value in arrays.values():
                value._mmap.close()
        for value in self._indices.values():
            value._mmap.close()
        self._packed, self._indices = {}, {}

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_images"], state["_consumed"] = OrderedDict(), {}
        state["_packed"], state["_indices"], state["_pid"] = {}, {}, None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._indices = {split: self._map(path, self._stamps[path])
                         for split, path in self._index_paths.items()}
        self._packed = {split: {key: self._map(path, self._stamps[path]) for key, path in paths.items()}
                        for split, paths in self._packed_paths.items()}
