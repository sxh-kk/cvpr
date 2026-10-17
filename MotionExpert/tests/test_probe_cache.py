"""Acceptance-bound probe reads avoid source scans and preserve split/time IDs."""
import hashlib
import json
import os
import pickle
from collections import Counter

import numpy as np
import pytest

from MotionExpert.dataset.probe_cache import AcceptedProbeCache, PROBE_ARRAYS
from MotionExpert.dataset.window_cache import CONTRACT, digest, stamp, window_protocol
from MotionExpert.dataset.vision_cache import CONTRACT as VISION_CONTRACT, storage_digest


def _save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _publication(tmp_path, *, overlapping_groups=False):
    root, vision, archive = tmp_path / "windows", tmp_path / "vision", tmp_path / "archive"
    root.mkdir(parents=True)
    process = {"profile_sha256": "profile-identity", "profile_id": "256x480"}
    encoder = {"model": "dinov2-with-registers-large", "image_process": process}
    source_binding = {"info_sha256": "info", "episodes_metadata_sha256": "episodes",
                      "split_metadata_sha256": {"splits.json": "splits"}}
    vision_manifest = {"contract": VISION_CONTRACT, "state": "complete", "scope": "all",
                       "validity_equal_source": True, "fps": 30, "native_fps": 30,
                       "sampling_step": 1, "frame_phase": 0, "shape": [5, 1024],
                       "storage_dtype": "float32", "source_binding": source_binding,
                       "image_process": process, "encoder": encoder}
    vision_sha = _save_json(vision / "manifest.json", vision_manifest)
    binding = {"archive_binding": source_binding, "archive_fingerprint": "archive-fingerprint",
               "split_fingerprint": "split-fingerprint", "vision_manifest_sha256": vision_sha,
               "protocol": window_protocol(), "image_profile_sha256": process["profile_sha256"]}
    records, descriptors = [], []
    for episode in range(8):
        split = "train" if episode < 4 else "val"
        source = "egosuite" if episode % 4 < 2 else "egoexo4d_hm"
        group = "shared" if overlapping_groups and episode in (0, 4) else str(episode)
        record = {"episode_index": episode, "split": split, "source": source,
                  "source_key": f"{source}:{episode}", "length": 151,
                  "split_group_ids": [group]}
        records.append(record)
        descriptors.append({"episode_index": episode, "split": split, "windows": 4,
                            "row_offset": (episode % 4) * 4,
                            "source_record_sha256": digest(record), "receipt_sha256": f"receipt-{episode}"})
        directory = vision / f"episodes/{episode:08d}"
        directory.mkdir(parents=True)
        features = np.broadcast_to(np.arange(151, dtype=np.float32)[:, None, None], (151, 5, 1024))
        np.save(directory / "features.npy", features)
        (directory / "join.npz").write_bytes(b"not needed for accepted packed selection")
        identities = {name: dict(zip(("size_bytes", "mtime_ns"), stamp(directory / name)))
                      for name in ("features.npy", "join.npz")}
        _save_json(directory / "receipt.json", {"contract": VISION_CONTRACT, "state": "complete",
                   "episode_index": episode, "source_key": record["source_key"],
                   "source_record_sha256": storage_digest(record), "frames": 151,
                   "valid_count": 151, "validity_equal_source": True, "encoder": encoder,
                   "output_identities": identities})
    records_sha = _save_json(root / "records.json", records)
    manifest = {"contract": CONTRACT, "state": "complete", "scope": "all", "diagnostic": False,
                "archive_root": str(archive), "binding": binding, "binding_sha256": digest(binding),
                "records_path": "records.json", "records_sha256": records_sha,
                "episodes": descriptors, "records_by_split": {}, "indices": {}, "packed": {}}
    for split in ("train", "val"):
        selected = [i for i, r in enumerate(records) if r["split"] == split]
        manifest["records_by_split"][split] = selected
        values = np.asarray([(i, start) for i in range(4) for start in (0, 30, 60, 90)], dtype=np.int64)
        np.save(root / f"index_{split}.npy", values)
        index_sha = hashlib.sha256((root / f"index_{split}.npy").read_bytes()).hexdigest()
        manifest["indices"][split] = {"path": f"index_{split}.npy", "sha256": index_sha, "windows": 16}
        arrays = {}
        directory = root / "packed" / split
        directory.mkdir(parents=True)
        for key, (dtype, tail) in PROBE_ARRAYS.items():
            value = np.ones((16, *tail), dtype=dtype)
            if key == "start_frames":
                value = values[:, 1].copy()
            path = directory / f"{key}.npy"
            np.save(path, value)
            arrays[key] = {"shape": list(value.shape), "dtype": dtype,
                           "path": str(path.relative_to(root)),
                           **dict(zip(("size_bytes", "mtime_ns"), stamp(path)))}
        receipts = [{"episode_index": descriptors[i]["episode_index"],
                     "receipt_sha256": descriptors[i]["receipt_sha256"]} for i in selected]
        manifest["packed"][split] = {"contract": CONTRACT, "state": "complete", "split": split,
            "windows": 16, "binding_sha256": digest(binding), "index_sha256": index_sha,
            "source_receipts_sha256": digest(receipts), "arrays": arrays}
    manifest_sha = _save_json(root / "manifest.json", manifest)
    state_path = archive / ".pipeline/dataset-state.json"
    _save_json(state_path, {"state": "accepted", "formal_training_ready": True,
                           "independent_acceptance_pending": False,
                           "window_cache_manifest_sha256": manifest_sha})
    return root, vision, state_path


def test_lazy_read_preserves_native_history_and_raw_labels(tmp_path):
    root, vision, _ = _publication(tmp_path)
    cache = AcceptedProbeCache(root, vision, file_cache=1)
    assert not any("/episodes/" in path or "parquet" in path for path in cache.constructor_open_paths)
    result = cache.read("val", 5, include_images=False)
    assert "image_features" not in result
    assert cache.assert_current()["consumed_dino_episodes"] == 0
    result = cache.read("val", 5)
    assert result["episode_index"] == 5
    assert result["start_frame"] == 30
    assert result["motion"].shape == (11, 245)
    assert result["motion_valid"].dtype == np.bool_
    np.testing.assert_array_equal(result["image_features"][:, 0, 0], [30, 36, 42, 48, 54, 60])
    assert result["image_features"].shape == (6, 5, 1024)
    result["motion"][0, 0] = 1234
    assert cache.read("val", 5, include_images=False)["motion"][0, 0] == 1
    with pytest.raises(ValueError, match="not explicitly opened"):
        cache.read("test", 0)
    cache.close()


def test_fixed_selection_balances_episode_caps_without_overlap(tmp_path):
    root, vision, _ = _publication(tmp_path)
    cache = AcceptedProbeCache(root, vision)
    options = dict(max_per_episode=16, max_episodes_per_source={"train": 2, "val": 1},
                   min_start_gap=60, seed=620017)
    selection = cache.select_balanced(**options)
    again = cache.select_balanced(**options)
    for split, rows in selection.items():
        np.testing.assert_array_equal(rows, again[split])
        population = cache.indices(split)[rows]
        episodes = Counter()
        for sequence in np.unique(population[:, 0]):
            record = cache.records[split][sequence]
            assert record["split"] == split
            episodes[record["source"]] += 1
            starts = population[population[:, 0] == sequence, 1]
            assert len(starts) <= 16
            assert np.all(np.diff(starts) >= 60)
        assert episodes == {"egosuite": options["max_episodes_per_source"][split],
                            "egoexo4d_hm": options["max_episodes_per_source"][split]}
    assert cache.assert_current()["consumed_dino_episodes"] == 0
    cache.close()


def test_rejects_unaccepted_or_cross_split_publications(tmp_path):
    root, vision, state_path = _publication(tmp_path)
    state = json.loads(state_path.read_text())
    state["window_cache_manifest_sha256"] = "changed"
    _save_json(state_path, state)
    with pytest.raises(ValueError, match="does not accept"):
        AcceptedProbeCache(root, vision)
    root2, vision2, _ = _publication(tmp_path / "other", overlapping_groups=True)
    with pytest.raises(ValueError, match="groups overlap"):
        AcceptedProbeCache(root2, vision2)


def test_consumed_feature_stamp_change_and_safe_pickle(tmp_path):
    root, vision, _ = _publication(tmp_path)
    cache = AcceptedProbeCache(root, vision)
    cache.read("train", 0)
    restored = pickle.loads(pickle.dumps(cache))
    assert not restored._images
    np.testing.assert_array_equal(restored.read("train", 0)["image_features"],
                                  cache.read("train", 0)["image_features"])
    feature = vision / "episodes/00000000/features.npy"
    original = feature.stat()
    os.utime(feature, ns=(original.st_atime_ns, original.st_mtime_ns + 1000000))
    with pytest.raises(ValueError, match="publication changed"):
        cache.assert_current()
    restored.close()
    cache.close()
