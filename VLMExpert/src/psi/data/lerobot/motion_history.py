"""Read native-frame DINO histories without changing LeRobot action queries.

Only metadata is indexed eagerly. Each DataLoader worker keeps a bounded LRU of
episode feature mmaps, and an item copies precisely six past/current frames.
"""
from __future__ import annotations

from collections import OrderedDict
import os
from pathlib import Path

import numpy as np

from ImageProcess import SharedImageProcessor
from .dino_cache import load_cache_manifest, load_episode_cache


def _close_episode(episode):
    mmap = getattr(episode.features, "_mmap", None)
    if mmap is not None:
        mmap.close()


def history_rows(frame_index, timestamp, source_frame_index, video_timestamp, valid, fps=30):
    """Return current-row and [6] history-row arrays, rejecting the whole gap span.

    Timestamp tolerance is a quarter native frame, accommodating float32 parquet
    timestamps without accepting a missing 30 Hz frame. Both decoded video PTS
    and original parquet time must be continuous throughout the preceding second.
    """
    if fps != 30:
        raise ValueError("Motion DINO histories require the original native 30 Hz source")
    frame_index = np.asarray(frame_index)
    timestamp = np.asarray(timestamp, dtype=np.float64)
    source_frame_index = np.asarray(source_frame_index)
    video_timestamp = np.asarray(video_timestamp, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    n = len(timestamp)
    if any(array.shape != (n,) for array in (frame_index, source_frame_index, video_timestamp, valid)):
        raise ValueError("DINO history sidecar columns must be one-dimensional and equal length")
    if not np.isfinite(timestamp).all() or not np.isfinite(video_timestamp).all():
        raise ValueError("DINO history timestamps must be finite")
    tolerance = 0.25 / fps
    if np.any(np.abs(timestamp - video_timestamp) > tolerance):
        raise ValueError("DINO video PTS do not match the current LeRobot timestamps")
    edges = ((np.diff(frame_index) == 1) & (np.diff(source_frame_index) == 1)
             & (np.abs(np.diff(timestamp) - 1 / fps) <= tolerance)
             & (np.abs(np.diff(video_timestamp) - 1 / fps) <= tolerance))
    # A bad point/edge ends the current valid run. Every candidate requires an
    # uninterrupted 31-frame run, including nonsampled intermediate native frames.
    rows = np.arange(n, dtype=np.int64)
    boundaries = np.ones(n, dtype=bool)
    if n:
        boundaries[1:] = ~edges | ~valid[:-1]
    run_start = np.maximum.accumulate(np.where(boundaries, rows, 0))
    current = rows[valid & (rows - run_start >= fps)]
    if not len(current):
        return current, np.empty((0, 6), dtype=np.int64)
    offsets = np.linspace(1.0, 0.0, 6, dtype=np.float64)
    targets = timestamp[current, None] - offsets
    right = np.minimum(np.searchsorted(timestamp, targets, side="left"), current[:, None])
    left = np.maximum(right - 1, 0)
    selected = np.where(np.abs(timestamp[left] - targets) <= np.abs(timestamp[right] - targets), left, right)
    keep = ((selected[:, 0] >= run_start[current]) & (selected[:, -1] == current)
            & np.all(np.diff(selected, axis=1) > 0, axis=1)
            & np.all(np.abs(timestamp[selected] - targets) <= tolerance, axis=1)
            & np.all(np.abs(video_timestamp[selected] - (video_timestamp[current, None] - offsets)) <= tolerance, axis=1))
    return current[keep], selected[keep]


class MotionHistoryIndex:
    """A filtered-to-original row map plus lazy episode-local feature access."""

    def __init__(self, cache_root, datasets, repo_ids, dataset_roots, image_key, *,
                 shared_image_profile="256x320", max_open_episodes=8):
        if type(max_open_episodes) is not int or max_open_episodes < 1:
            raise ValueError("max_open_episodes must be a positive integer")
        if not (len(datasets) == len(repo_ids) == len(dataset_roots)):
            raise ValueError("Each LeRobot pack needs its own cache/source identity")
        self.image_key = image_key
        self.processor = SharedImageProcessor(shared_image_profile)
        self.shared_image_profile = shared_image_profile
        self.max_open_episodes = max_open_episodes
        self._episodes = OrderedDict()
        self._pid = os.getpid()
        self.pack_info = []
        self.source_episode_ranges = []
        self.episode_slices = {}
        originals, pack_ids, episode_ids, history_indices = [], [], [], []
        raw_offset = filtered_offset = 0
        for pack_id, (dataset, repo_id, source_root) in enumerate(zip(datasets, repo_ids, dataset_roots)):
            root = Path(cache_root) / repo_id
            if len(datasets) == 1 and not (root / "manifest.json").is_file():
                root = Path(cache_root)
            manifest = load_cache_manifest(root, source_root, expected_profile=shared_image_profile)
            if manifest["source"]["video_key"] != image_key:
                raise ValueError("DINO cache video key differs from the configured current VLM image")
            if dataset.meta.fps != 30:
                raise ValueError("DINO history reading requires original native30Hz LeRobot data")
            self.pack_info.append((str(root), str(source_root), manifest))
            metadata = dataset.hf_dataset.with_format("numpy")
            source_columns = {name: np.asarray(metadata[name]).reshape(-1)
                              for name in ("episode_index", "frame_index", "timestamp")}
            if any(len(value) != len(dataset) for value in source_columns.values()):
                raise ValueError("LeRobot row metadata length differs from its source dataset")
            source_episodes = source_columns["episode_index"]
            if not len(source_episodes):
                raise ValueError("History-enabled LeRobot source pack is empty")
            starts = np.r_[0, np.flatnonzero(np.diff(source_episodes) != 0) + 1]
            stops = np.r_[starts[1:], len(source_episodes)]
            episode_order = [int(source_episodes[start]) for start in starts]
            if len(set(episode_order)) != len(episode_order):
                raise ValueError("An episode must occupy one contiguous range of original LeRobot rows")
            source_ranges = {}
            for episode_id, start, stop in zip(episode_order, starts, stops):
                raw_rows = np.arange(start, stop, dtype=np.int64)
                source_ranges[episode_id] = (int(start), int(stop))
                episode = load_episode_cache(root, episode_id, manifest=manifest, dataset_root=source_root)
                try:
                    if (not np.array_equal(source_columns["frame_index"][raw_rows], episode.frame_index)
                            or not np.allclose(source_columns["timestamp"][raw_rows], episode.timestamp,
                                               rtol=0, atol=1e-7)):
                        raise ValueError(f"DINO cache frame/timestamp identities differ from LeRobot episode {episode_id}")
                    current, selected = history_rows(episode.frame_index, episode.timestamp,
                                                    episode.source_frame_index, episode.video_timestamp,
                                                    episode.valid, fps=dataset.meta.fps)
                finally:
                    _close_episode(episode)
                count = len(current)
                if count:
                    originals.extend((raw_offset + raw_rows[current]).tolist())
                    pack_ids.extend([pack_id] * count)
                    episode_ids.extend([episode_id] * count)
                    history_indices.extend(selected.tolist())
                    self.episode_slices[(pack_id, episode_id)] = slice(filtered_offset, filtered_offset + count)
                    filtered_offset += count
            self.source_episode_ranges.append(source_ranges)
            raw_offset += len(dataset)
        self.original_indices = np.asarray(originals, dtype=np.int64)
        self.pack_ids = np.asarray(pack_ids, dtype=np.int64)
        self.episode_ids = np.asarray(episode_ids, dtype=np.int64)
        self.history_indices = np.asarray(history_indices, dtype=np.int64).reshape(-1, 6)
        if not len(self.original_indices):
            raise ValueError("No samples have a complete real one-second native30Hz DINO history")

    def selected_cache_rows(self, index):
        return self.history_indices[index].copy()

    def _episode(self, pack_id, episode_id):
        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()
        key = (int(pack_id), int(episode_id))
        if key not in self._episodes:
            root, source, manifest = self.pack_info[key[0]]
            self._episodes[key] = load_episode_cache(root, key[1], manifest=manifest, dataset_root=source)
        self._episodes.move_to_end(key)
        while len(self._episodes) > self.max_open_episodes:
            _, removed = self._episodes.popitem(last=False)
            _close_episode(removed)
        return self._episodes[key]

    def get_history(self, index):
        episode = self._episode(self.pack_ids[index], self.episode_ids[index])
        features = np.array(episode.features[self.history_indices[index]], copy=True)
        if features.shape != (6, 5, 1024) or not np.isfinite(features).all():
            raise ValueError("Selected DINO history must be finite [6,5,1024]")
        return features

    def process_current_rgb(self, sample, *, apply_geometry=True):
        """Restore decoded RGB bytes, optionally applying clean cache geometry.

        VLM-only color augmentation defers geometry to Psi0ModelTransform so
        colors are applied at the profile's intermediate size, before grid resize.
        The DINO history remains the original deterministic cached features.
        """
        import torch
        image = sample[self.image_key]
        if not isinstance(image, torch.Tensor) or image.ndim != 3 or image.shape[0] != 3:
            raise ValueError("History-enabled LeRobot must provide one current RGB float Tensor [3,H,W]")
        pixels = image.detach().cpu().numpy()
        if (not np.issubdtype(pixels.dtype, np.floating) or not np.isfinite(pixels).all()
                or np.any(pixels < 0) or np.any(pixels > 1)):
            raise ValueError("LeRobot current RGB must be a finite float image in [0,1]")
        rgb = np.rint(pixels.transpose(1, 2, 0) * 255).astype(np.uint8)
        processed = self.processor.process_image(rgb) if apply_geometry else rgb
        sample[self.image_key] = torch.from_numpy(processed.copy()).permute(2, 0, 1).float().div_(255)
        return sample

    def close(self):
        for episode in getattr(self, "_episodes", {}).values():
            _close_episode(episode)
        if hasattr(self, "_episodes"):
            self._episodes.clear()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_episodes"] = OrderedDict()
        return state

    def __del__(self):
        self.close()
