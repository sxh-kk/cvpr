#!/usr/bin/env python3
"""Compute training-only statistics for the WBT-native-v1 protocol.

The script reads parquet state/action columns through :class:`WBTDataset`, so
the same metadata checks and source-hand canonicalization used by training are
applied before statistics are accumulated.  Video files are never opened.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from psi.data.humanoid.wbt_dataset import WBTDataset
from psi.data.humanoid.wbt_hand_protocol import (
    NATIVE_DIM,
    PROTOCOL_GROUPS,
    PROTOCOL_NAMES,
    WBT_HAND_PROTOCOL,
    WBT_NATIVE_PROTOCOL,
)


@dataclass
class RunningStats:
    dim: int
    reservoir_size: int
    seed: int

    def __post_init__(self) -> None:
        if self.dim <= 0 or self.reservoir_size <= 0:
            raise ValueError("dim and reservoir_size must be positive")
        self.count = 0
        self.mean = np.zeros(self.dim, dtype=np.float64)
        self.m2 = np.zeros(self.dim, dtype=np.float64)
        self.minimum = np.full(self.dim, np.inf, dtype=np.float64)
        self.maximum = np.full(self.dim, -np.inf, dtype=np.float64)
        self.reservoir = np.empty((self.reservoir_size, self.dim), dtype=np.float32)
        self.reservoir_count = 0
        self.rng = np.random.default_rng(self.seed)

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.dim:
            raise ValueError(f"Expected (N,{self.dim}) statistics input, got {values.shape}")
        if values.shape[0] == 0:
            return
        if not np.isfinite(values).all():
            raise ValueError("Cannot compute WBT statistics from NaN or Inf")

        batch_count = values.shape[0]
        batch_mean = values.mean(axis=0)
        centered = values - batch_mean
        batch_m2 = np.sum(centered * centered, axis=0)
        previous_count = self.count
        if self.count == 0:
            self.mean = batch_mean
            self.m2 = batch_m2
        else:
            total = self.count + batch_count
            delta = batch_mean - self.mean
            self.m2 += batch_m2 + delta * delta * (self.count * batch_count / total)
            self.mean += delta * (batch_count / total)
        self.count += batch_count
        self.minimum = np.minimum(self.minimum, values.min(axis=0))
        self.maximum = np.maximum(self.maximum, values.max(axis=0))

        # Reservoir sampling keeps quantile memory bounded for multi-million
        # frame exports while preserving an unbiased sample of each dimension.
        for offset, row in enumerate(values.astype(np.float32, copy=False), start=1):
            seen = previous_count + offset
            if self.reservoir_count < self.reservoir_size:
                self.reservoir[self.reservoir_count] = row
                self.reservoir_count += 1
                continue
            slot = int(self.rng.integers(0, seen))
            if slot < self.reservoir_size:
                self.reservoir[slot] = row

    def finalize(self) -> dict[str, object]:
        if self.count == 0 or self.reservoir_count == 0:
            raise ValueError("No WBT samples were found")
        sample = self.reservoir[: self.reservoir_count].astype(np.float64)
        variance = self.m2 / max(self.count - 1, 1)
        return {
            "min": self.minimum.astype(np.float32).tolist(),
            "max": self.maximum.astype(np.float32).tolist(),
            "q01": np.quantile(sample, 0.01, axis=0).astype(np.float32).tolist(),
            "q99": np.quantile(sample, 0.99, axis=0).astype(np.float32).tolist(),
            "mean": self.mean.astype(np.float32).tolist(),
            "std": np.sqrt(np.maximum(variance, 0.0)).astype(np.float32).tolist(),
            "count": int(self.count),
            "quantile_sample_count": int(self.reservoir_count),
        }


def _parse_repo_ids(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        result.extend(part.strip() for part in value.split(",") if part.strip())
    if not result:
        raise ValueError("At least one --repo-id is required; these are training repositories")
    if len(set(result)) != len(result):
        raise ValueError("Duplicate training repository ids are not allowed")
    return result


def _load_episode_map(path: str | None) -> dict[str, list[int]] | None:
    if path is None:
        return None
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("--episode-map must contain a JSON object mapping repo ids to episode lists")
    result: dict[str, list[int]] = {}
    for repo, episodes in payload.items():
        if not isinstance(episodes, list):
            raise ValueError(f"Episode selection for {repo!r} must be a list")
        selected = [int(episode) for episode in episodes]
        if not selected or any(episode < 0 for episode in selected):
            raise ValueError(f"Episode selection for {repo!r} must contain non-negative ids")
        if len(set(selected)) != len(selected):
            raise ValueError(f"Episode selection for {repo!r} contains duplicates")
        result[str(repo)] = selected
    return result


def compute_stats(
    data_root: str | Path,
    repo_ids: Sequence[str],
    *,
    episode_map: dict[str, list[int]] | None = None,
    reservoir_size: int = 200_000,
    chunk_size: int = 8192,
    seed: int = 0,
) -> dict[str, object]:
    repo_ids = _parse_repo_ids(repo_ids)
    if episode_map is not None:
        missing = sorted(set(repo_ids) - set(episode_map))
        extra = sorted(set(episode_map) - set(repo_ids))
        if missing or extra:
            raise ValueError(
                "Episode map must cover exactly the selected training repositories; "
                f"missing={missing}, extra={extra}"
            )
    dataset = WBTDataset(
        data_root=data_root,
        repo_ids=repo_ids,
        split="train",
        episodes=episode_map,
        # No sampling/video access is needed for statistics.
        action_chunk_size=1,
        validate_data=True,
        frame_cache_size=0,
    )
    state_stats = RunningStats(NATIVE_DIM, reservoir_size, seed)
    action_stats = RunningStats(NATIVE_DIM, reservoir_size, seed + 1)
    selected_episodes: dict[str, list[int]] = {}
    layouts: dict[str, dict[str, str]] = {}
    hand_units: dict[str, dict[str, str]] = {}
    fps_by_repo: dict[str, float] = {}
    protocol_by_repo: dict[str, object] = {}
    for repo in dataset.repos:
        selected_episodes[repo.repo_id] = [episode.episode_index for episode in repo.episodes]
        layouts[repo.repo_id] = {
            "state": repo.state_hand_layout_id,
            "action": repo.action_hand_layout_id,
        }
        hand_units[repo.repo_id] = {
            "state": str(repo.protocol["state_hand_layout"]["units"]),
            "action": str(repo.protocol["action_hand_layout"]["units"]),
        }
        fps_by_repo[repo.repo_id] = float(repo.info["fps"])
        protocol_by_repo[repo.repo_id] = repo.protocol

    # Stream once over the selected data.  The public iterator applies exactly
    # the source adapters used by the training transform.
    for states, actions, _ in dataset.iter_protocol_vectors(chunk_size=chunk_size):
        state_stats.update(states)
        action_stats.update(actions)

    group_metadata = {
        name: {
            "start": int(group["start"]),
            "end": int(group["end"]),
            "names": list(group["names"]),
        }
        for name, group in PROTOCOL_GROUPS.items()
    }
    return {
        "version": 1,
        "protocol": WBT_NATIVE_PROTOCOL,
        "hand_protocol": WBT_HAND_PROTOCOL,
        "state_dim": NATIVE_DIM,
        "action_dim": NATIVE_DIM,
        "fps": float(dataset.expected_fps),
        "absolute_action": True,
        "use_delta_actions": False,
        "dimensions": list(PROTOCOL_NAMES),
        "groups": group_metadata,
        "hand_layout_ids": layouts,
        "units": {
            "root_position": "meters",
            "root_quaternion": "unitless",
            "body_joints": "radians",
            "source_hands": hand_units,
        },
        "quantiles": {
            "method": "deterministic_reservoir",
            "reservoir_size": int(reservoir_size),
            "seed": int(seed),
        },
        "states": state_stats.finalize(),
        "action": action_stats.finalize(),
        "source": {
            "split": "train",
            "repo_ids": list(repo_ids),
            "episodes": selected_episodes,
            "hand_layout_ids": layouts,
            "fps_by_repo": fps_by_repo,
            "protocol_metadata": protocol_by_repo,
            "note": "Statistics are computed after wbt-source-12d-v1 canonicalization; validation repositories must not be included.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="WBT root directory")
    parser.add_argument(
        "--repo-id",
        action="append",
        required=True,
        help="Training repository id (repeat or provide comma-separated ids)",
    )
    parser.add_argument("--episode-map", help="JSON mapping of training repo ids to episode ids")
    parser.add_argument("--output", default="assets/stats/wbt_native_stats.json")
    parser.add_argument("--reservoir-size", type=int, default=200_000)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    stats = compute_stats(
        args.root,
        args.repo_id,
        episode_map=_load_episode_map(args.episode_map),
        reservoir_size=args.reservoir_size,
        chunk_size=args.chunk_size,
        seed=args.seed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(stats, indent=2) + "\n")
    temporary.replace(output)
    print(f"Wrote {output} ({stats['states']['count']} frames, {len(args.repo_id)} repo arguments)")


if __name__ == "__main__":
    main()
