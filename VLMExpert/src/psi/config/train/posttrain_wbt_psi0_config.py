"""PSI0 action-expert posttraining configuration for WBT-native data."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import model_validator

from psi.config import transform as pt
from psi.config.config import LaunchConfig
from psi.config.data_wbt import WBTDataConfig
from psi.config.model_psi0 import Psi0ModelConfig
from psi.config.transform import DataTransform
from psi.data.humanoid.wbt_hand_protocol import (
    NATIVE_DIM,
    PROTOCOL_GROUPS,
    PROTOCOL_NAMES,
    WBT_HAND_PROTOCOL,
    WBT_NATIVE_PROTOCOL,
    get_hand_layout,
    infer_repo_hand_layout,
)
from psi.utils import resolve_path


def _load_wbt_stats(stat_path: str) -> tuple[Path, dict[str, Any]]:
    path = Path(resolve_path(stat_path))
    if not path.is_file():
        raise FileNotFoundError(
            f"WBT posttraining requires a statistics artifact, but {path} does not exist"
        )
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read WBT statistics at {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"WBT statistics at {path} must contain a JSON object")
    _validate_wbt_stats_payload(payload, path)
    return path, payload


def _validate_numeric_vector(
    section: Mapping[str, Any], key: str, path: Path
) -> list[float]:
    values = section.get(key)
    if not isinstance(values, list) or len(values) != NATIVE_DIM:
        raise ValueError(
            f"WBT statistics {path} field {key!r} must contain {NATIVE_DIM} values"
        )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in values
    ):
        raise ValueError(f"WBT statistics {path} field {key!r} contains non-finite values")
    return [float(value) for value in values]


def _validate_wbt_stats_payload(stats: Mapping[str, Any], path: Path) -> None:
    if stats.get("protocol") != WBT_NATIVE_PROTOCOL:
        raise ValueError(f"WBT statistics at {path} have the wrong protocol")
    if stats.get("hand_protocol") != WBT_HAND_PROTOCOL:
        raise ValueError(f"WBT statistics at {path} have the wrong hand protocol")
    if stats.get("state_dim") != NATIVE_DIM or stats.get("action_dim") != NATIVE_DIM:
        raise ValueError(f"WBT statistics at {path} must be {NATIVE_DIM}D")
    fps = stats.get("fps")
    if (
        isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not math.isfinite(float(fps))
        or float(fps) <= 0
    ):
        raise ValueError(f"WBT statistics at {path} have an invalid FPS")
    if stats.get("use_delta_actions") is not False or stats.get("absolute_action") is not True:
        raise ValueError(f"WBT statistics at {path} must describe absolute actions")
    if stats.get("dimensions") != list(PROTOCOL_NAMES):
        raise ValueError(f"WBT statistics at {path} have an incompatible dimension order")

    groups = stats.get("groups")
    if not isinstance(groups, Mapping):
        raise ValueError(f"WBT statistics at {path} have no protocol groups")
    for name, expected in PROTOCOL_GROUPS.items():
        actual = groups.get(name)
        if not isinstance(actual, Mapping) or (
            actual.get("start") != expected["start"]
            or actual.get("end") != expected["end"]
            or actual.get("names") != list(expected["names"])
        ):
            raise ValueError(f"WBT statistics at {path} have an invalid {name!r} group")

    counts: list[int] = []
    for section_name in ("states", "action"):
        section = stats.get(section_name)
        if not isinstance(section, Mapping):
            raise ValueError(f"WBT statistics at {path} have no {section_name!r} section")
        vectors = {
            key: _validate_numeric_vector(section, key, path)
            for key in ("min", "max", "q01", "q99", "mean", "std")
        }
        if any(lo > hi for lo, hi in zip(vectors["min"], vectors["max"])):
            raise ValueError(f"WBT statistics at {path} have min values above max values")
        if any(lo > hi for lo, hi in zip(vectors["q01"], vectors["q99"])):
            raise ValueError(f"WBT statistics at {path} have q01 values above q99 values")
        count = section.get("count")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"WBT statistics at {path} have an invalid sample count")
        counts.append(count)
    if counts[0] != counts[1]:
        raise ValueError(f"WBT state/action statistics at {path} use different sample counts")

    source = stats.get("source")
    if not isinstance(source, Mapping) or source.get("split") != "train":
        raise ValueError(f"WBT statistics at {path} must be computed from the train split")
    repo_ids = source.get("repo_ids")
    if (
        not isinstance(repo_ids, list)
        or not repo_ids
        or any(not isinstance(repo_id, str) or not repo_id for repo_id in repo_ids)
        or len(set(repo_ids)) != len(repo_ids)
    ):
        raise ValueError(f"WBT statistics at {path} have invalid source repository ids")
    episodes = source.get("episodes")
    if not isinstance(episodes, Mapping) or set(episodes) != set(repo_ids):
        raise ValueError(f"WBT statistics at {path} must record episodes for every source repository")
    for repo_id in repo_ids:
        values = episodes[repo_id]
        if (
            not isinstance(values, list)
            or not values
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values)
            or len(set(values)) != len(values)
        ):
            raise ValueError(f"WBT statistics at {path} have invalid episodes for {repo_id!r}")
    fps_by_repo = source.get("fps_by_repo")
    if not isinstance(fps_by_repo, Mapping) or set(fps_by_repo) != set(repo_ids):
        raise ValueError(f"WBT statistics at {path} must record FPS per source repository")
    if any(value != fps for value in fps_by_repo.values()):
        raise ValueError(f"WBT statistics at {path} mix incompatible sampling rates")

    layouts = stats.get("hand_layout_ids")
    if not isinstance(layouts, Mapping) or set(layouts) != set(repo_ids):
        raise ValueError(f"WBT statistics at {path} must record layouts for every source repository")
    if source.get("hand_layout_ids") != layouts:
        raise ValueError(f"WBT statistics at {path} contain inconsistent hand layout manifests")
    for repo_id in repo_ids:
        layout = layouts[repo_id]
        if not isinstance(layout, Mapping) or set(layout) != {"state", "action"}:
            raise ValueError(f"WBT statistics at {path} have invalid layouts for {repo_id!r}")
        get_hand_layout(str(layout["state"]))
        get_hand_layout(str(layout["action"]))

    units = stats.get("units")
    if (
        not isinstance(units, Mapping)
        or units.get("root_position") != "meters"
        or units.get("root_quaternion") != "unitless"
        or units.get("body_joints") != "radians"
    ):
        raise ValueError(f"WBT statistics at {path} have invalid root/body units")
    hand_units = units.get("source_hands")
    if not isinstance(hand_units, Mapping) or set(hand_units) != set(repo_ids):
        raise ValueError(f"WBT statistics at {path} must record source-hand units per repository")
    for repo_id in repo_ids:
        declared = hand_units[repo_id]
        layout = layouts[repo_id]
        expected = {
            "state": get_hand_layout(str(layout["state"])).units,
            "action": get_hand_layout(str(layout["action"])).units,
        }
        if declared != expected:
            raise ValueError(f"WBT statistics at {path} have invalid hand units for {repo_id!r}")

    protocol_by_repo = source.get("protocol_metadata")
    if not isinstance(protocol_by_repo, Mapping) or set(protocol_by_repo) != set(repo_ids):
        raise ValueError(f"WBT statistics at {path} must record protocol metadata per repository")
    for repo_id in repo_ids:
        metadata = protocol_by_repo[repo_id]
        if (
            not isinstance(metadata, Mapping)
            or metadata.get("protocol") != WBT_NATIVE_PROTOCOL
            or metadata.get("state_dim") != NATIVE_DIM
            or metadata.get("action_dim") != NATIVE_DIM
            or metadata.get("dimensions") != list(PROTOCOL_NAMES)
            or metadata.get("hand_layout_ids") != layouts[repo_id]
            or not isinstance(metadata.get("sampling"), Mapping)
            or metadata["sampling"].get("fps") != fps
        ):
            raise ValueError(
                f"WBT statistics at {path} have invalid protocol metadata for {repo_id!r}"
            )


def _repo_info(data_root: str, repo_id: str) -> tuple[Path, dict[str, Any]]:
    root = Path(data_root).expanduser()
    candidate = root if repo_id in {"", ".", root.name} else root / repo_id
    info_path = candidate / "meta" / "info.json"
    if not info_path.is_file():
        nested = sorted(candidate.glob("**/meta/info.json")) if candidate.exists() else []
        if len(nested) != 1:
            raise FileNotFoundError(
                f"Cannot validate WBT statistics: {repo_id!r} has no unique meta/info.json under {candidate}"
            )
        info_path = nested[0]
    try:
        info = json.loads(info_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read WBT metadata from {info_path}") from exc
    if not isinstance(info, dict):
        raise ValueError(f"WBT metadata at {info_path} must contain a JSON object")
    return info_path, info


def _repo_episode_ids(data_root: str, repo_id: str) -> list[int]:
    info_path, info = _repo_info(data_root, repo_id)
    try:
        total = int(info["total_episodes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Cannot read total_episodes from {info_path}") from exc
    if total <= 0:
        raise ValueError(f"WBT repository metadata at {info_path} has no episodes")
    return list(range(total))


def _configured_layout_ids(
    data: WBTDataConfig, repo_id: str
) -> tuple[str, str]:
    info_path, info = _repo_info(data.root_dir, repo_id)
    root_name = info_path.parent.parent.name
    known = data.repo_hand_layouts.get(repo_id)
    if known is None:
        known = data.repo_hand_layouts.get(root_name)
    if isinstance(known, Mapping):
        known_state = known.get("state")
        known_action = known.get("action")
    else:
        known_state = known_action = known
    metadata_layout = info.get("hand_layout_id")
    metadata_state = info.get("state_hand_layout_id", metadata_layout)
    metadata_action = info.get("action_hand_layout_id", metadata_layout)
    inferred = infer_repo_hand_layout(repo_id) or infer_repo_hand_layout(root_name)
    state = (
        data.state_hand_layout_id
        or known_state
        or data.hand_layout_id
        or metadata_state
        or inferred
    )
    action = (
        data.action_hand_layout_id
        or known_action
        or data.hand_layout_id
        or metadata_action
        or inferred
    )
    if state is None or action is None:
        raise ValueError(f"Cannot resolve WBT hand layouts for {repo_id!r}")
    get_hand_layout(str(state))
    get_hand_layout(str(action))
    return str(state), str(action)


class WBTActionStateTransform(pt.ActionStateTransform):
    """WBT defaults plus protocol validation for the normalization artifact."""

    stat_path: str = "assets/stats/wbt_native_stats.json"
    action_norm_type: str = "bounds_q99"
    use_norm_mask: bool = False
    normalize_state: bool = True

    @model_validator(mode="after")
    def validate_wbt_stats(self):
        if self.stat_action_key != "action" or self.stat_state_key != "states":
            raise ValueError(
                "WBT normalization requires stat_action_key='action' and "
                "stat_state_key='states'"
            )
        _load_wbt_stats(self.stat_path)
        return self


class DynamicDataTransform(DataTransform):
    repack: pt.WBTNativePosttrainRepackTransform
    field: WBTActionStateTransform
    model: pt.Psi0ModelTransform

    @model_validator(mode="after")
    def validate_wbt_normalization(self):
        if self.field.use_norm_mask and len(self.field.action_norm_masks) != 48:
            raise ValueError(
                "WBT use_norm_mask requires exactly 48 action_norm_masks entries"
            )
        for name, values in (
            ("action_min", self.field.action_min),
            ("action_max", self.field.action_max),
            ("state_min", self.field.state_min if self.field.normalize_state else None),
            ("state_max", self.field.state_max if self.field.normalize_state else None),
        ):
            if values is not None and len(values) != 48:
                raise ValueError(f"WBT {name} must contain exactly 48 values")
        return self


class DynamicDataConfig(WBTDataConfig):
    transform: DynamicDataTransform


class WBTPsi0ModelConfig(Psi0ModelConfig):
    """Defaults matching the fixed WBT-native-v1 48D interface."""

    action_dim: int = 48
    action_chunk_size: int = 30
    action_exec_horizon: int = 30
    observation_horizon: int = 1
    odim: int = 48
    noise_scheduler: str = "flow"
    train_diffusion_steps: int = 1000
    tune_vlm: bool = False

    @model_validator(mode="after")
    def validate_wbt_model(self):
        if self.action_dim != 48 or self.odim != 48:
            raise ValueError("WBT-native-v1 requires model action_dim=48 and odim=48")
        if self.action_chunk_size != self.action_exec_horizon:
            raise ValueError(
                "WBT action_chunk_size and action_exec_horizon must match"
            )
        if self.tune_vlm:
            raise ValueError("WBT posttraining freezes the VLM; tune_vlm must be false")
        if self.pretrained_action_header_path is not None:
            raise ValueError(
                "WBT-native-v1 initializes its 48D action header from scratch; "
                "a pre-existing HE action header cannot be loaded"
            )
        return self


class DynamicLaunchConfig(LaunchConfig):
    data: DynamicDataConfig
    model: WBTPsi0ModelConfig

    @model_validator(mode="after")
    def validate_wbt_pipeline(self):
        repack = self.data.transform.repack
        horizons = {
            self.data.action_chunk_size,
            repack.action_chunk_size,
            self.model.action_chunk_size,
        }
        if len(horizons) != 1:
            raise ValueError(
                "WBT data, repack, and model action_chunk_size values must match"
            )
        if self.model.observation_horizon != repack.num_past_frames + 1:
            raise ValueError(
                "model.observation_horizon must equal repack.num_past_frames + 1"
            )
        if self.data.num_past_frames != repack.num_past_frames:
            raise ValueError("WBT data and repack num_past_frames values must match")
        if self.data.expected_fps != repack.expected_fps:
            raise ValueError("WBT data and repack expected_fps values must match")

        path, stats = _load_wbt_stats(self.data.transform.field.stat_path)
        source = stats["source"]
        stat_repos = source["repo_ids"]
        if set(stat_repos) != set(self.data.train_repo_ids):
            raise ValueError(
                f"WBT statistics {path} repositories do not match the configured train split: "
                f"stats={stat_repos}, configured={self.data.train_repo_ids}"
            )
        stat_episodes = source["episodes"]
        stat_layouts = stats["hand_layout_ids"]
        if stats["fps"] != self.data.expected_fps:
            raise ValueError(
                f"WBT statistics {path} FPS does not match configured expected_fps"
            )
        for repo_id in self.data.train_repo_ids:
            configured = self.data._episode_ids(self.data.train_episodes, repo_id)
            expected = (
                configured
                if configured is not None
                else _repo_episode_ids(self.data.root_dir, repo_id)
            )
            if sorted(stat_episodes[repo_id]) != sorted(expected):
                raise ValueError(
                    f"WBT statistics {path} episodes for {repo_id!r} do not match "
                    f"the configured train split"
                )
            expected_state_layout, expected_action_layout = _configured_layout_ids(
                self.data, repo_id
            )
            if stat_layouts[repo_id] != {
                "state": expected_state_layout,
                "action": expected_action_layout,
            }:
                raise ValueError(
                    f"WBT statistics {path} hand layouts for {repo_id!r} do not "
                    "match the configured source adapters"
                )
            for attr, expected_layout in (
                ("state_hand_layout_id", expected_state_layout),
                ("action_hand_layout_id", expected_action_layout),
            ):
                configured_layout = getattr(repack, attr) or repack.hand_layout_id
                if configured_layout is not None and configured_layout != expected_layout:
                    raise ValueError(
                        f"WBT repack {attr}={configured_layout!r} conflicts with "
                        f"repository {repo_id!r} layout {expected_layout!r}"
                    )
        for repo_id in set(self.data.train_repo_ids) & set(self.data.val_repo_ids):
            val_episodes = self.data._episode_ids(self.data.val_episodes, repo_id)
            if val_episodes is None:  # also enforced by WBTDataConfig
                raise ValueError(
                    f"WBT validation episodes must be explicit for shared repository {repo_id!r}"
                )
            leaked = sorted(set(stat_episodes[repo_id]) & set(val_episodes))
            if leaked:
                raise ValueError(
                    f"WBT statistics {path} include validation episodes for {repo_id!r}: {leaked}"
                )
        return self


__all__ = [
    "DynamicDataTransform",
    "DynamicDataConfig",
    "WBTPsi0ModelConfig",
    "WBTActionStateTransform",
    "DynamicLaunchConfig",
]
