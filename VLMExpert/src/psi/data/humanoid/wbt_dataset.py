"""Native reader for the WBT LeRobot-v3 exports.

The WBT exports used by PSI0 are close to LeRobot v3, but they contain
``tasks.parquet`` (rather than ``tasks.jsonl``), no episode jsonl, and split
parquet/video files independently.  This reader handles that layout directly
and keeps all sampling episode-local.
"""

from __future__ import annotations

import bisect
import json
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is a project dependency
    torch = None  # type: ignore

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - reported when the dataset is opened
    pq = None  # type: ignore

from .wbt_hand_protocol import (
    NATIVE_DIM,
    WBT_HAND_PROTOCOL,
    WBT_NATIVE_PROTOCOL,
    canonicalize_hand,
    infer_repo_hand_layout,
    protocol_metadata,
    validate_robot_vector,
)


VECTOR_KEYS = (
    "observation.state.robot_q_current",
    "observation.state.hand_state",
    "action.robot_q_desired",
    "action.hand_cmd",
)
SCALAR_KEYS = ("timestamp", "frame_index", "episode_index", "index", "task_index")


@dataclass(frozen=True)
class _Episode:
    repo_id: str
    root: Path
    episode_index: int
    task_index: int
    length: int
    data_path: Path
    row_group: int
    row_start: int
    row_end: int
    global_index_start: int
    instruction: str


@dataclass(frozen=True)
class _VideoSegment:
    path: Path
    start: int
    count: int


@dataclass
class _Repo:
    repo_id: str
    root: Path
    info: dict[str, Any]
    tasks: dict[int, str]
    episodes: list[_Episode]
    state_hand_layout_id: str
    action_hand_layout_id: str
    protocol: dict[str, Any]
    camera_key: str
    video_segments: list[_VideoSegment] | None = None


_DatasetBase = torch.utils.data.Dataset if torch is not None else object


class WBTDataset(_DatasetBase):
    """Map-style WBT dataset with episode-safe temporal sampling.

    Parameters mirror the small interface used by ``HERawDataset`` while also
    accepting multiple LeRobot-style repository directories.  ``repo_ids`` may
    be omitted when ``data_root`` itself is a repository.
    """

    def __init__(
        self,
        data_root: str | Path,
        repo_ids: Sequence[str] | None = None,
        *,
        split: str = "train",
        episodes: Sequence[int] | Mapping[str, Sequence[int]] | None = None,
        num_past_frames: int = 0,
        action_chunk_size: int = 30,
        camera_key: str = "observation.images.head_stereo_left",
        hand_protocol: str = WBT_HAND_PROTOCOL,
        hand_layout_id: str | None = None,
        state_hand_layout_id: str | None = None,
        action_hand_layout_id: str | None = None,
        repo_hand_layouts: Mapping[str, str | Mapping[str, str]] | None = None,
        preserve_raw_hand: bool = True,
        use_delta_actions: bool = False,
        expected_fps: float = 30.0,
        validate_data: bool = True,
        quaternion_tolerance: float = 5e-2,
        video_backend: str | None = None,
        frame_cache_size: int = 32,
    ) -> None:
        if hand_protocol != WBT_HAND_PROTOCOL:
            raise ValueError(
                f"WBTDataset only supports {WBT_HAND_PROTOCOL!r}, got {hand_protocol!r}"
            )
        if use_delta_actions:
            raise ValueError(
                "WBT-native-v1 trains absolute robot_q_desired targets; "
                "use_delta_actions must be false"
            )
        if num_past_frames < 0:
            raise ValueError("num_past_frames must be non-negative")
        if action_chunk_size <= 0:
            raise ValueError("action_chunk_size must be positive")
        if not np.isfinite(expected_fps) or expected_fps <= 0:
            raise ValueError("expected_fps must be finite and positive")
        if not 0.0 < quaternion_tolerance < 1.0:
            raise ValueError("quaternion_tolerance must be in (0, 1)")
        if video_backend not in {None, "av", "opencv"}:
            raise ValueError("video_backend must be None, 'av', or 'opencv'")
        if split not in {"train", "val", "validation", "eval"}:
            raise ValueError(f"Unknown WBT split {split!r}")
        if frame_cache_size < 0:
            raise ValueError("frame_cache_size must be non-negative")

        self.data_root = Path(data_root).expanduser()
        self.split = split
        self.num_past_frames = int(num_past_frames)
        self.action_chunk_size = int(action_chunk_size)
        self.camera_key = self._normalize_camera_key(camera_key)
        self.hand_protocol = hand_protocol
        self.hand_layout_id = hand_layout_id
        self.state_hand_layout_id = state_hand_layout_id
        self.action_hand_layout_id = action_hand_layout_id
        self.repo_hand_layouts = dict(repo_hand_layouts or {})
        self.preserve_raw_hand = bool(preserve_raw_hand)
        self.expected_fps = float(expected_fps)
        self.validate_data = bool(validate_data)
        self.quaternion_tolerance = float(quaternion_tolerance)
        self.video_backend = video_backend
        self._frame_cache: OrderedDict[tuple[str, int], np.ndarray] = OrderedDict()
        self._frame_cache_size = int(frame_cache_size)
        self._parquet_cache: OrderedDict[tuple[str, int], dict[str, np.ndarray]] = OrderedDict()
        self._parquet_cache_size = 8
        self._parquet_files: dict[str, Any] = {}

        if pq is None:
            raise ImportError(
                "WBTDataset requires pyarrow; install the project dependencies before opening WBT data"
            )

        resolved_repos = self._resolve_repo_roots(self.data_root, repo_ids)
        if not resolved_repos:
            raise ValueError(f"No WBT repositories found under {self.data_root}")

        self.repos: list[_Repo] = []
        for repo_id, root in resolved_repos:
            selected = self._episodes_for_repo(episodes, repo_id, root)
            self.repos.append(
                self._load_repo(
                    repo_id,
                    root,
                    selected_episodes=selected,
                )
            )

        self._samples: list[tuple[int, int]] = []
        self._cumulative_lengths: list[int] = []
        total = 0
        for repo_idx, repo in enumerate(self.repos):
            for episode_pos, episode in enumerate(repo.episodes):
                self._samples.append((repo_idx, episode_pos))
                total += episode.length
                self._cumulative_lengths.append(total)
        if total <= 0:
            raise ValueError("WBT dataset contains no selected frames")

        # A compact, serializable manifest is useful for checkpoint metadata and
        # makes the source layout auditable after a training run.
        self.dataset_statistics = {
            "protocol": WBT_NATIVE_PROTOCOL,
            "version": WBT_NATIVE_PROTOCOL,
            "state_dim": NATIVE_DIM,
            "action_dim": NATIVE_DIM,
            "hand_protocol": WBT_HAND_PROTOCOL,
            "expected_fps": self.expected_fps,
            "split": split,
            "protocol_metadata": {
                repo.repo_id: repo.protocol for repo in self.repos
            },
            "hand_layout_ids": {
                repo.repo_id: {
                    "state": repo.state_hand_layout_id,
                    "action": repo.action_hand_layout_id,
                }
                for repo in self.repos
            },
            "repos": [
                {
                    "repo_id": repo.repo_id,
                    "root": str(repo.root),
                    "num_episodes": len(repo.episodes),
                    "num_frames": sum(ep.length for ep in repo.episodes),
                    "episode_ids": [ep.episode_index for ep in repo.episodes],
                    "fps": float(repo.info["fps"]),
                    "state_hand_layout_id": repo.state_hand_layout_id,
                    "action_hand_layout_id": repo.action_hand_layout_id,
                    "camera_key": repo.camera_key,
                }
                for repo in self.repos
            ],
        }

    @staticmethod
    def _normalize_camera_key(camera_key: str) -> str:
        key = str(camera_key)
        if key in {"head_stereo_left", "observation.images.head_stereo_left"}:
            return "observation.images.head_stereo_left"
        raise ValueError(
            "WBT-native-v1 uses only observation.images.head_stereo_left; "
            f"got {camera_key!r}"
        )

    @staticmethod
    def _resolve_repo_roots(
        data_root: Path,
        repo_ids: Sequence[str] | None,
    ) -> list[tuple[str, Path]]:
        if isinstance(repo_ids, (str, Path)):
            repo_ids = [str(repo_ids)]

        def find_info(root: Path) -> Path | None:
            candidate = root / "meta" / "info.json"
            return candidate if candidate.is_file() else None

        direct = find_info(data_root)
        if direct is not None:
            if repo_ids is None:
                return [(data_root.name or ".", data_root)]
            requested = [str(value) for value in repo_ids]
            if not requested:
                return [(data_root.name or ".", data_root)]
            if all(value in {"", ".", data_root.name} for value in requested):
                if len(requested) != 1:
                    raise ValueError(f"Duplicate WBT repository selection: {requested}")
                return [(requested[0] or data_root.name or ".", data_root)]
            raise ValueError(
                f"data_root {data_root} is already a repository; cannot resolve "
                f"other repo_ids {requested}"
            )

        if repo_ids is None:
            candidates = []
            if data_root.is_dir():
                for info_path in sorted(
                    data_root.glob("**/meta/info.json"), key=WBTDataset._natural_path_key
                ):
                    candidate = info_path.parent.parent
                    # Auto-discovery intentionally omits the documented 2D
                    # Dex1 exports; explicit selections still fail loudly in
                    # _validate_info so an accidental mismatch is visible.
                    try:
                        metadata = json.loads(info_path.read_text())
                        hand_shape = metadata["features"]["observation.state.hand_state"]["shape"]
                    except (OSError, ValueError, KeyError, TypeError):
                        continue
                    if hand_shape != [12]:
                        continue
                    if candidate not in candidates:
                        candidates.append(candidate)
            candidates.sort(key=lambda p: p.name.lower())
            resolved: list[tuple[str, Path]] = []
            for candidate in candidates:
                try:
                    relative = candidate.relative_to(data_root)
                except ValueError:
                    relative = Path(candidate.name)
                # Preserve the outer task id for nested downloads (the outer
                # name carries the Inspire/BrainCo hand-layout convention).
                repo_id = relative.parts[0] if len(relative.parts) > 1 else candidate.name
                resolved.append((repo_id, candidate))
            seen_ids: set[str] = set()
            for repo_id, _ in resolved:
                if repo_id in seen_ids:
                    raise ValueError(f"Multiple repositories resolve to the same id {repo_id!r}")
                seen_ids.add(repo_id)
            return resolved

        result: list[tuple[str, Path]] = []
        seen_paths: set[Path] = set()
        for repo_id in repo_ids:
            repo_text = str(repo_id)
            root = data_root if repo_text in {"", ".", data_root.name} else data_root / repo_text
            info = find_info(root)
            if info is None:
                # A few downloaded WBT repositories have an outer task folder
                # around the actual LeRobot repo. Resolve only an unambiguous
                # nested metadata directory; never silently choose among many.
                nested = sorted(root.glob("**/meta/info.json")) if root.exists() else []
                if len(nested) == 1:
                    root = nested[0].parent.parent
                else:
                    raise FileNotFoundError(
                        f"WBT repository {repo_id!r} has no unique meta/info.json under {root}"
                    )
            root = root.resolve()
            if root in seen_paths:
                raise ValueError(f"Duplicate WBT repository selection: {repo_text!r}")
            seen_paths.add(root)
            result.append((repo_text or root.name, root))
        return result

    @staticmethod
    def _episodes_for_repo(
        episodes: Sequence[int] | Mapping[str, Sequence[int]] | None,
        repo_id: str,
        root: Path,
    ) -> list[int] | None:
        if episodes is None:
            return None
        if isinstance(episodes, Mapping):
            values = episodes.get(repo_id)
            if values is None:
                values = episodes.get(root.name)
            if values is None:
                raise ValueError(
                    f"Episode selection mapping has no entry for WBT repository {repo_id!r}"
                )
            return [int(v) for v in values]
        return [int(v) for v in episodes]

    def _load_repo(
        self,
        repo_id: str,
        root: Path,
        *,
        selected_episodes: Sequence[int] | None,
    ) -> _Repo:
        info_path = root / "meta" / "info.json"
        try:
            info = json.loads(info_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Unable to read WBT metadata {info_path}") from exc

        camera_key = self._resolve_camera_key(info, self.camera_key, info_path)
        self._validate_info(info, info_path, camera_key, self.expected_fps)
        tasks = self._load_tasks(root / "meta" / "tasks.parquet")
        layout_state, layout_action = self._resolve_layout_ids(repo_id, root, info)
        manifest = protocol_metadata(
            state_hand_layout_id=layout_state,
            action_hand_layout_id=layout_action,
            use_delta_actions=False,
            fps=float(info["fps"]),
        )
        all_episodes = self._scan_episodes(repo_id, root, info, tasks)
        if not all_episodes:
            raise ValueError(f"No episodes found in {root / 'data'}")
        available = {episode.episode_index for episode in all_episodes}
        if selected_episodes is None:
            chosen_ids = sorted(available)
        else:
            chosen_ids = [int(x) for x in selected_episodes]
            if len(set(chosen_ids)) != len(chosen_ids):
                raise ValueError(f"Duplicate episode ids in {repo_id}: {chosen_ids}")
            missing = sorted(set(chosen_ids) - available)
            if missing:
                raise ValueError(f"Unknown episodes for {repo_id}: {missing}")
        by_id = {episode.episode_index: episode for episode in all_episodes}
        episodes = [by_id[idx] for idx in chosen_ids]

        repo = _Repo(
            repo_id=repo_id,
            root=root,
            info=info,
            tasks=tasks,
            episodes=episodes,
            state_hand_layout_id=layout_state,
            action_hand_layout_id=layout_action,
            protocol=manifest,
            camera_key=camera_key,
        )
        # Validate the selected metadata without decoding video frames. Video
        # files are opened lazily in workers, avoiding non-pickleable handles.
        if self.validate_data:
            self._validate_episode_metadata(repo)
        return repo

    @staticmethod
    def _resolve_camera_key(
        info: Mapping[str, Any], requested: str, path: Path
    ) -> str:
        features = info.get("features")
        if not isinstance(features, Mapping):
            raise ValueError(f"WBT metadata has no features map: {path}")
        feature = features.get(requested)
        if isinstance(feature, Mapping) and feature.get("dtype") == "video":
            return requested
        # Some MainCamOnly exports use the generic cam_0/cam_1 names.  cam_0
        # is the documented primary view and is the only implicit alias used.
        if requested == "observation.images.head_stereo_left":
            fallback = "observation.images.cam_0"
            feature = features.get(fallback)
            if isinstance(feature, Mapping) and feature.get("dtype") == "video":
                return fallback
        raise ValueError(
            f"{path}: requested camera {requested!r} is unavailable; "
            "WBT-native training requires a primary RGB camera"
        )

    @staticmethod
    def _validate_info(
        info: Mapping[str, Any],
        path: Path,
        camera_key: str,
        expected_fps: float,
    ) -> None:
        if str(info.get("codebase_version", "")) != "v3.0":
            raise ValueError(
                f"WBT reader expects LeRobot v3.0 metadata in {path}, "
                f"got {info.get('codebase_version')!r}"
            )
        if info.get("robot_type") != "unitree_g1":
            raise ValueError(
                f"WBT-native-v1 requires robot_type='unitree_g1', got {info.get('robot_type')!r}"
            )
        try:
            fps = float(info.get("fps", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid WBT fps in {path}: {info.get('fps')!r}") from exc
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"Invalid WBT fps in {path}: {info.get('fps')!r}")
        if abs(fps - expected_fps) > 1e-6:
            raise ValueError(
                f"WBT-native-v1 expects {expected_fps:g} FPS, but {path} declares {fps:g}"
            )
        features = info.get("features")
        if not isinstance(features, Mapping):
            raise ValueError(f"WBT metadata has no features map: {path}")
        required = {
            "observation.state.robot_q_current": 36,
            "observation.state.hand_state": 12,
            "action.robot_q_desired": 36,
            "action.hand_cmd": 12,
        }
        for key, dim in required.items():
            feature = features.get(key)
            shape = feature.get("shape") if isinstance(feature, Mapping) else None
            if shape != [dim]:
                if key.endswith("hand_state") or key.endswith("hand_cmd"):
                    raise ValueError(
                        f"{path}: {key} has shape {shape}; WBT-native-v1 requires "
                        "12D source hands. 2D Dex1 exports are excluded."
                    )
                raise ValueError(f"{path}: {key} must have shape [{dim}], got {shape}")
            if feature.get("dtype") not in {"float32", "float64"}:
                raise ValueError(f"{path}: {key} must use a floating dtype")
        camera = features.get(camera_key)
        if not isinstance(camera, Mapping) or camera.get("dtype") != "video":
            raise ValueError(f"{path}: missing usable camera feature {camera_key}")
        shape = camera.get("shape")
        if not isinstance(shape, list) or len(shape) != 3 or int(shape[-1]) != 3:
            raise ValueError(f"{path}: camera {camera_key} must have HxWx3 shape, got {shape}")
        video_info = camera.get("info")
        if isinstance(video_info, Mapping) and video_info.get("video.fps") is not None:
            try:
                video_fps = float(video_info["video.fps"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}: invalid camera video.fps") from exc
            if abs(video_fps - float(info["fps"])) > 1e-3:
                raise ValueError(
                    f"{path}: camera {camera_key} fps {video_fps} does not match metadata fps {info['fps']}"
                )

    @staticmethod
    def _load_tasks(path: Path) -> dict[int, str]:
        if not path.is_file():
            raise FileNotFoundError(f"WBT task metadata is required: {path}")
        table = pq.read_table(path)  # type: ignore[union-attr]
        columns = set(table.column_names)
        if "task_index" not in columns:
            raise ValueError(f"WBT tasks.parquet has no task_index column: {path}")
        task_column = "task" if "task" in columns else "__index_level_0__"
        if task_column not in columns:
            raise ValueError(f"WBT tasks.parquet has no task text column: {path}")
        indices = table["task_index"].to_pylist()
        values = table[task_column].to_pylist()
        result: dict[int, str] = {}
        for index, value in zip(indices, values):
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="replace")
            task_index = int(index)
            task = re.sub(r"\s+", " ", str(value)).strip()
            if not task:
                raise ValueError(f"WBT task {task_index} has empty text in {path}")
            if task_index in result:
                raise ValueError(f"Duplicate WBT task_index {task_index} in {path}")
            result[task_index] = task
        return result

    def _resolve_layout_ids(
        self,
        repo_id: str,
        root: Path,
        info: Mapping[str, Any],
    ) -> tuple[str, str]:
        metadata_layout = info.get("hand_layout_id")
        metadata_state = info.get("state_hand_layout_id", metadata_layout)
        metadata_action = info.get("action_hand_layout_id", metadata_layout)

        def mapped(*keys: str) -> str | Mapping[str, str] | None:
            for key in keys:
                value = self.repo_hand_layouts.get(key)
                if value is not None:
                    return value
            return None

        known = mapped(repo_id, root.name)
        if isinstance(known, Mapping):
            known_state = known.get("state")
            known_action = known.get("action")
        else:
            known_state = known_action = known
        inferred = infer_repo_hand_layout(repo_id) or infer_repo_hand_layout(root.name)
        state = self.state_hand_layout_id or known_state or self.hand_layout_id or metadata_state or inferred
        action = self.action_hand_layout_id or known_action or self.hand_layout_id or metadata_action or inferred
        if state is None or action is None:
            raise ValueError(
                f"No registered hand_layout_id for WBT repository {repo_id!r}; "
                "provide state_hand_layout_id/action_hand_layout_id or repo_hand_layouts"
            )
        # protocol_metadata resolves and validates the registry entries.
        return str(state), str(action)

    def _scan_episodes(
        self,
        repo_id: str,
        root: Path,
        info: Mapping[str, Any],
        tasks: Mapping[int, str],
    ) -> list[_Episode]:
        files = sorted((root / "data").glob("**/*.parquet"), key=self._natural_path_key)
        if not files:
            raise FileNotFoundError(f"No parquet data files under {root / 'data'}")
        records: list[_Episode] = []
        cumulative = 0
        fps = float(info["fps"])
        for path in files:
            parquet = self._get_parquet_file(path)
            for row_group in range(parquet.num_row_groups):
                columns = parquet.read_row_group(
                    row_group,
                    columns=["episode_index", "frame_index", "timestamp", "index", "task_index"],
                )
                values = {name: np.asarray(columns[name].to_pylist()) for name in columns.column_names}
                if not values.get("episode_index", np.array([])).size:
                    continue
                episode_values = values["episode_index"].astype(np.int64, copy=False)
                # A row group normally contains one episode. Split contiguous
                # runs as well, so the reader remains correct for other v3
                # writers that pack several episodes in one row group.
                boundaries = np.flatnonzero(episode_values[1:] != episode_values[:-1]) + 1
                starts = np.concatenate(([0], boundaries))
                ends = np.concatenate((boundaries, [len(episode_values)]))
                for row_start, row_end in zip(starts.tolist(), ends.tolist()):
                    episode_index = int(episode_values[row_start])
                    frame_values = values["frame_index"][row_start:row_end].astype(np.int64)
                    index_values = values["index"][row_start:row_end].astype(np.int64)
                    task_values = values["task_index"][row_start:row_end].astype(np.int64)
                    timestamp_values = values["timestamp"][row_start:row_end].astype(np.float64)
                    length = int(row_end - row_start)
                    if length <= 0:
                        continue
                    expected_frames = np.arange(length, dtype=np.int64)
                    if not np.array_equal(frame_values, expected_frames):
                        raise ValueError(
                            f"{path} row group {row_group} episode {episode_index} "
                            "does not have contiguous frame_index values"
                        )
                    if np.any(np.diff(index_values) != 1):
                        raise ValueError(
                            f"{path} row group {row_group} episode {episode_index} "
                            "does not have contiguous global index values"
                        )
                    if int(index_values[0]) != cumulative:
                        raise ValueError(
                            f"{path} row group {row_group} starts at global index "
                            f"{int(index_values[0])}, expected {cumulative}"
                        )
                    if np.any(task_values != task_values[0]):
                        raise ValueError(
                            f"{path} row group {row_group} episode {episode_index} "
                            "contains multiple task_index values"
                        )
                    if not np.isfinite(timestamp_values).all():
                        raise ValueError(f"Non-finite timestamps in {path} row group {row_group}")
                    expected_timestamps = frame_values.astype(np.float64) / fps
                    timestamp_tolerance = max(2e-3, 0.1 / fps)
                    if np.any(np.abs(timestamp_values - expected_timestamps) > timestamp_tolerance):
                        raise ValueError(
                            f"Timestamp/frame_index mismatch in {path} row group {row_group}"
                        )
                    if length > 1:
                        steps = np.diff(timestamp_values)
                        if np.any(np.abs(steps - 1.0 / fps) > timestamp_tolerance):
                            raise ValueError(
                                f"Timestamp/FPS mismatch in {path} row group {row_group} "
                                f"(expected {1.0 / fps:.6f}s frame spacing)"
                            )
                    task_index = int(task_values[0])
                    if task_index not in tasks:
                        raise ValueError(
                            f"Episode {episode_index} in {repo_id} references unknown "
                            f"task_index {task_index}"
                        )
                    instruction = tasks[task_index]
                    records.append(
                        _Episode(
                            repo_id=repo_id,
                            root=root,
                            episode_index=episode_index,
                            task_index=task_index,
                            length=length,
                            data_path=path,
                            row_group=row_group,
                            row_start=int(row_start),
                            row_end=int(row_end),
                            global_index_start=int(index_values[0]),
                            instruction=instruction,
                        )
                    )
                    cumulative += length

        records.sort(key=lambda episode: episode.episode_index)
        ids = [episode.episode_index for episode in records]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate episode_index values in {repo_id}: {ids}")
        if ids != list(range(len(ids))):
            raise ValueError(
                f"WBT episode_index values for {repo_id} must be contiguous from zero"
            )
        expected_total = info.get("total_episodes")
        if expected_total is not None and len(records) != int(expected_total):
            raise ValueError(
                f"{repo_id} metadata declares {expected_total} episodes but found {len(records)}"
            )
        expected_frames = info.get("total_frames")
        if expected_frames is not None and cumulative != int(expected_frames):
            raise ValueError(
                f"{repo_id} metadata declares {expected_frames} frames but found {cumulative}"
            )
        return records

    @staticmethod
    def _natural_path_key(path: Path) -> tuple[Any, ...]:
        return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(path)))

    def _get_parquet_file(self, path: Path):
        key = str(path)
        parquet = self._parquet_files.get(key)
        if parquet is None:
            parquet = pq.ParquetFile(path)  # type: ignore[union-attr]
            self._parquet_files[key] = parquet
        return parquet

    def _validate_episode_metadata(self, repo: _Repo) -> None:
        for episode in repo.episodes:
            rows = self._read_rows(episode, 0, min(episode.length, 1))
            for key in VECTOR_KEYS:
                array = rows[key]
                if not np.isfinite(array).all():
                    raise ValueError(f"Non-finite values in {repo.repo_id} episode {episode.episode_index} {key}")
            validate_robot_vector(
                rows["observation.state.robot_q_current"][0],
                name="observation.state.robot_q_current",
                quaternion_tolerance=self.quaternion_tolerance,
            )
            validate_robot_vector(
                rows["action.robot_q_desired"][0],
                name="action.robot_q_desired",
                quaternion_tolerance=self.quaternion_tolerance,
            )

    def _read_rows(self, episode: _Episode, start: int, end: int) -> dict[str, np.ndarray]:
        if start < 0 or end < start or end > episode.length:
            raise IndexError(f"row range [{start}, {end}) outside episode length {episode.length}")
        key = (str(episode.data_path), episode.row_group)
        rows = self._parquet_cache.get(key)
        if rows is None:
            parquet = self._get_parquet_file(episode.data_path)
            table = parquet.read_row_group(
                episode.row_group,
                columns=list(VECTOR_KEYS) + list(SCALAR_KEYS),
            )
            rows = {}
            for name in table.column_names:
                values = table[name].to_pylist()
                if name in VECTOR_KEYS:
                    rows[name] = np.asarray(values, dtype=np.float32)
                elif name == "timestamp":
                    rows[name] = np.asarray(values, dtype=np.float64)
                else:
                    rows[name] = np.asarray(values, dtype=np.int64)
            self._parquet_cache[key] = rows
            self._parquet_cache.move_to_end(key)
            while len(self._parquet_cache) > self._parquet_cache_size:
                self._parquet_cache.popitem(last=False)
        return {name: values[episode.row_start + start : episode.row_start + end] for name, values in rows.items()}

    def _load_video_segments(self, repo: _Repo) -> list[_VideoSegment]:
        if repo.video_segments is not None:
            return repo.video_segments
        video_root = repo.root / "videos" / repo.camera_key
        paths = sorted(video_root.glob("**/*.mp4"), key=self._natural_path_key)
        if not paths:
            raise FileNotFoundError(f"No video files for {repo.camera_key} under {video_root}")
        segments: list[_VideoSegment] = []
        start = 0
        for path in paths:
            count = self._video_frame_count(path)
            if count <= 0:
                raise ValueError(f"Video has no decodable frames: {path}")
            segments.append(_VideoSegment(path=path, start=start, count=count))
            start += count
        expected = int(repo.info.get("total_frames", start))
        if start != expected:
            raise ValueError(
                f"Video frame count for {repo.repo_id} is {start}, metadata total_frames is {expected}"
            )
        repo.video_segments = segments
        return segments

    def _video_frame_count(self, path: Path) -> int:
        if self.video_backend != "opencv":
            try:
                import av

                container = av.open(str(path))
                try:
                    stream = container.streams.video[0]
                    count = int(stream.frames or 0)
                    if count > 0:
                        return count
                    return sum(1 for _ in container.decode(video=0))
                finally:
                    container.close()
            except ImportError:
                if self.video_backend == "av":
                    raise ImportError("video_backend='av' requires PyAV")
        # OpenCV is the fallback for lightweight environments.
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover
            raise ImportError("WBT video decoding requires av or opencv-python") from exc
        cap = cv2.VideoCapture(str(path))
        try:
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()
        if count <= 0:
            raise ValueError(f"Unable to determine frame count for {path}")
        return count

    def _read_image(self, repo: _Repo, global_index: int) -> np.ndarray:
        cache_key = (repo.repo_id, int(global_index))
        if cache_key in self._frame_cache:
            value = self._frame_cache.pop(cache_key)
            self._frame_cache[cache_key] = value
            return value.copy()

        segments = self._load_video_segments(repo)
        starts = [segment.start for segment in segments]
        segment_index = bisect.bisect_right(starts, int(global_index)) - 1
        if segment_index < 0:
            raise IndexError(f"Global video index {global_index} is negative")
        segment = segments[segment_index]
        local_index = int(global_index) - segment.start
        if local_index < 0 or local_index >= segment.count:
            raise IndexError(f"Video index {global_index} outside {segment.path}")
        image = self._decode_video_frame(segment.path, local_index, repo.info["fps"])
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Decoded WBT frame has unexpected shape {image.shape}")
        image = np.asarray(image, dtype=np.uint8)
        if self._frame_cache_size:
            self._frame_cache[cache_key] = image
            while len(self._frame_cache) > self._frame_cache_size:
                self._frame_cache.popitem(last=False)
        return image.copy()

    def _decode_video_frame(self, path: Path, frame_index: int, fps: int) -> np.ndarray:
        if self.video_backend != "opencv":
            try:
                import av

                container = av.open(str(path))
                try:
                    stream = container.streams.video[0]
                    time_base = float(stream.time_base) if stream.time_base is not None else 0.0
                    target_time = frame_index / float(fps)
                    target_pts = int(round(target_time / time_base)) if time_base else frame_index
                    container.seek(max(0, target_pts), stream=stream, backward=True, any_frame=False)
                    best = None
                    best_distance = float("inf")
                    for frame in container.decode(stream):
                        if frame.pts is None or not time_base:
                            continue
                        current_time = float(frame.pts) * time_base
                        distance = abs(current_time - target_time)
                        if distance < best_distance:
                            best = frame
                            best_distance = distance
                        if current_time >= target_time and distance <= 0.75 / float(fps):
                            break
                        if current_time > target_time + 1.5 / float(fps):
                            break
                    if best is not None:
                        return best.to_ndarray(format="rgb24")
                finally:
                    container.close()
            except ImportError:
                if self.video_backend == "av":
                    raise ImportError("video_backend='av' requires PyAV")
            if self.video_backend == "av":
                raise RuntimeError(f"PyAV could not decode frame {frame_index} from {path}")

        try:
            import cv2
        except ImportError as exc:  # pragma: no cover
            raise ImportError("WBT video decoding requires av or opencv-python") from exc
        cap = cv2.VideoCapture(str(path))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
        finally:
            cap.release()
        if not ok or frame is None:
            raise RuntimeError(f"Unable to decode frame {frame_index} from {path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def __len__(self) -> int:
        return self._cumulative_lengths[-1]

    def __getstate__(self) -> dict[str, Any]:
        # ParquetFile instances and decoded frames should be opened afresh in a
        # spawned DataLoader worker.  All indexing metadata remains picklable.
        state = self.__dict__.copy()
        state["_parquet_files"] = {}
        state["_parquet_cache"] = OrderedDict()
        state["_frame_cache"] = OrderedDict()
        return state

    def _locate(self, index: int) -> tuple[_Repo, _Episode, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"WBT sample index {index} out of bounds for {len(self)} samples")
        sample_pos = bisect.bisect_right(self._cumulative_lengths, index)
        previous = self._cumulative_lengths[sample_pos - 1] if sample_pos else 0
        repo_idx, episode_pos = self._samples[sample_pos]
        repo = self.repos[repo_idx]
        episode = repo.episodes[episode_pos]
        return repo, episode, index - previous

    @staticmethod
    def _clamp_offset(offset: int, length: int) -> int:
        return min(max(int(offset), 0), length - 1)

    def __getitem__(self, index: int) -> dict[str, Any]:
        repo, episode, frame_offset = self._locate(int(index))
        obs_offsets = [
            frame_offset - self.num_past_frames + step
            for step in range(self.num_past_frames + 1)
        ]
        action_offsets = [frame_offset + step for step in range(self.action_chunk_size)]
        obs_mask = np.asarray(
            [float(0 <= offset < episode.length) for offset in obs_offsets], dtype=np.float32
        )
        action_mask = np.asarray(
            [float(0 <= offset < episode.length) for offset in action_offsets], dtype=np.float32
        )
        safe_obs = [self._clamp_offset(offset, episode.length) for offset in obs_offsets]
        safe_actions = [self._clamp_offset(offset, episode.length) for offset in action_offsets]

        obs_rows = [self._read_rows(episode, offset, offset + 1) for offset in safe_obs]
        action_rows = [self._read_rows(episode, offset, offset + 1) for offset in safe_actions]
        current_robot = np.concatenate(
            [row["observation.state.robot_q_current"] for row in obs_rows], axis=0
        ).astype(np.float32)
        current_hand = np.concatenate(
            [row["observation.state.hand_state"] for row in obs_rows], axis=0
        ).astype(np.float32)
        desired_robot = np.concatenate(
            [row["action.robot_q_desired"] for row in action_rows], axis=0
        ).astype(np.float32)
        desired_hand = np.concatenate(
            [row["action.hand_cmd"] for row in action_rows], axis=0
        ).astype(np.float32)
        if self.validate_data:
            validate_robot_vector(
                current_robot,
                name="observation.state.robot_q_current",
                quaternion_tolerance=self.quaternion_tolerance,
            )
            validate_robot_vector(
                desired_robot,
                name="action.robot_q_desired",
                quaternion_tolerance=self.quaternion_tolerance,
            )
            for name, value in (
                ("observation.state.hand_state", current_hand),
                ("action.hand_cmd", desired_hand),
            ):
                if value.shape[-1] != 12 or not np.isfinite(value).all():
                    raise ValueError(f"Invalid WBT hand values in {name}: {value.shape}")
            # Validate units/ranges through the registered source adapter while
            # leaving the raw values untouched in the returned sample.
            canonicalize_hand(
                current_hand,
                repo.state_hand_layout_id,
                validate_range=True,
            )
            canonicalize_hand(
                desired_hand,
                repo.action_hand_layout_id,
                validate_range=True,
            )

        image_indices = [episode.global_index_start + offset for offset in safe_obs]
        images = np.stack([self._read_image(repo, image_index) for image_index in image_indices], axis=0)

        output: dict[str, Any] = {
            repo.camera_key: images,
            "observation.state.robot_q_current": current_robot,
            "observation.state.hand_state": current_hand,
            "action.robot_q_desired": desired_robot,
            "action.hand_cmd": desired_hand,
            "timestamp": np.asarray(
                [row["timestamp"][0] for row in obs_rows], dtype=np.float32
            ),
            "action_timestamp": np.asarray(
                [row["timestamp"][0] for row in action_rows], dtype=np.float32
            ),
            "episode_index": np.int64(episode.episode_index),
            "frame_index": np.int64(frame_offset),
            "index": np.int64(episode.global_index_start + frame_offset),
            "task_index": np.int64(episode.task_index),
            "task": episode.instruction.lower(),
            "dataset_name": "wbt-native",
            "repo_id": repo.repo_id,
            "obs_mask": obs_mask,
            "action_mask": action_mask,
            "state_hand_layout_id": repo.state_hand_layout_id,
            "action_hand_layout_id": repo.action_hand_layout_id,
            "hand_protocol": self.hand_protocol,
            "protocol": WBT_NATIVE_PROTOCOL,
            "protocol_metadata": repo.protocol,
        }
        if self.preserve_raw_hand:
            output["raw_observation_hand"] = current_hand.copy()
            output["raw_action_hand"] = desired_hand.copy()
        # The model-side repack transform expects this canonical camera alias.
        output["observation.images.head_stereo_left"] = output[repo.camera_key]
        return output

    def iter_protocol_vectors(
        self,
        *,
        chunk_size: int = 8192,
    ) -> Iterator[tuple[np.ndarray, np.ndarray, dict[str, Any]]]:
        """Yield canonical state/action chunks without decoding video.

        This is used by the statistics generator and is also useful for cheap
        data audits.  Chunks never cross an episode boundary.
        """

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        for repo in self.repos:
            for episode in repo.episodes:
                for start in range(0, episode.length, chunk_size):
                    end = min(episode.length, start + chunk_size)
                    rows = self._read_rows(episode, start, end)
                    state_robot = validate_robot_vector(
                        rows["observation.state.robot_q_current"],
                        name="observation.state.robot_q_current",
                        quaternion_tolerance=self.quaternion_tolerance,
                    )
                    action_robot = validate_robot_vector(
                        rows["action.robot_q_desired"],
                        name="action.robot_q_desired",
                        quaternion_tolerance=self.quaternion_tolerance,
                    )
                    state_hand, _ = canonicalize_hand(
                        rows["observation.state.hand_state"],
                        repo.state_hand_layout_id,
                        validate_range=True,
                    )
                    action_hand, _ = canonicalize_hand(
                        rows["action.hand_cmd"],
                        repo.action_hand_layout_id,
                        validate_range=True,
                    )
                    states = np.concatenate([state_robot, state_hand], axis=-1).astype(
                        np.float32, copy=False
                    )
                    actions = np.concatenate([action_robot, action_hand], axis=-1).astype(
                        np.float32, copy=False
                    )
                    if states.shape[-1] != NATIVE_DIM or actions.shape[-1] != NATIVE_DIM:
                        raise ValueError("WBT canonicalization did not produce 48D vectors")
                    if not np.isfinite(states).all() or not np.isfinite(actions).all():
                        raise ValueError("WBT protocol vectors contain NaN or Inf")
                    yield states, actions, {
                        "repo_id": repo.repo_id,
                        "episode_index": episode.episode_index,
                        "start": start,
                        "end": end,
                        "state_hand_layout_id": repo.state_hand_layout_id,
                        "action_hand_layout_id": repo.action_hand_layout_id,
                        "protocol_metadata": repo.protocol,
                    }


__all__ = ["WBTDataset"]
