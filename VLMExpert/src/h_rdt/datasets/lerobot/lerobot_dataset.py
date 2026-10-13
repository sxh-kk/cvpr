import hashlib
import json
import os
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq
import torch

try:
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset_mod
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset as NativeLeRobotDataset
except Exception:  # pragma: no cover - runtime dependency
    lerobot_dataset_mod = None
    NativeLeRobotDataset = None


def _patch_native_lerobot_dataset() -> None:
    if NativeLeRobotDataset is None or getattr(NativeLeRobotDataset, "_hrdt_hf_column_patch", False):
        return

    def __init__(
        self,
        repo_id,
        root=None,
        episodes=None,
        image_transforms=None,
        delta_timestamps=None,
        tolerance_s=1e-4,
        revision=None,
        force_cache_sync=False,
        download_videos=True,
        video_backend=None,
    ):
        self.repo_id = repo_id
        self.root = Path(root) if root else lerobot_dataset_mod.HF_LEROBOT_HOME / repo_id
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.episodes = episodes
        self.tolerance_s = tolerance_s
        self.revision = revision if revision else lerobot_dataset_mod.CODEBASE_VERSION
        self.video_backend = video_backend if video_backend else lerobot_dataset_mod.get_safe_default_codec()
        self.delta_indices = None

        self.image_writer = None
        self.episode_buffer = None

        self.root.mkdir(exist_ok=True, parents=True)

        self.meta = lerobot_dataset_mod.LeRobotDatasetMetadata(
            self.repo_id, self.root, self.revision, force_cache_sync=force_cache_sync
        )
        if self.episodes is not None and self.meta._version >= lerobot_dataset_mod.packaging.version.parse("v2.1"):
            episodes_stats = [self.meta.episodes_stats[ep_idx] for ep_idx in self.episodes]
            self.stats = lerobot_dataset_mod.aggregate_stats(episodes_stats)

        try:
            if force_cache_sync:
                raise FileNotFoundError
            assert all((self.root / fpath).is_file() for fpath in self.get_episodes_file_paths())
            self.hf_dataset = self.load_hf_dataset()
        except (AssertionError, FileNotFoundError, NotADirectoryError):
            self.revision = lerobot_dataset_mod.get_safe_version(self.repo_id, self.revision)
            self.download_episodes(download_videos)
            self.hf_dataset = self.load_hf_dataset()

        self.episode_data_index = lerobot_dataset_mod.get_episode_data_index(self.meta.episodes, self.episodes)

        timestamps = np.asarray(self.hf_dataset["timestamp"], dtype=np.float32)
        episode_indices = np.asarray(self.hf_dataset["episode_index"], dtype=np.int64)
        ep_data_index_np = {
            k: t.numpy() if hasattr(t, "numpy") else np.asarray(t) for k, t in self.episode_data_index.items()
        }
        lerobot_dataset_mod.check_timestamps_sync(
            timestamps, episode_indices, ep_data_index_np, self.fps, self.tolerance_s
        )

        if self.delta_timestamps is not None:
            lerobot_dataset_mod.check_delta_timestamps(self.delta_timestamps, self.fps, self.tolerance_s)
            self.delta_indices = lerobot_dataset_mod.get_delta_indices(self.delta_timestamps, self.fps)

    NativeLeRobotDataset.__init__ = __init__
    NativeLeRobotDataset._hrdt_hf_column_patch = True


_patch_native_lerobot_dataset()


@dataclass
class EpisodeMeta:
    episode_index: int
    length: int
    instruction: str


class LeRobotDataset:
    DEFAULT_INSTRUCTION = (
        "Pick a dumpling toy and turn right, then walk towards the chair "
        "and place the toy on the chair."
    )

    def __init__(
        self,
        data_root: Optional[str] = None,
        config: Optional[Dict] = None,
        upsample_rate: int = 1,
        val: bool = False,
        use_precomp_lang_embed: bool = True,
        stat_path: Optional[str] = None,
        val_ratio: Optional[float] = None,
    ):
        if NativeLeRobotDataset is None:
            raise RuntimeError("LeRobot is not installed. Run `uv sync` from the nix dev shell first.")

        self.DATASET_NAME = "lerobot"
        self.data_root = Path(
            data_root or os.environ.get("LEROBOT_DATA_ROOT", "")
        ).expanduser()
        if not self.data_root.exists():
            raise FileNotFoundError(
                "LeRobot data root not found. Set LEROBOT_DATA_ROOT or pass --dataset_root."
            )

        self._native_dataset = NativeLeRobotDataset(
            repo_id=self.data_root.name,
            root=self.data_root,
            download_videos=False,
            video_backend=os.environ.get("LEROBOT_VIDEO_BACKEND"),
        )

        self.config = config or {}
        self.upsample_rate = max(int(upsample_rate or 1), 1)
        self.val = val
        self.use_precomp_lang_embed = use_precomp_lang_embed
        self.chunk_size = int(self.config.get("common", {}).get("action_chunk_size", 16))
        self.img_history_size = int(self.config.get("common", {}).get("img_history_size", 1))
        self.state_dim = self.config.get("common", {}).get("state_dim")
        self.action_dim = self.config.get("common", {}).get("action_dim")
        self.val_ratio = (
            float(os.environ.get("LEROBOT_VAL_RATIO", "0.1"))
            if val_ratio is None
            else float(val_ratio)
        )

        meta_dir = self.data_root / "meta"
        info_path = meta_dir / "info.json"
        episodes_path = meta_dir / "episodes.jsonl"
        if not info_path.exists() or not episodes_path.exists():
            raise FileNotFoundError("LeRobot metadata is incomplete.")

        with info_path.open("r") as f:
            info = json.load(f)
        self.chunks_size = int(info["chunks_size"])
        self.data_path_tpl = info["data_path"]
        self.video_path_tpl = info["video_path"]

        self.modality = self._load_modality(meta_dir / "modality.json")
        self.state_slices = self._get_modality_slices("state", expected_key="states")
        self.action_slices = self._get_modality_slices("action", expected_key="action")

        self.camera_key = "observation.images.egocentric"
        camera_keys = getattr(self._native_dataset.meta, "camera_keys", [])
        if self.camera_key not in camera_keys:
            raise RuntimeError(
                f"Camera key '{self.camera_key}' not found. Available keys: {camera_keys}"
            )

        self.episodes = self._load_episodes(episodes_path)
        if not self.episodes:
            raise RuntimeError("No episodes available after filtering.")

        self.action_min, self.action_max = self._load_action_stats(stat_path)
        self.lang_map_path = meta_dir / "lang_map.json"
        self.lang_embed_dir = meta_dir / "lang_embeddings"
        self.lang_map = {}
        if self.lang_map_path.exists():
            with self.lang_map_path.open("r") as f:
                self.lang_map = json.load(f)

        self._action_dim_from_data = self._infer_action_dim()
        if self.action_dim is None:
            self.action_dim = self._action_dim_from_data
        if self.state_dim is None:
            state_list = self._load_episode_states(self.episodes[0].episode_index)
            if state_list:
                self.state_dim = int(state_list[0].shape[0])

    def _load_modality(self, path: Path) -> Optional[Dict]:
        if not path.exists():
            return None
        with path.open("r") as f:
            return json.load(f)

    def _get_modality_slices(
        self, section: str, expected_key: Optional[str] = None
    ) -> Optional[List[Tuple[int, int]]]:
        if not self.modality:
            return None
        entries = self.modality.get(section, {})
        if not entries:
            return None
        if expected_key is not None:
            original_keys = {meta.get("original_key") for meta in entries.values()}
            if original_keys != {expected_key}:
                return None
        return [(int(meta["start"]), int(meta["end"])) for meta in entries.values()]

    def _reorder_by_slices(
        self, vec: np.ndarray, slices: Optional[List[Tuple[int, int]]]
    ) -> np.ndarray:
        if slices is None:
            return vec
        parts = [vec[start:end] for start, end in slices]
        return np.concatenate(parts, axis=0) if parts else vec

    def _load_episodes(self, episodes_path: Path) -> List[EpisodeMeta]:
        epsd_len_low = int(self.config.get("dataset", {}).get("epsd_len_thresh_low", 1))
        epsd_len_high = int(self.config.get("dataset", {}).get("epsd_len_thresh_high", 10**9))
        episodes: List[EpisodeMeta] = []
        with episodes_path.open("r") as f:
            for line in f:
                row = json.loads(line)
                length = min(int(row.get("length", 0)), epsd_len_high)
                if length < epsd_len_low:
                    continue
                instruction = str(row.get("instruction", "") or "").strip()
                if not instruction:
                    instruction = self.DEFAULT_INSTRUCTION
                episodes.append(
                    EpisodeMeta(
                        episode_index=int(row["episode_index"]),
                        length=length,
                        instruction=instruction,
                    )
                )

        if self.val_ratio <= 0 or self.val_ratio >= 1:
            return episodes

        split_mod = max(int(round(1.0 / self.val_ratio)), 1)
        if self.val:
            return [ep for ep in episodes if ep.episode_index % split_mod == 0]
        return [ep for ep in episodes if ep.episode_index % split_mod != 0]

    def _load_action_stats(
        self, stat_path: Optional[str]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        candidate = Path(stat_path).expanduser() if stat_path else None
        if candidate is None:
            env_path = os.environ.get("LEROBOT_STAT_PATH")
            candidate = Path(env_path).expanduser() if env_path else None

        if candidate is not None and candidate.exists():
            with candidate.open("r") as f:
                stats = json.load(f)
            if self.DATASET_NAME in stats:
                stats = stats[self.DATASET_NAME]
            return (
                np.asarray(stats["min"], dtype=np.float32),
                np.asarray(stats["max"], dtype=np.float32),
            )

        stats_path = self.data_root / "meta" / "episodes_stats.jsonl"
        if not stats_path.exists():
            return None, None

        min_val = None
        max_val = None
        with stats_path.open("r") as f:
            for line in f:
                row = json.loads(line)
                action_stats = row.get("stats", {}).get("action", {})
                if "min" not in action_stats or "max" not in action_stats:
                    continue
                ep_min = np.asarray(action_stats["min"], dtype=np.float32)
                ep_max = np.asarray(action_stats["max"], dtype=np.float32)
                min_val = ep_min if min_val is None else np.minimum(min_val, ep_min)
                max_val = ep_max if max_val is None else np.maximum(max_val, ep_max)
        return min_val, max_val

    def _data_path(self, episode_index: int) -> Path:
        chunk = episode_index // self.chunks_size
        return self.data_root / self.data_path_tpl.format(
            episode_chunk=chunk, episode_index=episode_index
        )

    @lru_cache(maxsize=64)
    def _load_episode_actions(self, episode_index: int) -> np.ndarray:
        table = pq.read_table(self._data_path(episode_index), columns=["action"])
        actions = np.asarray(table.column("action").to_pylist(), dtype=np.float32)
        if self.action_slices:
            actions = np.stack(
                [self._reorder_by_slices(np.asarray(action, dtype=np.float32), self.action_slices) for action in actions],
                axis=0,
            )
        return actions

    @lru_cache(maxsize=64)
    def _load_episode_states(self, episode_index: int) -> List[np.ndarray]:
        parquet_path = self._data_path(episode_index)
        schema = pq.read_schema(parquet_path)
        if "states" in schema.names:
            table = pq.read_table(parquet_path, columns=["states"])
            states = []
            for state in table.column("states").to_pylist():
                vec = np.asarray(state or [], dtype=np.float32)
                states.append(self._reorder_by_slices(vec, self.state_slices))
            return states

        table = pq.read_table(
            parquet_path,
            columns=[
                "observation.arm_joints",
                "observation.hand_joints",
                "observation.prev_height",
                "observation.prev_rpy",
                "observation.prev_vx",
                "observation.prev_vy",
                "observation.prev_vyaw",
                "observation.prev_dyaw",
            ],
        )
        states = []
        for arm, hand, height, rpy, vx, vy, vyaw, dyaw in zip(
            table.column("observation.arm_joints").to_pylist(),
            table.column("observation.hand_joints").to_pylist(),
            table.column("observation.prev_height").to_pylist(),
            table.column("observation.prev_rpy").to_pylist(),
            table.column("observation.prev_vx").to_pylist(),
            table.column("observation.prev_vy").to_pylist(),
            table.column("observation.prev_vyaw").to_pylist(),
            table.column("observation.prev_dyaw").to_pylist(),
        ):
            state_vec = np.concatenate(
                [
                    np.asarray(arm or [], dtype=np.float32),
                    np.asarray(hand or [], dtype=np.float32),
                    np.asarray([height], dtype=np.float32),
                    np.asarray(rpy or [0.0, 0.0, 0.0], dtype=np.float32),
                    np.asarray([vx], dtype=np.float32),
                    np.asarray([vy], dtype=np.float32),
                    np.asarray([vyaw], dtype=np.float32),
                    np.asarray([dyaw], dtype=np.float32),
                ],
                axis=0,
            )
            states.append(self._reorder_by_slices(state_vec, self.state_slices))
        return states

    def _infer_action_dim(self) -> int:
        sample_episode = self.episodes[0]
        return int(self._load_episode_actions(sample_episode.episode_index).shape[-1])

    def _select_action_indices(self, length: int) -> List[int]:
        max_start = max(length - (self.chunk_size - 1) * self.upsample_rate - 1, 0)
        start_idx = random.randint(0, max_start) if max_start > 0 else 0
        return [
            min(start_idx + i * self.upsample_rate, length - 1)
            for i in range(self.chunk_size)
        ]

    def _select_history_indices(self, frame_idx: int) -> List[int]:
        start = max(frame_idx - (self.img_history_size - 1) * self.upsample_rate, 0)
        indices = list(range(start, frame_idx + 1, self.upsample_rate))
        if not indices:
            indices = [0]
        while len(indices) < self.img_history_size:
            indices.insert(0, indices[0])
        return indices[-self.img_history_size:]

    def _to_uint8_hwc(self, frame) -> Optional[np.ndarray]:
        if frame is None:
            return None
        if torch.is_tensor(frame):
            frame = frame.detach().cpu()
            if frame.ndim == 3 and frame.shape[0] in (1, 3):
                frame = frame.permute(1, 2, 0)
            if frame.is_floating_point():
                frame = (frame * 255).clamp(0, 255)
            return frame.to(torch.uint8).numpy()
        if isinstance(frame, np.ndarray):
            if frame.ndim == 3 and frame.shape[0] in (1, 3) and frame.shape[0] != frame.shape[-1]:
                frame = np.transpose(frame, (1, 2, 0))
            if frame.dtype != np.uint8:
                frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
            return frame
        return None

    def _load_frames(self, episode_index: int, indices: List[int]) -> np.ndarray:
        ep_start = self._native_dataset.episode_data_index["from"][episode_index].item()
        ep_end = self._native_dataset.episode_data_index["to"][episode_index].item()
        frames = []
        for idx in indices:
            global_idx = min(ep_start + idx, ep_end - 1)
            item = self._native_dataset[global_idx]
            frames.append(self._to_uint8_hwc(item[self.camera_key]))

        valid_frames = [frame for frame in frames if frame is not None]
        if not valid_frames:
            return np.zeros((self.img_history_size, 480, 640, 3), dtype=np.uint8)

        fallback = valid_frames[0]
        return np.stack([frame if frame is not None else fallback for frame in frames], axis=0)

    def _instruction_hash(self, instruction: str) -> str:
        return hashlib.sha1(instruction.encode("utf-8")).hexdigest()

    @lru_cache(maxsize=512)
    def _load_lang_embed(self, instruction: str) -> torch.Tensor:
        if not self.use_precomp_lang_embed:
            raise RuntimeError("Language embeddings are disabled.")

        if self.lang_map:
            embed_name = self.lang_map.get(instruction)
            if embed_name is None:
                raise FileNotFoundError("Missing lang_map entry. Run datasets/lerobot/encode_lang_batch.py.")
            embed_path = self.lang_embed_dir / embed_name
        else:
            embed_path = self.lang_embed_dir / f"{self._instruction_hash(instruction)}.pt"

        if not embed_path.exists():
            raise FileNotFoundError(f"Missing language embedding: {embed_path}")
        return torch.load(embed_path, map_location="cpu")["embeddings"].squeeze(0)

    def _pad_or_trim(self, arr: np.ndarray, target_dim: int) -> np.ndarray:
        cur_dim = arr.shape[-1]
        if cur_dim == target_dim:
            return arr
        if cur_dim > target_dim:
            return arr[..., :target_dim]
        pad_width = [(0, 0)] * arr.ndim
        pad_width[-1] = (0, target_dim - cur_dim)
        return np.pad(arr, pad_width, mode="constant")

    def __len__(self) -> int:
        return len(self.episodes)

    def get_item(self, idx: Optional[int] = None) -> Optional[Dict]:
        if idx is None:
            idx = random.randint(0, len(self.episodes) - 1)

        episode = self.episodes[idx % len(self.episodes)]
        actions = self._load_episode_actions(episode.episode_index)
        length = min(len(actions), episode.length)
        if length <= 0:
            return None

        action_indices = self._select_action_indices(length)
        action_chunk = actions[action_indices]
        state_list = self._load_episode_states(episode.episode_index)
        state = state_list[action_indices[0]] if state_list else np.zeros((0,), dtype=np.float32)

        if self.action_dim is not None:
            action_chunk = self._pad_or_trim(action_chunk, int(self.action_dim))
        if self.state_dim is not None:
            state = self._pad_or_trim(state, int(self.state_dim))

        frame_idx = action_indices[0]
        history_indices = self._select_history_indices(frame_idx)
        image_frames = self._load_frames(episode.episode_index, history_indices)

        result = {
            "states": state.astype(np.float32)[None, :],
            "actions": action_chunk.astype(np.float32),
            "action_norm": np.ones_like(action_chunk, dtype=np.float32),
            "current_images": [image_frames],
            "current_images_mask": [np.ones(self.img_history_size, dtype=bool)],
            "instruction": episode.instruction,
            "dataset_name": self.DATASET_NAME,
            "file_info": {
                "episode_index": episode.episode_index,
                "frame_index": frame_idx,
            },
        }

        if self.use_precomp_lang_embed:
            result["lang_embeds"] = self._load_lang_embed(episode.instruction)

        return result

    def __getitem__(self, idx: int) -> Optional[Dict]:
        return self.get_item(idx)
