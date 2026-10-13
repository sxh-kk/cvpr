"""Configuration and dataset construction for WBT-native posttraining."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List

from pydantic import Field, model_validator

from psi.config.config import DataConfig
from psi.data.humanoid.wbt_hand_protocol import WBT_HAND_PROTOCOL, get_hand_layout


class WBTDataConfig(DataConfig):
    """Data source settings for the 48D ``wbt-native-v1`` protocol.

    Validation repositories and episode ids are intentionally explicit.  WBT
    exports commonly contain only a ``train`` split in ``info.json``; silently
    reusing the first training repository for validation would leak episodes.
    """

    root_dir: str = "/hfm/data/WBT"
    train_repo_ids: List[str] = Field(default_factory=list)
    val_repo_ids: List[str] = Field(default_factory=list)
    train_episodes: Dict[str, List[int]] | List[int] | None = None
    val_episodes: Dict[str, List[int]] | List[int] | None = None

    camera_key: str = "observation.images.head_stereo_left"
    hand_protocol: str = WBT_HAND_PROTOCOL
    hand_layout_id: str | None = None
    state_hand_layout_id: str | None = None
    action_hand_layout_id: str | None = None
    repo_hand_layouts: Dict[str, str | Dict[str, str]] = Field(default_factory=dict)
    preserve_raw_hand: bool = True

    num_past_frames: int = 0
    action_chunk_size: int = 30
    use_delta_actions: bool = False
    expected_fps: float = 30.0
    validate_data: bool = True
    quaternion_tolerance: float = 5e-2
    video_backend: str | None = None

    @model_validator(mode="after")
    def validate_wbt_config(self):
        # A direct repository path is a useful local-development shorthand, but
        # a directory containing multiple repositories must name them.
        direct_repo = Path(self.root_dir).expanduser() / "meta" / "info.json"
        if not self.train_repo_ids and direct_repo.is_file():
            self.train_repo_ids = ["."]
        if not self.train_repo_ids:
            raise ValueError(
                "WBT train_repo_ids must be provided (or root_dir must point to one repository)"
            )
        if not self.val_repo_ids:
            raise ValueError(
                "WBT val_repo_ids must be provided explicitly; validation cannot reuse a train repository"
            )
        if len(set(self.train_repo_ids)) != len(self.train_repo_ids):
            raise ValueError("WBT train_repo_ids contains duplicates")
        if len(set(self.val_repo_ids)) != len(self.val_repo_ids):
            raise ValueError("WBT val_repo_ids contains duplicates")
        if self.num_past_frames < 0:
            raise ValueError("num_past_frames must be non-negative")
        if self.action_chunk_size <= 0:
            raise ValueError("action_chunk_size must be positive")
        if not math.isfinite(self.expected_fps) or self.expected_fps <= 0:
            raise ValueError("expected_fps must be finite and positive")
        if self.camera_key not in {
            "head_stereo_left",
            "observation.images.head_stereo_left",
        }:
            raise ValueError(
                "WBT-native-v1 requires camera_key=observation.images.head_stereo_left"
            )
        if self.use_delta_actions:
            raise ValueError("WBT-native-v1 uses absolute actions; use_delta_actions must be false")
        if not 0.0 < self.quaternion_tolerance < 1.0:
            raise ValueError("quaternion_tolerance must be in (0, 1)")
        if self.hand_protocol != WBT_HAND_PROTOCOL:
            raise ValueError(
                f"WBTDataConfig requires hand_protocol={WBT_HAND_PROTOCOL!r}, "
                f"got {self.hand_protocol!r}"
            )
        for layout_id in (
            self.hand_layout_id,
            self.state_hand_layout_id,
            self.action_hand_layout_id,
        ):
            if layout_id is not None:
                get_hand_layout(layout_id)
        for repo_id, layout in self.repo_hand_layouts.items():
            if isinstance(layout, str):
                get_hand_layout(layout)
                continue
            unknown_keys = sorted(set(layout) - {"state", "action"})
            if unknown_keys or not {"state", "action"}.issubset(layout):
                raise ValueError(
                    f"repo_hand_layouts[{repo_id!r}] must contain exactly state/action layouts"
                )
            get_hand_layout(layout["state"])
            get_hand_layout(layout["action"])

        for split_name, repos, selection in (
            ("train", self.train_repo_ids, self.train_episodes),
            ("val", self.val_repo_ids, self.val_episodes),
        ):
            for repo_id in repos:
                self._episode_ids(selection, repo_id)
            if isinstance(selection, dict):
                missing = sorted(set(repos) - set(selection))
                unknown = sorted(set(selection) - set(repos))
                if missing or unknown:
                    raise ValueError(
                        f"WBT {split_name}_episodes must cover exactly the configured "
                        f"repositories: missing={missing}, unknown={unknown}"
                    )

        overlap = set(self.train_repo_ids) & set(self.val_repo_ids)
        for repo_id in sorted(overlap):
            train_episodes = self._episode_ids(self.train_episodes, repo_id)
            val_episodes = self._episode_ids(self.val_episodes, repo_id)
            if train_episodes is None or val_episodes is None:
                raise ValueError(
                    f"Repository {repo_id!r} is used for train and val; both splits "
                    "must provide explicit episode ids"
                )
            leaked = sorted(set(train_episodes) & set(val_episodes))
            if leaked:
                raise ValueError(
                    f"WBT train/val episode overlap for {repo_id!r}: {leaked}"
                )
        return self

    @staticmethod
    def _episode_ids(
        selection: Dict[str, List[int]] | List[int] | None,
        repo_id: str,
    ) -> List[int] | None:
        if selection is None:
            return None
        values = selection.get(repo_id) if isinstance(selection, dict) else selection
        if values is None:
            return None
        result = [int(value) for value in values]
        if not result:
            raise ValueError(f"WBT episode ids for {repo_id!r} cannot be empty")
        if any(value < 0 for value in result):
            raise ValueError(f"WBT episode ids for {repo_id!r} must be non-negative")
        if len(set(result)) != len(result):
            raise ValueError(f"WBT episode ids for {repo_id!r} contain duplicates")
        return result

    def _episodes_for_split(self, split: str):
        if split == "train":
            return self.train_episodes
        if split in {"val", "validation", "eval"}:
            return self.val_episodes
        raise ValueError(f"Unknown WBT split {split!r}; expected train or val")

    def __call__(self, split: str = "train", transform_kwargs=None, **kwargs) -> Any:
        from psi.data.dataset import Dataset as MapStyleDataset
        from psi.data.humanoid.wbt_dataset import WBTDataset

        if transform_kwargs is None:
            transform_kwargs = {}
        if split == "train":
            repo_ids = self.train_repo_ids
        elif split in {"val", "validation", "eval"}:
            repo_ids = self.val_repo_ids
        else:
            raise ValueError(f"Unknown WBT split {split!r}; expected train or val")
        if not repo_ids:
            raise ValueError(f"No repositories configured for WBT {split} split")

        dataset = WBTDataset(
            data_root=self.root_dir,
            repo_ids=repo_ids,
            split=split,
            episodes=self._episodes_for_split(split),
            num_past_frames=self.num_past_frames,
            action_chunk_size=self.action_chunk_size,
            camera_key=self.camera_key,
            hand_protocol=self.hand_protocol,
            hand_layout_id=self.hand_layout_id,
            state_hand_layout_id=self.state_hand_layout_id,
            action_hand_layout_id=self.action_hand_layout_id,
            repo_hand_layouts=self.repo_hand_layouts,
            preserve_raw_hand=self.preserve_raw_hand,
            use_delta_actions=self.use_delta_actions,
            expected_fps=self.expected_fps,
            validate_data=self.validate_data,
            quaternion_tolerance=self.quaternion_tolerance,
            video_backend=self.video_backend,
        )
        return MapStyleDataset(self, dataset, transform_kwargs=transform_kwargs, **kwargs)

    def mock(self, split: str = "train", transform_kwargs=None, **kwargs):
        return self(split=split, transform_kwargs=transform_kwargs, **kwargs)[0]


__all__ = ["WBTDataConfig"]
