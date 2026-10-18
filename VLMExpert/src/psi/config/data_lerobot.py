from __future__ import annotations
from pydantic import BaseModel, Field, model_validator
from typing import Any, Optional, Dict, List, TYPE_CHECKING
from psi.config.config import DataConfig
from pathlib import Path
from psi.utils import resolve_data_path
import os
import json
if TYPE_CHECKING:
    from psi.data.dataset import TransformableDataset

from psi.config.transform import ActionStateTransform, Psi0ModelTransform
class LerobotDataConfig(DataConfig):
    root_dir: str
    train_repo_ids: List[str] = Field(default_factory=list)
    val_repo_ids: List[str] = Field(default_factory=list)
    # Optional native30Hz offline DINO cache root (per-pack subdirectories).
    # The same named/JSON ImageProcess profile governs cached DINO and current VLM RGB.
    dino_cache_root: Optional[str] = None
    shared_image_profile: str = "256x320"
    dino_max_open_episodes: int = 8

    # Per-frame sampling weights by task, train split only: entries "<substring>=<weight>",
    # matched case-insensitively against the frame's task string (meta/tasks.jsonl). A frame
    # whose task matches several entries takes the largest weight; unmatched frames weigh 1.
    # Non-empty -> the train DataLoader uses a WeightedRandomSampler (with replacement) over
    # these weights instead of a uniform shuffle, so e.g. "trash=4" draws every frame of the
    # trash-can tasks 4x as often as any other frame.
    task_sample_weights: List[str] = Field(default_factory=list)

    def parsed_task_sample_weights(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for entry in self.task_sample_weights:
            if "=" not in entry:
                raise ValueError(f"task_sample_weights entry {entry!r} is not '<substring>=<weight>'")
            key, val = entry.rsplit("=", 1)
            w = float(val)
            if not key or w <= 0:
                raise ValueError(f"task_sample_weights entry {entry!r}: need a non-empty substring and weight > 0")
            out[key.strip().lower()] = w
        return out

    @model_validator(mode="after")
    def check_task_sample_weights(self) -> "LerobotDataConfig":
        self.parsed_task_sample_weights()
        return self

    @model_validator(mode="after")
    def _resolve_psi_home_paths(self) -> "LerobotDataConfig":
        psi_home = os.environ.get("PSI_HOME", "/psi")
        if psi_home is None:
            return self
        def _resolve(p: str | None) -> str | None:
            if p is None or Path(p).is_absolute():
                return p
            candidate = Path(psi_home) / p
            return str(candidate) if candidate.exists() else p
        self.root_dir = _resolve(self.root_dir)
        self.dino_cache_root = _resolve(self.dino_cache_root)
        return self

    @model_validator(mode="after")
    def check_dino_history(self) -> "LerobotDataConfig":
        if self.dino_cache_root is None:
            return self
        from ImageProcess import SharedImageProcessor
        model = self.transform.model
        repack = self.transform.repack
        if not isinstance(model, Psi0ModelTransform):
            raise ValueError("dino_cache_root is supported only by Psi0ModelTransform")
        if model.img_aug or model.view_aug or model.adaptive_resize:
            raise ValueError("Cached DINO requires one shared image profile; disable independent img_aug/view_aug/adaptive_resize")
        if getattr(repack, "num_past_frames", 0) != 0:
            raise ValueError("Cached DINO supplies history separately; VLM image must be the single current frame")
        if len(getattr(repack, "image_keys", [])) != 1:
            raise ValueError("DINO history requires exactly one configured current VLM image key")
        if type(self.dino_max_open_episodes) is not int or self.dino_max_open_episodes < 1:
            raise ValueError("dino_max_open_episodes must be a positive integer")
        profile = SharedImageProcessor(self.shared_image_profile).profile
        if model.shared_image_profile not in (None, self.shared_image_profile):
            raise ValueError("Psi0 model and DINO data must use the same shared_image_profile")
        model.shared_image_profile = self.shared_image_profile
        # Serialized model config describes the shared output rather than stale224.
        # Actual geometry runs once in the reader, and model transform bypasses it.
        model.resize.size = tuple(profile.output_hw)
        model.center_crop.size = tuple(profile.output_hw)
        return self

    @model_validator(mode="after")
    def check_repo_ids(self) -> "LerobotDataConfig":
        if len(self.train_repo_ids) == 0:
            raise ValueError("train_repo_ids must be provided")
        if len(self.val_repo_ids) == 0:
            self.val_repo_ids = [self.train_repo_ids[0]]
        return self
    
    @model_validator(mode="after")
    def load_stats(self) -> "LerobotDataConfig":
        if not isinstance(self.transform.field, ActionStateTransform):
            return self
        if (
            not Path(self.transform.field.stat_path).is_absolute() and 
            self.transform.field.action_max is None
        ):
            fpath = resolve_data_path(
                Path(self.root_dir) / self.train_repo_ids[0] / self.transform.field.stat_path
            )
            if not os.path.exists(fpath):
                return self
            with open(fpath, "r") as f:
                stats = json.load(f)
                self.transform.field.populate_stats(stats)
        return self

    def __call__(self, split: str = "train", transform_kwargs={}, **kwargs) -> TransformableDataset:
        from psi.data.lerobot import LeRobotDatasetWrapper
        from psi.data.dataset import Dataset as MapStyleDataset

        # no_aug switches off every augmentation in the transforms (img/view aug, temporal
        # state jitter, state noise). Default it from the split -- val gets the clean
        # pipeline -- but let an explicit caller value (mock clients pass no_aug=True) win.
        transform_kwargs = {"no_aug": split != "train", **transform_kwargs}
        train_dataset = LeRobotDatasetWrapper(self, split=split)
        dataset = MapStyleDataset(self, train_dataset, transform_kwargs=transform_kwargs)
        rules = self.parsed_task_sample_weights()
        if split == "train" and rules:
            dataset.sample_weights = train_dataset.frame_task_weights(rules)
        return dataset

    def mock(self, split: str = "train", transform_kwargs={}, **kwargs) -> Any:
        dataset = self.__call__(split, transform_kwargs=transform_kwargs, **kwargs)
        return dataset[0]
