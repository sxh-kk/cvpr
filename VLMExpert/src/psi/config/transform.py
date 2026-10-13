from __future__ import annotations
import os
import re
import base64
import copy
import json
import math
import time
from collections.abc import Mapping
from io import BytesIO
from typing import (TYPE_CHECKING, Annotated, Any, Dict, List, Optional, Tuple,
                    Union)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pydantic import BaseModel, Field, model_validator
from torchvision.transforms import v2
from psi.config.augmentation import *
from psi.utils import get_asset_dir, pt_to_pil, resolve_path, pad_to_len
from psi.data.humanoid.wbt_hand_protocol import (
    NATIVE_DIM as WBT_NATIVE_DIM,
    WBT_HAND_PROTOCOL,
    WBT_NATIVE_PROTOCOL,
    G1_BODY_JOINT_NAMES,
    PROTOCOL_NAMES,
    canonicalize_hand,
    protocol_metadata,
    validate_protocol_vector,
    validate_robot_vector,
)

IGNORE_INDEX = -100

class RepackTransform(BaseModel):
    def __call__(self, data: dict[str, Any], **kwargs) -> dict[str, Any]:
        return data

class LerobotRepackTransform(RepackTransform):

    def delta_timestamps(self, fps: int) -> dict[str, list[float]]:
        ...  

class ModelTransform(BaseModel):
    def __call__(self, data: dict[str, Any], no_aug: bool = False, **kwargs) -> dict[str, Any]:
        return data

class FieldTransform(BaseModel):
    def __call__(self, data: dict[str, Any], **kwargs) -> dict[str, Any]:
        return data

class DataTransform(BaseModel):
    repack: RepackTransform
    model: ModelTransform
    field: FieldTransform
    
    def __call__(self, data:dict[str, Any], **kwargs) -> dict[str, Any]:
        data = self.repack(data, **kwargs)
        data = self.field(data, **kwargs)
        data = self.model(data, **kwargs)
        return data
    
class EgodexRepackTransform(BaseModel):
    dataset_name: str = "egodex"
    stage:str = "pretrain"  # "pretrain" or "postpre"
    pad_action_dim: int | None = None
    pad_state_dim: int | None = None

    def __call__(self, data: dict[str, Any], **kwargs) -> dict[str, Any]:
        states, _ = pad_to_len(data["states"], self.pad_state_dim) if self.pad_state_dim is not None else (data["states"], None)
        if self.pad_action_dim is not None:
            actions, mask = pad_to_len(data["actions"], self.pad_action_dim) 
        else:
            actions = data["actions"]
            mask = np.ones_like(actions, dtype=bool)

        return dict(
            observations = [Image.fromarray(img[0]) for img in data["current_images"]],  # list of PIL Image
            states=states,  # (To, Da)
            actions=actions,  # (Tp, Da)
            instruction=data["instruction"].lower(),
            dataset=data.get("dataset_name", self.dataset_name),
            actions_mask=mask, #(Tp, Da)
        )

class HEPretrainRepackTransform(RepackTransform):
    dataset_name: str = "humanoid-everyday"
    num_past_frames: int = 0
    action_chunk_size: int = 1
    use_delta_actions: bool = True
    robot_type: str = "g1"
    # pad_action_dim: int | None = None
    # pad_state_dim: int | None = None

    def _to_numpy(self, value, dtype=np.float32):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=dtype)

    def _ensure_2d(self, value, dim: int, name: str):
        arr = self._to_numpy(value)
        if arr.ndim == 1:
            if arr.shape[0] != dim:
                raise ValueError(f"{name} expected shape ({dim},), got {arr.shape}")
            return arr.reshape(1, dim)
        if arr.ndim == 2:
            if arr.shape[1] != dim:
                raise ValueError(f"{name} expected shape (T,{dim}), got {arr.shape}")
            return arr
        raise ValueError(f"{name} expected 1D or 2D, got {arr.shape}")

    def _rpy_to_rot6d(self, rpy):
        from scipy.spatial.transform import Rotation as R
        rot = R.from_euler("xyz", rpy).as_matrix()
        return rot[:, :2].reshape(6)

    def _delta_rpy(self, rpy_seq):
        from scipy.spatial.transform import Rotation as R
        rot = R.from_euler("xyz", rpy_seq).as_matrix()
        rel = rot[1:] @ np.transpose(rot[:-1], (0, 2, 1))
        return R.from_matrix(rel).as_euler("xyz", degrees=False).astype(np.float32)

    def _build_actions(self, data: dict[str, Any]) -> np.ndarray:
        left_wrist_xyz = self._ensure_2d(
            data["action.wrists.left.xyz"], 3, "action.wrists.left.xyz"
        )
        left_wrist_rpy = self._ensure_2d(
            data["action.wrists.left.rpy"], 3, "action.wrists.left.rpy"
        )
        right_wrist_xyz = self._ensure_2d(
            data["action.wrists.right.xyz"], 3, "action.wrists.right.xyz"
        )
        right_wrist_rpy = self._ensure_2d(
            data["action.wrists.right.rpy"], 3, "action.wrists.right.rpy"
        )

        def tip_xyz(keys):
            for key in keys:
                if key in data:
                    return self._ensure_2d(data[key], 3, key)
            return np.zeros_like(left_wrist_xyz)

        left_thumb = tip_xyz(["action.hands.left_thumb.xyz"])
        left_index = tip_xyz(["action.hands.left_index.xyz"])
        left_middle = tip_xyz(
            ["action.hands.left_middle.xyz", "action.hands.left_middle_finger.xyz"]
        )
        left_ring = tip_xyz(["action.hands.left_ring_finger.xyz"])
        left_little = tip_xyz(["action.hands.left_little_finger.xyz"])

        right_thumb = tip_xyz(["action.hands.right_thumb.xyz"])
        right_index = tip_xyz(["action.hands.right_index.xyz"])
        right_middle = tip_xyz(
            ["action.hands.right_middle.xyz", "action.hands.right_middle_finger.xyz"]
        )
        right_ring = tip_xyz(["action.hands.right_ring_finger.xyz"])
        right_little = tip_xyz(["action.hands.right_little_finger.xyz"])

        if self.use_delta_actions:
            if left_wrist_xyz.shape[0] < 2:
                raise ValueError(
                    "use_delta_actions requires at least 2 action frames; "
                    "increase action_chunk_size or disable delta."
                )
            left_wrist_xyz = left_wrist_xyz[1:] - left_wrist_xyz[:-1]
            right_wrist_xyz = right_wrist_xyz[1:] - right_wrist_xyz[:-1]
            left_wrist_rpy = self._delta_rpy(left_wrist_rpy)
            right_wrist_rpy = self._delta_rpy(right_wrist_rpy)

            left_thumb = left_thumb[1:] - left_thumb[:-1]
            left_index = left_index[1:] - left_index[:-1]
            left_middle = left_middle[1:] - left_middle[:-1]
            left_ring = left_ring[1:] - left_ring[:-1]
            left_little = left_little[1:] - left_little[:-1]

            right_thumb = right_thumb[1:] - right_thumb[:-1]
            right_index = right_index[1:] - right_index[:-1]
            right_middle = right_middle[1:] - right_middle[:-1]
            right_ring = right_ring[1:] - right_ring[:-1]
            right_little = right_little[1:] - right_little[:-1]

            left_rot_pad = np.zeros_like(left_wrist_rpy)
            right_rot_pad = np.zeros_like(right_wrist_rpy)
        else:
            left_rot6d = np.stack(
                [self._rpy_to_rot6d(rpy) for rpy in left_wrist_rpy], axis=0
            )
            right_rot6d = np.stack(
                [self._rpy_to_rot6d(rpy) for rpy in right_wrist_rpy], axis=0
            )
            left_wrist_rpy = left_rot6d
            right_wrist_rpy = right_rot6d
            left_rot_pad = None
            right_rot_pad = None

        left_wrist = (
            np.concatenate([left_wrist_xyz, left_wrist_rpy, left_rot_pad], axis=1) #type:ignore
            if self.use_delta_actions
            else np.concatenate([left_wrist_xyz, left_wrist_rpy], axis=1)
        )
        right_wrist = (
            np.concatenate([right_wrist_xyz, right_wrist_rpy, right_rot_pad], axis=1) #type:ignore
            if self.use_delta_actions
            else np.concatenate([right_wrist_xyz, right_wrist_rpy], axis=1)
        )

        actions = np.concatenate(
            [
                left_wrist,
                left_thumb,
                left_index,
                left_middle,
                left_ring,
                left_little,
                right_wrist,
                right_thumb,
                right_index,
                right_middle,
                right_ring,
                right_little,
            ],
            axis=1,
        ).astype(np.float32)
        return actions

    def __call__(
        self,
        data: dict[str, Any],
        metadata=None,
        **kwargs,
    ) -> dict[str, Any]:
        hand_states = self._to_numpy(data["observation.hand_joints"])
        arm_states = self._to_numpy(data["observation.arm_joints"])
        if hand_states.ndim == 2:
            hand_states = hand_states[-1]
        if arm_states.ndim == 2:
            arm_states = arm_states[-1]
        states = np.concatenate((hand_states, arm_states), axis=0).astype(np.float32)

        try:
            actions = self._build_actions(data)
        except Exception as e:
            print("Error building actions:")
            print(data)

            import traceback
            traceback.print_exc()
            raise e

        instruction = ""
        if metadata is not None and "episode_index" in data:
            episode_idx = int(self._to_numpy(data["episode_index"]).reshape(-1)[0])
            if (
                hasattr(metadata, "episodes")
                and metadata.episodes is not None
                and episode_idx < len(metadata.episodes)
            ):
                instruction = metadata.episodes[episode_idx].get("instruction", "")
        if not instruction and "task" in data:
            instruction = data["task"]
        if isinstance(instruction, bytes):
            instruction = instruction.decode("utf-8")
            instruction = self.nice_instruction(instruction.split("/")[1])

        img = data["observation.images.egocentric"]
        if hasattr(img, "dim") and img.dim() == 4:
            img = img[0]
        elif isinstance(img, np.ndarray) and img.ndim == 4:
            img = img[0]

        return dict(
            observations=[pt_to_pil(img, normalized=False)],  # list of PIL Image
            states=states,
            actions=actions,
            instruction=str(instruction).lower(),
            dataset=data.get("dataset_name", self.dataset_name),
        )

    def nice_instruction(self, instruction: str) -> str:
        instruction = instruction.replace("_", " ").replace("-", " ")
        instruction = re.sub(r"\s+", " ", instruction)
        return instruction.strip()

    def delta_timestamps(self, fps: int):
        delta = {}
        delta["observation.hand_joints"] = [
            -t / fps for t in range(self.num_past_frames, -1, -1)
        ]
        delta["observation.arm_joints"] = [
            -t / fps for t in range(self.num_past_frames, -1, -1)
        ]

        action_keys = [
            "action.joint_angles",
            "action.wrists.left.xyz",
            "action.wrists.left.rpy",
            "action.wrists.right.xyz",
            "action.wrists.right.rpy",
            "action.hands.left_thumb.xyz",
            "action.hands.left_index.xyz",
            "action.hands.right_thumb.xyz",
            "action.hands.right_index.xyz",
        ]
        if self.robot_type == "h1":
            action_keys += [
                "action.hands.left_middle_finger.xyz",
                "action.hands.left_ring_finger.xyz",
                "action.hands.left_little_finger.xyz",
                "action.hands.right_middle_finger.xyz",
                "action.hands.right_ring_finger.xyz",
                "action.hands.right_little_finger.xyz",
            ]
        else:
            action_keys += [
                "action.hands.left_middle.xyz",
                "action.hands.right_middle.xyz",
            ]
        action_len = (
            self.action_chunk_size + 1 if self.use_delta_actions else self.action_chunk_size
        )
        for key in action_keys:
            delta[key] = [t / fps for t in range(action_len)]
        return delta

class RealRepackTransform(LerobotRepackTransform):
    dataset_name: str = "real"

    conditions: list[str] = Field(default_factory=lambda: [])
    image_keys: List[str] = Field(default_factory=lambda: ["observation.images.egocentric"])
    state_key: str = "states"
    action_key: str = "action"
    instruction_key: str = "task"

    num_past_frames: int = 0
    action_chunk_size: int = 30

    pad_action_dim: int | None = None
    pad_state_dim: int | None = None

    def delta_timestamps(self, fps):
        delta = {}
        for image_key in self.image_keys:
            delta[image_key] = [-t/fps for t in range(self.num_past_frames, -1, -1)]

        delta[self.state_key] = [-t/fps for t in range(self.num_past_frames, -1, -1)]
        delta[self.action_key] = [t/fps for t in range(self.action_chunk_size)]
        return delta

    def __call__(self, data: dict[str, Any], **kwargs) -> dict[str, Any]:
        states, _ = pad_to_len(data[self.state_key], self.pad_state_dim) if self.pad_state_dim is not None else (data[self.state_key], None)

        if self.pad_action_dim is not None:
            actions, mask = pad_to_len(data[self.action_key], self.pad_action_dim) 
        else:
            actions = data[self.action_key]
            mask = np.ones_like(actions, dtype=bool)

        result = {
            "observations": [ 
                pt_to_pil(data[key], normalized=False)
                for key in self.image_keys
            ], # list of PIL Image
            "states": np.array(states, dtype=np.float32), # (To, Da)
            "actions": np.array(actions, dtype=np.float32),  # (Tp, Da)
            "instruction": data[self.instruction_key].lower(),
            "actions_mask": mask, #(Tp, Da)
        } 
        return result

class SimpleRepackTransform(LerobotRepackTransform):
    dataset_name: str = "simple"

    num_past_frames: int = 0 # single current frame
    action_chunk_size: int = 30

    pad_action_dim: int | None = None
    pad_state_dim: int | None = None

    def delta_timestamps(self, fps: int):
        return {
            "observation.images.egocentric": [-t/fps for t in range(self.num_past_frames, -1, -1)],
            "states": [-t/fps for t in range(self.num_past_frames, -1, -1)],
            "action": [t/fps for t in range(self.action_chunk_size)]
        }
 

    def __call__(self, data: dict[str, Any], **kwargs) -> dict[str, Any]:
        image_key = "observation.images.egocentric"
        states = data["states"].numpy() # (To, Ds)
        actions = data["action"].numpy() # (Tp, Da)
        action_is_pad = data["action_is_pad"].numpy() # (Ta,) 
        pad_mask = np.ones_like(actions) * (1. - action_is_pad[...,None].astype(np.float32))

        states, _ = pad_to_len(states, self.pad_state_dim) if self.pad_state_dim is not None else (states, None)
        if self.pad_action_dim is not None:
            actions, _ = pad_to_len(actions, self.pad_action_dim) 
            mask,_ = pad_to_len(pad_mask, self.pad_action_dim)
        else:
            mask = np.ones_like(actions, dtype=np.float32)

        return {
            "observations": [pt_to_pil(data[image_key],normalized=False)], # single view
            "states": states.astype(np.float32), # (To, Do)
            "actions": actions.astype(np.float32), # (Tp, Da)
            # "action_is_pad": np.array(data["action_is_pad"], dtype=bool),
            "instruction": data["task"].lower(),
            "actions_mask": mask
        }

class IdentityTransform(BaseModel):
    # boilerplate transform config
    # define primitive data types only! eg., int, float, str
    # container types are allowed eg., list[int], dict[str, float]
    # default no-op transform
    # passing external dependency through **kwargs, eg., tokenizer, processor
    def __call__(self, data: dict[str, Any], **kwargs) -> dict[str, Any]:
        return data

class ActionStateTransform(FieldTransform):
    
    stat_path: str
    action_norm_type: str = "bounds"  # "bounds_q99"
    stat_action_key: str = "action"
    stat_state_key: str = "states"
    use_norm_mask: bool = False
    action_norm_masks: List[bool] = [
        True,
        True,
        True,
        True,
        True,
        True,
        False,
    ]  # delta-eef + gripper
    action_min: Optional[List[float]] = None
    action_max: Optional[List[float]] = None
    state_min: Optional[List[float]] = None
    state_max: Optional[List[float]] = None
    normalize_state: bool = False  # whether to normalize states

    pad_action_dim: int | None = None
    pad_state_dim: int | None = None

    def model_post_init(self, __context: Any) -> None:
        if not os.path.exists(resolve_path(self.stat_path)):
            return
        with open(resolve_path(self.stat_path), "r") as f:
            stats = json.load(f)
            self.populate_stats(stats)
        
    def populate_stats(self, stats: dict[str, Any]):
        if self.action_norm_type == "bounds_q99":
            self.action_min = stats[self.stat_action_key]["q01"]
            self.action_max = stats[self.stat_action_key]["q99"]
            if self.normalize_state:
                self.state_min = stats[self.stat_state_key]["q01"]
                self.state_max = stats[self.stat_state_key]["q99"]
        elif self.action_norm_type == "bounds":
            self.action_min = stats[self.stat_action_key]["min"]
            self.action_max = stats[self.stat_action_key]["max"]
            if self.normalize_state:
                self.state_min = stats[self.stat_state_key]["min"]
                self.state_max = stats[self.stat_state_key]["max"]
        else:
            raise ValueError(f"Unsupported action normalization type: {self.action_norm_type}")

        if self.pad_action_dim is not None:
            self.action_min= pad_to_len(np.array(self.action_min, dtype=np.float32), self.pad_action_dim, dim=0)[0].tolist()
            self.action_max = pad_to_len(np.array(self.action_max, dtype=np.float32), self.pad_action_dim, dim=0)[0].tolist()

        if self.pad_state_dim is not None and self.normalize_state:
            self.state_min = pad_to_len(np.array(self.state_min, dtype=np.float32), self.pad_state_dim, dim=0)[0].tolist()
            self.state_max = pad_to_len(np.array(self.state_max, dtype=np.float32), self.pad_state_dim, dim=0)[0].tolist()

    def __call__(self, data: dict[str, Any], **kwargs) -> dict[str, Any]:
        assert self.action_min is not None and self.action_max is not None, \
            f"{self.stat_path} is not loaded properly. Probably {resolve_path(self.stat_path)} does not exist."
        action_min = np.array(self.action_min, dtype=np.float32)
        action_max = np.array(self.action_max, dtype=np.float32)
        if self.normalize_state:
            data["states"] = self.normalize_state_func(data["states"])

        # tolerate near-zero, not just exact zero
        ill_mask = np.abs(action_max - action_min) < 1e-4 * (np.abs(action_max) + np.abs(action_min) + 1e-8) 
        action_max[ill_mask] = 1.0  # prevent division by zero
        actions_normalized = np.where(
            ill_mask, data["actions"], (data["actions"] - action_min) / (action_max - action_min) * 2 - 1
        )

        if self.use_norm_mask:
            actions = np.where(self.action_norm_masks, actions_normalized, data["actions"])
        else: 
            actions = actions_normalized

        # if (np.abs(actions) > 10.0).any():
        #     print("Error: values outside quantile range during normalization.")

        actions = np.clip(actions, -1, 1).astype(np.float32)
        # print(data["dataset"], actions.max(), actions.min(), actions.mean(), actions.std())
        data["raw_actions"] = data["actions"]
        data["actions"] = actions

        return data

    def normalize(self, action, **kwargs):
        data = {"actions": action}
        return self.__call__(data)["actions"]

    def normalize_state_func(self, states, **kwargs):
        state_min = np.array(self.state_min, dtype=np.float32)
        state_max = np.array(self.state_max, dtype=np.float32)
        # Normalize states, tolerate near-zero, not just exact zero
        ill_mask = np.abs(state_max - state_min) < 1e-4 * (np.abs(state_max) + np.abs(state_min) + 1e-8)
        state_max[ill_mask] = 1.0  # prevent division by zero
        current_state = np.where(
            ill_mask, 0, 
            (states - state_min) / (state_max - state_min) * 2 - 1
        )
        current_state = np.clip(current_state, -1, 1).astype(np.float32)
        if np.isnan(current_state).any():
            current_state = np.nan_to_num(current_state, nan=0.0)
        return current_state

    def reverse_call(self, array: Any, **kwargs) -> Any:
        assert self.action_min is not None and self.action_max is not None
        action_min = np.array(self.action_min)
        action_max = np.array(self.action_max)
        reversed_array = (array + 1) / 2 * (action_max - action_min) + action_min
        return reversed_array

    def denormalize_L1_action_err(self, L1_err):
        """return denormalized L1 err loss"""
        
        array_class = torch.tensor if torch.is_tensor(L1_err) else np.array
        where = torch.where if torch.is_tensor(L1_err) else np.where
        data_type = L1_err.dtype

        if self.action_norm_type == "bounds" or \
            self.action_norm_type == "bounds_q99":
            low = array_class(self.action_min, dtype=data_type)  # type: ignore
            high = array_class(self.action_max, dtype=data_type)  # type: ignore
            
            if self.use_norm_mask:
                if torch.is_tensor(L1_err):
                    low = low.to(L1_err.device)
                    high = high.to(L1_err.device)
                    action_norm_masks = torch.tensor(
                        self.action_norm_masks, device=L1_err.device
                    )
                else:
                    action_norm_masks = self.action_norm_masks
                result = where(action_norm_masks, 0.5 * L1_err * (high - low), L1_err) # type:ignore
            else:
                result = 0.5 * L1_err * (high - low)
            return result
        else:
            raise NotImplementedError
    
    
    def denormalize(self, normalized: np.ndarray|torch.Tensor) -> np.ndarray|torch.Tensor: 
        """ Denormalize the action. """
        array_class = torch.tensor if torch.is_tensor(normalized) else np.array
        where = torch.where if torch.is_tensor(normalized) else np.where
        data_type = normalized.dtype

        if self.action_norm_type == "bounds" or \
                self.action_norm_type == "bounds_q99":
            if self.action_norm_type == "bounds":
                low = array_class(self.action_min, dtype=data_type)  # type: ignore
                high = array_class(self.action_max, dtype=data_type)  # type: ignore
            elif self.action_norm_type == "bounds_q99":
                low = array_class(self.action_min, dtype=data_type)  # type: ignore
                high = array_class(self.action_max, dtype=data_type)  # type: ignore

            if not self.use_norm_mask:
                assert self.action_min is not None
                action_norm_masks = [True] * len(self.action_min)
            else:
                action_norm_masks = self.action_norm_masks

            if torch.is_tensor(normalized):
                low = low.to(normalized.device)  # type: ignore
                high = high.to(normalized.device)  # type: ignore
                action_norm_masks = torch.tensor(action_norm_masks, device=normalized.device)
            else:
                action_norm_masks = np.array(action_norm_masks)

            action = where(action_norm_masks, 0.5 * (normalized + 1) * (high - low) + low, normalized)  # type: ignore
        return action # type: ignore


class Qwen3vlModelTransform(ModelTransform):
    resize: ResizeImage = Field(default_factory=lambda: ResizeImage(size=(270, 480)))
    color_jitter: ColorJitter
    img_aug: bool = False
    # anchor_size: tuple[int, int] = (256, 256) # H, W

    # for mixed dataset with different image sizes
    adaptive_resize: bool = False
    img_sizes: dict[str, Any] = Field(default_factory=lambda: {
        "egodex": [270, 480],
        "he"    : [240, 320],
    })

    def __call__(
        self, 
        data: dict[str, Any], 
        no_aug: bool = False,
        vlm_processor = None, 
        action_tokenizer = None, 
        global_token_id = None,
        model_max_length = None,
        max_instruction_tokens = 512,
        **kwargs
    ) -> dict[str, Any]:
        if vlm_processor is None or action_tokenizer is None:
            return {"raw_actions": data["actions"], **data}
        """
        data: dict with keys:
            - instruction: str
            - observations: List[Image] (To x PIL)
            - states: (To, Da) torch.Tensor
            - action: (Tp, Da) torch.Tensor
        """
        do_img_aug = False if no_aug else self.img_aug
        if self.adaptive_resize:
            assert data["dataset"] is not None
            match data["dataset"]:
                case "egodex":
                    target_size = self.img_sizes["egodex"]
                case "humanoid-everyday":
                    target_size = self.img_sizes["he"]
                case _:
                    target_size = (256, 256)
            resizer = ResizeImage(size=tuple(target_size))() # type: ignore
        else:
            resizer = self.resize()
        t1 = v2.Compose([
            resizer,
            self.color_jitter() if do_img_aug else v2.Identity(),
        ])
        
        images = [t1(img) for img in data["observations"]]
        instruction = data["instruction"]
        state = data["states"]
        action = data["actions"]
        inputs, num_answer_tokens_list = self.build_qwenvl_inputs(
            vlm_processor, action_tokenizer, [images], [instruction], [state], [action],
            global_token_id=global_token_id, model_max_length=model_max_length,
            max_instruction_tokens=max_instruction_tokens,
        )
        inputs["dataset_name"] = data.get("dataset", "unknown")
        inputs["raw_actions"] = action
        # inputs["raw_instruction"] = instruction
        inputs["raw_images"] = np.stack([np.array(img) for img in images])
        return inputs

    def build_qwenvl_inputs(
        self,
        vlm_processor,
        action_tokenizer,
        images,
        instructions,
        states,
        actions,
        **kwargs,
    ):
        from psi.models.global_vlm import build_vlm_inputs

        action_texts = [action_tokenizer(action) for action in actions]
        num_answer_tokens_list = [len(vlm_processor.tokenizer.encode(
            text, add_special_tokens=False,
        )) for text in action_texts]
        inputs = build_vlm_inputs(
            vlm_processor, images, instructions, action_texts=action_texts,
            global_token_id=kwargs.get("global_token_id"),
            model_max_length=kwargs.get("model_max_length"),
            max_instruction_tokens=kwargs.get("max_instruction_tokens", 512),
        )
        return inputs, num_answer_tokens_list

class HEPosttrainRepackTransform(RepackTransform):
    dataset_name: str = "humanoid-everyday"
    num_past_frames: int = 0
    action_chunk_size: int = 16
    use_delta_actions: bool = True
    pad_action_dim: int | None = None
    pad_state_dim: int | None = None

    def __call__(
        self,
        data: dict[str, Any],
        **kwargs,
    ) -> dict[str, Any]:
        # HACK: different action format determines robot type
        if data["action.joint_angles"].shape[1] == 26:
            # H1 robot
            raw = data["action.joint_angles"]
            actions = np.concatenate([
                raw[:, 6:12][::-1], # from left thumb to little
                np.zeros((raw.shape[0], 1), dtype=np.float32), # pad finger to 7 dof
                raw[:, 0:6][::-1], # from right thumb to little
                np.zeros((raw.shape[0], 1), dtype=np.float32), # pad finger to 7 dof
                raw[:, 12:], # left arm + right arm (each from should to wrist)
            ], axis=1).astype(np.float32)
            hand = data["observation.hand_joints"]
            states = np.concatenate([
                hand[:, 6:12],
                np.zeros((hand.shape[0], 1), dtype=np.float32), # pad finger to 7 dof
                hand[:, 0:6],
                np.zeros((hand.shape[0], 1), dtype=np.float32), # pad finger to 7 dof 
                data["observation.arm_joints"]
            ], axis=1).astype(np.float32)
        else:
           actions = data["action.joint_angles"].astype(np.float32)
           states = np.concatenate((data["observation.hand_joints"], data["observation.arm_joints"]), axis=1).astype(np.float32)


        if self.use_delta_actions:
            actions = actions[1:] - actions[:-1]
            action_mask = data["action_mask"][1:]
        else:
            action_mask = data["action_mask"]

        # Expand action_mask to (T, Da) by repeating along last dimension
        if action_mask.ndim == 1:
            action_mask = np.repeat(action_mask[:, None], actions.shape[1], axis=1)

        states, _ = pad_to_len(states, self.pad_state_dim) if self.pad_state_dim is not None else (states, None)
        if self.pad_action_dim is not None:
            actions, _ = pad_to_len(actions, self.pad_action_dim)
            mask, _  = pad_to_len(action_mask, self.pad_action_dim)
        else:
            mask = np.ones_like(actions, dtype=bool)
            # mask = action_mask
        return {
            "observations": [pt_to_pil(img, normalized=False) for img in data["observation.images.egocentric"]],
            "states": states,
            "actions": actions,
            "instruction": str(data["task"]).lower(),
            "dataset": data.get("dataset_name", self.dataset_name),
            "actions_mask": mask,# torch.from_numpy(mask),
            "obs_mask": data["obs_mask"], #torch.from_numpy(data["obs_mask"]),
        }


class WBTNativePosttrainRepackTransform(RepackTransform):
    """Repack raw WBT fields into the native 48D action-expert protocol."""

    dataset_name: str = "wbt-native"
    protocol: str = WBT_NATIVE_PROTOCOL
    hand_protocol: str = WBT_HAND_PROTOCOL
    num_past_frames: int = 0
    action_chunk_size: int = 30
    use_delta_actions: bool = False
    hand_layout_id: str | None = None
    state_hand_layout_id: str | None = None
    action_hand_layout_id: str | None = None
    preserve_raw_hand: bool = True
    quaternion_tolerance: float = 5e-2
    expected_fps: float = 30.0
    pad_action_dim: int | None = None
    pad_state_dim: int | None = None

    @model_validator(mode="after")
    def validate_config(self):
        if self.protocol != WBT_NATIVE_PROTOCOL:
            raise ValueError(
                f"WBTNativePosttrainRepackTransform requires protocol={WBT_NATIVE_PROTOCOL!r}"
            )
        if self.hand_protocol != WBT_HAND_PROTOCOL:
            raise ValueError(
                f"WBTNativePosttrainRepackTransform requires hand_protocol={WBT_HAND_PROTOCOL!r}"
            )
        if self.num_past_frames < 0:
            raise ValueError("num_past_frames must be non-negative")
        if self.action_chunk_size <= 0:
            raise ValueError("action_chunk_size must be positive")
        if not np.isfinite(self.expected_fps) or self.expected_fps <= 0:
            raise ValueError("expected_fps must be finite and positive")
        if self.use_delta_actions:
            raise ValueError("WBT-native-v1 contains absolute desired targets; delta actions are unsupported")
        if not self.preserve_raw_hand:
            raise ValueError("WBT-native-v1 requires raw hand provenance")
        if not 0.0 < self.quaternion_tolerance < 1.0:
            raise ValueError("quaternion_tolerance must be in (0, 1)")
        if self.pad_action_dim not in (None, WBT_NATIVE_DIM):
            raise ValueError("WBT-native-v1 action dimension is fixed at 48; padding to another dimension is invalid")
        if self.pad_state_dim not in (None, WBT_NATIVE_DIM):
            raise ValueError("WBT-native-v1 state dimension is fixed at 48; padding to another dimension is invalid")
        return self

    @staticmethod
    def _to_numpy(value: Any, name: str) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        try:
            array = np.asarray(value, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} cannot be converted to float32") from exc
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains NaN or Inf")
        return array

    @classmethod
    def _sequence(cls, value: Any, dim: int, name: str) -> np.ndarray:
        array = cls._to_numpy(value, name)
        if array.ndim == 1:
            if array.shape[0] != dim:
                raise ValueError(f"{name} expected ({dim},), got {array.shape}")
            return array[None, :]
        if array.ndim != 2 or array.shape[1] != dim:
            raise ValueError(f"{name} expected shape (T,{dim}), got {array.shape}")
        return array

    @staticmethod
    def _fit_sequence(
        array: np.ndarray,
        target_len: int,
        *,
        name: str,
        pad_at_start: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        if array.ndim != 2:
            raise ValueError(f"{name} must be two-dimensional, got {array.shape}")
        if array.shape[0] == 0:
            raise ValueError(f"{name} cannot be empty")
        if array.shape[0] > target_len:
            raise ValueError(
                f"{name} has {array.shape[0]} frames but protocol expects at most {target_len}; "
                "check action_chunk_size/num_past_frames"
            )
        missing = target_len - array.shape[0]
        if missing == 0:
            return array.astype(np.float32, copy=False), np.ones(target_len, dtype=np.float32)
        if pad_at_start:
            padding = np.repeat(array[:1], missing, axis=0)
            result = np.concatenate([padding, array], axis=0)
            mask = np.concatenate(
                [np.zeros(missing, dtype=np.float32), np.ones(array.shape[0], dtype=np.float32)]
            )
        else:
            padding = np.repeat(array[-1:], missing, axis=0)
            result = np.concatenate([array, padding], axis=0)
            mask = np.concatenate(
                [np.ones(array.shape[0], dtype=np.float32), np.zeros(missing, dtype=np.float32)]
            )
        return result.astype(np.float32, copy=False), mask

    @staticmethod
    def _fit_mask(value: Any, target_len: int, name: str) -> np.ndarray:
        mask_array = np.asarray(value, dtype=np.float32)
        if mask_array.ndim == 2:
            if mask_array.shape[1] == 1:
                mask_array = mask_array[:, 0]
            else:
                # The protocol masks complete action time steps.  Accept a
                # repeated per-dimension mask only when all dimensions agree.
                if not np.allclose(mask_array, mask_array[:, :1], equal_nan=False):
                    raise ValueError(f"{name} must mask complete time steps, got {mask_array.shape}")
                mask_array = mask_array[:, 0]
        elif mask_array.ndim != 1:
            raise ValueError(f"{name} must have shape (T,) or (T,D), got {mask_array.shape}")
        mask = mask_array
        if mask.shape[0] == 0:
            raise ValueError(f"{name} cannot be empty")
        if mask.shape[0] > target_len:
            raise ValueError(f"{name} has {mask.shape[0]} entries but expected at most {target_len}")
        if mask.shape[0] < target_len:
            mask = np.concatenate([mask, np.zeros(target_len - mask.shape[0], dtype=np.float32)])
        if not np.isfinite(mask).all() or np.any(
            ~(np.isclose(mask, 0.0) | np.isclose(mask, 1.0))
        ):
            raise ValueError(f"{name} must contain only finite binary values")
        return np.where(mask > 0.5, 1.0, 0.0).astype(np.float32)

    @staticmethod
    def _image_to_pil(image: Any) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if hasattr(image, "detach"):
            image = image.detach().cpu().numpy()
        array = np.asarray(image)
        if array.ndim != 3:
            raise ValueError(f"WBT image must have three dimensions, got {array.shape}")
        if array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
            array = np.transpose(array, (1, 2, 0))
        if array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
        if array.shape[-1] != 3:
            raise ValueError(f"WBT image must have three channels, got {array.shape}")
        if array.size == 0:
            raise ValueError("WBT image cannot be empty")
        minimum = float(np.min(array))
        maximum = float(np.max(array))
        if np.issubdtype(array.dtype, np.floating):
            if not np.isfinite(array).all():
                raise ValueError("WBT image contains NaN or Inf")
            # WBT reader returns uint8; this branch supports synthetic tests and
            # alternate readers that return [0, 1] float images.
            if minimum >= -1e-6 and maximum <= 1.0 + 1e-6:
                array = array * 255.0
            elif minimum < 0.0 or maximum > 255.0:
                raise ValueError("WBT float image values must be in [0, 1] or [0, 255]")
        elif minimum < 0.0 or maximum > 255.0:
            raise ValueError("WBT integer image values must be in [0, 255]")
        return Image.fromarray(np.rint(array).astype(np.uint8))

    def _images(self, data: dict[str, Any], target_len: int, obs_mask: np.ndarray) -> list[Image.Image]:
        if obs_mask.shape != (target_len,):
            raise ValueError(
                f"obs_mask must have shape ({target_len},), got {obs_mask.shape}"
            )
        image_key = "observation.images.head_stereo_left"
        if image_key not in data:
            for candidate in (
                "observation.images.cam_0",
            ):
                if candidate in data:
                    image_key = candidate
                    break
        if image_key not in data:
            raise KeyError("WBT sample is missing observation.images.head_stereo_left")
        raw = data[image_key]
        if isinstance(raw, Image.Image):
            values = [raw]
        elif isinstance(raw, (list, tuple)):
            values = list(raw)
        elif hasattr(raw, "detach"):
            array = raw.detach().cpu()
            if array.ndim == 3:
                values = [array]
            elif array.ndim == 4:
                values = [array[i] for i in range(array.shape[0])]
            else:
                raise ValueError(f"WBT image sequence has unexpected shape {tuple(array.shape)}")
        else:
            array = np.asarray(raw)
            if array.ndim == 3:
                values = [array]
            elif array.ndim == 4:
                values = [array[i] for i in range(array.shape[0])]
            else:
                values = []
        if not values:
            raise ValueError("WBT sample contains no observation images")
        if len(values) > target_len:
            raise ValueError(f"WBT sample has {len(values)} observation images but expected at most {target_len}")
        if len(values) < target_len:
            missing = target_len - len(values)
            # Historical frames are padded with the earliest available frame.
            values = [values[0]] * missing + values
        return [self._image_to_pil(value) for value in values]

    @staticmethod
    def _instruction(data: dict[str, Any]) -> str:
        value = data.get("task", data.get("instruction", ""))
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        instruction = re.sub(r"\s+", " ", str(value)).strip().lower()
        if not instruction:
            raise ValueError("WBT task instruction cannot be empty")
        return instruction

    def __call__(self, data: dict[str, Any], **kwargs) -> dict[str, Any]:
        sample_dataset = data.get("dataset_name")
        if sample_dataset is not None and sample_dataset != self.dataset_name:
            raise ValueError(
                f"WBT transform received dataset_name={sample_dataset!r}; "
                f"expected {self.dataset_name!r}"
            )
        sample_protocol = data.get("protocol")
        if sample_protocol is not None and sample_protocol != self.protocol:
            raise ValueError(
                f"WBT sample protocol {data.get('protocol')!r} does not match {self.protocol!r}"
            )
        sample_hand_protocol = data.get("hand_protocol")
        if sample_hand_protocol is not None and sample_hand_protocol != self.hand_protocol:
            raise ValueError(
                "WBT sample hand protocol does not match "
                f"{self.hand_protocol!r}"
            )
        state_robot = self._sequence(
            data["observation.state.robot_q_current"], 36, "observation.state.robot_q_current"
        )
        state_hand = self._sequence(
            data["observation.state.hand_state"], 12, "observation.state.hand_state"
        )
        action_robot = self._sequence(
            data["action.robot_q_desired"], 36, "action.robot_q_desired"
        )
        action_hand = self._sequence(data["action.hand_cmd"], 12, "action.hand_cmd")

        # Validate root/body vectors before hand canonicalization so a malformed
        # quaternion cannot be hidden by later concatenation.
        validate_robot_vector(
            state_robot,
            name="observation.state.robot_q_current",
            quaternion_tolerance=self.quaternion_tolerance,
        )
        validate_robot_vector(
            action_robot,
            name="action.robot_q_desired",
            quaternion_tolerance=self.quaternion_tolerance,
        )

        state_layout = (
            self.state_hand_layout_id
            or data.get("state_hand_layout_id")
            or self.hand_layout_id
        )
        action_layout = (
            self.action_hand_layout_id
            or data.get("action_hand_layout_id")
            or self.hand_layout_id
        )
        if state_layout is None or action_layout is None:
            raise ValueError(
                "WBT hand layout is required; provide state/action_hand_layout_id or hand_layout_id"
            )
        canonical_state_hand, state_hand_meta = canonicalize_hand(
            state_hand, str(state_layout), validate_range=True
        )
        canonical_action_hand, action_hand_meta = canonicalize_hand(
            action_hand, str(action_layout), validate_range=True
        )

        states = np.concatenate([state_robot, canonical_state_hand], axis=-1).astype(np.float32)
        actions = np.concatenate([action_robot, canonical_action_hand], axis=-1).astype(np.float32)
        states, generated_obs_mask = self._fit_sequence(
            states,
            self.num_past_frames + 1,
            name="states",
            pad_at_start=True,
        )
        actions, generated_action_mask = self._fit_sequence(
            actions,
            self.action_chunk_size,
            name="actions",
            pad_at_start=False,
        )

        if "obs_mask" in data:
            obs_mask = self._fit_mask(data["obs_mask"], states.shape[0], "obs_mask")
        else:
            obs_mask = generated_obs_mask
        if "action_mask" in data:
            action_mask = self._fit_mask(data["action_mask"], actions.shape[0], "action_mask")
        elif "actions_mask" in data:
            action_mask = self._fit_mask(data["actions_mask"], actions.shape[0], "actions_mask")
        else:
            action_mask = generated_action_mask
        # A caller-provided mask may invalidate additional steps, but it may not
        # mark sequence-padding inserted above as valid.
        obs_mask = np.minimum(obs_mask, generated_obs_mask)
        action_mask = np.minimum(action_mask, generated_action_mask)

        validate_protocol_vector(
            states, name="states", quaternion_tolerance=self.quaternion_tolerance
        )
        validate_protocol_vector(
            actions, name="actions", quaternion_tolerance=self.quaternion_tolerance
        )
        supplied_metadata = data.get("protocol_metadata")
        if supplied_metadata is not None:
            if not isinstance(supplied_metadata, Mapping):
                raise ValueError("protocol_metadata must be a mapping")
            if supplied_metadata.get("protocol") != self.protocol:
                raise ValueError(
                    "WBT sample protocol metadata does not match "
                    f"{self.protocol!r}"
                )
            if supplied_metadata.get("state_dim") != WBT_NATIVE_DIM or supplied_metadata.get("action_dim") != WBT_NATIVE_DIM:
                raise ValueError("WBT protocol metadata must declare 48D state and action")
            if supplied_metadata.get("dimensions") != list(PROTOCOL_NAMES):
                raise ValueError("WBT protocol metadata has an incompatible dimension order")
            if supplied_metadata.get("hand_protocol") != self.hand_protocol:
                raise ValueError("WBT protocol metadata has an incompatible hand protocol")
            sampling = supplied_metadata.get("sampling")
            if (
                not isinstance(sampling, Mapping)
                or sampling.get("fps") != self.expected_fps
            ):
                raise ValueError("WBT protocol metadata has an incompatible sampling rate")
            if (
                supplied_metadata.get("use_delta_actions") is not False
                or supplied_metadata.get("absolute_action") is not True
            ):
                raise ValueError("WBT protocol metadata must declare absolute actions")
            root_metadata = supplied_metadata.get("root")
            if (
                not isinstance(root_metadata, Mapping)
                or root_metadata.get("quaternion_order") != "wxyz"
                or root_metadata.get("coordinate_frame") != "wbt_export"
                or root_metadata.get("episode_relative_transform") is not False
            ):
                raise ValueError("WBT protocol metadata has incompatible root-frame semantics")
            body_metadata = supplied_metadata.get("body")
            if (
                not isinstance(body_metadata, Mapping)
                or body_metadata.get("joint_names") != list(G1_BODY_JOINT_NAMES)
                or body_metadata.get("source_vector_slice") != [7, 36]
                or body_metadata.get("units") != "radians"
            ):
                raise ValueError("WBT protocol metadata has an incompatible G1 body joint order")
            metadata_layouts = supplied_metadata.get("hand_layout_ids", {})
            metadata_state = metadata_layouts.get("state") if isinstance(metadata_layouts, Mapping) else None
            metadata_action = metadata_layouts.get("action") if isinstance(metadata_layouts, Mapping) else None
            if metadata_state != str(state_layout):
                raise ValueError("WBT state hand layout conflicts with protocol metadata")
            if metadata_action != str(action_layout):
                raise ValueError("WBT action hand layout conflicts with protocol metadata")
        action_mask_2d = np.repeat(action_mask[:, None], WBT_NATIVE_DIM, axis=1).astype(np.float32)
        output: dict[str, Any] = {
            "observations": self._images(data, states.shape[0], obs_mask),
            "states": states,
            "actions": actions,
            "instruction": self._instruction(data),
            "dataset": data.get("dataset_name", self.dataset_name),
            "repo_id": data.get("repo_id"),
            "protocol": self.protocol,
            "hand_protocol": self.hand_protocol,
            "state_hand_layout_id": str(state_layout),
            "action_hand_layout_id": str(action_layout),
            "hand_layout_ids": {
                "state": str(state_layout),
                "action": str(action_layout),
            },
            "actions_mask": action_mask_2d,
            "obs_mask": obs_mask.astype(np.float32),
            "protocol_metadata": supplied_metadata
            or protocol_metadata(
                state_hand_layout_id=str(state_layout),
                action_hand_layout_id=str(action_layout),
                use_delta_actions=False,
                fps=self.expected_fps,
            ),
        }
        if self.preserve_raw_hand:
            output["raw_observation_hand"] = state_hand.copy()
            output["raw_action_hand"] = action_hand.copy()
            output["hand_provenance"] = {
                "state": state_hand_meta,
                "action": action_hand_meta,
            }
        return output

    def delta_timestamps(self, fps: int) -> dict[str, list[float]]:
        if fps <= 0:
            raise ValueError("fps must be positive")
        if abs(float(fps) - self.expected_fps) > 1e-6:
            raise ValueError(
                f"WBT-native-v1 expects {self.expected_fps:g} FPS, got {fps:g}"
            )
        return {
            "observation.state.robot_q_current": [
                -t / fps for t in range(self.num_past_frames, -1, -1)
            ],
            "observation.state.hand_state": [
                -t / fps for t in range(self.num_past_frames, -1, -1)
            ],
            "action.robot_q_desired": [t / fps for t in range(self.action_chunk_size)],
            "action.hand_cmd": [t / fps for t in range(self.action_chunk_size)],
        }

class MixedRepackTransform(RepackTransform):
    dataset_name: str = "mixed"

    num_past_frames: int = 0
    action_chunk_size: int = 1
    use_delta_actions: bool = True
    robot_type: str = "mixed"
    stage: str = "pretrain"  # "pretrain" or "postpre"
    pad_action_dim: int | None = None
    pad_state_dim: int | None = None

    def model_post_init(self, __context: Any) -> None:
        if self.stage == "pretrain":
            self._he_repacker = HEPretrainRepackTransform.model_validate(self.model_dump())
        else:
            self._he_repacker = HEPosttrainRepackTransform.model_validate(self.model_dump())
        self._egodex_repacker = EgodexRepackTransform.model_validate(self.model_dump())

    def __call__(self, data: dict[str, Any], **kwargs) -> dict[str, Any]:
        dataset_name = data.get("dataset_name", "default")
        match(dataset_name):
            case "egodex":
                return self._egodex_repacker(data, **kwargs)
            case "humanoid-everyday":
                return self._he_repacker(data, **kwargs)
        return data
    

class Psi0ModelTransform(ModelTransform):
    resize: ResizeImage = Field(default_factory=lambda: ResizeImage(size=(224)))
    center_crop: CenterCrop = Field(default_factory=lambda: CenterCrop(size=224))
    color_jitter: ColorJitter = Field(default_factory=lambda: ColorJitter())
    gaussian_noise: GaussianNoise = Field(default_factory=lambda: GaussianNoise(mean=0, std=3, prob_skip=0.1))
    img_aug: bool = False
    
    # for mixed dataset with different image sizes
    adaptive_resize: bool = False
    img_sizes: dict[str, Any] = Field(default_factory=lambda: {
        "egodex": [270, 480],
        "he"    : [240, 320],
    })

    def __call__(
        self, data: dict[str, Any], vlm_processor=None, no_aug=False, **kwargs
    ) -> dict[str, Any]:
        do_img_aug = False if no_aug else self.img_aug
        if self.adaptive_resize:
            assert data["dataset"] is not None
            match data["dataset"]:
                case "egodex":
                    target_size = self.img_sizes["egodex"]
                case "humanoid-everyday":
                    target_size = self.img_sizes["he"]
                case _:
                    target_size = (256, 256)
            resizer = ResizeImage(size=tuple(target_size))() # type: ignore
            center_crop = CenterCrop(size=tuple(target_size))() # type: ignore
        else:
            resizer = self.resize()
            center_crop = self.center_crop()
            
        t1 = v2.Compose([
            resizer,
            center_crop,
            self.color_jitter() if do_img_aug else v2.Identity(),
        ])

        images = [t1(img) for img in data["observations"]]
        instruction = data["instruction"]

        inputs: dict = self.build_qwenvl_inputs(
            vlm_processor, images, instruction,
        ) # type: ignore
        inputs["dataset_name"] = data.get("dataset", "unknown")
        inputs["raw_actions"] = data["raw_actions"]
        if "actions_mask" in data:
            inputs["actions_mask"] = data["actions_mask"]

        inputs["raw_images"] = images
        inputs['actions'] = data["actions"]
        inputs['states'] = data["states"]
        inputs['instruction'] = data["instruction"]
        return inputs

    def build_qwenvl_inputs(
        self,
        vlm_processor,
        imgs,
        instruction,
        # state,
        # action,
        **kwargs,
    ) -> tuple[dict, int]:
        from qwen_vl_utils import process_vision_info

        """adapted from Qwen_VL_Interface.build_qwenvl_inputs"""
        messages = []

        content = [{"type": "image", "image": img} for img in imgs]
        content.append({"type": "text", "text": instruction})
        user_msg = {"role": "user", "content": content}
        messages.append([user_msg])

        # Prepare text prompts using processor
        # default process is json --> message --> texts --> input_ids
        texts = [
            vlm_processor.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True
            )
            for m in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)  # type: ignore
        inputs = vlm_processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return inputs

class ACTModelTransform(ModelTransform):
    resize: ResizeImage
    center_crop: CenterCrop
    color_jitter: ColorJitter
    normalize: Normalize
    img_aug: bool = False
    
    def __call__(
        self,
        data: dict[str, Any],
        no_aug: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        do_img_aug = False if no_aug else self.img_aug
        
        # Image transform pipeline
        t = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                self.resize(),
                self.center_crop(),
                self.color_jitter() if do_img_aug else v2.Identity(),
                self.normalize(),
            ]
        )
        
        processed_images = torch.stack([t(img) for img in data["observations"]]) # (N*To, C, H, W)
        
        if "action_is_pad" in data:
            # LerobotRepackTransform format: (chunk_size,), True=padded
            action_is_pad = data["action_is_pad"]
        elif "actions_mask" in data:
            # WeGR00TRepackTransform format: (chunk_size, Da), True=original, False=padded
            actions_mask = data["actions_mask"]
            if isinstance(actions_mask, np.ndarray):
                actions_mask = actions_mask.astype(bool)
                # (chunk_size, Da) --> (chunk_size,)
                if actions_mask.ndim == 2:
                    action_is_pad = ~actions_mask[:, 0]  # Invert: True->False (original), False->True (padded)
                else:
                    action_is_pad = ~actions_mask
            else:
                actions_mask = actions_mask.bool() if hasattr(actions_mask, 'bool') else actions_mask
                action_is_pad = ~actions_mask[:, 0] if len(actions_mask.shape) == 2 else ~actions_mask
        else:
            action_is_pad = np.zeros(data["actions"].shape[0], dtype=bool)
        
        return {
            "observation.images": processed_images,  # (N*To, C, H, W)
            "observation.state": data["states"],  # (state_dim,) or (obs_horizon, state_dim)
            "action": data["actions"],  # (chunk_size, action_dim)
            "action_is_pad": action_is_pad,  # (chunk_size,), standardized: True=padded, False=original
        }
    
class DiffusionPolicyModelTransform(ModelTransform):
    resize: ResizeImage
    color_jitter: ColorJitter
    center_crop: CenterCrop
    normalize: Normalize
    img_aug: bool = False
    
    def __call__(
        self,
        data: dict[str, Any],
        no_aug: bool = False,
        **kwargs,
    ) -> dict[str, Any]:

        # import torch
        # try:
        # from torchvision.transforms import v2
        # except:
        #     from torchvision import transforms as v2
        do_img_aug = False if no_aug else self.img_aug
        t = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                self.resize(),
                self.center_crop(),
                self.color_jitter() if do_img_aug else v2.Identity(),
                self.normalize(),
            ]
        )
        processed_images = torch.stack([t(img) for img in data["observations"]]) # (To, 3, H, W)
        
        if "action_is_pad" in data:
            # LerobotRepackTransform format: (chunk_size,), True=padded
            action_is_pad = data["action_is_pad"]
        elif "actions_mask" in data:
            # WeGR00TRepackTransform format: (chunk_size, Da), True=original, False=padded
            actions_mask = data["actions_mask"]
            if isinstance(actions_mask, np.ndarray):
                actions_mask = actions_mask.astype(bool)
                # (chunk_size, Da) --> (chunk_size,)
                if actions_mask.ndim == 2:
                    action_is_pad = ~actions_mask[:, 0]  # Invert: True->False (original), False->True (padded)
                else:
                    action_is_pad = ~actions_mask
            else:
                actions_mask = actions_mask.bool() if hasattr(actions_mask, 'bool') else actions_mask
                action_is_pad = ~actions_mask[:, 0] if len(actions_mask.shape) == 2 else ~actions_mask
        else:
            action_is_pad = np.zeros(data["actions"].shape[0], dtype=bool)
        
        
        return dict(
            image=processed_images, # (To, 3, H, W)
            agent_pos=data["states"], # (To, Da)
            action=data["actions"], #(Tp, Da)
            action_is_pad=action_is_pad,  # (Tp,), standardized: True=padded, False=original
        )
        
        # return {
        #     "observation.images": processed_images,  # (To, C, H, W)
        #     "observation.state": data["states"],  # (To, state_dim)
        #     "action": data["actions"],  # (Tp, Da)
        # }
