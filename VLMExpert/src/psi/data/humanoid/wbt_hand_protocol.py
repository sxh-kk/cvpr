"""WBT source-hand protocol and canonicalization helpers.

The action expert is trained in a source protocol.  These helpers deliberately
do not contain target-robot joint limits or retargeting logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


WBT_HAND_PROTOCOL = "wbt-source-12d-v1"
WBT_NATIVE_PROTOCOL = "wbt-native-v1"
ROOT_DIM = 7
BODY_DIM = 29
HAND_DIM = 6
NATIVE_DIM = ROOT_DIM + BODY_DIM + 2 * HAND_DIM

ROOT_NAMES = (
    "root_x",
    "root_y",
    "root_z",
    "root_qw",
    "root_qx",
    "root_qy",
    "root_qz",
)

# WBT stores the G1 body vector in the same order used by the Unitree/SIMPLE
# whole-body interface.  Keeping this list here makes the numeric ordering an
# explicit part of the protocol instead of an implicit array-position guess.
G1_BODY_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

# The WBT README names the root/body split but does not expand the 29 joint
# names.  These repository-local sources are the control and whole-body model
# definitions used to interpret that exported vector, and agree index-for-index.
G1_BODY_JOINT_ORDER_SOURCES = (
    "real/teleop/robot_control/robot_arm_joints.py::G1_JOINT_NAME_MAP",
    "third_party/SIMPLE/src/simple/robots/g1_wholebody.py::WHOLE_BODY_JOINTS[:29]",
)

HAND_NAMES = (
    "thumb_flex",
    "thumb_lateral",
    "index",
    "middle",
    "ring",
    "little",
)

PROTOCOL_NAMES = ROOT_NAMES + G1_BODY_JOINT_NAMES + tuple(
    f"left_hand_{name}" for name in HAND_NAMES
) + tuple(f"right_hand_{name}" for name in HAND_NAMES)

PROTOCOL_GROUPS = {
    "root": {"start": 0, "end": ROOT_DIM, "names": ROOT_NAMES},
    "body": {"start": ROOT_DIM, "end": ROOT_DIM + BODY_DIM, "names": G1_BODY_JOINT_NAMES},
    "left_hand_source": {
        "start": ROOT_DIM + BODY_DIM,
        "end": ROOT_DIM + BODY_DIM + HAND_DIM,
        "names": tuple(f"left_hand_{name}" for name in HAND_NAMES),
    },
    "right_hand_source": {
        "start": ROOT_DIM + BODY_DIM + HAND_DIM,
        "end": NATIVE_DIM,
        "names": tuple(f"right_hand_{name}" for name in HAND_NAMES),
    },
}

if len(PROTOCOL_NAMES) != NATIVE_DIM:  # pragma: no cover - module invariant
    raise RuntimeError("wbt-native-v1 dimension names do not match its 48D shape")


@dataclass(frozen=True)
class HandLayout:
    """Description of one raw 12D hand layout.

    ``left_indices`` and ``right_indices`` address the raw concatenated vector;
    the permutations then reorder each side into :data:`HAND_NAMES` order.
    ``signs`` and ``offsets`` are applied after the permutation.
    """

    layout_id: str
    source_hand_type: str
    raw_left_order: tuple[str, ...]
    raw_right_order: tuple[str, ...]
    left_indices: tuple[int, ...]
    right_indices: tuple[int, ...]
    left_permutation: tuple[int, ...]
    right_permutation: tuple[int, ...]
    units: str = "normalized"
    signs: tuple[float, ...] = (1.0,) * HAND_DIM
    offsets: tuple[float, ...] = (0.0,) * HAND_DIM
    valid_range: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if len(self.left_indices) != HAND_DIM or len(self.right_indices) != HAND_DIM:
            raise ValueError(f"{self.layout_id}: each hand must have six raw indices")
        if len(self.raw_left_order) != HAND_DIM or len(self.raw_right_order) != HAND_DIM:
            raise ValueError(f"{self.layout_id}: raw hand names must be six-dimensional")
        if set(self.left_indices) & set(self.right_indices):
            raise ValueError(f"{self.layout_id}: left/right raw hand indices overlap")
        if min((*self.left_indices, *self.right_indices), default=-1) < 0:
            raise ValueError(f"{self.layout_id}: raw hand indices must be non-negative")
        if set((*self.left_indices, *self.right_indices)) != set(range(2 * HAND_DIM)):
            raise ValueError(f"{self.layout_id}: raw hand indices must cover all 12 values")
        if len(self.left_permutation) != HAND_DIM or len(self.right_permutation) != HAND_DIM:
            raise ValueError(f"{self.layout_id}: each hand permutation must have six entries")
        if len(self.signs) != HAND_DIM or len(self.offsets) != HAND_DIM:
            raise ValueError(f"{self.layout_id}: signs and offsets must be six-dimensional")
        if sorted(self.left_permutation) != list(range(HAND_DIM)):
            raise ValueError(f"{self.layout_id}: invalid left permutation")
        if sorted(self.right_permutation) != list(range(HAND_DIM)):
            raise ValueError(f"{self.layout_id}: invalid right permutation")
        if not np.isfinite(np.asarray(self.signs, dtype=np.float32)).all():
            raise ValueError(f"{self.layout_id}: signs must be finite")
        if not np.isfinite(np.asarray(self.offsets, dtype=np.float32)).all():
            raise ValueError(f"{self.layout_id}: offsets must be finite")
        if self.valid_range is not None:
            if len(self.valid_range) != 2 or not np.isfinite(self.valid_range).all():
                raise ValueError(f"{self.layout_id}: valid_range must contain two finite values")
            if self.valid_range[0] > self.valid_range[1]:
                raise ValueError(f"{self.layout_id}: valid_range lower bound exceeds upper bound")

    def metadata(self) -> dict[str, Any]:
        return {
            "hand_layout_id": self.layout_id,
            "source_hand_type": self.source_hand_type,
            "raw_left_order": list(self.raw_left_order),
            "raw_right_order": list(self.raw_right_order),
            "left_indices": list(self.left_indices),
            "right_indices": list(self.right_indices),
            "left_permutation": list(self.left_permutation),
            "right_permutation": list(self.right_permutation),
            "canonical_order": list(HAND_NAMES),
            "units": self.units,
            "signs": list(self.signs),
            "offsets": list(self.offsets),
            "valid_range": list(self.valid_range) if self.valid_range is not None else None,
            "invalid_value_policy": "error",
            "missing_value_policy": "error",
        }


_INSPIRE_RAW = (
    "index",
    "middle",
    "ring",
    "little",
    "thumb_flex",
    "thumb_lateral",
)
_BRAINCO_RAW = HAND_NAMES

# These layouts are the two 12D layouts documented by the WBT datasets.  A
# 2D Dex1 layout is intentionally absent: it cannot satisfy the 48D protocol.
HAND_LAYOUTS: dict[str, HandLayout] = {
    "inspire-6d": HandLayout(
        layout_id="inspire-6d",
        source_hand_type="inspire",
        raw_left_order=_INSPIRE_RAW,
        raw_right_order=_INSPIRE_RAW,
        left_indices=(0, 1, 2, 3, 4, 5),
        right_indices=(6, 7, 8, 9, 10, 11),
        left_permutation=(4, 5, 0, 1, 2, 3),
        right_permutation=(4, 5, 0, 1, 2, 3),
        units="normalized_open_to_close",
        valid_range=(0.0, 1.0),
    ),
    "brainco-6d": HandLayout(
        layout_id="brainco-6d",
        source_hand_type="brainco",
        raw_left_order=_BRAINCO_RAW,
        raw_right_order=_BRAINCO_RAW,
        left_indices=(0, 1, 2, 3, 4, 5),
        right_indices=(6, 7, 8, 9, 10, 11),
        left_permutation=(0, 1, 2, 3, 4, 5),
        right_permutation=(0, 1, 2, 3, 4, 5),
        units="normalized_open_to_close",
        valid_range=(0.0, 1.0),
    ),
}

# Dataset names are metadata, not dimensionality-based inference.  This map is
# used only when a WBT export omitted hand_layout_id from info.json.
KNOWN_REPO_LAYOUT_PREFIXES = (
    ("G1_WBT_Inspire_", "inspire-6d"),
    ("G1_WBT_Brainco_", "brainco-6d"),
)


def get_hand_layout(layout_id: str) -> HandLayout:
    try:
        return HAND_LAYOUTS[layout_id]
    except KeyError as exc:
        known = ", ".join(sorted(HAND_LAYOUTS))
        raise ValueError(f"Unknown WBT hand_layout_id {layout_id!r}; registered layouts: {known}") from exc


def infer_repo_hand_layout(repo_id: str) -> str | None:
    """Resolve a known repository naming convention to a registered layout."""

    name = str(repo_id).rstrip("/").split("/")[-1]
    for prefix, layout_id in KNOWN_REPO_LAYOUT_PREFIXES:
        if name.startswith(prefix):
            return layout_id
    return None


def _as_hand_array(raw_hand: Any) -> tuple[np.ndarray, tuple[int, ...]]:
    if hasattr(raw_hand, "detach"):
        raw_hand = raw_hand.detach().cpu().numpy()
    arr = np.asarray(raw_hand, dtype=np.float32)
    original_shape = tuple(arr.shape)
    if arr.ndim < 1 or arr.shape[-1] != 2 * HAND_DIM:
        raise ValueError(
            "WBT-native-v1 requires a 12D source hand vector "
            f"([left6,right6]); got shape {arr.shape}. 2D Dex1 data is excluded."
        )
    arr = arr.reshape(-1, 2 * HAND_DIM)
    if arr.shape[0] == 0:
        raise ValueError("WBT hand data cannot be empty")
    if not np.isfinite(arr).all():
        raise ValueError("WBT hand data contains NaN or Inf")
    return arr, original_shape


def canonicalize_hand(
    raw_hand: Any,
    hand_layout_id: str,
    *,
    validate_range: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Convert raw WBT hand values to ``[..., left6, right6]`` source order.

    The values remain source-hand values.  No target robot mapping or clipping
    is performed.  The returned metadata is suitable for a protocol manifest.
    """

    layout = get_hand_layout(hand_layout_id)
    arr, original_shape = _as_hand_array(raw_hand)
    left_raw = arr[:, layout.left_indices]
    right_raw = arr[:, layout.right_indices]
    left = left_raw[:, layout.left_permutation]
    right = right_raw[:, layout.right_permutation]
    signs = np.asarray(layout.signs, dtype=np.float32)
    offsets = np.asarray(layout.offsets, dtype=np.float32)
    canonical = np.concatenate(
        [left * signs + offsets, right * signs + offsets], axis=-1
    ).astype(np.float32, copy=False)
    if not np.isfinite(canonical).all():
        raise ValueError(f"Canonicalized {hand_layout_id} hand contains NaN or Inf")
    if validate_range and layout.valid_range is not None:
        lo, hi = layout.valid_range
        # A small tolerance accommodates float conversion while still catching
        # unit mistakes (for example degrees passed as normalized values).
        if np.nanmin(canonical) < lo - 1e-3 or np.nanmax(canonical) > hi + 1e-3:
            raise ValueError(
                f"{hand_layout_id} hand is outside declared range [{lo}, {hi}]: "
                f"min={np.nanmin(canonical):.5g}, max={np.nanmax(canonical):.5g}"
            )
    canonical = canonical.reshape(original_shape)
    return canonical, layout.metadata()


def protocol_metadata(
    *,
    state_hand_layout_id: str,
    action_hand_layout_id: str,
    use_delta_actions: bool = False,
    fps: float = 30.0,
) -> dict[str, Any]:
    if use_delta_actions:
        raise ValueError("wbt-native-v1 uses absolute robot_q_desired targets; delta actions are unsupported")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("wbt-native-v1 fps must be finite and positive")
    state_layout = get_hand_layout(state_hand_layout_id)
    action_layout = get_hand_layout(action_hand_layout_id)
    return {
        "protocol": WBT_NATIVE_PROTOCOL,
        "version": WBT_NATIVE_PROTOCOL,
        "state_dim": NATIVE_DIM,
        "action_dim": NATIVE_DIM,
        "dimensions": list(PROTOCOL_NAMES),
        "groups": {
            name: {
                "start": int(group["start"]),
                "end": int(group["end"]),
                "names": list(group["names"]),
            }
            for name, group in PROTOCOL_GROUPS.items()
        },
        "root": {
            "dim": ROOT_DIM,
            "names": list(ROOT_NAMES),
            "quaternion_order": "wxyz",
            "coordinate_frame": "wbt_export",
            "position_units": "meters",
            "quaternion_units": "unitless",
            "zero_reference": "as_exported",
            "episode_relative_transform": False,
            "order_source": "WBT dataset README: observation.state.robot_q_current/action.robot_q_desired",
        },
        "body": {
            "dim": BODY_DIM,
            "joint_names": list(G1_BODY_JOINT_NAMES),
            "source_vector_slice": [ROOT_DIM, ROOT_DIM + BODY_DIM],
            "joint_order_sources": list(G1_BODY_JOINT_ORDER_SOURCES),
            "units": "radians",
            "zero_reference": "as_exported_unitree_g1_joint_zero",
        },
        "hand_protocol": WBT_HAND_PROTOCOL,
        "hand_names": list(HAND_NAMES),
        "state_hand_layout": state_layout.metadata(),
        "action_hand_layout": action_layout.metadata(),
        "hand_layout_ids": {
            "state": state_layout.layout_id,
            "action": action_layout.layout_id,
        },
        "source_fields": {
            "state_robot": "observation.state.robot_q_current",
            "state_hand": "observation.state.hand_state",
            "action_robot": "action.robot_q_desired",
            "action_hand": "action.hand_cmd",
        },
        "sampling": {
            "fps": float(fps),
            "time_base": "seconds",
        },
        "absolute_action": True,
        "use_delta_actions": False,
    }


def validate_robot_vector(
    value: Any,
    *,
    name: str,
    quaternion_tolerance: float = 5e-2,
) -> np.ndarray:
    """Validate a WBT 36D root-plus-body vector without changing its order."""

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value, dtype=np.float32)
    original_shape = tuple(arr.shape)
    if arr.ndim < 1 or arr.shape[-1] != ROOT_DIM + BODY_DIM:
        raise ValueError(f"{name} must have shape (...,36), got {arr.shape}")
    arr = arr.reshape(-1, ROOT_DIM + BODY_DIM)
    if arr.shape[0] == 0:
        raise ValueError(f"{name} cannot be empty")
    if not 0.0 <= quaternion_tolerance < 1.0:
        raise ValueError("quaternion_tolerance must be in [0, 1)")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or Inf")
    quat_norm = np.linalg.norm(arr[:, 3:7], axis=-1)
    if np.any(~np.isfinite(quat_norm)) or np.any(np.abs(quat_norm - 1.0) > quaternion_tolerance):
        raise ValueError(
            f"{name} has invalid WXYZ root quaternion norm(s): "
            f"{quat_norm.tolist()}"
        )
    return arr.reshape(original_shape)


def validate_protocol_vector(
    value: Any,
    *,
    name: str,
    quaternion_tolerance: float = 5e-2,
) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value, dtype=np.float32)
    original_shape = tuple(arr.shape)
    if arr.ndim < 1 or arr.shape[-1] != NATIVE_DIM:
        raise ValueError(f"{name} must have shape (...,{NATIVE_DIM}), got {arr.shape}")
    arr = arr.reshape(-1, NATIVE_DIM)
    if arr.shape[0] == 0:
        raise ValueError(f"{name} cannot be empty")
    if not 0.0 <= quaternion_tolerance < 1.0:
        raise ValueError("quaternion_tolerance must be in [0, 1)")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or Inf")
    quat_norm = np.linalg.norm(arr[:, 3:7], axis=-1)
    if np.any(np.abs(quat_norm - 1.0) > quaternion_tolerance):
        raise ValueError(f"{name} has invalid root quaternion norm(s): {quat_norm.tolist()}")
    return arr.reshape(original_shape)
