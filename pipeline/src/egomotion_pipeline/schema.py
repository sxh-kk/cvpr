from dataclasses import dataclass, field
import numpy as np

LEGACY_SCHEMA_VERSION = "egomotion-body22-camera-floor-v3"
V4_SCHEMA_VERSION = "egomotion-body22-camera-floor-dino5-v4"
SCHEMA_VERSION = "egomotion-dense22-hands21-camera-floor-dino5-v5"
SUPPORTED_SCHEMA_VERSIONS = (LEGACY_SCHEMA_VERSION, V4_SCHEMA_VERSION, SCHEMA_VERSION)
DINO_CHECKPOINT = "facebook/dinov2-with-registers-large"
DINO_TOKEN_NAMES = ["cls", "register_0", "register_1", "register_2", "register_3"]
DINO_TOKEN_DIM = 1024
DINO_SHAPE = (len(DINO_TOKEN_NAMES), DINO_TOKEN_DIM)
DINO_FLAT_DIM = int(np.prod(DINO_SHAPE))


def dino_format():
    """Shared storage contract, independent of whether weights were loaded."""
    return {"checkpoint": DINO_CHECKPOINT, "token_order": list(DINO_TOKEN_NAMES),
            "shape": list(DINO_SHAPE), "flat_shape": [DINO_FLAT_DIM],
            "flatten_order": "token_major", "storage_dtype": "float32",
            "pooling": "last_hidden_state[:, :5]", "normalization": "none_after_model"}

BODY_NAMES = ["pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
              "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
              "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
              "left_elbow", "right_elbow", "left_wrist", "right_wrist"]
PARENTS = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19])
HAND_NAMES = ["wrist"] + [f"{finger}_{j}" for finger in ["thumb", "index", "middle", "ring", "pinky"] for j in range(1, 5)]
POSITION_SHAPES = {"body": (22, 3), "hand_left": (21, 3), "hand_right": (21, 3), "head": (1, 3), "camera": (1, 3)}
ROTATION_SHAPES = {"body": (22, 4), "head": (1, 4), "camera": (1, 4)}
for _key in ["head_reference", "head_reference_floor"]:
    POSITION_SHAPES[_key] = (1, 3)
    ROTATION_SHAPES[_key] = (1, 4)


@dataclass
class Episode:
    uid: str
    source: str
    timestamps: np.ndarray
    positions: dict
    rotations: dict
    position_valid: dict = field(default_factory=dict)
    rotation_valid: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    video: dict | None = None

    def check(self):
        from .geometry import normalize_quat
        ts = np.asarray(self.timestamps, dtype=np.float64)
        if ts.ndim != 1 or not len(ts) or not np.isfinite(ts).all() or np.any(np.diff(ts) <= 0):
            raise ValueError(f"{self.uid}: timestamps must be finite and strictly increasing")
        self.timestamps = ts
        for modality, shape in POSITION_SHAPES.items():
            value = np.asarray(self.positions.get(modality, np.full((len(ts), *shape), np.nan)), dtype=np.float64)
            if value.shape != (len(ts), *shape):
                raise ValueError(f"{self.uid}: {modality} position shape {value.shape}, expected {(len(ts), *shape)}")
            mask = np.asarray(self.position_valid.get(modality, np.ones(value.shape[:-1], bool)), bool)
            if mask.shape != value.shape[:-1]:
                raise ValueError(f"{modality} mask shape mismatch")
            self.position_valid[modality] = mask & np.isfinite(value).all(-1)
            self.positions[modality] = np.where(self.position_valid[modality][..., None], value, 0)
        for modality, shape in ROTATION_SHAPES.items():
            raw = np.asarray(self.rotations.get(modality, np.zeros((len(ts), *shape))), dtype=np.float64)
            if raw.shape != (len(ts), *shape):
                raise ValueError(f"{self.uid}: {modality} rotation shape mismatch")
            q, finite = normalize_quat(raw)
            mask = np.asarray(self.rotation_valid.get(modality, np.ones(q.shape[:-1], bool)), bool)
            if mask.shape != q.shape[:-1]:
                raise ValueError(f"{modality} rotation mask shape mismatch")
            self.rotation_valid[modality] = mask & finite
            q[~self.rotation_valid[modality]] = [1, 0, 0, 0]
            self.rotations[modality] = q
        return self


def _base_feature_specs(dino_dim):
    result = {"timestamp": {"dtype": "float32", "shape": [1], "names": None}}
    for key in ["index", "frame_index", "episode_index", "task_index"]:
        result[key] = {"dtype": "int64", "shape": [1], "names": None}
    result["observation.source_timestamp"] = {"dtype": "float64", "shape": [1], "names": None}
    result["observation.source_frame_left"] = {"dtype": "int64", "shape": [1], "names": None}
    result["observation.source_frame_right"] = {"dtype": "int64", "shape": [1], "names": None}
    result["observation.source_interpolation_weight"] = {"dtype": "float32", "shape": [1], "names": None}
    for modality, shape in POSITION_SHAPES.items():
        joints = BODY_NAMES if modality == "body" else HAND_NAMES if modality.startswith("hand") else [modality]
        result[f"observation.state.{modality}_world"] = {
            "dtype": "float32", "shape": [int(np.prod(shape))],
            "names": [f"{joint}.{axis}" for joint in joints for axis in "xyz"]}
        for suffix in ["valid", "interpolated"]:
            result[f"observation.{suffix}.{modality}_position"] = {"dtype": "bool", "shape": [shape[0]], "names": joints}
    for modality, shape in ROTATION_SHAPES.items():
        joints = BODY_NAMES if modality == "body" else [modality]
        result[f"observation.state.{modality}_world_rotation"] = {
            "dtype": "float32", "shape": [int(np.prod(shape))],
            "names": [f"{joint}.{axis}" for joint in joints for axis in "wxyz"]}
        for suffix in ["valid", "interpolated"]:
            result[f"observation.{suffix}.{modality}_rotation"] = {"dtype": "bool", "shape": [shape[0]], "names": joints}
    result["observation.features.dino"] = {"dtype": "float32", "shape": [dino_dim], "names": None}
    result["observation.valid.dino"] = {"dtype": "bool", "shape": [1], "names": None}
    result["observation.video_timestamp"] = {"dtype": "float64", "shape": [1], "names": None}
    result["observation.video_time_error"] = {"dtype": "float32", "shape": [1], "names": None}
    result["observation.environment.floor_height"] = {"dtype": "float32", "shape": [1], "names": None}
    result["observation.environment.floor_normal"] = {"dtype": "float32", "shape": [3], "names": ["x", "y", "z"]}
    for key in ["floor", "floor_verified", "head_reference_basis"]:
        result[f"observation.valid.{key}"] = {"dtype": "bool", "shape": [1], "names": None}
    result["observation.state.head_reference_height"] = {"dtype": "float32", "shape": [1], "names": None}
    result["observation.valid.head_reference_height"] = {"dtype": "bool", "shape": [1], "names": None}
    result["observation.state.body_head_floor"] = {
        "dtype": "float32", "shape": [66],
        "names": [f"{joint}.{axis}" for joint in BODY_NAMES for axis in "xyz"]}
    for suffix in ["valid", "interpolated"]:
        result[f"observation.{suffix}.body_head_floor_position"] = {
            "dtype": "bool", "shape": [22], "names": BODY_NAMES}
    return result


def feature_specs_for_schema(version):
    """Return the exact physical column contract for each supported archive."""
    if version == LEGACY_SCHEMA_VERSION:
        return _base_feature_specs(DINO_TOKEN_DIM)
    if version == V4_SCHEMA_VERSION:
        return _base_feature_specs(DINO_FLAT_DIM)
    if version == SCHEMA_VERSION:
        from .unified_dense22 import unified_specs
        result = _base_feature_specs(DINO_FLAT_DIM)
        result.update(unified_specs())
        return result
    raise ValueError(f"Unsupported archive schema: {version}")


def feature_specs(dino_dim=DINO_FLAT_DIM):
    """Current v5 features; the historical 1024-width call selects v3."""
    if dino_dim == DINO_TOKEN_DIM:
        return feature_specs_for_schema(LEGACY_SCHEMA_VERSION)
    if dino_dim != DINO_FLAT_DIM:
        raise ValueError("DINO width must be legacy 1024 or five-token 5120")
    return feature_specs_for_schema(SCHEMA_VERSION)
