"""Versioned dense motion contract and training configuration."""

import math
import os
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
K12_JOINTS = [0, 4, 5, 10, 11, 13, 14, 15, 18, 19, 20, 21]
MOTION_DIM = 245
TRAJECTORY_DIM = 18
IMAGE_TOKEN_COUNT = 5
IMAGE_FEATURE_DIM = 1024
REPRESENTATION_VERSION = "dense12_rot6d_hands21_contacts_v1"
LEGACY_ARCHITECTURE = "e7_dense12_hands21_v2"
GLOBAL_ARCHITECTURE = "e7_dense12_hands21_global_v2"
SMPLH_REPRESENTATION_VERSION = "smplh12_rot6d_hands21_contacts_v1"
SMPLH_MODEL_SHA256 = "be42bff1ff0589cf0467893bcfbbeb3f9b533ba151031c8cdfb64ca1dcb34947"
SMPLH_SCHEMA = "egomotion-smplh16-bodyparams-hands21-camera-floor-dino5-v1"


def _ns(**values):
    return SimpleNamespace(**values)


def motion_expert_config(*, global_token=True):
    """Dense E7 backbone; source-native axes are explicit and single-source."""
    archive = "/gaozt-test1/datasets/MixEgoMotion"
    data = _ns(
        DATA_DIR=archive, ARCHIVE_ROOT=archive, DATASET_NAME="egosuite",
        REPRE_TYPE=REPRESENTATION_VERSION, COND_BETAS=False,
        COND_IMG_FEAT=True, COND_TRAJ=True, IMG_FEAT_TYPE="dinov2_reg",
        FPS=5, FRAME_PHASE=0, WINDOW=11, FORECAST_PREFIX=6, SEGMENT_STRIDE=5,
        TRAIN_TASK="fore", REQUIRE_FULL_WINDOW=True, TRAJECTORY_CONDITION_PROB=0.0,
        MOTION_SUPERVISION="joint",
        SPLIT_MANIFEST=str(ROOT / "exp/dense_k12/splits.json"),
        STATS_PATH=str(ROOT / "exp/dense_k12_fore_5hz_joint/stats.pt"), STATS_MODE="dense_masked",
        NORMALIZATION_CLIP=None, ROW_GROUP_CACHE_MB=128, PARQUET_FILE_CACHE=8,
        ROTATION_BASIS_PATH=None, ROTATION_BASIS_MODE="source_native",
        TARGET_JOINT_BASIS="egosuite_native_v1", REQUIRE_VERIFIED_ROTATIONS=True,
        CONTACT_SPEED_M_S=0.2, CONTACT_HEIGHT_M=0.10,
        BATCH_SIZE=8, NUM_WORKERS=4, PIN_MEMORY=True,
        PERSISTENT_WORKERS=True, PREFETCH_FACTOR=2, DROP_LAST=True,
        MIN_VALID_IMAGE_FRACTION=0.5, SOURCE_BALANCE=False,
    )
    expert = _ns(
        ENABLED=True, NUM_EXPERTS=12, NUM_ROUTED_EXPERTS=11, TOP_K=2,
        ROUTER_JITTER=0.0, SHARED_EXPERT=True, CHUNK_SIZE=4,
        CONDITIONED_ROUTER=True, ROUTED_GATE_INIT=0.05,
        SHARED_UNFREEZE_EPOCH=0, UNFREEZE_RAMP_EPOCHS=0,
        ROUTED_LR_SCALE=1.0, UNFROZEN_LR_SCALE=1.0, LOAD_BALANCE_WEIGHT=0.005,
    )
    model = _ns(
        MODEL_NAME="uem", CKPT_PATH=None, INIT_CKPT_PATH=None,
        GENERATIVE_TYPE="flow", LEARN_TRAJ=False, ENCODER_TSFM=None,
        MOTION_EXPERT=expert, OUTPUT_BRANCH_MODE="single", ZERO_MASK_TOKEN=False,
        PREDICT_XSTART=True, GLOBAL_TOKEN_COUNT=1 if global_token else 0,
        INPUT_FEATS=MOTION_DIM, TRAJ_DIM=TRAJECTORY_DIM,
        IMAGE_TOKENS=IMAGE_TOKEN_COUNT, IMAGE_DIM=IMAGE_FEATURE_DIM,
    )
    flow = _ns(
        NUM_STEPS=10, SOLVER="euler", BETA_ALPHA=1.5, BETA_BETA=1.0,
        T_MIN=0.001, PREDICTION_TYPE="x0", GLOBAL_WEIGHT=8.0,
        GLOBAL_FEATURE_START=108, GLOBAL_FEATURE_END=117,
    )
    train = _ns(
        EXP_PATH=str(ROOT / "exp/dense_k12_fore_5hz_joint"), LR=1e-4, SEED=62,
        USE_CKPT_LR=True, FUSED_ADAMW=False, WEIGHT_DECAY=0.01,
        NUM_EPOCHS=300, NUM_GPUS=1, PRECISION="bf16-mixed",
        ACCUMULATE_GRAD_BATCHES=1, GRADIENT_CLIP_VAL=1.0, EMA_DECAY=0.992028,
        SCHEDULER="cosine_warmup", SCHEDULER_TOTAL_EPOCHS=300,
        WARMUP_EPOCHS=20, MIN_LR_RATIO=0.1, MAX_STEPS=-1,
        NUM_SANITY_VAL_STEPS=0, LOG_EVERY_N_STEPS=10, PROGRESS_REFRESH_RATE=10,
        CHECK_VAL_EVERY_N_EPOCHS=5, SAVE_EVERY_N_EPOCHS=10,
        SAMPLE_VAL_EVERY_N_EPOCHS=5, SAMPLE_VAL_MAX_SAMPLES=16,
    )
    return _ns(DATA=data, SPARSE_JOINTS=_ns(ENABLED=True, INDICES=list(K12_JOINTS)),
               MODEL=model, FLOW=flow, TRAIN=train)


def smplh_motion_expert_config(*, global_token=True):
    """Keep the K12/245D network; derive its labels from common SMPL-H FK."""
    cfg = motion_expert_config(global_token=global_token)
    archive = "/gaozt-test1/datasets/MixEgoMotion/smplh-parameters-v1"
    cfg.DATA.DATA_DIR = cfg.DATA.ARCHIVE_ROOT = archive
    cfg.DATA.DATASET_NAME = "mixegomotion_smplh"
    # Training-only historical trajectory; validation/deployment remain images.
    # A starting modality-dropout experiment, not a proven optimal probability.
    cfg.DATA.TRAJECTORY_CONDITION_PROB = 0.5
    cfg.DATA.REPRE_TYPE = SMPLH_REPRESENTATION_VERSION
    cfg.DATA.ROTATION_BASIS_MODE = "smplh_model_defined"
    cfg.DATA.TARGET_JOINT_BASIS = "smplh_body22_axes_v1"
    cfg.DATA.MODEL_SHA256 = SMPLH_MODEL_SHA256
    cfg.DATA.SMPLH_MODEL_PATH = None
    cfg.DATA.SPLIT_MANIFEST = archive + "/meta/splits.json"
    cfg.DATA.BATCH_SIZE = 64
    cfg.TRAIN.NUM_GPUS = 8
    cfg.TRAIN.EXP_PATH = str(ROOT / "exp/smplh_k12_fore_5hz_joint")
    cfg.DATA.STATS_PATH = cfg.TRAIN.EXP_PATH + "/stats.pt"
    return cfg


def validate_architecture(cfg):
    smplh = cfg.DATA.REPRE_TYPE == SMPLH_REPRESENTATION_VERSION
    dim, joints = MOTION_DIM, K12_JOINTS
    global_start, global_end = 108, 117
    expected = {
        "DATA.REPRE_TYPE": SMPLH_REPRESENTATION_VERSION if smplh else REPRESENTATION_VERSION,
        "DATA.COND_BETAS": False, "DATA.COND_IMG_FEAT": True,
        "DATA.COND_TRAJ": True, "DATA.IMG_FEAT_TYPE": "dinov2_reg",
        "DATA.FRAME_PHASE": 0,
        "SPARSE_JOINTS.ENABLED": True, "SPARSE_JOINTS.INDICES": joints,
        "MODEL.MODEL_NAME": "uem", "MODEL.GENERATIVE_TYPE": "flow",
        "MODEL.LEARN_TRAJ": False, "MODEL.ENCODER_TSFM": None,
        "MODEL.OUTPUT_BRANCH_MODE": "single", "MODEL.ZERO_MASK_TOKEN": False,
        "MODEL.PREDICT_XSTART": True, "MODEL.INPUT_FEATS": dim,
        "MODEL.TRAJ_DIM": TRAJECTORY_DIM, "MODEL.IMAGE_TOKENS": IMAGE_TOKEN_COUNT,
        "MODEL.IMAGE_DIM": IMAGE_FEATURE_DIM,
        "MODEL.MOTION_EXPERT.ENABLED": True,
        "MODEL.MOTION_EXPERT.NUM_EXPERTS": 12,
        "MODEL.MOTION_EXPERT.NUM_ROUTED_EXPERTS": 11,
        "MODEL.MOTION_EXPERT.TOP_K": 2, "MODEL.MOTION_EXPERT.SHARED_EXPERT": True,
        "MODEL.MOTION_EXPERT.CHUNK_SIZE": 4,
        "MODEL.MOTION_EXPERT.CONDITIONED_ROUTER": True,
        "FLOW.PREDICTION_TYPE": "x0", "FLOW.SOLVER": "euler", "FLOW.NUM_STEPS": 10,
        "FLOW.GLOBAL_FEATURE_START": global_start, "FLOW.GLOBAL_FEATURE_END": global_end,
    }
    for path, value in expected.items():
        actual = cfg
        for key in path.split("."):
            actual = getattr(actual, key, None)
        if actual != value:
            raise ValueError(f"Dense MotionExpert requires {path}={value!r}, got {actual!r}")
    count = cfg.MODEL.GLOBAL_TOKEN_COUNT
    if type(count) is not int or count not in (0, 1):
        raise ValueError("GLOBAL_TOKEN_COUNT must be 0 or 1")
    if type(cfg.DATA.FPS) is not int or cfg.DATA.FPS not in (5, 15):
        raise ValueError("Dense MotionExpert supports 5Hz forecasting and legacy 15Hz profiles")
    task = getattr(cfg.DATA, "TRAIN_TASK", None)
    if task not in (None, "recon", "fore", "gen"):
        raise ValueError("DATA.TRAIN_TASK must be recon, fore, gen or null for legacy training")
    if getattr(cfg.DATA, "MOTION_SUPERVISION", "future_only") not in ("joint", "future_only"):
        raise ValueError("DATA.MOTION_SUPERVISION must be joint or future_only")
    probability = getattr(cfg.DATA, "TRAJECTORY_CONDITION_PROB", 0.0)
    if (isinstance(probability, bool) or not isinstance(probability, (int, float))
            or not math.isfinite(probability) or not 0 <= probability <= 1):
        raise ValueError("DATA.TRAJECTORY_CONDITION_PROB must be between 0 and 1")
    if type(getattr(cfg.DATA, "REQUIRE_FULL_WINDOW", False)) is not bool:
        raise ValueError("DATA.REQUIRE_FULL_WINDOW must be a boolean")
    for key in ("WINDOW", "FORECAST_PREFIX", "SEGMENT_STRIDE"):
        value = getattr(cfg.DATA, key)
        if type(value) is not int or value < 1:
            raise ValueError(f"DATA.{key} must be a positive integer")
    if cfg.DATA.WINDOW < 2 or cfg.DATA.FORECAST_PREFIX >= cfg.DATA.WINDOW:
        raise ValueError("FORECAST_PREFIX must leave at least one future frame")
    if cfg.DATA.FPS == 5 and (cfg.DATA.WINDOW != 11 or cfg.DATA.FORECAST_PREFIX != 6
                            or task != "fore" or not cfg.DATA.REQUIRE_FULL_WINDOW):
        raise ValueError("5Hz forecasting requires WINDOW=11, FORECAST_PREFIX=6, TRAIN_TASK=fore and complete windows")
    if cfg.DATA.MOTION_SUPERVISION == "joint" and cfg.DATA.FPS != 5:
        raise ValueError("Joint history/future supervision requires the 5Hz forecasting profile")
    weight = cfg.FLOW.GLOBAL_WEIGHT
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
        raise ValueError("FLOW.GLOBAL_WEIGHT must be finite and positive")
    if getattr(cfg.DATA, "COND_TEXT", False) or getattr(getattr(cfg.MODEL, "TEXT_CONDITION", None), "ENABLED", False):
        raise ValueError("Text-conditioned legacy configurations are incompatible with dense MotionExpert")
    if smplh:
        if (cfg.DATA.ROTATION_BASIS_MODE != "smplh_model_defined"
                or cfg.DATA.TARGET_JOINT_BASIS != "smplh_body22_axes_v1"
                or getattr(cfg.DATA, "MODEL_SHA256", None) != SMPLH_MODEL_SHA256):
            raise ValueError("SMPL-H requires the matching model-defined body22 axes and model SHA256")
        if cfg.DATA.ROTATION_BASIS_PATH:
            raise ValueError("SMPL-H model-defined axes do not use native-axis calibration mappings")
    elif cfg.DATA.ROTATION_BASIS_MODE not in {"source_native", "unified"}:
        raise ValueError("ROTATION_BASIS_MODE must be source_native or unified")
    if not cfg.DATA.TARGET_JOINT_BASIS:
        raise ValueError("TARGET_JOINT_BASIS must explicitly identify the supervised joint axes")


def architecture_metadata(cfg):
    validate_architecture(cfg)
    count = cfg.MODEL.GLOBAL_TOKEN_COUNT
    smplh = cfg.DATA.REPRE_TYPE == SMPLH_REPRESENTATION_VERSION
    metadata = {
        "architecture": GLOBAL_ARCHITECTURE if count else LEGACY_ARCHITECTURE,
        "representation": cfg.DATA.REPRE_TYPE, "global_token_count": count,
        "num_prefix_tokens": 1 + count, "hidden_dim": 768,
        "motion_dim": cfg.MODEL.INPUT_FEATS, "trajectory_dim": TRAJECTORY_DIM,
        "joint_indices": list(cfg.SPARSE_JOINTS.INDICES), "hand_joint_count": 21,
        "rotation_encoding": "matrix_first_two_rows", "fps": cfg.DATA.FPS,
        "frame_phase": cfg.DATA.FRAME_PHASE, "window": cfg.DATA.WINDOW,
        "forecast_prefix": cfg.DATA.FORECAST_PREFIX,
        "image_token_count": IMAGE_TOKEN_COUNT, "image_feature_dim": IMAGE_FEATURE_DIM,
        "image_token_order": ["cls", "register_0", "register_1", "register_2", "register_3"],
        "reference": "camera_floor", "rotation_basis_mode": cfg.DATA.ROTATION_BASIS_MODE,
        "target_joint_basis": cfg.DATA.TARGET_JOINT_BASIS,
        "feature_slices": {"body": [0,108], "global": [108,117],
                           "left_hand": [117,180], "right_hand": [180,243], "contacts": [243,245]},
        "feature_t": 1.0, "prediction_type": cfg.FLOW.PREDICTION_TYPE,
        "num_steps": cfg.FLOW.NUM_STEPS, "solver": cfg.FLOW.SOLVER,
    }
    if smplh:
        metadata["architecture"] = "e7_smplh12_hands21_global_v1" if count else "e7_smplh12_hands21_v1"
        metadata["body_model"] = {"type": "smplh", "betas": 16,
                                  "model_sha256": cfg.DATA.MODEL_SHA256,
                                  "shape_condition": False, "shape_supervision": None,
                                  "shape_usage": "source_fk_only"}
    if cfg.DATA.FPS == 5:
        joint = getattr(cfg.DATA, "MOTION_SUPERVISION", "future_only") == "joint"
        metadata["forecast_protocol"] = {
            "version": "image_history_joint_forecast_v1" if joint else "image_history_forecast_v1",
            "train_task": cfg.DATA.TRAIN_TASK,
            "trajectory_condition_prob": cfg.DATA.TRAJECTORY_CONDITION_PROB,
            "complete_windows_only": cfg.DATA.REQUIRE_FULL_WINDOW,
            "segment_stride": cfg.DATA.SEGMENT_STRIDE,
            "history_latent": "joint_denoising" if joint else "fixed_gaussian_context",
            "supervision": "all_valid_frames" if joint else "future_only",
            "frame_interval_s": 1.0 / cfg.DATA.FPS,
            "history_duration_s": (cfg.DATA.FORECAST_PREFIX - 1) / cfg.DATA.FPS,
            "future_duration_s": (cfg.DATA.WINDOW - cfg.DATA.FORECAST_PREFIX) / cfg.DATA.FPS,
        }
    return metadata


def as_dict(cfg):
    if isinstance(cfg, SimpleNamespace):
        return {key: as_dict(value) for key, value in vars(cfg).items()}
    if isinstance(cfg, dict):
        return {key: as_dict(value) for key, value in cfg.items()}
    if isinstance(cfg, (list, tuple)):
        return [as_dict(value) for value in cfg]
    return cfg


def config_from_dict(values):
    cfg = (smplh_motion_expert_config() if values.get("DATA", {}).get("REPRE_TYPE") == SMPLH_REPRESENTATION_VERSION
           else motion_expert_config())
    # Old checkpoints/YAMLs must not inherit the new forecasting task defaults.
    supplied_data = values.get("DATA", {})
    if "MOTION_SUPERVISION" not in supplied_data:
        # Existing checkpoints retain their loss and sampling protocol.
        cfg.DATA.MOTION_SUPERVISION = "future_only"
    if supplied_data.get("FPS") == 15:
        cfg.DATA.TRAIN_TASK = None
        cfg.DATA.REQUIRE_FULL_WINDOW = False
        cfg.DATA.TRAJECTORY_CONDITION_PROB = 0.0
    def merge(target, source):
        for key, value in source.items():
            if isinstance(value, dict):
                child = getattr(target, key, SimpleNamespace())
                if not isinstance(child, SimpleNamespace):
                    child = SimpleNamespace()
                merge(child, value)
                setattr(target, key, child)
            else:
                setattr(target, key, value)
    merge(cfg, values)
    return cfg


def load_config(path=None, *, data_dir=None):
    import yaml
    with open(path or ROOT / "configs/smplh_k12_global.yaml", encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    if not isinstance(values, dict):
        raise ValueError("Configuration must be a YAML mapping")
    cfg = config_from_dict(values)
    supplied_data = values.get("DATA", {})
    archive = data_dir or os.environ.get("MOTIONEXPERT_DATA_DIR") or supplied_data.get("ARCHIVE_ROOT") or supplied_data.get("DATA_DIR") or cfg.DATA.ARCHIVE_ROOT
    def resolve(value):
        p = Path(value).expanduser()
        return str((p if p.is_absolute() else ROOT / p).resolve())
    cfg.DATA.DATA_DIR = cfg.DATA.ARCHIVE_ROOT = resolve(archive)
    if (cfg.DATA.REPRE_TYPE == SMPLH_REPRESENTATION_VERSION
            and (data_dir or os.environ.get("MOTIONEXPERT_DATA_DIR"))):
        # This archive owns its split membership; relocating data relocates
        # that manifest too, rather than borrowing an old absolute path.
        cfg.DATA.SPLIT_MANIFEST = str(Path(cfg.DATA.ARCHIVE_ROOT) / "meta/splits.json")
    for key in ("SPLIT_MANIFEST", "STATS_PATH", "ROTATION_BASIS_PATH", "SMPLH_MODEL_PATH"):
        value = getattr(cfg.DATA, key, None)
        if value:
            setattr(cfg.DATA, key, resolve(value))
    for section, key in ((cfg.MODEL, "INIT_CKPT_PATH"), (cfg.MODEL, "CKPT_PATH"), (cfg.TRAIN, "EXP_PATH")):
        value = getattr(section, key, None)
        if value:
            setattr(section, key, resolve(value))
    return cfg
