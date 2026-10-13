import os
from pathlib import Path
from types import SimpleNamespace


def _ns(**values):
    return SimpleNamespace(**values)


ROOT = Path(__file__).resolve().parent
K12_JOINTS = [0, 4, 5, 10, 11, 13, 14, 15, 18, 19, 20, 21]
LEGACY_ARCHITECTURE = "e7_k12_400m_text_v1"
GLOBAL_ARCHITECTURE = "e7_k12_400m_global_text_v1"


def motion_expert_config(*, global_token=False):
    """Return the E7-K12-400M text-conditioned architecture."""
    data = _ns(
        REPRE_TYPE="v4_beta",
        COND_BETAS=False,
        COND_IMG_FEAT=True,
        COND_TRAJ=True,
        IMG_FEAT_TYPE="dinov2",
        WINDOW=80,
        COND_TEXT=True,
        TEXT_FEATURE_DIM=768,
        TEXT_TOKEN_COUNT=16,
        TEXT_DROPOUT=0.10,
    )
    sparse = _ns(
        ENABLED=True,
        INDICES=list(K12_JOINTS),
    )
    expert = _ns(
        ENABLED=True,
        NUM_EXPERTS=12,
        NUM_ROUTED_EXPERTS=11,
        TOP_K=2,
        ROUTER_JITTER=0.0,
        SHARED_EXPERT=True,
        CHUNK_SIZE=4,
        CONDITIONED_ROUTER=True,
        ROUTED_GATE_INIT=0.05,
    )
    model = _ns(
        MODEL_NAME="uem",
        CKPT_PATH=None,
        GENERATIVE_TYPE="flow",
        LEARN_TRAJ=False,
        ENCODER_TSFM=None,
        TEXT_CONDITION=_ns(
            ENABLED=True, TOKEN_DIM=768, NUM_TOKENS=16,
            ROUTER_ENABLED=False, CROSS_ATTENTION=True, ADALN=False,
        ),
        MOTION_EXPERT=expert,
        OUTPUT_BRANCH_MODE="single",
        ZERO_MASK_TOKEN=False,
        PREDICT_XSTART=True,
        GLOBAL_TOKEN_COUNT=1 if global_token else 0,
    )
    flow = _ns(
        NUM_STEPS=10,
        SOLVER="euler",
        BETA_ALPHA=1.5,
        BETA_BETA=1.0,
        T_MIN=0.001,
        PREDICTION_TYPE="x0",
        GLOBAL_WEIGHT=8.0,
        GLOBAL_FEATURE_START=108,
        GLOBAL_FEATURE_END=117,
    )
    train = _ns(
        USE_CKPT_LR=True, FUSED_ADAMW=False,
        SCHEDULER="cosine_warmup", SCHEDULER_TOTAL_EPOCHS=300,
        MAX_STEPS=-1,
    )
    return _ns(DATA=data, SPARSE_JOINTS=sparse, MODEL=model, FLOW=flow, TRAIN=train)


def validate_architecture(cfg):
    count = cfg.MODEL.GLOBAL_TOKEN_COUNT
    if type(count) is not int or count not in (0, 1):
        raise ValueError("E7 GLOBAL_TOKEN_COUNT must be 0 (legacy) or 1")
    expected = {
        "DATA.REPRE_TYPE": "v4_beta", "DATA.COND_BETAS": False,
        "DATA.COND_IMG_FEAT": True, "DATA.COND_TRAJ": True,
        "DATA.IMG_FEAT_TYPE": "dinov2", "DATA.COND_TEXT": True,
        "DATA.TEXT_FEATURE_DIM": 768, "DATA.TEXT_TOKEN_COUNT": 16,
        "SPARSE_JOINTS.ENABLED": True, "SPARSE_JOINTS.INDICES": K12_JOINTS,
        "MODEL.MODEL_NAME": "uem", "MODEL.GENERATIVE_TYPE": "flow",
        "MODEL.LEARN_TRAJ": False, "MODEL.ENCODER_TSFM": None,
        "MODEL.OUTPUT_BRANCH_MODE": "single", "MODEL.ZERO_MASK_TOKEN": False,
        "MODEL.TEXT_CONDITION.ENABLED": True, "MODEL.TEXT_CONDITION.TOKEN_DIM": 768,
        "MODEL.TEXT_CONDITION.NUM_TOKENS": 16, "MODEL.TEXT_CONDITION.ROUTER_ENABLED": False,
        "MODEL.TEXT_CONDITION.CROSS_ATTENTION": True, "MODEL.TEXT_CONDITION.ADALN": False,
        "MODEL.MOTION_EXPERT.ENABLED": True, "MODEL.MOTION_EXPERT.NUM_EXPERTS": 12,
        "MODEL.MOTION_EXPERT.NUM_ROUTED_EXPERTS": 11, "MODEL.MOTION_EXPERT.TOP_K": 2,
        "MODEL.MOTION_EXPERT.SHARED_EXPERT": True, "MODEL.MOTION_EXPERT.CHUNK_SIZE": 4,
        "MODEL.MOTION_EXPERT.CONDITIONED_ROUTER": True,
        "FLOW.PREDICTION_TYPE": "x0", "FLOW.SOLVER": "euler", "FLOW.NUM_STEPS": 10,
        "FLOW.GLOBAL_FEATURE_START": 108, "FLOW.GLOBAL_FEATURE_END": 117, "FLOW.GLOBAL_WEIGHT": 8.0,
    }
    for path, value in expected.items():
        actual = cfg
        for key in path.split("."):
            actual = getattr(actual, key, None)
        if actual != value:
            raise ValueError(f"E7 K12 400M requires {path}={value!r}, got {actual!r}")


def architecture_metadata(cfg):
    validate_architecture(cfg)
    count = cfg.MODEL.GLOBAL_TOKEN_COUNT
    return {
        "architecture": GLOBAL_ARCHITECTURE if count else LEGACY_ARCHITECTURE,
        "global_token_count": count,
        "num_prefix_tokens": 1 + count,
        "hidden_dim": 768,
        "motion_dim": 153,
        "joint_indices": list(cfg.SPARSE_JOINTS.INDICES),
        "feature_t": 1.0,
        "prediction_type": cfg.FLOW.PREDICTION_TYPE,
        "num_steps": cfg.FLOW.NUM_STEPS,
        "solver": cfg.FLOW.SOLVER,
    }


def as_dict(cfg):
    if isinstance(cfg, SimpleNamespace):
        return {key: as_dict(value) for key, value in vars(cfg).items()}
    if isinstance(cfg, dict):
        return {key: as_dict(value) for key, value in cfg.items()}
    if isinstance(cfg, (list, tuple)):
        return [as_dict(value) for value in cfg]
    return cfg


def config_from_dict(values):
    cfg = motion_expert_config()
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

    with open(path or ROOT / "configs/e7_k12_400m_global_text.yaml", encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    if not isinstance(values, dict):
        raise ValueError("Configuration must be a YAML mapping")
    cfg = config_from_dict(values)
    root = Path(data_dir or os.environ.get("MOTIONEXPERT_DATA_DIR", cfg.DATA.DATA_DIR)).expanduser()
    cfg.DATA.DATA_DIR = str((ROOT / root).resolve())
    for key in ("EE4D_PACK", "EE4D_DINO_PACK", "NYMERIA_PACK", "NYMERIA_DINO_PACK", "TEXT_FEATURES_PATH", "STATS_PATH", "SEQUENCE_MANIFEST"):
        value = getattr(cfg.DATA, key, None)
        if value:
            setattr(cfg.DATA, key, str(Path(cfg.DATA.DATA_DIR) / Path(value).expanduser()))
    for section, key in ((cfg.MODEL, "INIT_CKPT_PATH"), (cfg.MODEL, "CKPT_PATH"), (cfg.TRAIN, "EXP_PATH")):
        value = getattr(section, key, None)
        if value:
            setattr(section, key, str((ROOT / Path(value).expanduser()).resolve()))
    return cfg
