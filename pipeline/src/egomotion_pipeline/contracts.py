"""Current read-only SMPL-H source and DINO token contracts."""

PARAMETER_SCHEMA_VERSION = "egomotion-smplh16-bodyparams-hands21-camera-floor-dino5-v1"
NATIVE_FPS = 30
DINO_CHECKPOINT = "facebook/dinov2-with-registers-large"
DINO_TOKEN_NAMES = ("cls", "register_0", "register_1", "register_2", "register_3")
DINO_SHAPE = (5, 1024)


def dino_format():
    return {
        "checkpoint": DINO_CHECKPOINT,
        "token_order": list(DINO_TOKEN_NAMES),
        "shape": list(DINO_SHAPE),
        "flat_shape": [5120],
        "flatten_order": "token_major",
        "storage_dtype": "float32",
        "pooling": "last_hidden_state[:, :5]",
        "normalization": "none_after_model",
    }
