# Source after conda activate. Runtime files and resources live outside the checkout.
export CVPR_SIMPLE_RUNTIME="${CVPR_SIMPLE_RUNTIME:-${HOME}/CVPR_Evaluation_runtime}"
export PYTHONNOUSERSITE=1
export HF_HOME="${CVPR_SIMPLE_RUNTIME}/cache/huggingface"
export TORCH_HOME="${CVPR_SIMPLE_RUNTIME}/cache/torch"
export TORCH_EXTENSIONS_DIR="${CVPR_SIMPLE_RUNTIME}/cache/torch_extensions"
export UV_CACHE_DIR="${CVPR_SIMPLE_RUNTIME}/cache/uv"
export XDG_CACHE_HOME="${CVPR_SIMPLE_RUNTIME}/cache/xdg"
export SIMPLE_DATA_DIR="${CVPR_SIMPLE_RUNTIME}/resources/simple"
export SIMPLE_ISAAC_EXPERIENCE="${CVPR_SIMPLE_RUNTIME}/apps/simple4090.kit"
export SIMPLE_ISAAC_PRELOAD_PINOCCHIO=1
export CVPR_JOINT_20K_DIR="${CVPR_SIMPLE_RUNTIME}/resources/joint20k"
export CVPR_DINO_DIR="${CVPR_SIMPLE_RUNTIME}/resources/dinov2-with-registers-large"
export CVPR_SONIC_DIR="${CVPR_SIMPLE_RUNTIME}/resources/sonic-body-token-v1.1"
export CVPR_MODEL_HOST=127.0.0.1
export CVPR_MODEL_PORT="${CVPR_MODEL_PORT:-18765}"
export MUJOCO_GL=egl
# Full offline evaluation additionally needs populated Isaac extension/material caches.
# Enable HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 only once resources are present.
