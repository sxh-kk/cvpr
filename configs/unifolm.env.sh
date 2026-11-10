#!/usr/bin/env bash
# Default paths loaded by the joint training launcher; edit as needed.
export CVPR_DATA_ROOT="${CVPR_DATA_ROOT:-${HOME}/sigma0/data}"
export CVPR_MODEL_ROOT="${CVPR_MODEL_ROOT:-${HOME}/sigma0/models}"
export CVPR_RUN_ROOT="${CVPR_RUN_ROOT:-${HOME}/sigma0/runs}"

export PSI_HOME="${PSI_HOME:-${CVPR_RUN_ROOT}/psi}"
export DATA_HOME="${DATA_HOME:-${CVPR_DATA_ROOT}}"
export HF_HOME="${HF_HOME:-${CVPR_RUN_ROOT}/cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${CVPR_RUN_ROOT}/cache/torch}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${CVPR_RUN_ROOT}/cache/uv}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${CVPR_DATA_ROOT}/lerobot}"

# VLM weights and the Motion training output used for initialization.
export EGO_DEX_VLM_PATH="${EGO_DEX_VLM_PATH:-${CVPR_MODEL_ROOT}/egodex-vlm}"
export MOTION_EXPERT_CHECKPOINT="${MOTION_EXPERT_CHECKPOINT:-${CVPR_RUN_ROOT}/motion/smplh_k12_psi0_256x480_5hz_camera_quality_v4_meanpool_from_scratch/final.ckpt}"
export MOTION_USE_EMA="${MOTION_USE_EMA:-1}"

export UNIFOLM_DINO_ROOT="${UNIFOLM_DINO_ROOT:-${CVPR_DATA_ROOT}/UnifoLM}"
export UNIFOLM_DINO_TRAIN_ID="${UNIFOLM_DINO_TRAIN_ID:-unifolm_sonic_lerobot_train}"
export UNIFOLM_DINO_VAL_ID="${UNIFOLM_DINO_VAL_ID:-unifolm_sonic_lerobot_val}"
export UNIFOLM_DINO_CACHE="${UNIFOLM_DINO_CACHE:-${UNIFOLM_DINO_ROOT}/dino/256x320}"
export UNIFOLM_RUN_ROOT="${UNIFOLM_RUN_ROOT:-${CVPR_RUN_ROOT}/unifolm}"
# Use the active Python 3.11 environment, or set PSI_PYTHON explicitly.
export PSI_PYTHON="${PSI_PYTHON:-python3}"
