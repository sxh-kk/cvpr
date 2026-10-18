#!/usr/bin/env bash
# Source this file from the repository root. It only exports the fixed local inputs;
# it does not create an environment, load models, or start training.

export EGO_DEX_VLM_PATH='/gaozt-test1/guanzerong/Psi0/cache/checkpoints/psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k'
export MOTION_EXPERT_CHECKPOINT='/gaozt-test1/datasets/MixEgoMotion/training/smplh_k12_psi0_256x480_5hz_camera_quality_v4_meanpool_from_scratch/epoch=99-step=48700.ckpt'
export MOTION_USE_EMA=1

export UNIFOLM_DINO_ROOT='/gaozt-test1/datasets/UnifoLM'
export UNIFOLM_DINO_TRAIN_ID='unifolm_sonic_lerobot_train'
export UNIFOLM_DINO_VAL_ID='unifolm_sonic_lerobot_val'
export UNIFOLM_DINO_CACHE='/gaozt-test1/datasets/UnifoLM/dino/256x320'
export PSI_PYTHON='/gaozt-test1/datasets/UnifoLM/.pipeline/runtimes/psi0-py311-torch270-cu126/bin/python'
export UNIFOLM_RUN_ROOT='/gaozt-test1/datasets/UnifoLM/.pipeline/training'

# Actual local payload identities from the initialization acceptance record.
# The VLM SHA differs from the legacy HF download sidecar; retain the selected
# local file and do not describe this SHA as an official remote verification.
export UNIFOLM_INIT_WEIGHTS_RECORD='/gaozt-test1/datasets/UnifoLM/.pipeline/posttrain_initialization_weights.json'
export EGO_DEX_VLM_SHA256='db31e5dd953f68febc11330f5a436a37d3bd02b735e206503f2cf000faad616c'
export EGO_DEX_VLM_SIZE_BYTES=4262742488
export EGO_DEX_VLM_MTIME_NS=1786858566379009472
export MOTION_EXPERT_SHA256='e627cddede53588a7adfd5dba2a521f635927e62813798e9cf0502c994951528'
export MOTION_EXPERT_SIZE_BYTES=6415019251
export MOTION_EXPERT_MTIME_NS=1792211661700937107
