#!/bin/bash

set -euo pipefail

PSI_VENV=${PSI_VENV:-.venv-psi}
source "${PSI_VENV}/bin/activate"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export TF_CPP_MIN_LOG_LEVEL=${TF_CPP_MIN_LOG_LEVEL:-3}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

NPROC_PER_NODE=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

# Keep the split explicit.  WBT metadata commonly exposes only a train split,
# so validation repositories must be supplied separately and never default to
# the first training repository.  Override these lists for a local split.
WBT_ROOT=${WBT_ROOT:-/hfm/data/WBT}
WBT_TRAIN_REPOS=${WBT_TRAIN_REPOS:-"G1_WBT_Brainco_Make_The_Bed G1_WBT_Brainco_Pick_Up_Medicine G1_WBT_Inspire_Clean_The_Living_Room G1_WBT_Inspire_Collect_Clothes_MainCamOnly G1_WBT_Inspire_Pick_Up_Drinks G1_WBT_Inspire_Pickup_Pillow_MainCamOnly G1_WBT_Inspire_Put_Clothes_Into_Basket G1_WBT_Inspire_Put_Clothes_into_Washing_Machine G1_WBT_Inspire_Put_Clothes_into_Washing_Machine_MainCamOnly G1_WBT_Inspire_Put_Drinks_Into_Fridge G1_WBT_Inspire_Put_Vegetables_Into_Basket"}
WBT_VAL_REPOS=${WBT_VAL_REPOS:-"G1_WBT_Brainco_Collect_Plates_Into_Dishwasher G1_WBT_Brainco_Pickup_Pillow"}
PSI0_VLM_CHECKPOINT=${PSI0_VLM_CHECKPOINT:-/hfm/cache/checkpoints/psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k}

args="posttrain_wbt_psi0_config \
--seed=292285 \
--exp=posttrain-wbt-native \
--timestamp=$(date +"%y%m%d%H%M") \
--train.name=posttrain \
--train.data-parallel=ddp \
--train.mixed-precision=bf16 \
--train.train-batch-size=64 \
--train.max-checkpoints-to-keep=5 \
--train.gradient-accumulation-steps=1 \
--train.learning-rate=1e-4 \
--train.max-training-steps=1000000 \
--train.warmup-ratio=None \
--train.warmup-steps=1000 \
--train.checkpointing-steps=100 \
--train.validation-steps=1000 \
--train.val-num-batches=20 \
--train.max-grad-norm=1.0 \
--train.lr-scheduler-type=constant \
--train.lr-scheduler-kwargs.betas 0.9 0.999 \
--train.lr-scheduler-kwargs.weight-decay=0.0 \
--train.lr-scheduler-kwargs.eps=1e-8 \
--log.report-to=wandb \
--data.root-dir=${WBT_ROOT} \
--data.train-repo-ids ${WBT_TRAIN_REPOS} \
--data.val-repo-ids ${WBT_VAL_REPOS} \
--data.camera-key=observation.images.head_stereo_left \
--data.hand-protocol=wbt-source-12d-v1 \
--data.action-chunk-size=30 \
--data.expected-fps=30 \
--data.no-use-delta-actions \
--data.validate-data \
--data.transform.repack.protocol=wbt-native-v1 \
--data.transform.repack.hand-protocol=wbt-source-12d-v1 \
--data.transform.repack.action-chunk-size=30 \
--data.transform.repack.expected-fps=30 \
--data.transform.repack.no-use-delta-actions \
--data.transform.repack.preserve-raw-hand \
--data.transform.repack.quaternion-tolerance=0.05 \
--data.transform.field.action-norm-type=bounds_q99 \
--data.transform.field.stat-path=assets/stats/wbt_native_stats.json \
--data.transform.field.no-use-norm-mask \
--data.transform.field.normalize-state \
--data.transform.model.resize.size=224 \
--data.transform.model.center-crop.size=224 \
--data.transform.model.no-img-aug \
--model.model-name-or-path=${PSI0_VLM_CHECKPOINT} \
--model.noise-scheduler=flow \
--model.n-conditions=0 \
--model.action-chunk-size=30 \
--model.action-dim=48 \
--model.action-exec-horizon=30 \
--model.observation-horizon=1 \
--model.odim=48 \
--model.no-tune-vlm \
--model.no-use-film \
--model.no-combined-temb"

torchrun --nproc_per_node="$NPROC_PER_NODE" --master_port="${MASTER_PORT:-29501}" scripts/train.py \
    ${args}
