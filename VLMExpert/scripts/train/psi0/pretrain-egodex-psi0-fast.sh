#!/bin/bash
set -euo pipefail

PSI_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PSI_REPO_ROOT"
export PYTHONPATH="$PSI_REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PSI_PYTHON="${PSI_PYTHON:-$PSI_REPO_ROOT/.venv-psi/bin/python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
IFS=, read -ra PSI_GPUS <<< "$CUDA_VISIBLE_DEVICES"
PSI_NPROC="${NPROC_PER_NODE:-${#PSI_GPUS[@]}}"

args=(
  pretrain_egodex_qwen3vl_config model.action-tokenizer:fast
  --seed=7 --exp=global-egodex
  --train.name=pretrain --train.data_parallel=deepspeed
  --train.deepspeed_config=scripts/deepspeed/zero3.json
  --train.mixed_precision=bf16
  "--train.train_batch_size=${TRAIN_BATCH_SIZE:-16}"
  "--train.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS:-1}"
  --train.learning_rate=1e-4 --train.max_training_steps=200000
  --train.warmup_ratio=None --train.warmup_steps=0
  --train.checkpointing_steps=5000 --train.max_checkpoints_to_keep=5
  --train.validation_steps=0 --train.max_grad_norm=1.0
  --train.lr_scheduler_type=constant
  --train.lr_scheduler_kwargs.betas 0.9 0.999
  --train.lr_scheduler_kwargs.weight_decay=0.0 --train.lr_scheduler_kwargs.eps=1e-8
  --log.report_to=wandb
  --data.chunk_size=1 --data.upsample_rate=3 --data.use-delta-actions
  --data.transform.field.action_norm_type=bounds_q99
  --data.transform.field.stat-action-key=egodex
  --data.transform.field.stat-path=assets/stats/egodex_stat_all.json
  --data.transform.model.resize.size 270 480
  "--data.root-dir=${EGODEX_DATA_ROOT:-/hfm/data/egodex}"
  --data.transform.model.no-img-aug
  "--model.model_name_or_path=${QWEN_CHECKPOINT:-Qwen/Qwen3-VL-2B-Instruct}"
  --model.action_tokenizer.bins=2048
  "--model.action_tokenizer.pretrained_checkpoint=${FAST_CHECKPOINT:-src/fast/egodex-rel-50w-1x48-v2048-s100}"
  --model.tune-mm-llm --model.tune-mm-vision --model.tune-mm-mlp
  --model.mm_projector_lr=1e-5 --model.vision_tower_lr=1e-5
  --model.use-global-token --model.pretraining-stage=egodex
)

"$PSI_PYTHON" -m torch.distributed.run --nproc_per_node="$PSI_NPROC" \
  --master_port="${MASTER_PORT:-29500}" scripts/train.py "${args[@]}" "$@"
