#!/usr/bin/env bash
# New action head + trainable EgoDex VLM/MotionExpert. Does not modify the baseline recipe.
# Required: EGO_DEX_VLM_PATH, MOTION_EXPERT_CHECKPOINT, UNIFOLM_DINO_ROOT,
#           UNIFOLM_DINO_TRAIN_ID, UNIFOLM_DINO_VAL_ID, UNIFOLM_DINO_CACHE.
# Cache packs are native30Hz offline DINO, using the same canonical256x320 profile.
# Only the VLM current frame receives Psi0 color augmentation; Motion keeps clean history.
# Review without starting workers: bash "$0" --dry-run [experiment]
set -euo pipefail

dry_run=0
if [[ "${1:-}" == "--dry-run" ]]; then dry_run=1; shift; fi
experiment="${1:-dual_source_unifolm}"
if (( $# > 1 )); then echo "Usage: $0 [--dry-run] [experiment]" >&2; exit 2; fi

: "${EGO_DEX_VLM_PATH:?Set the explicit EgoDex-pretrained VLM directory}"
: "${MOTION_EXPERT_CHECKPOINT:?Set an explicit compatible MotionExpert checkpoint file}"
: "${UNIFOLM_DINO_ROOT:?Set the UnifoLM data root, e.g. /gaozt-test1/datasets/UnifoLM}"
: "${UNIFOLM_DINO_TRAIN_ID:?Set the original training pack ID}"
: "${UNIFOLM_DINO_VAL_ID:?Set the original validation pack ID}"
: "${UNIFOLM_DINO_CACHE:?Set the completed native30Hz DINO cache root with per-pack subdirectories}"

recipe_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$recipe_root"
export PYTHONPATH="$recipe_root/src:$recipe_root/..:$recipe_root${PYTHONPATH:+:$PYTHONPATH}"
python_bin="${PSI_PYTHON:-python3}"
train_stats="$UNIFOLM_DINO_ROOT/$UNIFOLM_DINO_TRAIN_ID/meta/stats_psi0.json"
# Request EMA initialization when available by default; set MOTION_USE_EMA=0 for raw weights.
case "${MOTION_USE_EMA:-1}" in
    1|true) motion_ema_arg=--model.motion-use-ema ;;
    0|false) motion_ema_arg=--model.no-motion-use-ema ;;
    *) echo "MOTION_USE_EMA must be 0/1 or false/true" >&2; exit 2 ;;
esac

if (( ! dry_run )); then
    [[ -d "$EGO_DEX_VLM_PATH" ]] || { echo "Missing VLM directory: $EGO_DEX_VLM_PATH" >&2; exit 1; }
    [[ -f "$MOTION_EXPERT_CHECKPOINT" ]] || { echo "Missing MotionExpert checkpoint: $MOTION_EXPERT_CHECKPOINT" >&2; exit 1; }
    for pack_id in "$UNIFOLM_DINO_TRAIN_ID" "$UNIFOLM_DINO_VAL_ID"; do
        [[ -s "$UNIFOLM_DINO_CACHE/$pack_id/manifest.json" ]] || {
            echo "Missing completed DINO cache manifest: $UNIFOLM_DINO_CACHE/$pack_id/manifest.json" >&2; exit 1;
        }
        "$python_bin" - "$UNIFOLM_DINO_CACHE/$pack_id/manifest.json" <<'PY'
import json
from pathlib import Path
import sys
path = Path(sys.argv[1])
if json.loads(path.read_text()).get("state") != "complete":
    raise SystemExit(f"Incomplete DINO cache: {path}; finish extraction before training")
PY
        for metadata in modality.json stats_psi0.json; do
            [[ -s "$UNIFOLM_DINO_ROOT/$pack_id/meta/$metadata" ]] || {
                echo "Missing pack metadata: $pack_id/meta/$metadata" >&2; exit 1;
            }
        done
    done
    cmp -s "$train_stats" "$UNIFOLM_DINO_ROOT/$UNIFOLM_DINO_VAL_ID/meta/stats_psi0.json" || {
        echo "Train and validation packs must share stats_psi0.json" >&2; exit 1;
    }
fi

command=("$python_bin" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE:-1}"
    scripts/train.py posttrain_sonic_psi0_config
    --seed=292285 "--exp=$experiment" --train.name=posttrain
    --train.data_parallel=ddp --train.mixed_precision=bf16
    "--train.num_workers=${NUM_WORKERS:-8}" "--train.train_batch_size=${BATCH_SIZE:-16}"
    "--train.gradient_accumulation_steps=${GRAD_ACCUM_STEPS:-2}"
    --train.learning_rate=1e-4 "--train.max_training_steps=${MAX_STEPS:-40000}"
    --train.warmup_ratio=None --train.warmup_steps=1000
    --train.max_checkpoints_to_keep=5 --train.checkpointing_steps=2000
    --train.validation_steps=5000 --train.val_num_batches=20 --train.max_grad_norm=1.0
    --train.lr_scheduler_type=cosine --train.lr_scheduler_kwargs.weight_decay=1e-6
    --train.lr_scheduler_kwargs.betas 0.95 0.999
    "--log.report_to=${REPORT_TO:-wandb}"
    "--data.root_dir=$UNIFOLM_DINO_ROOT" "--data.train_repo_ids=$UNIFOLM_DINO_TRAIN_ID"
    "--data.val_repo_ids=$UNIFOLM_DINO_VAL_ID"
    "--data.dino-cache-root=$UNIFOLM_DINO_CACHE" --data.shared-image-profile=256x320
    --data.transform.repack.dataset-name=unifolm_sonic
    --data.transform.repack.image-keys observation.images.egocentric
    --data.transform.repack.instruction-key=task_description
    --data.transform.repack.state-keys observation.state
    --data.transform.repack.action-keys action.body_token_v1_1 'action[:14]'
    --data.transform.repack.action-chunk-size=30
    --data.transform.repack.pad-action-dim=80 --data.transform.repack.pad-state-dim=45
    "--data.transform.field.stat-path=$train_stats"
    --data.transform.field.stat-state-keys observation.state
    --data.transform.field.stat-action-keys action.body_token_v1_1 'action[:14]'
    --data.transform.field.action_norm_type=bounds --data.transform.field.normalize-state
    --data.transform.field.pad-action-dim=80 --data.transform.field.pad-state-dim=45
    --data.transform.model.motion-image-policy=vlm_color_clean_motion
    --data.transform.model.img-aug --data.transform.model.no-view-aug
    --model.action-architecture=dual_source_v1
    "--model.model_name_or_path=$EGO_DEX_VLM_PATH"
    "--model.motion-checkpoint=$MOTION_EXPERT_CHECKPOINT"
    "$motion_ema_arg"
    --model.pretrained-action-header-path=None
    --model.tune-vlm --model.tune-motion --model.motion-feature-seed=0
    # Motion LR is our initial choice; original Psi0 has no MotionExpert branch.
    --model.motion-expert-lr=1e-5
    --model.lang-backbone-lr=1e-6 --model.vision-tower-lr=1e-5 --model.mm-projector-lr=1e-4
    --model.noise-scheduler=flow --model.train-diffusion-steps=1000 --model.n_conditions=0
    --model.action-chunk-size=30 --model.action-dim=80 --model.action-exec-horizon=30
    --model.observation-horizon=1 --model.odim=45 --model.num-blocks=6
    --model.dropout=0.0 --model.state-feature-dropout=0.0 --model.view_feature_dim=2048
    --model.gradient-checkpointing --model.qk-norm=rms_norm
    --model.no-use-film --model.no-use-dit --model.no-combined-temb
    --model.vlm-layer-indices=None --model.pooled-text-encoder=None
    --model.state-drop-prob=0.5 --model.state-null-token --model.no-rtc)

printf 'Command:'
printf ' %q' "${command[@]}"
printf '\n'
if (( dry_run )); then exit 0; fi
exec "${command[@]}"
