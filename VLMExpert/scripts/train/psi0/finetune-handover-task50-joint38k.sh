#!/usr/bin/env bash
# User-selected38k initialization with a fresh20k-update task schedule.
set -euo pipefail
recipe_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
repository="$(cd -- "$recipe_root/.." && pwd)"
task_root="${TASK50_ROOT:-/gaozt-test1/datasets/SIMPLE_sonic_v1_1}"
stage="$task_root/.pipeline/task50_38k_preparation"
python_bin="${TASK50_PYTHON:-$task_root/.pipeline/envs/task-finetune-py311-cu126/bin/python}"
prepared="${TASK50_RUN_DIR:-$stage/handover_joint38000_b32_s20000}"
source_run="${TASK50_SOURCE_RUN:-/gaozt-test1/datasets/UnifoLM/.pipeline/training/posttrain/dual_source_unifolm_joint_motion100_8gpu_recovery.us.flow1000.cosine.lr1.0e-04.b256.gpus8.261020043942}"
export PYTHONPATH="$recipe_root/src:$repository:$recipe_root"
export PYTHONNOUSERSITE=1
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
[[ -x "$python_bin" ]] || { echo "Missing independent Conda task environment: $python_bin" >&2; exit 1; }
case "${1:-prepare}" in
  prepare)
    exec "$python_bin" "$repository/scripts/4090_simple/a100_task_finetune.py" prepare \
      --source-run "$source_run" --source-step 38000 \
      --dataset-root "$task_root/handover_task50_30hz" \
      --dino-root "$task_root/dino/handover_task50_30hz" --output "$prepared" \
      --python "$python_bin" --weight-audit "$stage/joint38000_cpu_audit.json" \
      --max-steps 20000 --warmup-steps 1000 \
      --checkpoint-steps 4000 --validation-steps 1000 --early-checkpoint-steps ;;
  check|run)
    exec "$python_bin" "$repository/scripts/4090_simple/a100_task_finetune.py" "$1" --prepared "$prepared" ;;
  status)
    exec "$python_bin" "$repository/VLMExpert/scripts/analysis/monitor_task_finetune.py" \
      --state "$prepared/training_state.json" --output-dir "$prepared/progress" --once ;;
  *) echo "Usage: $0 [prepare|check|run|status]" >&2; exit 2 ;;
esac
