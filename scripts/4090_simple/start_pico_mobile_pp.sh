#!/usr/bin/env bash
# Separate Mobile P&P raw collection. Never post-process it with Handover gates.
set -euo pipefail
task_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
task_runtime="${CVPR_SIMPLE_RUNTIME:-/home/zxc/CVPR_Evaluation_runtime}"
task_python="${PICO_SIM_PYTHON:-/home/zxc/miniconda3/envs/cvpr-simple-sim310/bin/python}"
if [[ ! -f "$task_runtime/local.env.sh" || ! -x "$task_python" ]]; then
  echo 'Missing local runtime configuration or conda simulation Python.' >&2
  exit 1
fi
source "$task_runtime/local.env.sh"
export OMNI_KIT_ACCEPT_EULA=YES HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
task_resources="$task_runtime/mobile_pp_teleop_20261027/resources"
task_output="$task_runtime/mobile_pp_teleop_20261027/collection/live_$(date +%Y%m%d_%H%M%S)"
task_arguments=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output)
      [[ $# -ge 2 ]] || { echo 'Missing --output path.' >&2; exit 2; }
      task_output="$2"; shift 2 ;;
    --output=*) task_output="${1#--output=}"; shift ;;
    *) task_arguments+=("$1"); shift ;;
  esac
done
cd "$task_repo"
"$task_python" scripts/4090_simple/pico_simple_teleop.py \
  --task mobile_pp --output "$task_output" \
  --data-root "$task_resources/data_root" \
  --environment-config "$task_resources/official_train_episode000_reset_environment_config.json" \
  --mobile-cold-start "$task_resources/mobile_pp_cold_start_profile.json" \
  --planner-model "$task_runtime/handover_random100_20261024/official-planner-v1/planner_sonic.onnx" \
  "${task_arguments[@]}"
"$task_python" scripts/4090_simple/pico_episode_review.py \
  --index "$(dirname -- "$task_output")/pico_episode_history.json" --wait-session "$task_output"
echo "Mobile P&P raw evidence sealed: $task_output"
echo 'RTX rendering and independent Mobile data admission remain pending; no training upload was started.'
