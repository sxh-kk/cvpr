#!/usr/bin/env bash
# One new human demonstration per distinct official TRAIN initialization.
set -euo pipefail
task_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
task_runtime="${CVPR_SIMPLE_RUNTIME:-/home/zxc/CVPR_Evaluation_runtime}"
task_python="${PICO_SIM_PYTHON:-/home/zxc/miniconda3/envs/cvpr-simple-sim310/bin/python}"
source "$task_runtime/local.env.sh"
export OMNI_KIT_ACCEPT_EULA=YES HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
task_round="$task_runtime/mobile_pp_official99_$(date +%Y%m%d)/round_001"
task_arguments=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --round-root)
      [[ $# -ge 2 ]] || { echo 'Missing --round-root.' >&2; exit 2; }
      task_round="$2"; shift 2 ;;
    *) task_arguments+=("$1"); shift ;;
  esac
done
[[ "$task_round" = /* ]] || { echo 'Round root must be absolute.' >&2; exit 2; }
mkdir -p "$task_round/collection" "$task_round/tmp"
export TMPDIR="$task_round/tmp"
task_resources="$task_runtime/mobile_pp_teleop_20261027/resources"
task_plan="$task_round/plan/manifest.json"
task_cold_profile="$task_round/resources/mobile_pp_cold_start_profile.json"
[[ -f "$task_cold_profile" ]] || { echo 'Missing this round validated Mobile cold-start profile.' >&2; exit 2; }
cd "$task_repo"
if [[ ! -f "$task_plan" ]]; then
  task_source="$(realpath -- "$task_runtime/resources/simple_finetune_sources/G1WholebodyLocomotionPickBetweenTablesTeleop-v0/meta/episodes.jsonl")"
  "$task_python" scripts/4090_simple/pico_mobile_collection_plan.py build \
    --source-jsonl "$task_source" \
    --output-dir "$task_round/plan"
fi
task_output="$task_round/collection/live_$(date +%Y%m%d_%H%M%S)"
"$task_python" scripts/4090_simple/pico_simple_teleop.py \
  --task mobile_pp --output "$task_output" \
  --data-root "$task_resources/data_root" \
  --mobile-collection-plan "$task_plan" --mobile-collection-progress "$task_round/progress.json" \
  --mobile-cold-start "$task_cold_profile" \
  --planner-model "$task_runtime/handover_random100_20261024/official-planner-v1/planner_sonic.onnx" \
  "${task_arguments[@]}"
"$task_python" scripts/4090_simple/pico_episode_review.py \
  --index "$task_round/collection/pico_episode_history.json" --wait-session "$task_output"
echo "Raw evidence: $task_output"
echo "Official initialization progress: $task_round/progress.json"
echo 'RTX, DINO, packing and upload require the later collection-complete workflow.'
