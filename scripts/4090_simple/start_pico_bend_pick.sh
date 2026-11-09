#!/usr/bin/env bash
# PICO full-body SMPL mode2 + SONIC, explicit native BendPick TRAIN100 adapter.
set -euo pipefail
task_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
task_runtime="${CVPR_SIMPLE_RUNTIME:-/home/zxc/CVPR_Evaluation_runtime}"
task_python="${PICO_SIM_PYTHON:-/home/zxc/miniconda3/envs/cvpr-simple-sim310/bin/python}"
source "$task_runtime/local.env.sh"
export OMNI_KIT_ACCEPT_EULA=YES HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
task_round=""
task_resources=""
task_plan=""
task_arguments=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --round-root|--data-root|--collection-plan)
      [[ $# -ge 2 ]] || { echo "Missing $1 value." >&2; exit 2; }
      case "$1" in
        --round-root) task_round="$2" ;;
        --data-root) task_resources="$2" ;;
        --collection-plan) task_plan="$2" ;;
      esac
      shift 2 ;;
    *) task_arguments+=("$1"); shift ;;
  esac
done
[[ "$task_round" = /* && "$task_resources" = /* && "$task_plan" = /* ]] || {
  echo 'Specify absolute --round-root, --data-root and --collection-plan.' >&2; exit 2;
}
cd "$task_repo"
"$task_python" scripts/4090_simple/pico_bend_pick_collection_plan.py check --plan "$task_plan"
mkdir -p "$task_round/collection" "$task_round/tmp"
export TMPDIR="$task_round/tmp"
task_output="$task_round/collection/live_$(date +%Y%m%d_%H%M%S)"
"$task_python" scripts/4090_simple/pico_simple_teleop.py \
  --task bend_pick --output "$task_output" --data-root "$task_resources" \
  --bend-pick-collection-plan "$task_plan" --bend-pick-collection-progress "$task_round/progress.json" \
  --evidence-max-pending 1024 "${task_arguments[@]}"
"$task_python" scripts/4090_simple/pico_episode_review.py \
  --index "$task_round/collection/pico_episode_history.json" --wait-session "$task_output"
echo "Raw BendPick evidence: $task_output"
echo "Official initialization progress: $task_round/progress.json"
echo 'Raw capture only; deferred RTX, independent admission, DINO and upload remain separate.'
