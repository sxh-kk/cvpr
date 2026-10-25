#!/usr/bin/env bash
# Use the independent conda simulation environment; no model/training service.
set -euo pipefail
task_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
task_runtime="${CVPR_SIMPLE_RUNTIME:-/home/zxc/CVPR_Evaluation_runtime}"
task_python="${PICO_SIM_PYTHON:-/home/zxc/miniconda3/envs/cvpr-simple-sim310/bin/python}"
if [[ ! -f "$task_runtime/local.env.sh" || ! -x "$task_python" ]]; then
  echo 'Missing local runtime configuration or conda simulation Python.' >&2
  exit 1
fi
source "$task_runtime/local.env.sh"
# Set before either collector or finisher Python imports native CPU libraries.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OMNI_KIT_ACCEPT_EULA=YES
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
task_output="$task_runtime/pico_human_demos/$(date +%Y%m%d_%H%M%S)"
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
  --output "$task_output" "${task_arguments[@]}"
# Run expensive RTX only after the live collector exits, preserving real time.
"$task_python" scripts/4090_simple/finish_pico_session.py "$task_output"
