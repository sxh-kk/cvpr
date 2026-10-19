#!/usr/bin/env bash
# Preparation entry point: no workers are launched without explicit --run.
# Source unifolm-local.env.sh first. Uses the same batch/LR/model as the baseline.
set -euo pipefail

mode="${1:---check-only}"
case "$mode" in
    --dry-run|--check-only|--run) if (( $# )); then shift; fi ;;
    *) echo "Usage: $0 [--dry-run|--check-only|--run] [experiment]" >&2; exit 2 ;;
esac
if (( $# > 1 )); then
    echo "Usage: $0 [--dry-run|--check-only|--run] [experiment]" >&2; exit 2
fi

export UNIFOLM_SPEEDUP_PROFILE=1
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
# Generate once in the parent shell so every rank uses the same output directory.
export UNIFOLM_RUN_TIMESTAMP="${UNIFOLM_RUN_TIMESTAMP:-$(date -u +%y%m%d%H%M%S)}"
recipe_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
args=()
if [[ "$mode" != --run ]]; then args+=("$mode"); fi
exec bash "$recipe_dir/posttrain-dual-source-unifolm.sh" "${args[@]}" "${1:-dual_source_unifolm_speedup}"
