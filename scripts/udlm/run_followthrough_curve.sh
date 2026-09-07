#!/usr/bin/env bash
set -euo pipefail
study_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$study_root"
if [[ "${1:-}" != --inside ]]; then
    if tmux has-session -t genmol-udlm-followthrough-curve 2>/dev/null; then
        echo 'Existing curve evaluation session retained; refusing duplicate launch.' >&2
        exit 1
    fi
    exec tmux new-session -d -s genmol-udlm-followthrough-curve -c "$study_root" \
        'bash scripts/udlm/run_followthrough_curve.sh --inside'
fi
[[ -n "${TMUX:-}" ]] || { echo 'Execution requires tmux.' >&2; exit 1; }
mkdir -p output/logs
set -o noclobber
exec > output/logs/followthrough-curve-pipeline.log 2>&1
unset CUDA_VISIBLE_DEVICES
export PYTHONDONTWRITEBYTECODE=1
exec /home/aidar.alimbayev/Documents/genmolv2/.venv/bin/python -u \
    scripts/udlm/run_followthrough_curve_evaluation.py
