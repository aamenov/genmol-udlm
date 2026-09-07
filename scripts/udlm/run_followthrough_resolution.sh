#!/usr/bin/env bash
set -euo pipefail

study_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
study_python=/home/aidar.alimbayev/Documents/genmolv2/.venv/bin/python
study_session=genmol-udlm-followthrough-resolution
cd "$study_root"

if [[ "${1:-}" == --dry-run ]]; then
    exec "$study_python" scripts/udlm/launch_exploration.py \
        --protocol experiments/udlm/protocols/followthrough_resolution.json \
        --output-root output/udlm/followthrough_resolution \
        --log-root output/logs/followthrough_resolution --dry-run
fi

if [[ "${1:-}" != --inside ]]; then
    "$study_python" -c 'from scripts.exps.denovo.launch_benchmark import _require_clean_pushed_source; print(_require_clean_pushed_source())'
    if tmux has-session -t "$study_session" 2>/dev/null; then
        echo "Existing resolution session retained; refusing a duplicate launch." >&2
        exit 1
    fi
    exec tmux new-session -d -s "$study_session" -c "$study_root" \
        "bash scripts/udlm/run_followthrough_resolution.sh --inside"
fi

[[ -n "${TMUX:-}" ]] || { echo 'Execution requires tmux.' >&2; exit 1; }
mkdir -p output/logs
set -o noclobber
exec > output/logs/followthrough-resolution-pipeline.log 2>&1
unset CUDA_VISIBLE_DEVICES
export PYTHONDONTWRITEBYTECODE=1
"$study_python" -u scripts/udlm/launch_exploration.py \
    --protocol experiments/udlm/protocols/followthrough_resolution.json \
    --output-root output/udlm/followthrough_resolution \
    --log-root output/logs/followthrough_resolution
CUDA_VISIBLE_DEVICES='' "$study_python" -u scripts/udlm/report_followthrough_resolution.py
