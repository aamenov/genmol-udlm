#!/usr/bin/env bash
set -euo pipefail
study_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
study_python=/home/aidar.alimbayev/Documents/genmolv2/.venv/bin/python
study_arm="${1:-}"
[[ "$study_arm" == ct || "$study_arm" == ce ]] || { echo 'Specify ct or ce.' >&2; exit 1; }
cd "$study_root"
if [[ "${2:-}" != --inside ]]; then
    for arm in "$study_arm" mdlm; do
        "$study_python" scripts/udlm/launch_followthrough_learning.py --arm "$arm" --gpu-count 2 --dry-run > /dev/null
    done
    if tmux has-session -t genmol-udlm-followthrough-learning-r1 2>/dev/null; then
        echo 'Existing learning-curve session retained; refusing duplicate launch.' >&2
        exit 1
    fi
    exec tmux new-session -d -s genmol-udlm-followthrough-learning-r1 -c "$study_root" \
        "bash scripts/udlm/run_followthrough_learning.sh $study_arm --inside"
fi
[[ -n "${TMUX:-}" ]] || { echo 'Execution requires tmux.' >&2; exit 1; }
mkdir -p output/logs
set -o noclobber
exec > output/logs/followthrough-learning-r1-pipeline.log 2>&1
unset CUDA_VISIBLE_DEVICES
export PYTHONDONTWRITEBYTECODE=1
for arm in "$study_arm" mdlm; do
    "$study_python" -u scripts/udlm/launch_followthrough_learning.py --arm "$arm" --gpu-count 2
done
echo '{"event":"both_learning_arms_completed","evaluation_pending":true}'
