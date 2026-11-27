#!/bin/bash
#SBATCH --job-name=allnom-final-315
#SBATCH --partition=short-simple
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

set -euo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
runs="$HOME/allnomalous-runs"
run="$runs/final_eval_315"
python="$runs/venv/bin/python"
cd "$run"
sha256sum -c SHA256SUMS
cd code
"$python" -m unittest discover -s tests -p test_anomaly_metrics.py
"$python" -m unittest discover -s tests -p test_evaluate_final.py
args=(
    --data-root "$runs/allnomalous"
    --output-dir "$run/results"
    --baseline-checkpoint "$run/baseline.pt" --baseline-training-metadata "$run/baseline.json"
    --complete-checkpoint "$run/complete.pt" --complete-training-metadata "$run/complete.json"
    --expected-complete-epoch 315
    --expected-complete-sha256 "$(awk '$2=="complete.pt"{print $1}' "$run/SHA256SUMS")"
    --expected-baseline-sha256 "$(awk '$2=="baseline.pt"{print $1}' "$run/SHA256SUMS")"
    --device cuda --batch-size 2 --num-workers 2
)
"$python" -u scripts/evaluate_final.py "${args[@]}" --preflight
"$python" -u scripts/evaluate_final.py "${args[@]}" --confirm-final-test
