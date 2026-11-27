#!/bin/bash
#SBATCH --job-name=allnom-complete
#SBATCH --partition=long-complex
#SBATCH --qos=complex
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00

set -euo pipefail
umask 077
runs="$HOME/allnomalous-runs"
output="$runs/checkpoints_train_complete_1000"
repo="$runs/allnomalous"
mkdir -p "$output"
exec 9>"$output/training.lock"
flock -n 9 || { echo "A training process already holds $output/training.lock" >&2; exit 1; }
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
exec "$runs/venv/bin/python" -u "$output/train_apuana_complete.py" "$repo" "$output"
