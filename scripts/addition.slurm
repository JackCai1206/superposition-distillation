#!/bin/bash
#SBATCH --job-name=add10m
#SBATCH --partition=all
#SBATCH --account=arora
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=outputs/%x-%j.out

# Usage: sbatch scripts/addition.slurm [n_digits] [steps]
set -euo pipefail
cd /scratch/gpfs/ARORA/zc5794/superposition_distillation

NDIG=${1:-4}
STEPS=${2:-8000}
export HF_HOME=${HF_HOME:-/scratch/gpfs/ARORA/zc5794/.cache/huggingface}
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false

echo "n_digits=$NDIG steps=$STEPS node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

.venv/bin/python -u train_addition.py \
  --n_digits "$NDIG" --steps "$STEPS" --batch_size 512 \
  --device cuda --dtype bfloat16 --eval_every 1000
