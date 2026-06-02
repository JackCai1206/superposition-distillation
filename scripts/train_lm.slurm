#!/bin/bash
#SBATCH --job-name=lm_teacher
#SBATCH --partition=all
#SBATCH --account=arora
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=outputs/%x-%j.out
set -uo pipefail
cd /scratch/gpfs/ARORA/zc5794/superposition_distillation
export HF_HOME=${HF_HOME:-/scratch/gpfs/ARORA/zc5794/.cache/huggingface}
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
STEPS=${1:-6000}
echo "node=$(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader
.venv/bin/python -u train_lm.py --steps "$STEPS" --hidden 512 --layers 8 --heads 8 \
  --inter 2048 --batch_size 64 --seq_len 512 --lr 6e-4 --eval_every 500 --device cuda
