#!/bin/bash
#SBATCH --job-name=distadd
#SBATCH --partition=all
#SBATCH --account=arora
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --output=outputs/%x-%j.out

# Usage: sbatch scripts/distill_addition.slurm <method> <teacher_dir> [n_digits] [lambda]
set -euo pipefail
cd /scratch/gpfs/ARORA/zc5794/superposition_distillation

METHOD=${1:-cross_seq}
TEACHER=${2:?need teacher checkpoint dir}
NDIG=${3:-4}
LAMBDA=${4:-0.7}
S1=${5:-6000}
S2=${6:-1500}
export HF_HOME=${HF_HOME:-/scratch/gpfs/ARORA/zc5794/.cache/huggingface}
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false

echo "method=$METHOD teacher=$TEACHER n_digits=$NDIG lambda=$LAMBDA node=$(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader

.venv/bin/python -u distill_addition.py \
  --teacher "$TEACHER" --method "$METHOD" \
  --n_digits "$NDIG" --fixed_lambda "$LAMBDA" \
  --stage1_steps "$S1" --stage2_steps "$S2" --batch_size 512 \
  --device cuda --eval_every 250
