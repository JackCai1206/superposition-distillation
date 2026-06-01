#!/bin/bash
#SBATCH --job-name=supdistill
#SBATCH --partition=all
#SBATCH --account=arora
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/%x-%j.out

# Usage: sbatch scripts/train.slurm <method> [data_kind] [stage1_steps] [stage2_steps]
#   method:    none | cross_seq | token_merge
#   data_kind: reasoning (default) | pretrain
set -euo pipefail
cd /scratch/gpfs/ARORA/zc5794/superposition_distillation

METHOD=${1:-cross_seq}
KIND=${2:-reasoning}
S1=${3:-1000}
S2=${4:-300}
LAMBDA=${5:-0.7}   # analysis: ~0.7 keeps the frozen teacher coherent (collapses at 0.5)

export HF_HOME=${HF_HOME:-/scratch/gpfs/ARORA/zc5794/.cache/huggingface}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

echo "method=$METHOD kind=$KIND s1=$S1 s2=$S2 node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

.venv/bin/python train.py \
  --method "$METHOD" --data_kind "$KIND" \
  --teacher deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --student Qwen/Qwen2.5-0.5B \
  --device cuda \
  --stage1_steps "$S1" --stage2_steps "$S2" \
  --fixed_lambda "$LAMBDA" \
  --seq_len 1024 --batch_size 8
