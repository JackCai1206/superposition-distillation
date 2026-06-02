#!/bin/bash
#SBATCH --job-name=packlm
#SBATCH --partition=all
#SBATCH --account=arora
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:50:00
#SBATCH --output=outputs/%x-%j.out

# Pack several TinyStories distill runs onto ONE GPU.
# Usage: sbatch pack_lm.slurm <teacher> "m:lam:s1:s2:seed" ...
set -uo pipefail
cd /scratch/gpfs/ARORA/zc5794/superposition_distillation
TEACHER=$1; shift
export HF_HOME=${HF_HOME:-/scratch/gpfs/ARORA/zc5794/.cache/huggingface}
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "packing $# NL runs on $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
pids=()
for spec in "$@"; do
  IFS=: read -r m lam s1 s2 seed <<< "$spec"
  s2=${s2:-3000}; seed=${seed:-0}
  echo "  launch $m lam=$lam s1=$s1 s2=$s2 seed=$seed"
  .venv/bin/python -u distill_lm.py --teacher "$TEACHER" --method "$m" \
    --fixed_lambda "$lam" --stage1_steps "$s1" --stage2_steps "$s2" --seed "$seed" \
    --batch_size 32 --seq_len 512 --device cuda --eval_every 100 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "PACK COMPLETE (fail=$fail)"; exit $fail
