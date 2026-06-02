#!/bin/bash
#SBATCH --job-name=packdist
#SBATCH --partition=all
#SBATCH --account=arora
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=outputs/%x-%j.out

# Pack several distill runs onto ONE GPU (tiny models share it fine).
# Usage: sbatch pack_distill.slurm <teacher> <ndig> "m:lam:s1:s2:seed" ...
#   spec = method:lambda:stage1:stage2:seed   (s2 default 3000, seed default 0)
set -uo pipefail
cd /scratch/gpfs/ARORA/zc5794/superposition_distillation
TEACHER=$1; NDIG=$2; shift 2
export HF_HOME=${HF_HOME:-/scratch/gpfs/ARORA/zc5794/.cache/huggingface}
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false

echo "packing $# runs on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
pids=()
for spec in "$@"; do
  IFS=: read -r m lam s1 s2 seed <<< "$spec"
  s2=${s2:-3000}; seed=${seed:-0}
  echo "  launch $m lam=$lam s1=$s1 s2=$s2 seed=$seed"
  .venv/bin/python -u distill_addition.py --teacher "$TEACHER" --method "$m" \
    --n_digits "$NDIG" --fixed_lambda "$lam" --stage1_steps "$s1" --stage2_steps "$s2" \
    --seed "$seed" --batch_size 512 --device cuda --eval_every 50 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "PACK COMPLETE (fail=$fail)"
exit $fail
