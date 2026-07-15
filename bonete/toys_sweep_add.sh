#!/usr/bin/env bash
# Addition 2-D sweep (clean pure-KL): cross_seq over S1-FRACTION x LAMBDA x seeds,
# plus the `none` baseline. Fixed TOTAL steps so varying S1 fraction isolates "how
# much superposition" at ~constant compute. Finds where (if anywhere) cross_seq's
# iso-FLOP edge is maximized vs the baseline. MPS-packed on 1-2 B200.
set -uo pipefail
echo "== addition 2-D sweep (S1-frac x lambda) =="; date -u
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

USER_ALIAS=${USER_ALIAS:-${USER%@*}}; PVC=/mnt/pvc/${USER_ALIAS}
export HF_HOME=${HF_HOME:-$PVC/hf-cache} PIP_CACHE_DIR=${PIP_CACHE_DIR:-$PVC/pip-cache} TOKENIZERS_PARALLELISM=false
TOY=$PVC/toys; mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$TOY/outputs" "$TOY/teachers"
rm -rf outputs; ln -sfn "$TOY/outputs" outputs
[ -L outputs ] || { echo "FATAL: outputs symlink"; exit 1; }
LOG="$TOY/sweep_$(date -u +%Y%m%d-%H%M%S).log"; exec > >(tee -a "$LOG") 2>&1
python3 -m pip install -q --cache-dir "$PIP_CACHE_DIR" "transformers==5.9.0" numpy 2>&1 | tail -1

# MPS so the many tiny cells share each GPU concurrently
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
command -v nvidia-cuda-mps-control >/dev/null 2>&1 && nvidia-cuda-mps-control -d >/dev/null 2>&1 && sleep 2 && echo "[mps] on" || echo "[mps] off"

export SLURM_JOB_ID=${SWEEP_TAG:-add2d}        # tags output dirs uniquely (distadd_..._<tag>)
TOTAL=${TOTAL:-2000}; EVAL=${EVAL:-50}   # baseline crosses 0.99 ~step 1200; 2000 leaves a tail (40 eval pts resolves the f99 crossing)
SEEDS=${SEEDS:-"0 1 2 3 4"}
FRACS=${FRACS:-"0.1 0.2 0.3 0.4 0.5 0.6"}   # S1 = 200..1200 steps, all pre-convergence
LAMBDAS=${LAMBDAS:-"0.5 0.7 0.9"}
NDIGITS=${NDIGITS:-10}; TEACHER_STEPS=${TEACHER_STEPS:-12000}   # reversed 10-digit add groks ~step 7500
TASK=${TASK:-addition}; THID=${THID:-320}; TLAY=${TLAY:-6}      # teacher arch (mult is harder -> bump)
DPRE=$([ "$TASK" = multiplication ] && echo distmul || echo distadd)
MAXPAR=${MAXPAR:-8}; NG=$(nvidia-smi -L 2>/dev/null | wc -l); [ "${NG:-0}" -lt 1 ] && NG=1; CAP=$(( MAXPAR*NG ))
echo "TASK=$TASK NDIGITS=$NDIGITS teacher(h$THID L$TLAY $TEACHER_STEPS) TOTAL=$TOTAL EVAL=$EVAL fracs='$FRACS' lambdas='$LAMBDAS' seeds='$SEEDS' GPUs=$NG cap=$CAP"

ADD_T=$TOY/teachers/${TASK}_rev_d${NDIGITS}_h${THID}L${TLAY}   # all-numbers-reversed (LSB-first) format
if [ ! -s "$ADD_T/model.safetensors" ]; then
  echo "== train $TASK teacher (n_digits=$NDIGITS, h$THID L$TLAY, steps=$TEACHER_STEPS) =="
  python3 train_addition.py --task "$TASK" --hidden "$THID" --layers "$TLAY" --n_digits "$NDIGITS" --steps "$TEACHER_STEPS" --out "$ADD_T"
fi

n=0; running=0
cell(){ # method s1 s2 lam seed [loss_mode=pure_kd]
  local lm=${6:-pure_kd}; local mt
  case "$lm" in pure_kd) mt=kd;; kd_ce) mt=ce;; ce_only) mt=ceo;; *) mt=kd;; esac
  # RESUME: skip cells whose result already exists (distill_addition out-dir name)
  local outdir="outputs/${DPRE}_${mt}_${1}_d${NDIGITS}_l${4}_s1${2}_seed${5}_${SLURM_JOB_ID}"
  [ -s "$outdir/results.json" ] && return
  # ROLLING POOL: keep exactly CAP cells in flight; launch a replacement the instant
  # any finishes (vs a barrier that waits for the slowest of each batch).
  while [ "$running" -ge "$CAP" ]; do wait -n 2>/dev/null; running=$((running-1)); done
  local gpu=$(( n % NG )); n=$((n+1)); running=$((running+1))
  CUDA_VISIBLE_DEVICES=$gpu python3 distill_addition.py --teacher "$ADD_T" --task "$TASK" --method "$1" --loss_mode "$lm" \
    --n_digits "$NDIGITS" --stage1_steps "$2" --stage2_steps "$3" --fixed_lambda "$4" --seed "$5" --eval_every "$EVAL" --device cuda \
    > "outputs/swcell_${mt}_${1}_s1${2}_l${4}_seed${5}.log" 2>&1 &
}
echo "== launching sweep cells =="
half=$(( TOTAL/2 ))
for s in $SEEDS; do cell none "$half" "$half" 0.7 "$s"; done                 # none-KD baseline (teacher, no superposition)
for s in $SEEDS; do cell none "$half" "$half" 0.7 "$s" ce_only; done         # CE-only FLOOR (no teacher at all)
# cross_seq AND token_merge over the same S1-frac x lambda grid (token_merge keeps
# its default merge_k=2 -- not passing --merge_k -- per request).
for f in $FRACS; do
  s1=$(awk "BEGIN{print int($TOTAL*$f)}"); s2=$(( TOTAL - s1 ))
  for lam in $LAMBDAS; do for s in $SEEDS; do
    cell cross_seq  "$s1" "$s2" "$lam" "$s"
    cell token_merge "$s1" "$s2" "$lam" "$s"
  done; done
done
wait
echo "== sweep cells done =="
ls -d outputs/distadd_*${SLURM_JOB_ID} 2>/dev/null | wc -l
