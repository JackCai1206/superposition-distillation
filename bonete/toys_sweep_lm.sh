#!/usr/bin/env bash
# TinyStories 2-D sweep (clean pure-KL): cross_seq AND token_merge over S1-FRACTION
# x LAMBDA x seeds, plus the `none` baseline. Same design as the addition sweep, but
# the metric is continuous val cross-entropy (no saturation), so the iso-FLOP question
# is "val-loss reached at a common FLOP budget" / "FLOPs to reach a val-loss target".
# Heavier cells (30M teacher) -> conservative packing + 200Gi host RAM.
set -uo pipefail
echo "== TinyStories 2-D sweep (S1-frac x lambda) =="; date -u
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

USER_ALIAS=${USER_ALIAS:-${USER%@*}}; PVC=/mnt/pvc/${USER_ALIAS}
export HF_HOME=${HF_HOME:-$PVC/hf-cache} HF_HUB_CACHE=${HF_HUB_CACHE:-$PVC/hf-cache/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$PVC/hf-cache/datasets} PIP_CACHE_DIR=${PIP_CACHE_DIR:-$PVC/pip-cache} TOKENIZERS_PARALLELISM=false
TOY=$PVC/toys; mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$PIP_CACHE_DIR" "$TOY/outputs" "$TOY/data_cache" "$TOY/teachers"
rm -rf outputs data_cache
ln -sfn "$TOY/outputs" outputs; ln -sfn "$TOY/data_cache" data_cache
[ -L outputs ] || { echo "FATAL: outputs symlink"; exit 1; }
LOG="$TOY/sweeplm_$(date -u +%Y%m%d-%H%M%S).log"; exec > >(tee -a "$LOG") 2>&1
python3 -m pip install -q --cache-dir "$PIP_CACHE_DIR" "transformers==5.9.0" "datasets==4.8.5" numpy 2>&1 | tail -1

# MPS for concurrent sharing
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
command -v nvidia-cuda-mps-control >/dev/null 2>&1 && nvidia-cuda-mps-control -d >/dev/null 2>&1 && sleep 2 && echo "[mps] on" || echo "[mps] off"

export SLURM_JOB_ID=${SWEEP_TAG:-lm2d}
TOTAL=${TOTAL:-3000}; EVAL=${EVAL:-100}
SEEDS=${SEEDS:-"0 1 2 3 4"}; FRACS=${FRACS:-"0.2 0.4 0.6 0.8"}; LAMBDAS=${LAMBDAS:-"0.5 0.7 0.9"}
MAXPAR=${MAXPAR:-12}; NG=$(nvidia-smi -L 2>/dev/null | wc -l); [ "${NG:-0}" -lt 1 ] && NG=1; CAP=$(( MAXPAR*NG ))
echo "TOTAL=$TOTAL EVAL=$EVAL fracs='$FRACS' lambdas='$LAMBDAS' seeds='$SEEDS' GPUs=$NG cap=$CAP"

# data
[ -s data_cache/tinystories/train.bin ] || { echo "== prep TinyStories =="; python3 prepare_tinystories.py; }
# TEACHERS axis: ctrl = custom from-scratch (h512/l8), gpt2 = pretrained HF (124M realism check).
# Both share the GPT-2 vocab (50257) so KD logits are directly comparable.
TEACHERS=${TEACHERS:-"ctrl gpt2"}
LM_CTRL=$TOY/teachers/lm_ctrl_h512l8
if echo " $TEACHERS " | grep -q " ctrl "; then
  [ -s "$LM_CTRL/model.safetensors" ] || { echo "== train LM ctrl teacher =="; python3 train_lm.py --hidden 512 --layers 8 --steps 6000 --out "$LM_CTRL"; }
fi

n=0; running=0
cell(){ # ttag teacher method s1 s2 lam seed [loss_mode=pure_kd]
  local tt="$1" teach="$2" method="$3" s1="$4" s2="$5" lam="$6" seed="$7" lm=${8:-pure_kd} mt
  case "$lm" in pure_kd) mt=kd;; kd_ce) mt=ce;; ce_only) mt=ceo; tt=noT;; *) mt=kd;; esac
  local outdir="outputs/lmdist_${tt}_${mt}_${method}_l${lam}_s1${s1}_seed${seed}_${SLURM_JOB_ID}"
  [ -s "$outdir/results.json" ] && return
  while [ "$running" -ge "$CAP" ]; do wait -n 2>/dev/null; running=$((running-1)); done
  local gpu=$(( n % NG )); n=$((n+1)); running=$((running+1))
  CUDA_VISIBLE_DEVICES=$gpu python3 distill_lm.py --teacher "$teach" --method "$method" --loss_mode "$lm" \
    --stage1_steps "$s1" --stage2_steps "$s2" --fixed_lambda "$lam" --seed "$seed" --eval_every "$EVAL" --device cuda \
    > "outputs/swlm_${tt}_${mt}_${method}_s1${s1}_l${lam}_seed${seed}.log" 2>&1 &
}
echo "== launching sweep cells (teachers: $TEACHERS) =="
half=$(( TOTAL/2 ))
# CE-only FLOOR: no teacher -> teacher-independent, run ONCE (skip-if-done makes a 2nd job a no-op)
for s in $SEEDS; do cell noT none none "$half" "$half" 0.7 "$s" ce_only; done
for T in $TEACHERS; do
  case "$T" in ctrl) TARG="$LM_CTRL"; TT=ctrl;; gpt2) TARG="gpt2"; TT=gpt2;; *) echo "skip unknown teacher $T"; continue;; esac
  echo "== teacher=$TT ($TARG) =="
  for s in $SEEDS; do cell "$TT" "$TARG" none "$half" "$half" 0.7 "$s"; done   # none-KD baseline per teacher
  for f in $FRACS; do
    s1=$(awk "BEGIN{print int($TOTAL*$f)}"); s2=$(( TOTAL - s1 ))
    for lam in $LAMBDAS; do for s in $SEEDS; do
      cell "$TT" "$TARG" cross_seq   "$s1" "$s2" "$lam" "$s"
      cell "$TT" "$TARG" token_merge "$s1" "$s2" "$lam" "$s"
    done; done
  done
done
wait
echo "== sweep cells done =="
ls -d outputs/lmdist_*${SLURM_JOB_ID} 2>/dev/null | wc -l
