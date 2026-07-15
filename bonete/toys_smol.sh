#!/usr/bin/env bash
# SmolLM2 TinyStories sweep: STRONG in-domain teacher (SmolLM2-135M finetuned on
# TinyStories) -> RANDOM-INIT scaled-down student of the SAME arch. Clean pure-KL,
# none/cross_seq/token_merge over S1-frac x lambda + CE-only floor. The proper
# strong-teacher regime (fixes the weak ctrl/gpt2 teachers).
set -uo pipefail
echo "== SmolLM2 TinyStories sweep =="; date -u
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

USER_ALIAS=${USER_ALIAS:-${USER%@*}}; PVC=/mnt/pvc/${USER_ALIAS}
export HF_HOME=${HF_HOME:-$PVC/hf-cache} HF_HUB_CACHE=${HF_HUB_CACHE:-$PVC/hf-cache/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$PVC/hf-cache/datasets} PIP_CACHE_DIR=${PIP_CACHE_DIR:-$PVC/pip-cache} TOKENIZERS_PARALLELISM=false
TOY=$PVC/toys; mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$PIP_CACHE_DIR" "$TOY/outputs" "$TOY/data_cache" "$TOY/teachers"
rm -rf outputs data_cache; ln -sfn "$TOY/outputs" outputs; ln -sfn "$TOY/data_cache" data_cache
[ -L outputs ] || { echo "FATAL: outputs symlink"; exit 1; }
LOG="$TOY/smol_$(date -u +%Y%m%d-%H%M%S).log"; exec > >(tee -a "$LOG") 2>&1
python3 -m pip install -q --cache-dir "$PIP_CACHE_DIR" "transformers==5.9.0" "datasets==4.8.5" numpy 2>&1 | tail -1

# --- SmolLM2 data/vocab ---
export SD_TOKENIZER=${SD_TOKENIZER:-HuggingFaceTB/SmolLM2-135M}
export SD_DATA_DIR=${SD_DATA_DIR:-data_cache/tinystories_smol}
export SD_VOCAB=${SD_VOCAB:-49152}
REF=${REF:-HuggingFaceTB/SmolLM2-135M}

# MPS
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
command -v nvidia-cuda-mps-control >/dev/null 2>&1 && nvidia-cuda-mps-control -d >/dev/null 2>&1 && sleep 2 && echo "[mps] on" || echo "[mps] off"

export SLURM_JOB_ID=${SWEEP_TAG:-smol}
TOTAL=${TOTAL:-3000}; EVAL=${EVAL:-100}
SEEDS=${SEEDS:-"0"}; FRACS=${FRACS:-"0.2 0.4 0.6 0.8"}; LAMBDAS=${LAMBDAS:-"0.5 0.7 0.9"}
# scaled student (~17M): h256 L6 heads4 kv2 (head_dim 64, SmolLM2 family)
SH=${SH:-256}; SL=${SL:-6}; SHEADS=${SHEADS:-4}; SKV=${SKV:-2}
FT_STEPS=${FT_STEPS:-3000}
MAXPAR=${MAXPAR:-8}; NG=$(nvidia-smi -L 2>/dev/null | wc -l); [ "${NG:-0}" -lt 1 ] && NG=1; CAP=$(( MAXPAR*NG ))
echo "REF=$REF student h$SH L$SL kv$SKV | TOTAL=$TOTAL EVAL=$EVAL fracs='$FRACS' lambdas='$LAMBDAS' GPUs=$NG cap=$CAP"

# --- data: tokenize TinyStories with SmolLM2 tokenizer ---
[ -s "$SD_DATA_DIR/train.bin" ] || { echo "== prep TinyStories (SmolLM2 tok) =="; python3 prepare_tinystories.py; }
# --- teacher: finetune SmolLM2-135M on TinyStories ---
SMOL_T=$TOY/teachers/smol135_ft
[ -s "$SMOL_T/model.safetensors" ] || { echo "== finetune SmolLM2-135M teacher ($FT_STEPS steps) =="; \
  CUDA_VISIBLE_DEVICES=0 python3 finetune_lm.py --ref "$REF" --out "$SMOL_T" --steps "$FT_STEPS" --eval_every 200; }

stu="--student_ref $REF --student_hidden $SH --student_layers $SL --student_heads $SHEADS --student_kv_heads $SKV"
n=0; running=0
cell(){ # method s1 s2 lam seed [loss_mode=pure_kd]
  local lm=${6:-pure_kd}; local mt tt
  case "$lm" in pure_kd) mt=kd;; kd_ce) mt=ce;; ce_only) mt=ceo;; *) mt=kd;; esac
  if [ "$lm" = ce_only ]; then tt=noT; else tt=smol; fi
  local outdir="outputs/lmdist_${tt}_${mt}_${1}_l${4}_s1${2}_seed${5}_${SLURM_JOB_ID}"
  [ -s "$outdir/results.json" ] && return
  while [ "$running" -ge "$CAP" ]; do wait -n 2>/dev/null; running=$((running-1)); done
  local gpu=$(( n % NG )); n=$((n+1)); running=$((running+1))
  CUDA_VISIBLE_DEVICES=$gpu python3 distill_lm.py --teacher "$SMOL_T" $stu --method "$1" --loss_mode "$lm" \
    --temperature "${KD_TEMP:-2.0}" \
    --stage1_steps "$2" --stage2_steps "$3" --fixed_lambda "$4" --seed "$5" --eval_every "$EVAL" --device cuda \
    > "outputs/swsmol_${tt}_${mt}_${1}_s1${2}_l${4}_seed${5}.log" 2>&1 &
}
echo "== launching sweep cells =="
half=$(( TOTAL/2 ))
for s in $SEEDS; do cell none "$half" "$half" 0.7 "$s" ce_only; done    # CE-only floor (no teacher)
for s in $SEEDS; do cell none "$half" "$half" 0.7 "$s"; done            # none-KD baseline
for f in $FRACS; do
  s1=$(awk "BEGIN{print int($TOTAL*$f)}"); s2=$(( TOTAL - s1 ))
  for lam in $LAMBDAS; do for s in $SEEDS; do
    cell cross_seq   "$s1" "$s2" "$lam" "$s"
    cell token_merge "$s1" "$s2" "$lam" "$s"
  done; done
done
wait
echo "== smol sweep cells done =="
ls -d outputs/lmdist_*_${SLURM_JOB_ID} 2>/dev/null | wc -l
