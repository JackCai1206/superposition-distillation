#!/usr/bin/env bash
# FineWeb-Edu distillation at pretraining scale (nanoGPT recipe).
# Teacher = pretrained SmolLM2-1.7B (already trained on FineWeb-Edu, FROZEN, no finetune).
# Student = RANDOM-INIT SmolLM2-135M arch. Shared SmolLM2 vocab (49152).
# Recipe: ~0.5M-token effective batch via grad-accum, seq 1024, cosine LR 6e-4->6e-5,
# warmup 10%, AdamW(.9,.95) wd .1 clip 1.0. ~1B tokens (~2000 steps) to start.
# Methods: none / cross_seq / token_merge + CE-only floor (batch held equal in tokens).
set -uo pipefail
echo "== FineWeb-Edu distill =="; date -u
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
USER_ALIAS=${USER_ALIAS:-${USER%@*}}; PVC=/mnt/pvc/${USER_ALIAS}
export HF_HOME=${HF_HOME:-$PVC/hf-cache} HF_HUB_CACHE=${HF_HUB_CACHE:-$PVC/hf-cache/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$PVC/hf-cache/datasets} PIP_CACHE_DIR=${PIP_CACHE_DIR:-$PVC/pip-cache} TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
TOY=$PVC/toys; mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$PIP_CACHE_DIR" "$TOY/outputs"
rm -rf outputs; ln -sfn "$TOY/outputs" outputs
LOG="$TOY/fineweb_$(date -u +%Y%m%d-%H%M%S).log"; exec > >(tee -a "$LOG") 2>&1
python3 -m pip install -q --cache-dir "$PIP_CACHE_DIR" "transformers==5.9.0" "datasets==4.8.5" numpy 2>&1 | tail -1
# GPU saturation logger: POWER draw vs limit (the real saturation signal, not util% which
# just means "a kernel ran") + util/mem for reference, every 20s -> PVC.
UTILLOG="$TOY/gpuutil_$(date -u +%Y%m%d-%H%M%S).log"
(nvidia-smi --query-gpu=index,power.draw,power.limit,utilization.gpu,memory.used --format=csv,noheader -l 20 >> "$UTILLOG" 2>&1 &)
echo "[util] logging GPU power+util -> $UTILLOG"

# FineWeb-Edu SmolLM2-tokenized data (from prep_fineweb.sh)
export SD_DATA_DIR=${SD_DATA_DIR:-$PVC/toys/data_cache/fineweb_edu_smol}
export SD_VOCAB=${SD_VOCAB:-49152}
[ -s "$SD_DATA_DIR/train.bin" ] || { echo "FATAL: FineWeb data not prepped at $SD_DATA_DIR (run prep_fineweb.sh)"; exit 1; }

TEACHER=${TEACHER:-HuggingFaceTB/SmolLM2-1.7B}
REF=${REF:-HuggingFaceTB/SmolLM2-135M}                      # student arch (random init)
# SmolLM2-135M config: h576 L30 heads9 kv3 inter1536
SH=${SH:-576}; SL=${SL:-30}; SHEADS=${SHEADS:-9}; SKV=${SKV:-3}; SINTER=${SINTER:-1536}
SEQ=${SEQ:-1024}; MB=${MB:-16}; GA=${GA:-32}               # eff batch = MB*GA*SEQ = 16*32*1024 = 524288 tokens
STEPS=${STEPS:-2000}; LR=${LR:-6e-4}; EVAL=${EVAL:-100}
# FAIR DESIGN: every arm at the SAME matched config (lambda, S1-fraction) + the SAME
# seed set -> compare MEANS, not best-of-N; >=3 seeds incl. the none baseline so one
# unlucky run can't manufacture a gap. Batch held equal in tokens (grad-accum + the
# cross_seq 2x / token_merge k x sampling -> identical eff batch for all methods).
SEEDS=${SEEDS:-"0 1 2"}; LAMBDA=${LAMBDA:-0.5}; FRAC=${FRAC:-0.3}
METHODS=${METHODS:-"none cross_seq token_merge ce_only"}
MAXPAR=${MAXPAR:-1}                                        # 1.7B teacher is heavy -> few per GPU
echo "teacher=$TEACHER student=$REF(h$SH L$SL) seq=$SEQ eff_batch=$((MB*GA*SEQ)) steps=$STEPS lr=$LR"
echo "FAIR: methods='$METHODS' lambda=$LAMBDA S1frac=$FRAC seeds='$SEEDS' (matched config, multi-seed, means)"

export SLURM_JOB_ID=${SWEEP_TAG:-fineweb}
stu="--student_ref $REF --student_hidden $SH --student_layers $SL --student_heads $SHEADS --student_kv_heads $SKV --student_inter $SINTER"
NG=$(nvidia-smi -L 2>/dev/null | wc -l); [ "${NG:-0}" -lt 1 ] && NG=1; CAP=$(( MAXPAR*NG )); n=0; running=0
s1=$(awk "BEGIN{print int($STEPS*$FRAC)}"); s2=$(( STEPS - s1 ))
cell(){ # method loss_mode s1 s2 seed
  local m="$1" lm="$2" cs1="$3" cs2="$4" sd="$5"
  local mt; case "$lm" in pure_kd) mt=kd;; ce_only) mt=ceo;; mce) mt=mce;; *) mt=kd;; esac
  local tt; case "$lm" in ce_only|mce) tt=noT;; *) tt=smol;; esac
  local outdir="outputs/lmdist_${tt}_${mt}_${m}_l${LAMBDA}_s1${cs1}_seed${sd}_${SLURM_JOB_ID}"
  [ -s "$outdir/results.json" ] && return                 # resume: skip done cells
  while [ "$running" -ge "$CAP" ]; do wait -n 2>/dev/null; running=$((running-1)); done
  local gpu=$(( n % NG )); n=$((n+1)); running=$((running+1))
  CUDA_VISIBLE_DEVICES=$gpu python3 distill_lm.py --teacher "$TEACHER" $stu --method "$m" --loss_mode "$lm" \
    --seq_len "$SEQ" --batch_size "$MB" --grad_accum "$GA" --lr "$LR" --lr_sched "${LR_SCHED:-cosine}" --warmup_frac 0.1 \
    --fixed_lambda "$LAMBDA" --stage1_steps "$cs1" --stage2_steps "$cs2" --temperature 1.0 \
    --eval_every "$EVAL" --ckpt_every "${CKPT_EVERY:-0}" --seed "$sd" --device cuda \
    > "outputs/swfw_${tt}_${mt}_${m}_l${LAMBDA}_seed${sd}.log" 2>&1 &
}
echo "== launching FineWeb cells (cap=$CAP) =="
for sd in $SEEDS; do
  for M in $METHODS; do
    if [ "$M" = ce_only ]; then cell none ce_only $((STEPS/2)) $((STEPS/2)) "$sd"
    else cell "$M" "${LOSS_MODE:-pure_kd}" "$s1" "$s2" "$sd"; fi
  done
done
wait
echo "== fineweb done =="; ls -d outputs/lmdist_*_${SLURM_JOB_ID} 2>/dev/null | wc -l
