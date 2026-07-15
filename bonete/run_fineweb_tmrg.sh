#!/usr/bin/env bash
# ISO-TOKEN MCE comparison: token_merge(k=3) vs cross_seq, no teacher (MCE).
# --iso_token 1 holds RAW TOKENS/step == baseline (no bs scaling): the superposed
# stage runs at 1/k (token_merge) or 1/2 (cross_seq) the POSITIONS/FLOPs but sees the
# SAME data as none-KD / vanilla-NTP. So any win is from the packing, NOT extra data
# (the iso-FLOP fw4bwsd arms ate ~1.3x tokens in S1 -- this controls for that).
# Separate SWEEP_TAG=fw4biso so dirs do NOT collide with the iso-FLOP fw4bwsd runs.
# token_merge + KD is intentionally ABSENT: a standard teacher emits one next-token
# distribution per position, but a merged position must predict a BAG -> no teacher
# target. token_merge is a no-teacher (MCE) technique here.
set -uo pipefail
echo "== FineWeb ISO-TOKEN tmrg =="; date -u
USER_ALIAS=${USER_ALIAS:-${USER%@*}}; PVC=/mnt/pvc/${USER_ALIAS}
export HF_HOME=${HF_HOME:-$PVC/hf-cache} HF_HUB_CACHE=${HF_HUB_CACHE:-$PVC/hf-cache/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$PVC/hf-cache/datasets} PIP_CACHE_DIR=${PIP_CACHE_DIR:-$PVC/pip-cache} TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
TOY=$PVC/toys; mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$PIP_CACHE_DIR" "$TOY/outputs"
rm -rf outputs; ln -sfn "$TOY/outputs" outputs
LOG="$TOY/fwtmrg_$(date -u +%Y%m%d-%H%M%S).log"; exec > >(tee -a "$LOG") 2>&1
python3 -m pip install -q --cache-dir "$PIP_CACHE_DIR" "transformers==5.9.0" "datasets==4.8.5" numpy 2>&1 | tail -1
UTILLOG="$TOY/gpuutil_$(date -u +%Y%m%d-%H%M%S).log"
(nvidia-smi --query-gpu=index,power.draw,power.limit,utilization.gpu,memory.used --format=csv,noheader -l 30 >> "$UTILLOG" 2>&1 &)
export SD_DATA_DIR=${SD_DATA_DIR:-$PVC/toys/data_cache/fineweb_edu_smol} SD_VOCAB=${SD_VOCAB:-49152}
[ -s "$SD_DATA_DIR/train.bin" ] || { echo "FATAL: no FineWeb data at $SD_DATA_DIR"; exit 1; }

SWEEP_TAG=${SWEEP_TAG:-fw4biso}; LR_SCHED=${LR_SCHED:-wsd}; MERGE_K=${MERGE_K:-3}
SEQ=1024; MB=64; GA=8; STEPS=8000; FRAC=0.3; EVAL=200; CKPT=200; LR=6e-4
TEACHER=HuggingFaceTB/SmolLM2-1.7B
stu="--student_ref HuggingFaceTB/SmolLM2-135M --student_hidden 576 --student_layers 30 --student_heads 9 --student_kv_heads 3 --student_inter 1536"
s1=$(awk "BEGIN{print int($STEPS*$FRAC)}"); s2=$(( STEPS - s1 ))
export SLURM_JOB_ID=$SWEEP_TAG
NG=$(nvidia-smi -L 2>/dev/null | wc -l); echo "GPUs=$NG tag=$SWEEP_TAG sched=$LR_SCHED merge_k=$MERGE_K iso_token=1"
# arms: "method:lambda:seed"  (all MCE, all iso-token). token_merge uses MERGE_K.
ARMS=${ARMS:-"token_merge:0.5:1 token_merge:0.5:2 token_merge:0.7:1 token_merge:0.7:2 cross_seq:0.5:1 cross_seq:0.5:2 cross_seq:0.7:1 cross_seq:0.7:2"}
gpu=0
for spec in $ARMS; do
  IFS=: read -r m lam sd <<< "$spec"; sd=${sd:-1}
  echo "[tmrg] gpu$gpu <- mce $m l$lam seed$sd iso_token=1 (k=$MERGE_K if token_merge)"
  CUDA_VISIBLE_DEVICES=$gpu python3 distill_lm.py --teacher "$TEACHER" $stu --method "$m" --loss_mode mce \
    --seq_len $SEQ --batch_size $MB --grad_accum $GA --lr $LR --lr_sched "$LR_SCHED" --warmup_frac 0.1 \
    --fixed_lambda "$lam" --merge_k $MERGE_K --iso_token 1 --stage1_steps $s1 --stage2_steps $s2 --temperature 1.0 \
    --eval_every $EVAL --ckpt_every $CKPT --seed "$sd" --device cuda \
    > "outputs/swfw_${SWEEP_TAG}_noT_mce_${m}_l${lam}_seed${sd}.log" 2>&1 &
  gpu=$((gpu+1)); sleep 10
done
echo "== launched $gpu arms on $NG GPUs =="; wait; echo "== tmrg done =="
