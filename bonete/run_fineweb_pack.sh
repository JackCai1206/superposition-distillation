#!/usr/bin/env bash
# Pack N heterogeneous arms onto ONE multi-GPU node (one arm per GPU) so the whole
# node is used -> no fragmentation, no defrag-reaping of 1-GPU jobs. Each arm is a
# spec "lossmode:method:lambda[:seed]" placed on CUDA_VISIBLE_DEVICES=i. Arms write
# the SAME lmdist_*_<SWEEP_TAG> dirs as the 1-GPU jobs, so in-flight ones RESUME from
# their ckpt. (No DDP; each arm trains at 1-GPU speed but co-located on one node.)
set -uo pipefail
echo "== FineWeb PACK =="; date -u
USER_ALIAS=${USER_ALIAS:-${USER%@*}}; PVC=/mnt/pvc/${USER_ALIAS}
export HF_HOME=${HF_HOME:-$PVC/hf-cache} HF_HUB_CACHE=${HF_HUB_CACHE:-$PVC/hf-cache/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$PVC/hf-cache/datasets} PIP_CACHE_DIR=${PIP_CACHE_DIR:-$PVC/pip-cache} TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
TOY=$PVC/toys; mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$PIP_CACHE_DIR" "$TOY/outputs"
rm -rf outputs; ln -sfn "$TOY/outputs" outputs
LOG="$TOY/fwpack_$(date -u +%Y%m%d-%H%M%S).log"; exec > >(tee -a "$LOG") 2>&1
python3 -m pip install -q --cache-dir "$PIP_CACHE_DIR" "transformers==5.9.0" "datasets==4.8.5" numpy 2>&1 | tail -1
UTILLOG="$TOY/gpuutil_$(date -u +%Y%m%d-%H%M%S).log"
(nvidia-smi --query-gpu=index,power.draw,power.limit,utilization.gpu,memory.used --format=csv,noheader -l 30 >> "$UTILLOG" 2>&1 &)
export SD_DATA_DIR=${SD_DATA_DIR:-$PVC/toys/data_cache/fineweb_edu_smol} SD_VOCAB=${SD_VOCAB:-49152}
[ -s "$SD_DATA_DIR/train.bin" ] || { echo "FATAL: no FineWeb data at $SD_DATA_DIR"; exit 1; }

SWEEP_TAG=${SWEEP_TAG:-fw4bwsd}; LR_SCHED=${LR_SCHED:-wsd}
SEQ=1024; MB=64; GA=8; STEPS=8000; FRAC=0.3; EVAL=200; CKPT=200; LR=6e-4
TEACHER=HuggingFaceTB/SmolLM2-1.7B
stu="--student_ref HuggingFaceTB/SmolLM2-135M --student_hidden 576 --student_layers 30 --student_heads 9 --student_kv_heads 3 --student_inter 1536"
s1=$(awk "BEGIN{print int($STEPS*$FRAC)}"); s2=$(( STEPS - s1 ))
export SLURM_JOB_ID=$SWEEP_TAG
NG=$(nvidia-smi -L 2>/dev/null | wc -l); echo "GPUs=$NG  tag=$SWEEP_TAG sched=$LR_SCHED"
# arms: "lossmode:method:lambda[:seed]"  (ce_only ignores method/lambda, uses STEPS/2 stages)
ARMS=${ARMS:-"pure_kd:cross_seq:0.6:0 ce_only:none:0.5:0 mce:cross_seq:0.5:0 mce:cross_seq:0.6:0 mce:cross_seq:0.7:0 mce:cross_seq:0.8:0 mce:cross_seq:0.9:0 ce_only:none:0.5:1"}
gpu=0
for spec in $ARMS; do
  IFS=: read -r lm m lam sd <<< "$spec"; sd=${sd:-0}
  if [ "$lm" = ce_only ]; then m=none; cs1=$((STEPS/2)); cs2=$((STEPS/2)); else cs1=$s1; cs2=$s2; fi
  case "$lm" in pure_kd) mt=kd;; ce_only) mt=ceo;; mce) mt=mce;; *) mt=kd;; esac
  case "$lm" in ce_only|mce) tt=noT;; *) tt=smol;; esac
  echo "[pack] gpu$gpu <- $lm $m l$lam seed$sd"
  CUDA_VISIBLE_DEVICES=$gpu python3 distill_lm.py --teacher "$TEACHER" $stu --method "$m" --loss_mode "$lm" \
    --seq_len $SEQ --batch_size $MB --grad_accum $GA --lr $LR --lr_sched "$LR_SCHED" --warmup_frac 0.1 \
    --fixed_lambda "$lam" --stage1_steps $cs1 --stage2_steps $cs2 --temperature 1.0 \
    --eval_every $EVAL --ckpt_every $CKPT --seed "$sd" --device cuda \
    > "outputs/swfw_${tt}_${mt}_${m}_l${lam}_seed${sd}.log" 2>&1 &
  gpu=$((gpu+1))
  sleep 10        # stagger model-load peaks across processes
done
echo "== launched $gpu arms on $NG GPUs =="; wait; echo "== pack done =="
