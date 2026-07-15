#!/usr/bin/env bash
# Run ONE arm data-parallel across all GPUs of the node (torchrun DDP). 8 GPUs with
# grad_accum=1 == the single-GPU grad_accum=8 effective batch (512), ~8x faster wall-clock.
# Whole-node job -> no fragmentation/defrag-reaping. Arm via env LOSS_MODE/METHOD/LAMBDA/SEED.
set -uo pipefail
echo "== FineWeb DDP =="; date -u
USER_ALIAS=${USER_ALIAS:-${USER%@*}}; PVC=/mnt/pvc/${USER_ALIAS}
export HF_HOME=${HF_HOME:-$PVC/hf-cache} HF_HUB_CACHE=${HF_HUB_CACHE:-$PVC/hf-cache/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$PVC/hf-cache/datasets} PIP_CACHE_DIR=${PIP_CACHE_DIR:-$PVC/pip-cache} TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
TOY=$PVC/toys; mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$PIP_CACHE_DIR" "$TOY/outputs"
rm -rf outputs; ln -sfn "$TOY/outputs" outputs
python3 -m pip install -q --cache-dir "$PIP_CACHE_DIR" "transformers==5.9.0" "datasets==4.8.5" numpy 2>&1 | tail -1
export SD_DATA_DIR=${SD_DATA_DIR:-$PVC/toys/data_cache/fineweb_edu_smol} SD_VOCAB=${SD_VOCAB:-49152}
[ -s "$SD_DATA_DIR/train.bin" ] || { echo "FATAL: no FineWeb data at $SD_DATA_DIR"; exit 1; }
SWEEP_TAG=${SWEEP_TAG:-fw4bddp}; LR_SCHED=${LR_SCHED:-wsd}
SEQ=1024; MB=${MB:-64}; STEPS=${STEPS:-8000}; FRAC=${FRAC:-0.3}; EVAL=${EVAL:-200}; CKPT=${CKPT:-200}; LR=${LR:-6e-4}
NG=$(nvidia-smi -L 2>/dev/null | wc -l); [ "${NG:-0}" -lt 1 ] && NG=1
s1=$(awk "BEGIN{print int($STEPS*$FRAC)}"); s2=$(( STEPS - s1 ))
export SLURM_JOB_ID=$SWEEP_TAG
TEACHER=${TEACHER:-HuggingFaceTB/SmolLM2-1.7B}
stu="--student_ref HuggingFaceTB/SmolLM2-135M --student_hidden 576 --student_layers 30 --student_heads 9 --student_kv_heads 3 --student_inter 1536"
LM=${LOSS_MODE:-pure_kd}; M=${METHOD:-cross_seq}; LAM=${LAMBDA:-0.5}; SD=${SEED:-0}
if [ "$LM" = ce_only ]; then M=none; cs1=$((STEPS/2)); cs2=$((STEPS/2)); else cs1=$s1; cs2=$s2; fi
case "$LM" in pure_kd) mt=kd;; ce_only) mt=ceo;; mce) mt=mce;; *) mt=kd;; esac
case "$LM" in ce_only|mce) tt=noT;; *) tt=smol;; esac
GPUUTIL="$TOY/gpuutil_$(date -u +%Y%m%d-%H%M%S).log"
(nvidia-smi --query-gpu=index,power.draw,utilization.gpu --format=csv,noheader -l 30 >> "$GPUUTIL" 2>&1 &)
echo "DDP: $NG GPUs | arm $LM $M l$LAM seed$SD | eff_batch=$((NG*MB)) (=NG*MB, GA=1) tag=$SWEEP_TAG"
# Reap-forwarding: kubelet SIGTERMs only PID 1 (this bash); forward it to torchrun so
# distill_lm.py's REAP_SIGTERM hook can checkpoint within the grace window (else the workers
# see only the final uncatchable SIGKILL). See run_fineweb_ddp_seq.sh for the same pattern.
TORCH_PID=""
trap '[ -n "$TORCH_PID" ] && { echo "[ddp] SIGTERM -> torchrun $TORCH_PID (reap)"; kill -TERM "$TORCH_PID" 2>/dev/null; wait "$TORCH_PID" 2>/dev/null; }; exit 143' TERM INT
torchrun --standalone --nproc_per_node=$NG distill_lm.py --teacher "$TEACHER" $stu \
  --method "$M" --loss_mode "$LM" --seq_len $SEQ --batch_size $MB --grad_accum 1 \
  --lr $LR --lr_sched "$LR_SCHED" --warmup_frac 0.1 --fixed_lambda "$LAM" \
  --stage1_steps $cs1 --stage2_steps $cs2 --temperature 1.0 \
  --eval_every $EVAL --ckpt_every $CKPT --seed $SD --device cuda \
  > "outputs/swfw_${tt}_${mt}_${M}_l${LAM}_seed${SD}.log" 2>&1 &
TORCH_PID=$!
wait "$TORCH_PID"
echo "== DDP arm done =="
