#!/usr/bin/env bash
# Bonete (B200, k8s+Volcano) pod entrypoint for superposition-distillation.
# Mirrors the verified hf-sft-qwen-openmath workflow: PVC-backed HF caches,
# minimal deps layered on the NVIDIA base image, then train.py.
#
# Knobs (all via --extra-env-vars on submit_job.sh):
#   METHOD          none | cross_seq | token_merge   (default cross_seq)
#   DATA_KIND       reasoning | pretrain             (default reasoning)
#   TEACHER         HF id   (default deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)
#   STUDENT         HF id   (default Qwen/Qwen2.5-0.5B)
#   STAGE1_STEPS / STAGE2_STEPS / SEQ_LEN / BATCH_SIZE / FIXED_LAMBDA
#   EVAL_MATH_N     # MATH-500 problems for the auto-eval (default 100)
set -euo pipefail

echo "== superposition-distillation cluster entrypoint =="
date -u; echo "hostname=$(hostname) user=$(whoami) pwd=$PWD"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

export USER_ALIAS=${USER_ALIAS:-${USER%@*}}
export PVC_MOUNT=${PVC_MOUNT:-/mnt/pvc}
export PVC_USER_ROOT=${PVC_USER_ROOT:-${PVC_MOUNT}/${USER_ALIAS}}
export HF_HOME=${HF_HOME:-${PVC_USER_ROOT}/hf-cache}
export HF_HUB_CACHE=${HF_HUB_CACHE:-${HF_HOME}/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${HF_HOME}/datasets}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-${PVC_USER_ROOT}/pip-cache}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# STABLE output root across resubmits: drop submit_job.sh's random 5-char suffix so a
# resubmitted job finds (and resumes from) its predecessor's checkpoints.
JOB_BASE=$(echo "${JOB_NAME:-supdistill}" | sed -E 's/-[a-z0-9]{5}$//')
export OUTPUT_ROOT=${OUTPUT_ROOT:-${PVC_USER_ROOT}/outputs/${JOB_BASE}}
# train.py writes its run dir + periodic checkpoints DIRECTLY under here (on PVC),
# so nothing large lands on the small ephemeral pod disk and checkpoints survive preempt.
export SD_OUTPUT_BASE=${SD_OUTPUT_BASE:-${OUTPUT_ROOT}}
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$PIP_CACHE_DIR" "$OUTPUT_ROOT"

# --- deps: keep the image's torch/flash-attn; layer only what the repo needs.
# Repo is written against transformers 5.x (uses the `dtype=` from_pretrained kwarg).
export WANDB_PROJECT=${WANDB_PROJECT:-superposition-distillation}
python3 -m pip install --upgrade pip
python3 -m pip install --cache-dir "$PIP_CACHE_DIR" \
  "transformers==5.9.0" "datasets==4.8.5" "accelerate>=1.0" \
  "math-verify==0.9.0" "latex2sympy2-extended==1.11.0" wandb

# --- run config (env-overridable) ---
METHOD=${METHOD:-cross_seq}
DATA_KIND=${DATA_KIND:-reasoning}
TEACHER=${TEACHER:-nvidia/OpenMath-Nemotron-7B}
STUDENT=${STUDENT:-Qwen/Qwen2.5-Math-1.5B-Instruct}   # +RoPE fix (config.py) for long CoT
STAGE1_STEPS=${STAGE1_STEPS:-8}
STAGE2_STEPS=${STAGE2_STEPS:-4}
SEQ_LEN=${SEQ_LEN:-16384}      # unpacked reasoning: one full CoT per seq; >cap SKIPPED (~4%)
BATCH_SIZE=${BATCH_SIZE:-8}    # MICRO batch per GPU
GRAD_ACCUM=${GRAD_ACCUM:-4}    # effective batch = GPUS * BATCH_SIZE * GRAD_ACCUM
GPUS=${GPUS:-1}                # torchrun procs (single-node)
FIXED_LAMBDA=${FIXED_LAMBDA:-0.7}
LOSS_MODE=${LOSS_MODE:-kd_ce}
LR=${LR:-3e-4}
KD_TEMP=${KD_TEMP:-2.0}
STUDENT_MAX_POS=${STUDENT_MAX_POS:--1}      # -1 config default; 0 disables RoPE fix (native)
STUDENT_ROPE_THETA=${STUDENT_ROPE_THETA:--1}
STUDENT_INIT=${STUDENT_INIT:-pretrained}    # 'random' = from-scratch (pretraining distill)
export EVAL_MATH_N=${EVAL_MATH_N:-20}

MEASURE_FLOPS_FLAG=""
[ "${MEASURE_FLOPS:-0}" = "1" ] && MEASURE_FLOPS_FLAG="--measure_flops"
CKPT_EVERY=${CKPT_EVERY:-0}

echo "method=$METHOD kind=$DATA_KIND teacher=$TEACHER student=$STUDENT loss_mode=$LOSS_MODE lr=$LR"
echo "s1=$STAGE1_STEPS s2=$STAGE2_STEPS seq=$SEQ_LEN micro=$BATCH_SIZE accum=$GRAD_ACCUM gpus=$GPUS lambda=$FIXED_LAMBDA eval_n=$EVAL_MATH_N measure_flops=${MEASURE_FLOPS:-0}"

if [ "$GPUS" -gt 1 ]; then
  LAUNCH="torchrun --standalone --nproc_per_node=$GPUS"
else
  LAUNCH="python3"
fi
$LAUNCH train.py \
  --method "$METHOD" --data_kind "$DATA_KIND" \
  --teacher "$TEACHER" --student "$STUDENT" \
  --stage1_steps "$STAGE1_STEPS" --stage2_steps "$STAGE2_STEPS" \
  --fixed_lambda "$FIXED_LAMBDA" --loss_mode "$LOSS_MODE" --lr "$LR" --temperature "$KD_TEMP" \
  --seq_len "$SEQ_LEN" --batch_size "$BATCH_SIZE" --grad_accum "$GRAD_ACCUM" \
  --student_max_pos "$STUDENT_MAX_POS" --student_rope_theta "$STUDENT_ROPE_THETA" \
  --student_init "$STUDENT_INIT" \
  --ckpt_every "$CKPT_EVERY" $MEASURE_FLOPS_FLAG \
  2>&1 | tee "${OUTPUT_ROOT}/train_${METHOD}_${LOSS_MODE}.log"

# train.py wrote its student + checkpoints + results.json DIRECTLY under
# $SD_OUTPUT_BASE (= $OUTPUT_ROOT on PVC) -- nothing to copy off the ephemeral disk.
echo "== done; artifacts under ${OUTPUT_ROOT} =="
ls -la "${SD_OUTPUT_BASE}"/* 2>/dev/null | tail -40 || true
