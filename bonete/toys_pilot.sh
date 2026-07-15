#!/usr/bin/env bash
# PILOT: train a 10-digit addition teacher + one `none` student distillation, both with
# dense eval, to locate the phase transition (grokking) before designing the sweep.
set -uo pipefail
echo "== 10-digit addition PILOT =="; date -u
USER_ALIAS=${USER_ALIAS:-${USER%@*}}; PVC=/mnt/pvc/${USER_ALIAS}
export HF_HOME=${HF_HOME:-$PVC/hf-cache} PIP_CACHE_DIR=${PIP_CACHE_DIR:-$PVC/pip-cache} TOKENIZERS_PARALLELISM=false
TOY=$PVC/toys; mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$TOY/outputs" "$TOY/teachers"
rm -rf outputs; ln -sfn "$TOY/outputs" outputs
LOG="$TOY/pilot_$(date -u +%Y%m%d-%H%M%S).log"; exec > >(tee -a "$LOG") 2>&1
python3 -m pip install -q --cache-dir "$PIP_CACHE_DIR" "transformers==5.9.0" numpy 2>&1 | tail -1
ND=${ND:-10}; TSTEPS=${TSTEPS:-40000}; SSTEPS=${SSTEPS:-40000}; TEVAL=${TEVAL:-1000}; SEVAL=${SEVAL:-500}
echo "ND=$ND TSTEPS=$TSTEPS SSTEPS=$SSTEPS"
ADD_T=$TOY/teachers/addition_rev_d${ND}
echo "== TRAIN TEACHER (n_digits=$ND, $TSTEPS steps, eval/$TEVAL) =="
[ -s "$ADD_T/model.safetensors" ] || python3 train_addition.py --n_digits "$ND" --steps "$TSTEPS" --eval_every "$TEVAL" --out "$ADD_T" 2>&1 | tee "$TOY/outputs/pilot_teacher_d${ND}.log"
echo "== ONE STUDENT (none, pure_kd, $SSTEPS steps, eval/$SEVAL) =="
SLURM_JOB_ID=pilot python3 distill_addition.py --teacher "$ADD_T" --method none --loss_mode pure_kd \
  --n_digits "$ND" --stage1_steps 0 --stage2_steps "$SSTEPS" --eval_every "$SEVAL" --seed 0 --device cuda 2>&1 | tee "$TOY/outputs/pilot_student_d${ND}.log"
echo "== pilot done =="
