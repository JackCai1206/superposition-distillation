#!/usr/bin/env bash
# MULT PILOT: train a 3x3 multiplication teacher + one `none` student distillation,
# both with dense eval, to (a) confirm a small model can LEARN 3-digit x 3-digit mult
# (it's genuinely hard) and (b) locate the transition before designing the sweep.
set -uo pipefail
echo "== 3x3 multiplication PILOT =="; date -u
USER_ALIAS=${USER_ALIAS:-${USER%@*}}; PVC=/mnt/pvc/${USER_ALIAS}
export HF_HOME=${HF_HOME:-$PVC/hf-cache} PIP_CACHE_DIR=${PIP_CACHE_DIR:-$PVC/pip-cache} TOKENIZERS_PARALLELISM=false
TOY=$PVC/toys; mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$TOY/outputs" "$TOY/teachers"
rm -rf outputs; ln -sfn "$TOY/outputs" outputs
LOG="$TOY/pilotmul_$(date -u +%Y%m%d-%H%M%S).log"; exec > >(tee -a "$LOG") 2>&1
python3 -m pip install -q --cache-dir "$PIP_CACHE_DIR" "transformers==5.9.0" numpy 2>&1 | tail -1

ND=${ND:-3}; THID=${THID:-512}; TLAY=${TLAY:-8}
TSTEPS=${TSTEPS:-40000}; SSTEPS=${SSTEPS:-20000}; TEVAL=${TEVAL:-1000}; SEVAL=${SEVAL:-500}
echo "ND=$ND teacher(h$THID L$TLAY $TSTEPS) student($SSTEPS)"
MUL_T=$TOY/teachers/multiplication_rev_d${ND}_h${THID}L${TLAY}
echo "== TRAIN TEACHER (3x3 mult, h$THID L$TLAY, $TSTEPS steps, eval/$TEVAL) =="
[ -s "$MUL_T/model.safetensors" ] || python3 train_addition.py --task multiplication --hidden "$THID" --layers "$TLAY" \
  --n_digits "$ND" --steps "$TSTEPS" --eval_every "$TEVAL" --out "$MUL_T" 2>&1 | tee "$TOY/outputs/pilotmul_teacher_d${ND}.log"
echo "== ONE STUDENT (none, pure_kd, $SSTEPS steps, eval/$SEVAL) =="
SLURM_JOB_ID=pilotmul python3 distill_addition.py --teacher "$MUL_T" --task multiplication --method none --loss_mode pure_kd \
  --n_digits "$ND" --stage1_steps 0 --stage2_steps "$SSTEPS" --eval_every "$SEVAL" --seed 0 --device cuda 2>&1 | tee "$TOY/outputs/pilotmul_student_d${ND}.log"
echo "== mult pilot done =="
