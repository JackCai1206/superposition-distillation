#!/usr/bin/env bash
# LM PILOT: distill one `none`-student from EACH teacher (ctrl custom + gpt2 pretrained),
# full TOTAL-step curve with dense eval, to (a) confirm both teacher paths load+superpose+
# distill end-to-end (gpt2 path is new/untested) and (b) set TOTAL + the val-loss target
# before committing the 255-cell sweep.
set -uo pipefail
echo "== TinyStories LM PILOT (ctrl + gpt2 teachers) =="; date -u
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

USER_ALIAS=${USER_ALIAS:-${USER%@*}}; PVC=/mnt/pvc/${USER_ALIAS}
export HF_HOME=${HF_HOME:-$PVC/hf-cache} HF_HUB_CACHE=${HF_HUB_CACHE:-$PVC/hf-cache/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$PVC/hf-cache/datasets} PIP_CACHE_DIR=${PIP_CACHE_DIR:-$PVC/pip-cache} TOKENIZERS_PARALLELISM=false
TOY=$PVC/toys; mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$PIP_CACHE_DIR" "$TOY/outputs" "$TOY/data_cache" "$TOY/teachers"
rm -rf outputs data_cache
ln -sfn "$TOY/outputs" outputs; ln -sfn "$TOY/data_cache" data_cache
[ -L outputs ] || { echo "FATAL: outputs symlink"; exit 1; }
LOG="$TOY/pilotlm_$(date -u +%Y%m%d-%H%M%S).log"; exec > >(tee -a "$LOG") 2>&1
python3 -m pip install -q --cache-dir "$PIP_CACHE_DIR" "transformers==5.9.0" "datasets==4.8.5" numpy 2>&1 | tail -1

TOTAL=${TOTAL:-3000}; EVAL=${EVAL:-100}
echo "TOTAL=$TOTAL EVAL=$EVAL"

# data
[ -s data_cache/tinystories/train.bin ] || { echo "== prep TinyStories =="; python3 prepare_tinystories.py; }
# ctrl teacher (train if absent)
LM_CTRL=$TOY/teachers/lm_ctrl_h512l8
[ -s "$LM_CTRL/model.safetensors" ] || { echo "== train LM ctrl teacher =="; python3 train_lm.py --hidden 512 --layers 8 --steps 6000 --out "$LM_CTRL"; }

export SLURM_JOB_ID=pilotlm
echo "== CELL 1: ctrl none-student (pure_kd) =="
SLURM_JOB_ID=pilotlm CUDA_VISIBLE_DEVICES=0 python3 distill_lm.py --teacher "$LM_CTRL" --method none --loss_mode pure_kd \
  --stage1_steps 0 --stage2_steps "$TOTAL" --eval_every "$EVAL" --seed 0 --device cuda 2>&1 | tee "$TOY/outputs/pilotlm_ctrl_none.log"

echo "== CELL 2: gpt2 none-student (pure_kd) -- NEW teacher path =="
SLURM_JOB_ID=pilotlm CUDA_VISIBLE_DEVICES=0 python3 distill_lm.py --teacher gpt2 --method none --loss_mode pure_kd \
  --stage1_steps 0 --stage2_steps "$TOTAL" --eval_every "$EVAL" --seed 0 --device cuda 2>&1 | tee "$TOY/outputs/pilotlm_gpt2_none.log"

echo "== CELL 3: gpt2 cross_seq SMOKE (200 steps) -- confirm superposition works on the pretrained teacher =="
SLURM_JOB_ID=pilotlm CUDA_VISIBLE_DEVICES=0 python3 distill_lm.py --teacher gpt2 --method cross_seq --loss_mode pure_kd \
  --stage1_steps 200 --stage2_steps 0 --fixed_lambda 0.7 --eval_every 50 --seed 0 --device cuda 2>&1 | tee "$TOY/outputs/pilotlm_gpt2_cs_smoke.log"
echo "== LM pilot done =="
