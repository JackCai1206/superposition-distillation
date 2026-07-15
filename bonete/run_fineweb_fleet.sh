#!/usr/bin/env bash
# FLEET mode: run many arms in PARALLEL, ONE GPU each, all co-located on the SAME node
# (a single NG-GPU allocation). Each arm is an independent 1-GPU process at MB=64,
# grad_accum=8 -> eff_batch = 512 == the DDP arms and the 3.172 baseline (iso-FLOP,
# same optimization). Concurrency is capped at NG (one arm per physical GPU) so we never
# oversubscribe; extra arms queue and start as GPUs free. Guarantees same-node by
# construction (it IS one node). arm spec: "lossmode:method:lambda:merge_k:seed:noise_sigma:anchor:noise_mode".
set -uo pipefail
echo "== FineWeb FLEET (1-GPU/arm, co-located) =="; date -u
USER_ALIAS=${USER_ALIAS:-${USER%@*}}; PVC=${PVC_ROOT:-/mnt/pvc}/${USER_ALIAS}
# data-path auto-detect (survives PVC reorgs; see run_fineweb_ddp_seq.sh)
_b=${PVC_ROOT:-/mnt/pvc}
for _c in "$_b/${USER_ALIAS}" "$_b/experiments/${USER_ALIAS}" "/data/experiments/${USER_ALIAS}" "/mnt/pvc/experiments/${USER_ALIAS}" "/mnt/pvc/${USER_ALIAS}"; do
  if [ -s "$_c/toys/data_cache/fineweb_edu_smol/train.bin" ]; then PVC="$_c"; echo "[fleet] FineWeb data found -> PVC=$PVC"; break; fi
done
export HF_HOME=${HF_HOME:-$PVC/hf-cache} HF_HUB_CACHE=${HF_HUB_CACHE:-$PVC/hf-cache/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$PVC/hf-cache/datasets} PIP_CACHE_DIR=${PIP_CACHE_DIR:-$PVC/pip-cache} TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
TOY=$PVC/toys; mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$PIP_CACHE_DIR" "$TOY/outputs"
rm -rf outputs; ln -sfn "$TOY/outputs" outputs
LOG="$TOY/fwfleet_$(date -u +%Y%m%d-%H%M%S).log"; exec > >(tee -a "$LOG") 2>&1
python3 -m pip install -q --cache-dir "$PIP_CACHE_DIR" "transformers==5.9.0" "datasets==4.8.5" numpy 2>&1 | tail -1
UTILLOG="$TOY/gpuutil_$(date -u +%Y%m%d-%H%M%S).log"
(nvidia-smi --query-gpu=index,power.draw,power.limit,utilization.gpu,memory.used --format=csv,noheader -l 30 >> "$UTILLOG" 2>&1 &)
export SD_DATA_DIR=${SD_DATA_DIR:-$PVC/toys/data_cache/fineweb_edu_smol} SD_VOCAB=${SD_VOCAB:-49152}
[ -s "$SD_DATA_DIR/train.bin" ] || { echo "FATAL: no FineWeb data at $SD_DATA_DIR"; exit 1; }

SWEEP_TAG=${SWEEP_TAG:-fw4bfleet}; LR_SCHED=${LR_SCHED:-wsd}
SEQ=1024; MB=${MB:-64}; GA=${GA:-8}; STEPS=${STEPS:-8000}; FRAC=0.3; EVAL=200; CKPT=200; LR=6e-4
NOISE_K=${NOISE_K:-8}   # random-token components for onehot perturbation (bounds embed-lookup memory)
TEACHER=HuggingFaceTB/SmolLM2-1.7B
stu="--student_ref HuggingFaceTB/SmolLM2-135M --student_hidden 576 --student_layers 30 --student_heads 9 --student_kv_heads 3 --student_inter 1536"
s1=$(awk "BEGIN{print int($STEPS*$FRAC)}"); s2=$(( STEPS - s1 ))
NG=$(nvidia-smi -L 2>/dev/null | wc -l); [ "${NG:-0}" -lt 1 ] && NG=1
echo "GPUs=$NG tag=$SWEEP_TAG sched=$LR_SCHED eff_batch=$((MB*GA)) (MB*GA, 1 GPU/arm), max $NG concurrent"
ARMS=${ARMS:-"pure_kd:none:0.7:2:1:0:0:onehot"}

launch() {  # $1=spec $2=gpu
  local spec=$1 gpu=$2
  IFS=: read -r lm m lam mk sd nz an nm <<< "$spec"
  sd=${sd:-1}; mk=${mk:-2}; nz=${nz:-0}; an=${an:-0}; nm=${nm:-onehot}
  local cs1 cs2 mt tt
  if [ "$lm" = ce_only ]; then m=none; cs1=$((STEPS/2)); cs2=$((STEPS/2)); else cs1=$s1; cs2=$s2; fi
  case "$lm" in pure_kd) mt=kd;; ce_only) mt=ceo;; mce) mt=mce;; *) mt=kd;; esac
  case "$lm" in ce_only|mce) tt=noT;; *) tt=smol;; esac
  local outdir="outputs/lmdist_${tt}_${mt}_${m}_l${lam}_nz${nz}_a${an}_${nm}_s1${cs1}_seed${sd}_${SWEEP_TAG}"
  if [ -s "$outdir/results.json" ]; then echo "[skip] $spec already done ($outdir)"; return; fi
  local armlog="outputs/swfw_${SWEEP_TAG}_${tt}_${mt}_${m}_nz${nz}_a${an}_${nm}_seed${sd}.log"
  echo "[fleet] START gpu$gpu :: $lm $m nz$nz a$an $nm seed$sd -> $outdir"; date -u
  # torchrun (nproc=1) NOT bare python3: base-image torch lives in torchrun's python, not
  # system python3 (fleet arms otherwise ImportError instantly). world=1 -> single-GPU path.
  # distinct master_port per GPU so 8 concurrent single-proc runs don't collide on rdzv.
  CUDA_VISIBLE_DEVICES=$gpu torchrun --nproc_per_node=1 --master_addr=127.0.0.1 \
    --master_port=$((29500 + gpu)) distill_lm.py --teacher "$TEACHER" $stu \
    --method "$m" --loss_mode "$lm" --seq_len $SEQ --batch_size $MB --grad_accum $GA \
    --lr $LR --lr_sched "$LR_SCHED" --warmup_frac 0.1 --fixed_lambda "$lam" --merge_k $mk \
    --noise_sigma "$nz" --anchor "$an" --noise_mode "$nm" --noise_k "$NOISE_K" \
    --stage1_steps $cs1 --stage2_steps $cs2 --temperature 1.0 \
    --eval_every $EVAL --ckpt_every $CKPT --seed "$sd" --device cuda \
    > "$armlog" 2>&1 \
    && { echo "[fleet] DONE gpu$gpu :: $spec"; echo "  result: $(grep -ioE 'final[^,}]*loss[^,}]*[0-9.]+|val_loss[^,}]*[0-9.]+' "$armlog" | tail -1)"; } \
    || { echo "[fleet] FAIL gpu$gpu :: $spec"; echo "---- $armlog (last 100) ----"; tail -100 "$armlog"; echo "---- end ----"; }
}

# dispatch: assign each arm to a genuinely FREE GPU slot (slot whose prior process
# has exited). Never places two arms on the same GPU. Handles #arms > NG via reuse.
declare -a SLOT_PID
for ((g=0; g<NG; g++)); do SLOT_PID[$g]=0; done
for spec in $ARMS; do
  placed=0
  while [ $placed -eq 0 ]; do
    for ((g=0; g<NG; g++)); do
      pid=${SLOT_PID[$g]}
      if [ "$pid" = "0" ] || ! kill -0 "$pid" 2>/dev/null; then
        launch "$spec" "$g" & SLOT_PID[$g]=$!
        placed=1; break
      fi
    done
    [ $placed -eq 0 ] && sleep 10
  done
done
wait
echo "== FLEET all arms done =="

if [ "${AZBACKUP:-1}" = "1" ]; then
  ( command -v azcopy >/dev/null 2>&1 || { cd /tmp && curl -sL https://aka.ms/downloadazcopy-v10-linux -o az.tgz && tar xzf az.tgz && cp azcopy_linux_*/azcopy /usr/local/bin/; } ) >/dev/null 2>&1
  AZDEST=${AZDEST:-https://aifrontiers.blob.core.windows.net/data/bonete/t-jackcai/supd-backup}
  AZCOPY_AUTO_LOGIN_TYPE=WORKLOAD azcopy copy "$TOY/outputs" "$AZDEST/" --recursive --overwrite=ifSourceNewer \
    --include-pattern "*${SWEEP_TAG}*" >/dev/null 2>&1 && echo "[azbackup] mirrored $SWEEP_TAG -> Azure" || echo "[azbackup] skipped"
fi
