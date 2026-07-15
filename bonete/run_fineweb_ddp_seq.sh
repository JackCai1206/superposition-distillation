#!/usr/bin/env bash
# Run N arms SEQUENTIALLY, each DATA-PARALLEL across the WHOLE node (torchrun DDP,
# NG GPUs x grad_accum=1). eff_batch = NG*MB = 512 == the packed 1-GPU (MB=64, GA=8)
# AND positions/step = NG*MB*L == baseline, for EVERY method (iso-FLOP, default mode --
# NOT --iso_token): so COMPUTE and batch-position ("tokens in the batch") count are
# equalized across arms; raw DATA differs (token_merge eats k x, cross_seq 2x) -- we are
# explicitly NOT controlling data here. token_merge KD runs the teacher on the merged
# input (OOD but defined) and distills it. arm spec: "lossmode:method:lambda:merge_k:seed".
set -uo pipefail
echo "== FineWeb DDP-SEQ =="; date -u
USER_ALIAS=${USER_ALIAS:-${USER%@*}}; PVC=${PVC_ROOT:-/mnt/pvc}/${USER_ALIAS}  # cjob mounts PVC at /data (pass PVC_ROOT=/data); old submit_job.sh uses /mnt/pvc
# data-path auto-detect: survives PVC reorgs (2026-07-03 moved /<root>/<user>/* ->
# /<root>/experiments/<user>/*). Pick the root that actually holds the FineWeb data so a
# relocation doesn't instant-FATAL. Outputs follow the same PVC root (data+outputs co-located).
_b=${PVC_ROOT:-/mnt/pvc}
for _c in "$_b/${USER_ALIAS}" "$_b/experiments/${USER_ALIAS}" "/data/experiments/${USER_ALIAS}" "/mnt/pvc/experiments/${USER_ALIAS}" "/mnt/pvc/${USER_ALIAS}"; do
  if [ -s "$_c/toys/data_cache/${DATA_SUBDIR:-fineweb_edu_smol}/train.bin" ]; then PVC="$_c"; echo "[ddp-seq] FineWeb data found -> PVC=$PVC"; break; fi
done
export HF_HOME=${HF_HOME:-$PVC/hf-cache} HF_HUB_CACHE=${HF_HUB_CACHE:-$PVC/hf-cache/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$PVC/hf-cache/datasets} PIP_CACHE_DIR=${PIP_CACHE_DIR:-$PVC/pip-cache} TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
TOY=$PVC/toys; mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$PIP_CACHE_DIR" "$TOY/outputs"
rm -rf outputs; ln -sfn "$TOY/outputs" outputs
LOG="$TOY/fwddpseq_$(date -u +%Y%m%d-%H%M%S).log"; exec > >(tee -a "$LOG") 2>&1
python3 -m pip install -q --cache-dir "$PIP_CACHE_DIR" "transformers==5.9.0" "datasets==4.8.5" numpy 2>&1 | tail -1
# transformers 5.9.0 imports the `kernels` hub package at module load; a newer kernels release
# made LayerRepository REQUIRE a revision/version -> import crash that killed every fw4bnoise4 arm
# (2026-07-08). We don't use hub kernels for a 135M student, so remove it (transformers guards it
# as optional and falls back cleanly). Unpinned kernels drift is the cause, not our code.
python3 -m pip uninstall -y kernels 2>&1 | tail -1 || true
UTILLOG="$TOY/gpuutil_$(date -u +%Y%m%d-%H%M%S).log"
(nvidia-smi --query-gpu=index,power.draw,power.limit,utilization.gpu,memory.used --format=csv,noheader -l 30 >> "$UTILLOG" 2>&1 &)
export SD_DATA_DIR=${SD_DATA_DIR:-$PVC/toys/data_cache/${DATA_SUBDIR:-fineweb_edu_smol}} SD_VOCAB=${SD_VOCAB:-49152}
[ -s "$SD_DATA_DIR/train.bin" ] || { echo "FATAL: no FineWeb data at $SD_DATA_DIR"; exit 1; }

SWEEP_TAG=${SWEEP_TAG:-fw4bddp}; LR_SCHED=${LR_SCHED:-wsd}
SEQ=${SEQ:-1024}; MB=${MB:-64}; GA=${GA:-1}; STEPS=${STEPS:-8000}; FRAC=${FRAC:-0.3}; EVAL=200; CKPT=200; LR=${LR:-6e-4}
TEACHER=${TEACHER:-HuggingFaceTB/SmolLM2-1.7B}
stu="${STU:---student_ref HuggingFaceTB/SmolLM2-135M --student_hidden 576 --student_layers 30 --student_heads 9 --student_kv_heads 3 --student_inter 1536}"
s1=$(awk "BEGIN{print int($STEPS*$FRAC)}"); s2=$(( STEPS - s1 ))
export SLURM_JOB_ID=$SWEEP_TAG
NG=$(nvidia-smi -L 2>/dev/null | wc -l); [ "${NG:-0}" -lt 1 ] && NG=1
echo "GPUs=$NG tag=$SWEEP_TAG sched=$LR_SCHED eff_batch=$((NG*MB)) (NG*MB, GA=1)"
# --- automatic micro-batch search (AUTO_MICRO=1, on by default): faithful single-GPU OOM probe on
# the REAL fwd+bwd+opt.step path (honors compile/grad_ckpt), then hold the target eff-batch via GA.
AUTO_MICRO=${AUTO_MICRO:-1}; EFF_BATCH_SEQ=${EFF_BATCH_SEQ:-$((NG*64))}
if [ "$AUTO_MICRO" = "1" ]; then
  echo "[ddp-seq] micro-batch search (cap ${FIND_MICRO_CAP:-128}, target eff ${EFF_BATCH_SEQ} seq)…"
  flog="$TOY/find_micro_${SWEEP_TAG}_$(date -u +%H%M%S).log"
  CUDA_VISIBLE_DEVICES=0 python3 distill_lm.py --teacher "$TEACHER" $stu \
    --method none --loss_mode pure_kd --seq_len $SEQ --batch_size 1 --grad_accum 1 \
    --stage1_steps 1 --stage2_steps 0 --noise_sigma 0.4 --noise_k "${NOISE_K:-8}" \
    --eval_every 0 --ckpt_every 0 --device cuda --find_micro "${FIND_MICRO_CAP:-128}" \
    ${EXTRA_TRAIN_ARGS:-} > "$flog" 2>&1 || true
  grep -E "\[find_micro\]|MAX_MICRO" "$flog" 2>/dev/null | tail -20
  fm=$(grep -oE "MAX_MICRO=[0-9]+" "$flog" 2>/dev/null | tail -1 | cut -d= -f2)
  if [ -n "${fm:-}" ] && [ "${fm:-0}" -ge 1 ]; then
    MB=$fm; GA=$(( EFF_BATCH_SEQ / (NG*MB) )); [ "$GA" -lt 1 ] && GA=1
    echo "[ddp-seq] max micro=$MB -> MB=$MB GA=$GA (eff $((NG*MB*GA)) seq)"
  else
    echo "[ddp-seq] micro search inconclusive -> keep MB=$MB GA=$GA (see $flog)"
  fi
fi
# default: k3 MCE (no teacher) then k2 distillation (teacher on merged input)
ARMS=${ARMS:-"mce:token_merge:0.5:3:1 pure_kd:token_merge:0.5:2:1"}
# Reap-forwarding: on eviction the kubelet sends SIGTERM only to PID 1 (this bash); the
# torchrun workers get nothing until the final (uncatchable) SIGKILL. Forward SIGTERM to the
# live torchrun so distill_lm.py's REAP_SIGTERM hook can checkpoint inside the grace window.
TORCH_PID=""
_forward_term() { [ -n "$TORCH_PID" ] && { echo "[ddp-seq] SIGTERM -> torchrun $TORCH_PID (reap)"; kill -TERM "$TORCH_PID" 2>/dev/null; wait "$TORCH_PID" 2>/dev/null; }; exit 143; }
trap _forward_term TERM INT
for spec in $ARMS; do
  # arm spec: "lossmode:method:lambda:merge_k:seed[:noise_sigma[:anchor[:noise_mode[:s1_override]]]]"
  IFS=: read -r lm m lam mk sd nz an nm s1o nk <<< "$spec"; sd=${sd:-1}; mk=${mk:-2}; nz=${nz:-0}; an=${an:-0}; nm=${nm:-onehot}; nk=${nk:-${NOISE_K:-8}}
  if [ "$lm" = ce_only ]; then m=none; cs1=$((STEPS/2)); cs2=$((STEPS/2));
  elif [ -n "$s1o" ]; then cs1=$s1o; cs2=$(( STEPS - s1o ));   # per-arm S1 (noise-phase length) override, total STEPS fixed
  else cs1=$s1; cs2=$s2; fi
  case "$lm" in pure_kd) mt=kd;; ce_only) mt=ceo;; mce) mt=mce;; *) mt=kd;; esac
  case "$lm" in ce_only|mce) tt=noT;; *) tt=smol;; esac
  # noise tag MUST match distill_lm.py's `out` exactly (else skip-check + azbackup point at a
  # nonexistent dir). Empty when all-default so it matches the sigma=0 baseline's historical name.
  if [ "$nz" = 0 ] && [ "$an" = 0 ] && [ "$nm" = onehot ]; then ntag=""; else ntag="_nz${nz}_a${an}_${nm}"; fi
  if [ -n "$ntag" ] && [ "$nk" != 8 ]; then ntag="${ntag}_k${nk}"; fi   # K-sweep: match distill_lm.py's out exactly
  outdir="outputs/lmdist_${tt}_${mt}_${m}_l${lam}${ntag}_s1${cs1}_seed${sd}_${SWEEP_TAG}"
  armlog="outputs/swfw_${SWEEP_TAG}_${tt}_${mt}_${m}_k${mk}_l${lam}_nz${nz}_a${an}_${nm}_s1${cs1}_nk${nk}_seed${sd}.log"
  if [ -s "$outdir/results.json" ]; then echo "[skip] $spec already done ($outdir)"; continue; fi
  echo "[ddp-seq] START arm $lm $m l$lam k$mk seed$sd nz$nz a$an $nm nk$nk on $NG GPUs -> $outdir"; date -u
  torchrun --standalone --nproc_per_node=$NG distill_lm.py --teacher "$TEACHER" $stu \
    --method "$m" --loss_mode "$lm" --seq_len $SEQ --batch_size $MB --grad_accum $GA \
    --lr $LR --lr_sched "$LR_SCHED" --warmup_frac 0.1 --fixed_lambda "$lam" --merge_k $mk \
    --noise_sigma "$nz" --anchor "$an" --noise_mode "$nm" --noise_k "$nk" \
    --stage1_steps $cs1 --stage2_steps $cs2 --temperature 1.0 \
    --eval_every $EVAL --ckpt_every $CKPT --seed "$sd" --device cuda ${EXTRA_TRAIN_ARGS:-} \
    > "$armlog" 2>&1 &
  TORCH_PID=$!                                    # background + wait so the TERM trap can forward
  if wait "$TORCH_PID"; then
    echo "[ddp-seq] DONE arm $spec :: result: $(grep -ioE 'final[^,}]*loss[^,}]*[0-9.]+' "$armlog" | tail -1)"
  else
    echo "[ddp-seq] FAIL arm $spec"; echo "---- $armlog (last 100) ----"; tail -100 "$armlog"; echo "---- end ----"
  fi
  TORCH_PID=""
  date -u
  # Azure auto-mirror: back up this arm's results+log to blob (durable vs PVC reorg/purge).
  # Best-effort, never fails the run. Needs workload identity (aif-bonete-uami SA). AZBACKUP=0 to disable.
  if [ "${AZBACKUP:-1}" = "1" ]; then
    ( command -v azcopy >/dev/null 2>&1 || { cd /tmp && curl -sL https://aka.ms/downloadazcopy-v10-linux -o az.tgz && tar xzf az.tgz && cp azcopy_linux_*/azcopy /usr/local/bin/; } ) >/dev/null 2>&1
    AZDEST=${AZDEST:-https://aifrontiers.blob.core.windows.net/data/bonete/t-jackcai/supd-backup}
    AZCOPY_AUTO_LOGIN_TYPE=WORKLOAD azcopy copy "$TOY/outputs/$(basename $outdir)" "$AZDEST/outputs/" --recursive --overwrite=ifSourceNewer >/dev/null 2>&1 \
      && echo "[azbackup] mirrored $(basename $outdir) -> Azure" || echo "[azbackup] skipped (no workload identity?)"
  fi
done
echo "== DDP-SEQ all done =="
