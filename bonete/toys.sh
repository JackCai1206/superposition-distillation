#!/usr/bin/env bash
# Toy superposition-distillation (addition + TinyStories) with the CLEAN
# pure-forward-KL setup: NO alpha/CE mix anywhere -> baseline(none) vs cross_seq
# differ ONLY by input superposition. Multi-seed, none/cross_seq/token_merge, iso-FLOP.
# Tiny models -> packs many cells per B200; 1-2 GPU is plenty.
set -uo pipefail
echo "== TOY pure-KL distillation =="; date -u
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

USER_ALIAS=${USER_ALIAS:-${USER%@*}}
PVC=/mnt/pvc/${USER_ALIAS}
export HF_HOME=${HF_HOME:-$PVC/hf-cache}; export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}; export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$PVC/pip-cache}
export TOKENIZERS_PARALLELISM=false
TOY=$PVC/toys
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$PIP_CACHE_DIR" "$TOY/outputs" "$TOY/data_cache" "$TOY/teachers"
# the repo ships an outputs/ dir; ln -sfn into an existing dir nests the link inside
# it (outputs/outputs) -> cells would write to the EPHEMERAL pod disk. Force-replace.
rm -rf outputs data_cache
ln -sfn "$TOY/outputs" outputs
ln -sfn "$TOY/data_cache" data_cache
[ -L outputs ] && [ "$(readlink outputs)" = "$TOY/outputs" ] || { echo "FATAL: outputs symlink not set"; exit 1; }
LOG="$TOY/run_$(date -u +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "outputs -> $TOY/outputs ; log -> $LOG"

python3 -m pip install -q --cache-dir "$PIP_CACHE_DIR" "transformers==5.9.0" "datasets==4.8.5" numpy matplotlib 2>&1 | tail -1

# CUDA MPS: lets the many tiny cells SHARE each GPU concurrently instead of
# timeslicing (timeslicing ~= running them serially -> the slowness). Big win for
# packing many small models per card. Graceful fallback if MPS can't start.
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
if command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
  nvidia-cuda-mps-control -d >/dev/null 2>&1 && sleep 2 && echo "[mps] daemon started" || echo "[mps] start failed -> timeslice"
else
  echo "[mps] unavailable -> timeslice"
fi

SEEDS=${SEEDS:-"0 1 2 3 4"}
METHODS=${METHODS:-"none cross_seq token_merge"}
TASKS=${TASKS:-"add lm"}                     # which toys to run
# fewer total steps (cut the saturated tail) + dense eval (fine curve / threshold res)
ADD_S1=${ADD_S1:-6000}; ADD_S2=${ADD_S2:-1500}; ADD_EVAL=${ADD_EVAL:-1000}
LM_S1=${LM_S1:-1500};   LM_S2=${LM_S2:-3000};   LM_EVAL=${LM_EVAL:-100}
MAXPAR=${MAXPAR:-8}                         # cells packed per GPU
NG=$(nvidia-smi -L 2>/dev/null | wc -l); [ "${NG:-0}" -lt 1 ] && NG=1
CAP=$(( MAXPAR * NG ))
echo "GPUs=$NG seeds='$SEEDS' methods='$METHODS' cap=$CAP"

# 1) TinyStories data (cache on PVC)
[ -s data_cache/tinystories/train.bin ] || { echo "== prep TinyStories =="; python3 prepare_tinystories.py; }

# 2) Teachers (cache on PVC)
ADD_T=$TOY/teachers/addition_d4
LM_T=$TOY/teachers/lm_ctrl_h512l8
[ -s "$ADD_T/model.safetensors" ] || { echo "== train addition teacher =="; python3 train_addition.py --n_digits 4 --steps 8000 --out "$ADD_T"; }
[ -s "$LM_T/model.safetensors" ]  || { echo "== train LM teacher =="; python3 train_lm.py --hidden 512 --layers 8 --steps 6000 --out "$LM_T"; }

# 3) Distill cells in parallel (pure_kd), round-robin across GPUs, capped
echo "== launching distill cells (pure_kd) =="; n=0
launch(){ local gpu=$(( n % NG )); n=$((n+1))
  if [ "$1" = add ]; then
    CUDA_VISIBLE_DEVICES=$gpu python3 distill_addition.py --teacher "$ADD_T" --method "$2" --loss_mode pure_kd --seed "$3" \
      --stage1_steps "$ADD_S1" --stage2_steps "$ADD_S2" --eval_every "$ADD_EVAL" --device cuda > "outputs/cell_add_${2}_s${3}.log" 2>&1 &
  else
    CUDA_VISIBLE_DEVICES=$gpu python3 distill_lm.py --teacher "$LM_T" --method "$2" --loss_mode pure_kd --seed "$3" \
      --stage1_steps "$LM_S1" --stage2_steps "$LM_S2" --eval_every "$LM_EVAL" --device cuda > "outputs/cell_lm_${2}_s${3}.log" 2>&1 &
  fi
  [ $(( n % CAP )) -eq 0 ] && wait
}
for task in $TASKS; do for m in $METHODS; do for s in $SEEDS; do launch "$task" "$m" "$s"; done; done; done
wait
echo "== all cells done =="

# 4) Multi-seed iso-FLOP readout (printed to log; full results.json on PVC)
python3 - <<'PY'
import glob, json, statistics as st
def load(pat):
    R={}
    for p in sorted(glob.glob(pat)):
        try: r=json.load(open(p))
        except Exception: continue
        R.setdefault(r["method"],[]).append(r)
    return R
def at_flops(hist, budget, key):
    best=None
    for h in hist:
        if h["flops"]<=budget: best=h[key]
    return best
def report(name, pat, key, better):
    R=load(pat)
    if not R: print(f"\n[{name}] no results"); return
    print(f"\n===== {name}  (pure-KL, multi-seed; metric={key}, better={better}) =====")
    # common iso-FLOP budget = min final total_flops across all runs
    budget=min(r["flops"]["total_flops"] for rs in R.values() for r in rs)
    print(f"{'method':>11} {'n':>2} {'final_'+key:>16} {'@isoFLOP':>16}  (mean±std)")
    base=None
    for m,rs in R.items():
        fin=[r["final"][key] if "final" in r and r["final"] else r["history"][-1][key] for r in rs]
        iso=[at_flops(r["history"], budget, key) for r in rs]
        iso=[x for x in iso if x is not None]
        fm,fs=st.mean(fin),(st.pstdev(fin) if len(fin)>1 else 0)
        im,isd=(st.mean(iso),(st.pstdev(iso) if len(iso)>1 else 0)) if iso else (float('nan'),0)
        print(f"{m:>11} {len(rs):>2}   {fm:>7.4f}±{fs:<6.4f}   {im:>7.4f}±{isd:<6.4f}")
        if m=="none": base=(im,isd)
    if base and "cross_seq" in R:
        cs=[at_flops(r["history"], budget, key) for r in R["cross_seq"]]; cs=[x for x in cs if x is not None]
        if cs:
            d=st.mean(cs)-base[0]
            sign="+" if d>=0 else ""
            print(f"  iso-FLOP cross_seq - none = {sign}{d:.4f}  ({'cross_seq better' if (d>0)==(better=='higher') else 'none better'})")
    print(f"  (iso-FLOP budget = {budget:.3e})")
report("ADDITION",     "outputs/distadd_kd_*/results.json", "exact_match", "higher")
report("TINYSTORIES",  "outputs/lmdist_*_kd_*/results.json", "val_loss",   "lower")
PY
echo "== done; results + cell logs under $TOY/outputs =="
