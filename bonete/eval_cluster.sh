#!/usr/bin/env bash
# Bonete vLLM eval sweep: for a training run dir on PVC (has checkpoints.json from
# train.py), eval every FLOP-tagged checkpoint on GSM8K + MATH-500 and merge into an
# accuracy-vs-FLOPs curve. Decoupled from training (slow 16K-token generation).
#
#   RUN_DIR=/mnt/pvc/t-jackcai/outputs/<job>/<run>  bash bonete/eval_cluster.sh
# Optional: BENCHMARKS=gsm8k,math500  N=200  MAX_NEW_TOKENS=16384
set -euo pipefail

echo "== superposition-distillation vLLM eval sweep =="; date -u; nvidia-smi -L || true

export USER_ALIAS=${USER_ALIAS:-${USER%@*}}
export PVC_MOUNT=${PVC_MOUNT:-/mnt/pvc}
export PVC_USER_ROOT=${PVC_USER_ROOT:-${PVC_MOUNT}/${USER_ALIAS}}
export HF_HOME=${HF_HOME:-${PVC_USER_ROOT}/hf-cache}
export HF_HUB_CACHE=${HF_HUB_CACHE:-${HF_HOME}/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${HF_HOME}/datasets}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-${PVC_USER_ROOT}/pip-cache}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-${PVC_USER_ROOT}/.cache}   # catches vLLM/flashinfer
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-${PVC_USER_ROOT}/.cache/vllm}
export FLASHINFER_DISABLE_VERSION_CHECK=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME"

RUN_DIR=${RUN_DIR:?set RUN_DIR=<PVC run dir with checkpoints.json>}
BENCHMARKS=${BENCHMARKS:-gsm8k,math500}
BENCHMARKS=${BENCHMARKS//+/,}   # submit_job.sh --extra-env-vars splits on ',' -> pass 'gsm8k+math500'
EVAL_N=${EVAL_N:-200}    # NB: don't name this 'N' -- YAML 1.1 parses N as boolean
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-16384}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}   # model ctx ceiling (gen capped to fit)
# checkpoints were saved by transformers 5.x whose tokenizer_config.json is unreadable
# under vLLM's transformers<5 pin -> load the tokenizer from the hub id instead
TOKENIZER=${TOKENIZER:-Qwen/Qwen2.5-Math-1.5B-Instruct}
MANIFEST="${RUN_DIR}/checkpoints.json"
[ -f "$MANIFEST" ] || { echo "ERROR: no $MANIFEST"; exit 1; }

# --- vLLM via a FRESH per-pod uv venv on container-local disk, wheels cached on PVC.
# Why not pip-into-image: vllm swaps torch 2.8->2.9 under image-resident native libs
# -> C++ std::bad_alloc at import. Why not a prebuilt PVC venv (tried): image lacks
# ensurepip; --system-site-packages leaks the image's flash_attn (ABI crash, and pip
# inside a venv silently refuses to uninstall it); venvs are not relocatable; PVC
# venvs need cross-pod locks + self-heal and import slowly over NFS. A FULLY-ISOLATED
# uv venv per pod kills the whole class: stateless, no image leakage, local-NVMe
# imports; ~2-5 min/pod after the first pod warms the PVC uv cache.
export UV_CACHE_DIR=${UV_CACHE_DIR:-${PVC_USER_ROOT}/uv-cache}
mkdir -p "$UV_CACHE_DIR"
python3 -m pip install --quiet uv
EVAL_VENV=/tmp/supd-eval-venv          # container-local: fast, dies with the pod
uv venv --python python3 "$EVAL_VENV"
uv pip install --python "${EVAL_VENV}/bin/python" \
  "vllm==0.13.0" "datasets==4.8.5" "math-verify==0.9.0" "latex2sympy2-extended==1.11.0" wandb
VPY="${EVAL_VENV}/bin/python3"
"$VPY" -c "import vllm" || { echo "[eval] venv self-check FAILED"; exit 1; }
echo "[eval] using uv venv python: $VPY (vllm $("$VPY" -c 'import vllm;print(vllm.__version__)'))"

# ANCHOR=1: eval the RAW student (zero-FLOP control) instead of the checkpoint sweep.
# Downloads the hub model and patches its config with the SAME RoPE extension the
# training runs use (theta 500000, 32K) so the protocol matches gstep-0 exactly.
if [ "${ANCHOR:-0}" = "1" ]; then
  # ANCHOR_MODEL = hub id of the raw student (default math-instruct).
  # ANCHOR_PATCH_ROPE=1 applies the training RoPE extension (Math models, native 4096);
  # =0 evals the model AS-IS (Qwen2.5-1.5B-Instruct etc. are already 32K -> no patch).
  AM=${ANCHOR_MODEL:-Qwen/Qwen2.5-Math-1.5B-Instruct}
  if [ "${ANCHOR_PATCH_ROPE:-1}" = "1" ]; then
    AD="${RUN_DIR}/anchor-raw"
    "$VPY" - "$AD" "$AM" <<'PY'
import json, os, shutil, sys
from huggingface_hub import snapshot_download
ad, model = sys.argv[1], sys.argv[2]
d = snapshot_download(model)
os.makedirs(ad, exist_ok=True)
for f in os.listdir(d):
    p = os.path.join(d, f)
    if os.path.isfile(p) and not f.startswith("."):
        shutil.copy(p, ad)
c = json.load(open(os.path.join(ad, "config.json")))
c["max_position_embeddings"] = 32768
rs = c.get("rope_scaling") or {}
rs.update({"rope_theta": 500000.0, "rope_type": "default"})
c["rope_scaling"] = rs; c["rope_theta"] = 500000.0
json.dump(c, open(os.path.join(ad, "config.json"), "w"), indent=2)
print("anchor (rope-patched) prepared at", ad)
PY
    TARGET="$AD"
  else
    TARGET="$AM"     # native; vLLM downloads it
    echo "anchor (native, no patch): $AM"
  fi
  "$VPY" eval_vllm.py --checkpoint "$TARGET" --benchmarks "$BENCHMARKS" --n "$EVAL_N" \
    --tokenizer "${TOKENIZER}" --max_new_tokens "$MAX_NEW_TOKENS" --max_model_len "$MAX_MODEL_LEN"
  echo "== anchor eval done =="
  exit 0
fi

# eval each checkpoint in its own process (clean GPU/parallel state per model).
# FINALS_ONLY=1 -> just the highest-gstep checkpoint (the run's final model) — for
# heavy protocols (full set / multi-rollout) where a full sweep is too costly.
if [ "${FINALS_ONLY:-0}" = "1" ]; then
  mapfile -t CKPTS < <(python3 -c "import json; m=json.load(open('$MANIFEST')); print(max(m,key=lambda e:e['gstep'])['dir'])")
else
  mapfile -t CKPTS < <(python3 -c "import json;[print(e['dir']) for e in json.load(open('$MANIFEST'))]")
fi
echo "found ${#CKPTS[@]} checkpoints in $MANIFEST"
# protocol-aware skip/resume: a checkpoint is DONE only if its eval_vllm.json matches
# the CURRENT max_new_tokens AND every requested benchmark's k -> old-protocol files
# get re-run, and a killed sweep resumes where it left off. FORCE_EVAL=1 overrides.
need_eval() {
  "$VPY" - "$1/eval_vllm.json" "$MAX_NEW_TOKENS" "$BENCHMARKS" <<'PY'
import json, os, sys
p, mnt, bms = sys.argv[1], int(sys.argv[2]), sys.argv[3]
if not os.path.exists(p): sys.exit(0)
try: r = json.load(open(p))
except Exception: sys.exit(0)
if r.get("max_new_tokens") != mnt: sys.exit(0)
res = r.get("results", {})
for item in bms.split(","):
    pp = item.split(":"); name = pp[0]; k = int(pp[2]) if len(pp) > 2 else 1
    if name not in res or res[name].get("k", 1) != k: sys.exit(0)
sys.exit(1)   # all present at this protocol -> skip
PY
}
for d in "${CKPTS[@]}"; do
  if [ "${FORCE_EVAL:-0}" != "1" ] && ! need_eval "$d"; then echo "=== skip (already at protocol) $d ==="; continue; fi
  echo "=== eval $d ==="
  "$VPY" eval_vllm.py --checkpoint "$d" --benchmarks "$BENCHMARKS" --n "$EVAL_N" --tokenizer "$TOKENIZER" \
    --max_new_tokens "$MAX_NEW_TOKENS" --max_model_len "$MAX_MODEL_LEN" || echo "[warn] eval failed for $d (continuing)"
done

# merge per-checkpoint eval_vllm.json with the FLOP manifest -> accuracy-vs-FLOPs
# curve, and SYNC it back into the training run's wandb (run id from results.json)
"$VPY" - "$MANIFEST" "$RUN_DIR/eval_curve.json" "$RUN_DIR/results.json" <<'PY'
import json, os, sys
manifest, out, results_path = sys.argv[1], sys.argv[2], sys.argv[3]
curve = []
for e in json.load(open(manifest)):
    rec = {"gstep": e["gstep"], "total_flops": e["total_flops"],
           "teacher_flops": e.get("teacher_flops"), "val_loss": e.get("val_loss")}
    ej = os.path.join(e["dir"], "eval_vllm.json")
    if os.path.exists(ej):
        rec["accuracy"] = {k: v["accuracy"] for k, v in json.load(open(ej))["results"].items()}
    curve.append(rec)
json.dump(curve, open(out, "w"), indent=2)
print("wrote", out)
for c in curve:
    print(f"  flops={c['total_flops']:.3e} val={c['val_loss']} acc={c.get('accuracy')}")

run_id = None
try:
    run_id = json.load(open(results_path)).get("wandb_run_id")
except Exception:
    pass
if run_id and os.environ.get("WANDB_API_KEY"):
    try:
        import wandb
        run = wandb.init(project=os.environ.get("WANDB_PROJECT", "superposition-distillation"),
                         id=run_id, resume="allow")
        for c in curve:
            if "accuracy" in c:
                run.log({"flops": c["total_flops"],
                         **{f"eval/{k}": v for k, v in c["accuracy"].items()}})
        finals = [c for c in curve if "accuracy" in c]
        if finals:
            run.summary.update({f"final/{k}": v for k, v in finals[-1]["accuracy"].items()})
        run.finish()
        print(f"synced eval curve into wandb run {run_id}")
    except Exception as ex:
        print(f"[wandb] eval sync skipped ({type(ex).__name__}: {str(ex)[:80]})")
else:
    print("[wandb] no run id or key; eval curve not synced")
PY
echo "== eval sweep done: ${RUN_DIR}/eval_curve.json =="
