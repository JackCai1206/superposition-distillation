#!/usr/bin/env bash
# CPU-only entrypoint: build the capped reasoning-pool cache to PVC so GPU training arms
# skip the slow idle build. No GPU requested -> not a target for the GPU-budget reaper.
set -euo pipefail
echo "== prebuild pool cache =="; date -u
export USER_ALIAS=${USER_ALIAS:-${USER%@*}}
export PVC_MOUNT=${PVC_MOUNT:-/mnt/pvc}
export PVC_USER_ROOT=${PVC_USER_ROOT:-${PVC_MOUNT}/${USER_ALIAS}}
export HF_HOME=${HF_HOME:-${PVC_USER_ROOT}/hf-cache}        # MUST match run_cluster.sh
export HF_HUB_CACHE=${HF_HUB_CACHE:-${HF_HOME}/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${HF_HOME}/datasets}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-${PVC_USER_ROOT}/pip-cache}
export TOKENIZERS_PARALLELISM=true
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$PIP_CACHE_DIR"
python3 -m pip install --upgrade pip
python3 -m pip install --cache-dir "$PIP_CACHE_DIR" "transformers==5.9.0" "datasets==4.8.5" "accelerate>=1.0"
python3 prebuild_pool.py
echo "== prebuild done =="; ls -la "${HF_HOME}/sd_pool" 2>/dev/null || true
