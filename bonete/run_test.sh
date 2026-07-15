#!/usr/bin/env bash
set -uo pipefail
PVC=/mnt/pvc/t-jackcai
export HF_HOME=${HF_HOME:-$PVC/hf-cache} PIP_CACHE_DIR=${PIP_CACHE_DIR:-$PVC/pip-cache} TOKENIZERS_PARALLELISM=false
python3 -m pip install -q --cache-dir "$PIP_CACHE_DIR" "transformers==5.9.0" numpy 2>&1 | tail -1
echo "=== running infra tests (CPU) ==="
python3 test_infra.py
