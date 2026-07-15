#!/usr/bin/env bash
# CPU job: tokenize a ~1.2B-token FineWeb-Edu slice with the SmolLM2 tokenizer -> PVC.
set -uo pipefail
echo "== FineWeb-Edu prep =="; date -u
USER_ALIAS=${USER_ALIAS:-${USER%@*}}; PVC=/mnt/pvc/${USER_ALIAS}
export HF_HOME=${HF_HOME:-$PVC/hf-cache} HF_HUB_CACHE=${HF_HUB_CACHE:-$PVC/hf-cache/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$PVC/hf-cache/datasets} PIP_CACHE_DIR=${PIP_CACHE_DIR:-$PVC/pip-cache} TOKENIZERS_PARALLELISM=true
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$PIP_CACHE_DIR"
export SD_TOKENIZER=${SD_TOKENIZER:-HuggingFaceTB/SmolLM2-135M}
export SD_DATA_DIR=${SD_DATA_DIR:-$PVC/toys/data_cache/fineweb_edu_smol}
export SD_TRAIN_TOKENS=${SD_TRAIN_TOKENS:-1200000000}
export SD_VAL_TOKENS=${SD_VAL_TOKENS:-5000000}
export FW_CONFIG=${FW_CONFIG:-sample-10BT}
mkdir -p "$SD_DATA_DIR"
LOG="$PVC/toys/fineweb_prep_$(date -u +%Y%m%d-%H%M%S).log"; mkdir -p "$PVC/toys"; exec > >(tee -a "$LOG") 2>&1
python3 -m pip install -q --cache-dir "$PIP_CACHE_DIR" "transformers==5.9.0" "datasets==4.8.5" numpy 2>&1 | tail -1
echo "tok=$SD_TOKENIZER out=$SD_DATA_DIR train=$SD_TRAIN_TOKENS"
[ -s "$SD_DATA_DIR/train.bin" ] && { echo "already prepped"; exit 0; }
python3 prepare_fineweb.py
echo "== prep done =="; ls -la "$SD_DATA_DIR"
