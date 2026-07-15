#!/usr/bin/env bash
# ONE-TIME: re-tokenize FineWeb-Edu with the Qwen2.5 tokenizer -> uint32 .bin memmaps.
# Qwen vocab (152k) exceeds uint16 (65535), so SD_DTYPE=uint32 (prepare_fineweb.py + nl_data.py
# both honor SD_DTYPE). Writes <PVC>/toys/data_cache/fineweb_edu_qwen/{train,val}.bin, which the
# Qwen noise-KD launcher reads via DATA_SUBDIR=fineweb_edu_qwen. CPU-bound (stream + tokenize).
set -uo pipefail
echo "== FineWeb-Edu QWEN prep =="; date -u
USER_ALIAS=${USER_ALIAS:-${USER%@*}}
_b=${PVC_ROOT:-/mnt/pvc}
PVC=""
for _c in "$_b/${USER_ALIAS}" "$_b/experiments/${USER_ALIAS}" "/mnt/pvc/experiments/${USER_ALIAS}" "/mnt/pvc/${USER_ALIAS}"; do
  [ -d "$_c/toys" ] && { PVC="$_c"; break; }
done
PVC="${PVC:-$_b/${USER_ALIAS}}"
export HF_HOME=${HF_HOME:-$PVC/hf-cache} HF_HUB_CACHE=${HF_HUB_CACHE:-$PVC/hf-cache/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$PVC/hf-cache/datasets} PIP_CACHE_DIR=${PIP_CACHE_DIR:-$PVC/pip-cache} TOKENIZERS_PARALLELISM=true
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$PIP_CACHE_DIR"
export SD_TOKENIZER=${SD_TOKENIZER:-Qwen/Qwen2.5-1.5B}   # same Qwen2.5 tokenizer as the 7B teacher
export SD_DTYPE=uint32
export SD_DATA_DIR=${SD_DATA_DIR:-$PVC/toys/data_cache/fineweb_edu_qwen}
export SD_TRAIN_TOKENS=${SD_TRAIN_TOKENS:-4000000000}    # 4B tok (> the ~2.1B the small-ablation run needs; no repeats)
export SD_VAL_TOKENS=${SD_VAL_TOKENS:-5000000}
export FW_CONFIG=${FW_CONFIG:-sample-10BT}
mkdir -p "$SD_DATA_DIR"
[ -s "$SD_DATA_DIR/train.bin" ] && { echo "already prepped: $SD_DATA_DIR"; exit 0; }
python3 -m pip install -q --cache-dir "$PIP_CACHE_DIR" "transformers==5.9.0" "datasets==4.8.5" numpy 2>&1 | tail -1
python3 -m pip uninstall -y kernels 2>&1 | tail -1 || true
echo "tok=$SD_TOKENIZER dtype=$SD_DTYPE out=$SD_DATA_DIR train=$SD_TRAIN_TOKENS"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" && python3 prepare_fineweb.py
echo "== qwen prep done =="; ls -la "$SD_DATA_DIR"
