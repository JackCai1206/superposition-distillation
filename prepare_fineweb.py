"""Stream FineWeb-Edu and pre-tokenize a ~1.2B-token slice to uint16 .bin memmaps
with the SmolLM2 tokenizer (so a pretrained SmolLM2 teacher + scaled SmolLM2 student
share the vocab). Writes <SD_DATA_DIR>/{train,val}.bin which distill reads via nl_data.

Env: SD_TOKENIZER (SmolLM2), SD_DATA_DIR, SD_TRAIN_TOKENS, SD_VAL_TOKENS, FW_CONFIG.
"""
from __future__ import annotations

import os
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

OUT = os.environ.get("SD_DATA_DIR", "data_cache/fineweb_edu_smol")
TOKENIZER = os.environ.get("SD_TOKENIZER", "HuggingFaceTB/SmolLM2-135M")
CONFIG = os.environ.get("FW_CONFIG", "sample-10BT")          # 10B-token sample subset
TRAIN_TOKENS = int(os.environ.get("SD_TRAIN_TOKENS", 1_200_000_000))
VAL_TOKENS = int(os.environ.get("SD_VAL_TOKENS", 5_000_000))
os.makedirs(OUT, exist_ok=True)
# uint16 (default) caps vocab at 65535 (SmolLM2 49152 fits). For big-vocab tokenizers
# (Qwen2.5 152k, Llama-3 128k, Gemma 256k) set SD_DTYPE=uint32. nl_data.py MUST read
# the same dtype (also via SD_DTYPE) or the memmap is garbage.
DTYPE = np.uint32 if os.environ.get("SD_DTYPE", "uint16") == "uint32" else np.uint16

tok = AutoTokenizer.from_pretrained(TOKENIZER)
EOT = tok.eos_token_id if tok.eos_token_id is not None else 0
assert tok.vocab_size <= np.iinfo(DTYPE).max, f"vocab {tok.vocab_size} > {DTYPE.__name__} max; set SD_DTYPE=uint32"
print(f"[prep] dataset=HuggingFaceFW/fineweb-edu config={CONFIG} tok={TOKENIZER} vocab={tok.vocab_size}")
print(f"[prep] val={VAL_TOKENS:,} then train={TRAIN_TOKENS:,} tokens -> {OUT}")

# FineWeb-Edu has only a 'train' split; stream once, fill val first, then train.
ds = load_dataset("HuggingFaceFW/fineweb-edu", name=CONFIG, split="train", streaming=True)
it = iter(ds)


def dump(budget, path):
    buf = np.empty(budget + 4096, dtype=DTYPE)
    n = 0
    for ex in it:
        ids = tok(ex["text"], add_special_tokens=False)["input_ids"]
        ids.append(EOT)
        if n + len(ids) > budget:
            ids = ids[: budget - n]
        buf[n:n + len(ids)] = np.array(ids, dtype=DTYPE)
        n += len(ids)
        if n % 50_000_000 < len(ids):
            print(f"  ...{n:,}/{budget:,}", flush=True)
        if n >= budget:
            break
    buf[:n].tofile(path)
    print(f"wrote {n:,} tokens -> {path}", flush=True)


dump(VAL_TOKENS, os.path.join(OUT, "val.bin"))
dump(TRAIN_TOKENS, os.path.join(OUT, "train.bin"))
print("done. vocab_size =", tok.vocab_size, flush=True)
# The datasets streaming (parquet) generator raises a harmless AttributeError in __del__ during
# interpreter shutdown and core-dumps -> a job would falsely report failure. Data is fully written
# above, so exit hard here to skip that teardown path.
os._exit(0)
