"""Download TinyStories and pre-tokenize a subset to uint16 .bin memmaps.

Run on the LOGIN node (needs internet). Writes data_cache/tinystories/{train,val}.bin
which the offline training jobs memmap. GPT-2 tokenizer (cached).
"""

from __future__ import annotations

import os

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

OUT = os.environ.get("SD_DATA_DIR", "data_cache/tinystories")
TOKENIZER = os.environ.get("SD_TOKENIZER", "gpt2")
TRAIN_TOKENS = int(os.environ.get("SD_TRAIN_TOKENS", 50_000_000))
VAL_TOKENS = int(os.environ.get("SD_VAL_TOKENS", 1_000_000))
os.makedirs(OUT, exist_ok=True)

tok = AutoTokenizer.from_pretrained(TOKENIZER)
EOT = tok.eos_token_id if tok.eos_token_id is not None else 0
assert tok.vocab_size <= 65535, f"vocab {tok.vocab_size} > uint16; widen dtype"


def dump(split, budget, path):
    ds = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
    buf = np.empty(budget + 2048, dtype=np.uint16)
    n = 0
    for ex in ds:
        ids = tok(ex["text"], add_special_tokens=False)["input_ids"]
        ids.append(EOT)
        if n + len(ids) > budget:
            ids = ids[: budget - n]
        buf[n:n + len(ids)] = np.array(ids, dtype=np.uint16)
        n += len(ids)
        if n >= budget:
            break
    buf[:n].tofile(path)
    print(f"{split}: wrote {n:,} tokens -> {path}")


dump("validation", VAL_TOKENS, os.path.join(OUT, "val.bin"))
dump("train", TRAIN_TOKENS, os.path.join(OUT, "train.bin"))
print("vocab_size =", tok.vocab_size)
