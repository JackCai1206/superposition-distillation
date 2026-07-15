"""TinyStories memmap loader (nanoGPT-style). Reads the uint16 .bin files written
by prepare_tinystories.py. Random contiguous blocks; no padding (dense text), so
attention_mask is all ones and the same superpose.py collators apply.
"""

from __future__ import annotations

import os

import numpy as np
import torch

DATA = os.environ.get("SD_DATA_DIR", "data_cache/tinystories")
VOCAB_SIZE = int(os.environ.get("SD_VOCAB", 50257))   # gpt2 default; SmolLM2=49152; Qwen2.5=151936
# uint16 caps vocab at 65535. Big-vocab tokenizers (Qwen/Llama-3/Gemma) need uint32 -> set
# SD_DTYPE=uint32 (MUST match prepare_fineweb.py, else the memmap decodes to garbage token ids).
DTYPE = np.uint32 if os.environ.get("SD_DTYPE", "uint16") == "uint32" else np.uint16


def load_split(split: str):
    path = os.path.join(DATA, f"{split}.bin")
    return np.memmap(path, dtype=DTYPE, mode="r")


def get_batch(data, batch: int, seq_len: int, device, generator: torch.Generator):
    ix = torch.randint(len(data) - seq_len - 1, (batch,), generator=generator)
    x = torch.stack([torch.from_numpy(data[i:i + seq_len].astype(np.int64)) for i in ix])
    mask = torch.ones_like(x)
    return x.to(device), mask.to(device)


@torch.no_grad()
def eval_lm_loss(model, data, device, n_batches=50, batch=32, seq_len=512, generator=None):
    import torch.nn.functional as F
    model.eval()
    g = generator or torch.Generator().manual_seed(1234)
    tot, n = 0.0, 0
    for _ in range(n_batches):
        ids, _ = get_batch(data, batch, seq_len, device, g)
        logits = model(input_ids=ids).logits
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1))
        tot += loss.item(); n += 1
    return tot / max(n, 1)
