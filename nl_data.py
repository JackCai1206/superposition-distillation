"""TinyStories memmap loader (nanoGPT-style). Reads the uint16 .bin files written
by prepare_tinystories.py. Random contiguous blocks; no padding (dense text), so
attention_mask is all ones and the same superpose.py collators apply.
"""

from __future__ import annotations

import os

import numpy as np
import torch

DATA = "data_cache/tinystories"
VOCAB_SIZE = 50257   # gpt2


def load_split(split: str):
    path = os.path.join(DATA, f"{split}.bin")
    return np.memmap(path, dtype=np.uint16, mode="r")


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
