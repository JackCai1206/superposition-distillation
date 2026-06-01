"""Data sources -> fixed-length token blocks, plus the Superposed dispatcher.

The data layer is deliberately dumb: it yields plain (input_ids, attention_mask)
batches of shape (B, L). Turning a batch into a Superposed input (baseline /
cross_seq / token_merge) is done by `build_superposed`, so the same stream feeds
every condition for a fair iso-FLOP comparison.
"""

from __future__ import annotations

import itertools

import torch
from datasets import load_dataset

from config import Config
from superpose import (Superposed, superpose_cross_seq, superpose_none,
                       superpose_token_merge)


# ----------------------------- token streams -----------------------------

def _packed_blocks(token_iter, seq_len):
    """Greedily pack a stream of token-id lists into fixed-length blocks."""
    buf = []
    for toks in token_iter:
        buf.extend(toks)
        while len(buf) >= seq_len:
            yield buf[:seq_len]
            buf = buf[seq_len:]


def pretrain_stream(cfg: Config, tokenizer):
    """Streaming FineWeb-Edu -> packed seq_len blocks."""
    ds = load_dataset(cfg.data.pretrain_dataset, name=cfg.data.pretrain_subset,
                      split="train", streaming=True)

    def toks():
        for ex in ds:
            ids = tokenizer(ex["text"], add_special_tokens=False)["input_ids"]
            if ids:
                yield ids + [tokenizer.eos_token_id]

    yield from _packed_blocks(toks(), cfg.data.seq_len)


def reasoning_stream(cfg: Config, tokenizer):
    """Math-reasoning CoT traces -> packed seq_len blocks (problem + solution).

    Default nvidia/OpenMathReasoning split 'cot' = DeepSeek-R1 generated <think>
    traces, fields problem / generated_solution.
    """
    ds = load_dataset(cfg.data.reasoning_dataset, split=cfg.data.reasoning_split,
                      streaming=True)

    def field(ex, *names):
        for n in names:
            if ex.get(n):
                return ex[n]
        return ""

    def toks():
        for ex in ds:
            prob = field(ex, "problem", "question", "prompt")
            sol = field(ex, "generated_solution", "solution", "generation", "answer")
            ids = tokenizer(prob + "\n" + sol, add_special_tokens=False)["input_ids"]
            if ids:
                yield ids + [tokenizer.eos_token_id]

    yield from _packed_blocks(toks(), cfg.data.seq_len)


def synthetic_stream(cfg: Config, vocab_size: int):
    """Debug source: random blocks, no download."""
    g = torch.Generator().manual_seed(cfg.train.seed)
    while True:
        yield torch.randint(0, vocab_size, (cfg.data.seq_len,), generator=g).tolist()


def batched(block_iter, batch_size):
    """Group blocks into (B, L) tensors."""
    it = iter(block_iter)
    while True:
        chunk = list(itertools.islice(it, batch_size))
        if len(chunk) < batch_size:
            return
        ids = torch.tensor(chunk, dtype=torch.long)
        yield ids, torch.ones_like(ids)


# --------------------------- superposed dispatch ---------------------------

def build_superposed(method: str, ids: torch.Tensor, mask: torch.Tensor,
                     cfg: Config) -> tuple[Superposed, float]:
    """Return (Superposed, effective_sequences_per_output_example).

    cross_seq halves the batch (pairs first/second half), so the caller should
    pass an even batch; output batch is B//2 carrying 2 sequences each.
    """
    sc = cfg.superpose
    if method == "none":
        return superpose_none(ids, mask), 1.0
    if method == "cross_seq":
        B = ids.shape[0]
        h = B // 2
        sup = superpose_cross_seq(ids[:h], mask[:h], ids[h:2 * h], mask[h:2 * h],
                                  mix_alpha=sc.mix_alpha, fixed=sc.fixed_lambda)
        return sup, 2.0
    if method == "token_merge":
        return superpose_token_merge(ids, mask, k=sc.merge_k,
                                     mix_alpha=sc.mix_alpha, fixed=sc.fixed_lambda), 1.0
    raise ValueError(f"unknown method {method}")
