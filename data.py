"""Data sources -> fixed-length token blocks, plus the Superposed dispatcher.

The data layer is deliberately dumb: it yields plain (input_ids, attention_mask)
batches of shape (B, L). Turning a batch into a Superposed input (baseline /
cross_seq / token_merge) is done by `build_superposed`, so the same stream feeds
every condition for a fair iso-FLOP comparison.
"""

from __future__ import annotations

import itertools
import os

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


def reasoning_stream(cfg: Config, tokenizer, rank=0, world=1):
    """Math-reasoning CoT traces (problem + solution), sharded ACROSS RANKS BEFORE
    tokenization (rank takes every world-th raw example) -- sharding after the
    tokenizer would make every rank tokenize ALL examples and discard (world-1)/world.

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

    # SD_MAX_EXAMPLES>0: restrict to the first N raw examples (sharded across ranks)
    # and REPEAT them across epochs. Data-constrained training repeats ~as well as
    # fresh tokens up to ~4 epochs (Muennighoff et al. 2023), and a curated, fully-
    # cycled pool beats <1 epoch of a giant set. 0 = stream once (no repetition).
    cap = int(os.environ.get("SD_MAX_EXAMPLES", "0"))

    def toks():
        for i, ex in enumerate(ds):
            if cap and i >= cap:
                break
            if i % world != rank:
                continue
            prob = field(ex, "problem", "question", "prompt")
            sol = field(ex, "generated_solution", "solution", "generation", "answer")
            ids = tokenizer(prob + "\n" + sol, add_special_tokens=False)["input_ids"]
            if ids:
                yield ids + [tokenizer.eos_token_id]

    min_len = int(os.environ.get("SD_MIN_LEN", "0"))   # smoke knob: probe the LONG end
    seq_len = cfg.data.seq_len
    packed = getattr(cfg.data, "reasoning_packed", False)

    if not cap:
        # original single-pass streaming behavior (one full CoT per example; SKIP
        # overlong instead of truncating -- a cut trace loses its \boxed answer).
        if packed:
            yield from _packed_blocks(toks(), seq_len)
        else:
            for ids in toks():
                if min_len <= len(ids) <= seq_len:
                    yield ids
        return

    # capped + repeated pool. Each rank INDEPENDENTLY builds (or loads from PVC cache)
    # its OWN shard: stream, keep every world-th raw example, BATCH-tokenize (parallel),
    # filter, cache. NO cross-rank coordination/barrier (that races on shared files);
    # all ranks tokenize their 1/world concurrently (~4 min cold, instant on resubmit/
    # resume). Then cycle the shard across epochs, reshuffled each pass.
    import random, pickle, hashlib, time
    cache_dir = os.environ.get(
        "SD_POOL_CACHE", os.path.join(os.environ.get("HF_HOME", "/tmp"), "sd_pool"))
    os.makedirs(cache_dir, exist_ok=True)
    key = hashlib.md5(
        f"{cfg.data.reasoning_dataset}|{cfg.data.reasoning_split}|{cap}|{seq_len}|{min_len}|{world}|{rank}"
        .encode()).hexdigest()[:12]
    cache = os.path.join(cache_dir, f"pool_{key}.pkl")

    pool = None
    if os.path.exists(cache):
        try:
            with open(cache, "rb") as f:
                pool = pickle.load(f)
        except Exception:
            pool = None   # corrupt/partial cache -> rebuild
    if pool is None:
        t0 = time.time()
        os.environ["TOKENIZERS_PARALLELISM"] = "true"   # parallel batch tokenization
        eos = tokenizer.eos_token_id
        texts = []
        for i, ex in enumerate(ds):
            if i >= cap:
                break
            if i % world != rank:
                continue
            texts.append(field(ex, "problem", "question", "prompt") + "\n"
                         + field(ex, "generated_solution", "solution", "generation", "answer"))
        pool = []
        for j in range(0, len(texts), 1000):
            for ids in tokenizer(texts[j:j+1000], add_special_tokens=False)["input_ids"]:
                if not ids:
                    continue
                ids = ids + [eos]
                if min_len <= len(ids) <= seq_len:
                    pool.append(ids)
        del texts
        tmp = cache + f".tmp{os.getpid()}"
        with open(tmp, "wb") as f:
            pickle.dump(pool, f, protocol=4)
        os.rename(tmp, cache)   # atomic -> readers never see a partial file
        print(f"[pool] rank {rank} built {len(pool)} ex ({time.time()-t0:.0f}s)", flush=True)
    if not pool:
        raise RuntimeError(f"reasoning_stream: empty pool (cap={cap}) rank {rank}")
    print(f"[pool] rank {rank} pool={len(pool)} examples", flush=True)
    rng = random.Random(cfg.train.seed + rank)
    if packed:
        while True:
            rng.shuffle(pool)
            yield from _packed_blocks(iter(pool), seq_len)
    else:
        while True:
            rng.shuffle(pool)
            for ids in pool:
                yield ids


def synthetic_stream(cfg: Config, vocab_size: int):
    """Debug source: random blocks, no download."""
    g = torch.Generator().manual_seed(cfg.train.seed)
    while True:
        yield torch.randint(0, vocab_size, (cfg.data.seq_len,), generator=g).tolist()


def batched(block_iter, batch_size, bucket_batches=64):
    """Group blocks into (B, L) tensors, right-padding to the batch max length,
    with LENGTH BUCKETING: buffer ~bucket_batches batches worth of examples, sort
    by length, emit neighbor batches. With high length variance (OpenMathReasoning
    cot: mean ~7.9K, cap 16K) naive batching wastes ~50% of positions as padding;
    bucketing makes padded ~= real tokens and naturally length-pairs cross_seq.
    Packed blocks are all seq_len -> sort is a no-op (== old behaviour)."""
    # knobs for future sweeps (defaults preserve current behaviour):
    #   SD_BUCKET_BATCHES=1  -> no cross-batch bucketing (naive batching; ~50% padding
    #                           waste at 16K variance but no ordering artifact)
    #   SD_BUCKET_SHUFFLE=1  -> keep bucketing efficiency but emit the buffer's batches
    #                           in random order (kills the 16-step short->long sawtooth)
    bucket_batches = int(os.environ.get("SD_BUCKET_BATCHES", bucket_batches))
    shuffle = os.environ.get("SD_BUCKET_SHUFFLE", "0") == "1"
    rng = torch.Generator().manual_seed(0)
    it = iter(block_iter)
    while True:
        buf = list(itertools.islice(it, batch_size * bucket_batches))
        if len(buf) < batch_size:
            return
        buf.sort(key=len)
        starts = list(range(0, len(buf) - batch_size + 1, batch_size))
        if shuffle:
            starts = [starts[i] for i in torch.randperm(len(starts), generator=rng).tolist()]
        for i in starts:
            chunk = buf[i:i + batch_size]
            maxlen = max(len(b) for b in chunk)
            ids = torch.zeros((batch_size, maxlen), dtype=torch.long)
            mask = torch.zeros((batch_size, maxlen), dtype=torch.long)
            for j, b in enumerate(chunk):
                ids[j, :len(b)] = torch.tensor(b, dtype=torch.long)
                mask[j, :len(b)] = 1
            yield ids, mask


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
        # pair length-NEIGHBORS (sort by real length, take alternating indices):
        # the superposed mask is the AND of the pair's masks, so mismatched lengths
        # waste positions where only one sequence is real.
        order = mask.sum(1).argsort()
        a, b = order[0::2], order[1::2]
        sup = superpose_cross_seq(ids[a], mask[a], ids[b], mask[b],
                                  mix_alpha=sc.mix_alpha, fixed=sc.fixed_lambda)
        return sup, 2.0
    if method == "token_merge":
        return superpose_token_merge(ids, mask, k=sc.merge_k,
                                     mix_alpha=sc.mix_alpha, fixed=sc.fixed_lambda), 1.0
    raise ValueError(f"unknown method {method}")
