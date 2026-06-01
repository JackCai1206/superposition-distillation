"""Char-level integer-addition task with LSB-first (digit-reversed) answers.

Format:  "457+638=5901;"   where the answer 1095 is written reversed -> "5901"
and ';' is EOS. The reversed answer lets a causal model emit least-significant
digit first, so carries propagate left-to-right.

Produces plain token-id tensors so the same superpose.py collators
(none/cross_seq/token_merge) and kd_loss apply unchanged in the distillation step.
"""

from __future__ import annotations

import torch

VOCAB = list("0123456789+=") + ["_", ";"]   # '_' = pad, ';' = eos
STOI = {c: i for i, c in enumerate(VOCAB)}
ITOS = {i: c for c, i in STOI.items()}
VOCAB_SIZE = len(VOCAB)
PAD_ID = STOI["_"]
EOS_ID = STOI[";"]


def encode(s: str):
    return [STOI[c] for c in s]


def decode(ids):
    return "".join(ITOS[int(i)] for i in ids)


def seq_len_for(n_digits: int) -> int:
    # "D...D+D...D=" + reversed sum (<= n_digits+1) + eos, padded
    return (2 * n_digits + 2) + (n_digits + 1) + 1


def sample_batch(batch: int, n_digits: int, device, generator: torch.Generator):
    """Returns (ids, loss_mask) shape (B, L). loss_mask[t]=True where the target
    ids[t+1] is an answer/eos token (i.e. we only train the answer)."""
    L = seq_len_for(n_digits)
    hi = 10 ** n_digits - 1
    ops = torch.randint(0, hi + 1, (batch, 2), generator=generator)
    ids = torch.full((batch, L), PAD_ID, dtype=torch.long)
    loss_mask = torch.zeros((batch, L), dtype=torch.bool)
    for i in range(batch):
        a, b = ops[i].tolist()
        prompt = f"{a}+{b}="
        ans = str(a + b)[::-1]                 # LSB-first
        seq = encode(prompt) + encode(ans) + [EOS_ID]
        seq = seq[:L]
        ids[i, :len(seq)] = torch.tensor(seq)
        p = len(prompt)                        # answer starts at index p
        loss_mask[i, p - 1:len(seq) - 1] = True
    return ids.to(device), loss_mask.to(device)


@torch.no_grad()
def exact_match(model, n_digits: int, device, n: int = 512, generator=None):
    """Greedy-decode the answer from the prompt; exact-match accuracy.

    Prompts are grouped by length so every batch is equal-length and needs no
    padding (right-padding between '=' and the answer would corrupt decoding).
    """
    from collections import defaultdict
    model.eval()
    hi = 10 ** n_digits - 1
    g = generator or torch.Generator().manual_seed(1234)
    ops = torch.randint(0, hi + 1, (n, 2), generator=g)
    ans_len = n_digits + 2                      # reversed sum + eos slack
    groups = defaultdict(list)
    for a, b in ops.tolist():
        groups[len(f"{a}+{b}=")].append((a, b))
    correct = 0
    for plen, pairs in groups.items():
        for s in range(0, len(pairs), 256):
            chunk = pairs[s:s + 256]
            out = torch.tensor([encode(f"{a}+{b}=") for a, b in chunk],
                               dtype=torch.long, device=device)
            for _ in range(ans_len):
                nxt = model(input_ids=out).logits[:, -1].argmax(-1, keepdim=True)
                out = torch.cat([out, nxt], dim=1)
            for i, (a, b) in enumerate(chunk):
                pred = decode(out[i, plen:].tolist()).split(";")[0][::-1]
                try:
                    if int(pred) == a + b:
                        correct += 1
                except ValueError:
                    pass
    return correct / n
