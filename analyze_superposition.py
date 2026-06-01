"""Probe: what does a FROZEN teacher's next-token distribution look like on
SUPERPOSED inputs? If it collapses to garbage, the iso-FLOP comparison is moot,
so this is the make-or-break diagnostic before launching real runs.

For cross_seq we sweep the mixing weight lambda from 1.0 (= clean sequence A) down
to 0.5 (= equal blend) and measure how fast the distribution degrades away from
the clean prediction:
  - entropy(nats) of the next-token distribution
  - KL(P_clean_A || P_superposed)   (0 at lambda=1 by construction = sanity)
  - top-1 agreement and top-5 overlap with the clean-A prediction
For token_merge we report entropy + top-1 agreement vs the clean model at the
aligned original position.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from model import load_model, load_tokenizer, superposed_logits
from superpose import (superpose_cross_seq, superpose_none, superpose_token_merge)

PAIRS = [
    ("The mitochondrion is the powerhouse of the cell, generating most of the chemical energy needed to power the cell's biochemical reactions.",
     "In 1969 the Apollo 11 mission landed the first humans on the Moon, a milestone in the history of space exploration and engineering."),
    ("Let x be a positive integer such that x squared plus three x minus ten equals zero. Solving the quadratic gives the value of x.",
     "A train travels at sixty miles per hour for two hours and then at forty miles per hour for three hours, covering a total distance."),
]


def entropy(p, mask):
    e = -(p * (p.clamp_min(1e-12)).log()).sum(-1)      # (B,T)
    return (e * mask).sum() / mask.sum().clamp_min(1)


def topk_overlap(a_logits, b_logits, mask, k=5):
    ta = a_logits.topk(k, dim=-1).indices                # (B,T,k)
    tb = b_logits.topk(k, dim=-1).indices
    inter = (ta.unsqueeze(-1) == tb.unsqueeze(-2)).any(-1).float().sum(-1)  # (B,T)
    return ((inter / k) * mask).sum() / mask.sum().clamp_min(1)


def top1_agree(a_logits, b_logits, mask):
    agree = (a_logits.argmax(-1) == b_logits.argmax(-1)).float()
    return (agree * mask).sum() / mask.sum().clamp_min(1)


@torch.no_grad()
def analyze(teacher, tok, device):
    print("\n================  CROSS_SEQ  (mix two sequences position-wise)  ================")
    print(f"{'lambda':>7} {'entropy':>9} {'KL(clean||sup)':>15} {'top1=clean':>11} {'top5overlap':>12}")
    for a_text, b_text in PAIRS:
        a = tok(a_text, return_tensors="pt", add_special_tokens=False).input_ids
        b = tok(b_text, return_tensors="pt", add_special_tokens=False).input_ids
        L = min(a.shape[1], b.shape[1])
        a, b = a[:, :L].to(device), b[:, :L].to(device)
        am = torch.ones_like(a); bm = torch.ones_like(b)
        # clean reference = sequence A alone
        clean = superposed_logits(teacher, superpose_none(a, am))
        cmask = am.float()
        print(f"  pair: '{a_text[:42]}...' x '{b_text[:42]}...'")
        for lam in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
            sup = superpose_cross_seq(a, am, b, bm, fixed=lam)
            logits = superposed_logits(teacher, sup)
            p = F.softmax(logits, -1)
            ent = entropy(p, cmask)
            logp_sup = F.log_softmax(logits, -1)
            p_clean = F.softmax(clean, -1)
            kl = ((p_clean * (F.log_softmax(clean, -1) - logp_sup)).sum(-1) * cmask).sum() / cmask.sum()
            t1 = top1_agree(clean, logits, cmask)
            t5 = topk_overlap(clean, logits, cmask)
            print(f"{lam:>7.2f} {ent.item():>9.3f} {kl.item():>15.3f} {t1.item():>11.2f} {t5.item():>12.2f}")

    print("\n================  TOKEN_MERGE  (merge k adjacent tokens -> 1)  ================")
    print(f"{'k':>3} {'lambda':>7} {'entropy':>9} {'top1=clean@2j+1':>16}")
    for a_text, _ in PAIRS:
        a = tok(a_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        am = torch.ones_like(a)
        clean = superposed_logits(teacher, superpose_none(a, am))   # (1,L,V)
        print(f"  seq: '{a_text[:50]}...'")
        for k in [2, 3]:
            for lam in [0.9, 0.7, 0.5]:
                sup = superpose_token_merge(a, am, k=k, fixed=lam)
                logits = superposed_logits(teacher, sup)            # (1,L//k,V)
                T = logits.shape[1]
                p = F.softmax(logits, -1)
                ent = entropy(p, sup.mask.float().to(device))
                # align merged pos j -> clean pos (j+1)*k-1 (last token of block)
                idx = torch.arange(T, device=device) * k + (k - 1)
                idx = idx.clamp_max(clean.shape[1] - 1)
                clean_aligned = clean[:, idx]
                t1 = top1_agree(clean_aligned, logits, torch.ones(1, T, device=device))
                print(f"{k:>3} {lam:>7.2f} {ent.item():>9.3f} {t1.item():>16.2f}")

    # eyeball: print top-5 tokens at one position, clean vs lambda=0.6
    print("\n================  EYEBALL (cross_seq, first position)  ================")
    a_text, b_text = PAIRS[0]
    a = tok(a_text, return_tensors="pt", add_special_tokens=False).input_ids
    b = tok(b_text, return_tensors="pt", add_special_tokens=False).input_ids
    L = min(a.shape[1], b.shape[1]); a, b = a[:, :L].to(device), b[:, :L].to(device)
    am = torch.ones_like(a); bm = torch.ones_like(b)
    pos = 8
    clean = superposed_logits(teacher, superpose_none(a, am))[0, pos]
    sup06 = superposed_logits(teacher, superpose_cross_seq(a, am, b, bm, fixed=0.6))[0, pos]
    ctx = tok.decode(a[0, :pos + 1]); ctxb = tok.decode(b[0, :pos + 1])
    print(f"context A: ...{ctx!r}")
    print(f"context B: ...{ctxb!r}")
    print("clean-A top5 :", [tok.decode([i]) for i in clean.topk(5).indices.tolist()])
    print("lambda0.6 top5:", [tok.decode([i]) for i in sup06.topk(5).indices.tolist()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()
    dt = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    print(f"Loading teacher {args.teacher} ...")
    tok = load_tokenizer(args.teacher)
    teacher = load_model(args.teacher, dtype=dt, device=args.device, frozen=True)
    analyze(teacher, tok, args.device)
    print("\nDONE. Read: at lambda=1.0 KL must be ~0 (sanity). How fast top1/top5 fall "
          "as lambda->0.5 tells you how OOD the superposed input is to the frozen teacher.")


if __name__ == "__main__":
    main()
