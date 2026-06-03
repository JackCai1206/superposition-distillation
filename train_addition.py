"""Train a ~10M transformer from scratch on LSB-first integer addition.

This trains the CLEAN model (next-token CE on the answer tokens). The resulting
checkpoint is the controlled FROZEN TEACHER for the superposition-distillation
step (cross_seq / token_merge) on the same task.
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn.functional as F

from addition import (VOCAB_SIZE, exact_match, sample_batch, seq_len_for)
from flops import FlopCounter, model_flops_from_config
from model import tiny_model


def build_10m(device, dtype):
    # ~10M params: 16*H^2 per layer * L  ->  H=320,L=6 ~ 9.8M (+ tiny char embeds)
    m = tiny_model(VOCAB_SIZE, hidden=320, layers=6, heads=8, inter=1280,
                   dtype=dtype, device=device)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_digits", type=int, default=4)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.1,
                    help="higher (e.g. 1.0) reliably accelerates addition grokking")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dt = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    dev = args.device
    jid = os.environ.get("SLURM_JOB_ID", "local")
    out = args.out or f"outputs/addition_10m_d{args.n_digits}_{jid}"
    os.makedirs(out, exist_ok=True)

    model = build_10m(dev, dt)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params/1e6:.2f}M | seq_len={seq_len_for(args.n_digits)} | vocab={VOCAB_SIZE}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    fc = FlopCounter(model_flops_from_config(model.config), opt_params=sum(p.numel() for p in model.parameters() if p.requires_grad))
    g = torch.Generator().manual_seed(args.seed)

    L = seq_len_for(args.n_digits)
    hist = []
    model.train()
    for step in range(args.steps):
        ids, lmask = sample_batch(args.batch_size, args.n_digits, dev, g)
        logits = model(input_ids=ids).logits
        sl = logits[:, :-1]
        tgt = ids[:, 1:]
        m = lmask[:, :-1]
        loss = F.cross_entropy(sl[m], tgt[m])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        fc.add_step(seq_len=L, batch=args.batch_size, effective_sequences=args.batch_size)

        if step % args.eval_every == 0 or step == args.steps - 1:
            acc = exact_match(model, args.n_digits, dev, n=512)
            model.train()
            s = fc.summary()
            print(f"step {step:>6} loss={loss.item():.4f} exact_match={acc:.3f} "
                  f"flops={s['total_flops']:.3e}")
            hist.append({"step": step, "loss": loss.item(), "exact_match": acc,
                         "flops": s["total_flops"]})
            # periodic checkpoint: lets a saturated run be cut short without losing weights
            if acc >= 0.999:
                model.save_pretrained(out)
                print(f"  [checkpoint @ step {step}, acc {acc:.3f}]")

    model.save_pretrained(out)
    results = {"task": "lsb_addition", "n_digits": args.n_digits,
               "n_params": n_params, "steps": args.steps, "batch_size": args.batch_size,
               "final": hist[-1] if hist else None, "history": hist,
               "flops": fc.summary()}
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("saved ->", out, "| final exact_match:", hist[-1]["exact_match"] if hist else None)


if __name__ == "__main__":
    main()
