"""Train a small GPT-2-style teacher LM on TinyStories (next-token CE).

The checkpoint is the controlled FROZEN TEACHER for the NL superposition-distillation
de-risk. Mirrors train_addition.py but with real tokens + val-loss metric.
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn.functional as F

from flops import FlopCounter, model_flops_from_config
from kd_loss import wsd_lr_mult
from model import tiny_model
from nl_data import VOCAB_SIZE, eval_lm_loss, get_batch, load_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--inter", type=int, default=2048)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dt = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    dev = args.device
    torch.manual_seed(args.seed)
    jid = os.environ.get("SLURM_JOB_ID", "local")
    out = args.out or f"outputs/lm_teacher_h{args.hidden}l{args.layers}_{jid}"
    os.makedirs(out, exist_ok=True)

    train = load_split("train"); val = load_split("val")
    model = tiny_model(VOCAB_SIZE, hidden=args.hidden, layers=args.layers, heads=args.heads,
                       inter=args.inter, dtype=dt, device=dev, tie_embeddings=True,
                       max_pos=args.seq_len)
    npar = sum(p.numel() for p in model.parameters())
    print(f"teacher params: {npar/1e6:.1f}M (tied) | vocab={VOCAB_SIZE} | seq={args.seq_len}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))
    fc = FlopCounter(model_flops_from_config(model.config), opt_params=sum(p.numel() for p in model.parameters() if p.requires_grad))
    g = torch.Generator().manual_seed(args.seed)
    hist = []

    model.train()
    for step in range(args.steps):
        lr = args.lr * wsd_lr_mult(step, args.steps, warmup=100, decay=500)
        for grp in opt.param_groups:
            grp["lr"] = lr
        ids, _ = get_batch(train, args.batch_size, args.seq_len, dev, g)
        logits = model(input_ids=ids).logits
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        fc.add_step(seq_len=args.seq_len, batch=args.batch_size, effective_sequences=args.batch_size)
        if step % args.eval_every == 0 or step == args.steps - 1:
            vloss = eval_lm_loss(model, val, dev, n_batches=40, batch=32, seq_len=args.seq_len)
            model.train()
            s = fc.summary()
            print(f"step {step:>6} train={loss.item():.3f} val={vloss:.3f} flops={s['total_flops']:.3e}")
            hist.append({"step": step, "train_loss": loss.item(), "val_loss": vloss,
                         "flops": s["total_flops"]})

    model.save_pretrained(out)
    json.dump({"n_params": npar, "hidden": args.hidden, "layers": args.layers,
               "final_val_loss": hist[-1]["val_loss"], "history": hist},
              open(os.path.join(out, "results.json"), "w"), indent=2)
    print("saved ->", out, "| final val loss:", hist[-1]["val_loss"])


if __name__ == "__main__":
    main()
