"""Finetune a pretrained HF LM (e.g. SmolLM2-135M) on TinyStories -> a STRONG
in-domain teacher for distillation. Plain next-token CE on the SmolLM2-tokenized
.bin (SD_DATA_DIR). Saves to --out; distill_lm then loads it as a frozen teacher
and the student is a random-init scaled-down version of the SAME arch.
"""
from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn.functional as F

from model import load_model
from nl_data import eval_lm_loss, get_batch, load_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = args.device
    torch.manual_seed(args.seed)
    train = load_split("train"); val = load_split("val")
    model = load_model(args.ref, dtype=torch.bfloat16, device=dev, frozen=False)
    model.gradient_checkpointing_enable()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))
    g = torch.Generator().manual_seed(args.seed)
    hist = []
    v0 = eval_lm_loss(model, val, dev, n_batches=20, batch=16, seq_len=args.seq_len); model.train()
    print(f"[ft] ref={args.ref} init val={v0:.3f}")
    for step in range(args.steps):
        lr = args.lr * min(1.0, (step + 1) / args.warmup)
        for grp in opt.param_groups:
            grp["lr"] = lr
        ids, _ = get_batch(train, args.batch_size, args.seq_len, dev, g)
        logits = model(input_ids=ids).logits
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % args.eval_every == 0 or step == args.steps - 1:
            v = eval_lm_loss(model, val, dev, n_batches=20, batch=16, seq_len=args.seq_len); model.train()
            print(f"[ft] step {step:>5} loss={loss.item():.3f} val={v:.3f}")
            hist.append({"step": step, "loss": loss.item(), "val": v})
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    json.dump({"ref": args.ref, "steps": args.steps, "final_val": hist[-1]["val"], "history": hist},
              open(os.path.join(args.out, "ft.json"), "w"), indent=2)
    print("saved teacher ->", args.out, "| final val:", hist[-1]["val"])


if __name__ == "__main__":
    main()
