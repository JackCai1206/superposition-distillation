"""Superposition distillation on the LSB-first addition testbed.

Frozen teacher = a clean addition model (train_addition.py checkpoint). A student
is trained from scratch by logit-KD where the teacher consumes SUPERPOSED inputs:
  none       : baseline single-sequence KD
  cross_seq  : two addition problems mixed position-wise (compute packing)
  token_merge: k adjacent tokens merged (sequence shortening)
Stage 1 = superposed (pure forward-KL). Stage 2 = normal-data recovery
(KD + CE on the answer tokens). Compares methods at equal student FLOPs on
exact-match accuracy.
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn.functional as F

from addition import (PAD_ID, VOCAB_SIZE, exact_match, sample_batch, seq_len_for)
from flops import FlopCounter, model_flops_from_config
from kd_loss import forward_kl, wsd_alpha
from model import load_model, superposed_logits, tiny_model
from superpose import (superpose_cross_seq, superpose_none, superpose_token_merge)


def make_superposed(method, ids, mask, k, lam):
    if method == "none":
        return superpose_none(ids, mask), 1.0
    if method == "cross_seq":
        h = ids.shape[0] // 2
        sup = superpose_cross_seq(ids[:h], mask[:h], ids[h:2 * h], mask[h:2 * h], fixed=lam)
        return sup, 2.0
    if method == "token_merge":
        return superpose_token_merge(ids, mask, k=k, fixed=lam), 1.0
    raise ValueError(method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, help="dir of trained addition model")
    ap.add_argument("--method", default="cross_seq", choices=["none", "cross_seq", "token_merge"])
    ap.add_argument("--n_digits", type=int, default=4)
    ap.add_argument("--student_hidden", type=int, default=256)
    ap.add_argument("--student_layers", type=int, default=4)
    ap.add_argument("--stage1_steps", type=int, default=6000)
    ap.add_argument("--stage2_steps", type=int, default=1500)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--fixed_lambda", type=float, default=0.7)
    ap.add_argument("--merge_k", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = args.device
    jid = os.environ.get("SLURM_JOB_ID", "local")
    out = f"outputs/distadd_{args.method}_d{args.n_digits}_{jid}"
    os.makedirs(out, exist_ok=True)

    teacher = load_model(args.teacher, dtype=torch.bfloat16, device=dev, frozen=True)
    student = tiny_model(VOCAB_SIZE, hidden=args.student_hidden, layers=args.student_layers,
                         heads=8, inter=4 * args.student_hidden, dtype=torch.bfloat16, device=dev)
    np_s = sum(p.numel() for p in student.parameters())
    print(f"teacher={args.teacher} | student {np_s/1e6:.2f}M | method={args.method}")
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.1)
    # count the frozen teacher's forward FLOPs too (cross_seq halves its batch)
    fc = FlopCounter(model_flops_from_config(student.config),
                     teacher_fm=model_flops_from_config(teacher.config))
    g = torch.Generator().manual_seed(args.seed)
    L = seq_len_for(args.n_digits)
    hist = []

    def step_once(step, normal, steps):
        ids, lmask = sample_batch(args.batch_size, args.n_digits, dev, g)
        attn = (ids != PAD_ID).long()           # structural mask = real (non-pad) tokens
        if normal:
            sup, eff = superpose_none(ids, attn), 1.0
        else:
            sup, eff = make_superposed(args.method, ids, attn, args.merge_k, args.fixed_lambda)
        with torch.no_grad():
            t_logits = superposed_logits(teacher, sup)
        s_logits = superposed_logits(student, sup)
        kd = forward_kl(s_logits, t_logits, args.temperature, sup.mask)
        if normal:
            # add CE on the answer tokens (loss_mask), WSD-weighted vs KD
            a = wsd_alpha(step, steps, alpha_max=0.9)
            sl, tgt, m = s_logits[:, :-1], ids[:, 1:], lmask[:, :-1]
            ce = F.cross_entropy(sl[m], tgt[m]) if m.any() else torch.zeros((), device=dev)
            loss = a * kd + (1 - a) * ce
        else:
            loss, ce = kd, torch.zeros((), device=dev)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        T = sup.ids.shape[1]
        fc.add_step(seq_len=T, batch=sup.ids.shape[0], effective_sequences=eff * sup.ids.shape[0])
        return loss.item(), kd.item(), float(ce)

    student.train()
    for tag, steps, normal in [("S1", args.stage1_steps, False), ("S2", args.stage2_steps, True)]:
        print(f"== {tag} ({'normal' if normal else args.method}) ==")
        for step in range(steps):
            loss, kd, ce = step_once(step, normal, steps)
            if step % args.eval_every == 0 or step == steps - 1:
                acc = exact_match(student, args.n_digits, dev, n=512); student.train()
                s = fc.summary()
                print(f"[{tag}] step {step:>5} loss={loss:.4f} kd={kd:.4f} ce={ce:.4f} "
                      f"exact_match={acc:.3f} flops={s['total_flops']:.3e}")
                hist.append({"stage": tag, "step": step, "exact_match": acc,
                             "flops": s["total_flops"], "loss": loss})

    student.save_pretrained(out)
    results = {"task": "lsb_addition_distill", "method": args.method,
               "n_digits": args.n_digits, "fixed_lambda": args.fixed_lambda,
               "merge_k": args.merge_k, "student_params": np_s,
               "stage1_steps": args.stage1_steps, "stage2_steps": args.stage2_steps,
               "final": hist[-1] if hist else None, "history": hist, "flops": fc.summary()}
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("saved ->", out, "| final exact_match:", hist[-1]["exact_match"] if hist else None)


if __name__ == "__main__":
    main()
