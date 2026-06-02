"""Two-stage superposition distillation on TinyStories (NL de-risk).

Frozen LM teacher -> student, where the teacher consumes SUPERPOSED inputs
(none / cross_seq / token_merge). Stage 1 superposed (pure forward-KL), Stage 2
normal recovery (KD + next-token CE), WSD-LR over the whole run. Metric: val
cross-entropy loss; iso-FLOP = total (student+teacher) FLOPs to reach a val-loss
threshold.
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn.functional as F

from flops import FlopCounter, measure_step_flops, model_flops_from_config
from kd_loss import chunked_distill_loss, wsd_alpha, wsd_lr_mult
from model import load_model, superposed_hidden, tiny_model
from nl_data import VOCAB_SIZE, eval_lm_loss, get_batch, load_split
from superpose import superpose_cross_seq, superpose_none, superpose_token_merge


def make_superposed(method, ids, mask, k, lam):
    if method == "none":
        return superpose_none(ids, mask), 1.0
    if method == "cross_seq":
        h = ids.shape[0] // 2
        return superpose_cross_seq(ids[:h], mask[:h], ids[h:2 * h], mask[h:2 * h], fixed=lam), 2.0
    if method == "token_merge":
        return superpose_token_merge(ids, mask, k=k, fixed=lam), 1.0
    raise ValueError(method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--method", default="cross_seq", choices=["none", "cross_seq", "token_merge"])
    ap.add_argument("--student_hidden", type=int, default=320)
    ap.add_argument("--student_layers", type=int, default=6)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--stage1_steps", type=int, default=1500)
    ap.add_argument("--stage2_steps", type=int, default=3000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--fixed_lambda", type=float, default=0.7)
    ap.add_argument("--merge_k", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = args.device
    torch.manual_seed(args.seed)
    jid = os.environ.get("SLURM_JOB_ID", "local")
    ttag = "gpt2" if "gpt2" in args.teacher.lower() else "ctrl"   # which teacher
    out = f"outputs/lmdist_{ttag}_{args.method}_l{args.fixed_lambda}_s1{args.stage1_steps}_seed{args.seed}_{jid}"
    os.makedirs(out, exist_ok=True)

    train = load_split("train"); val = load_split("val")
    teacher = load_model(args.teacher, dtype=torch.bfloat16, device=dev, frozen=True)
    student = tiny_model(VOCAB_SIZE, hidden=args.student_hidden, layers=args.student_layers,
                         heads=8, inter=4 * args.student_hidden, dtype=torch.bfloat16,
                         device=dev, tie_embeddings=True, max_pos=args.seq_len)
    np_s = sum(p.numel() for p in student.parameters())
    print(f"teacher={args.teacher} | student {np_s/1e6:.1f}M | method={args.method}")
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))
    fc = FlopCounter(model_flops_from_config(student.config),
                     teacher_fm=model_flops_from_config(teacher.config))
    g = torch.Generator().manual_seed(args.seed)
    total_steps = args.stage1_steps + args.stage2_steps
    hist = []

    s_head = student.get_output_embeddings().weight
    t_head = teacher.get_output_embeddings().weight

    def step_once(normal, measure=False):
        ids, mask = get_batch(train, args.batch_size, args.seq_len, dev, g)
        if normal:
            sup, eff = superpose_none(ids, mask), 1.0
        else:
            sup, eff = make_superposed(args.method, ids, mask, args.merge_k, args.fixed_lambda)
        h = {}

        def fwd_bwd():
            with torch.no_grad():
                t_hidden = superposed_hidden(teacher, sup)
            s_hidden = superposed_hidden(student, sup)
            if normal:
                labels = ids.clone()
                labels[:, :-1] = ids[:, 1:]
                labels[:, -1] = -100                   # next-token CE, last has no target
                loss, parts = chunked_distill_loss(s_hidden, s_head, t_hidden, t_head,
                                                   args.temperature, mask=sup.mask,
                                                   labels=labels, alpha=wsd_alpha_step)
            else:
                loss, parts = chunked_distill_loss(s_hidden, s_head, t_hidden, t_head,
                                                   args.temperature, mask=sup.mask)
            opt.zero_grad(); loss.backward()
            h.update(loss=loss.item(), kd=float(parts["kd"]), ce=float(parts["ce"]))

        measured = measure_step_flops(fwd_bwd) if measure else (fwd_bwd() or None)
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        return h["loss"], h["kd"], h["ce"], sup.ids.shape[1], sup.ids.shape[0], eff, measured

    gstep = 0
    student.train()
    for tag, steps, normal in [("S1", args.stage1_steps, False), ("S2", args.stage2_steps, True)]:
        print(f"== {tag} ({'normal' if normal else args.method}) ==")
        for step in range(steps):
            lr = args.lr * wsd_lr_mult(gstep, total_steps, warmup=100, decay=400)
            for grp in opt.param_groups:
                grp["lr"] = lr
            wsd_alpha_step = wsd_alpha(step, steps, alpha_max=0.9)
            loss, kd, ce, T, outB, eff, measured = step_once(normal, measure=(step == 0))
            fc.add_step(seq_len=T, batch=outB, effective_sequences=eff * outB, measured_step=measured)
            gstep += 1
            if step % args.eval_every == 0 or step == steps - 1:
                vloss = eval_lm_loss(student, val, dev, n_batches=30, batch=16, seq_len=args.seq_len)
                student.train()
                s = fc.summary()
                print(f"[{tag}] step {step:>5} loss={loss:.3f} ce={ce:.3f} val={vloss:.3f} "
                      f"est={s['total_flops']:.3e} rec={s['recorded_flops']:.3e}")
                hist.append({"stage": tag, "step": step, "val_loss": vloss,
                             "flops": s["total_flops"], "recorded_flops": s["recorded_flops"], "loss": loss})

    student.save_pretrained(out)
    json.dump({"task": "tinystories_distill", "teacher": args.teacher, "teacher_tag": ttag,
               "method": args.method, "fixed_lambda": args.fixed_lambda,
               "merge_k": args.merge_k, "stage1_steps": args.stage1_steps, "stage2_steps": args.stage2_steps,
               "seed": args.seed, "lr_sched": "wsd", "student_params": np_s,
               "final_val_loss": hist[-1]["val_loss"], "history": hist, "flops": fc.summary()},
              open(os.path.join(out, "results.json"), "w"), indent=2)
    print("saved ->", out, "| final val loss:", hist[-1]["val_loss"])


if __name__ == "__main__":
    main()
