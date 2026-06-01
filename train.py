"""Two-stage superposition-distillation training.

Stage 1: train the student on SUPERPOSED inputs (OOD for the frozen teacher) with
         pure forward-KL. This is where the compute-packing happens.
Stage 2: short "recovery" stage on NORMAL single-sequence inputs (standard KD,
         WSD-scheduled alpha) so the student re-adapts to in-distribution data.

The frozen teacher provides logits live (no_grad). Conditions (none / cross_seq /
token_merge) share the same data stream and are compared at equal student FLOPs.
"""

from __future__ import annotations

import argparse
import json
import math
import os

import torch

from config import Config
from data import (batched, build_superposed, pretrain_stream, reasoning_stream,
                  synthetic_stream)
from flops import FlopCounter, model_flops_from_config
from kd_loss import kd_loss, wsd_alpha
from model import (load_model, load_tokenizer, superposed_logits, tiny_model)

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def build_models(cfg: Config, tokenizer):
    dev = cfg.train.device
    if cfg.debug:
        V = len(tokenizer) if tokenizer is not None else 256
        teacher = tiny_model(V, hidden=96, layers=3, device=dev)
        student = tiny_model(V, hidden=48, layers=2, device=dev)
        for p in teacher.parameters():
            p.requires_grad_(False)
        teacher.eval()
    else:
        dt = DTYPES[cfg.model.dtype]
        teacher = load_model(cfg.model.teacher, dtype=dt, device=dev, frozen=True)
        student = load_model(cfg.model.student, dtype=dt, device=dev, frozen=False)
    return teacher, student


def make_stream(cfg: Config, tokenizer, kind: str):
    if cfg.debug or getattr(cfg.train, "synth_data", False):
        V = len(tokenizer) if tokenizer is not None else 256
        return synthetic_stream(cfg, V)
    return pretrain_stream(cfg, tokenizer) if kind == "pretrain" else reasoning_stream(cfg, tokenizer)


def run_stage(cfg, teacher, student, opt, fc, stream, *, method, steps, normal, tag):
    """One training stage. normal=True -> single-seq inputs + CE-mixed KD."""
    student.train()
    # cross_seq consumes 2 source seqs per output example -> pull a double batch
    eff_batch = cfg.train.batch_size * (2 if (method == "cross_seq" and not normal) else 1)
    loader = batched(stream, eff_batch)
    logf = []
    for step in range(steps):
        try:
            ids, mask = next(loader)
        except StopIteration:
            print(f"[{tag}] stream exhausted at step {step}"); break
        ids, mask = ids.to(cfg.train.device), mask.to(cfg.train.device)

        if normal:
            from superpose import superpose_none
            sup, eff = superpose_none(ids, mask), 1.0
            labels = ids.clone()                      # next-token CE valid here
            alpha = wsd_alpha(step, steps, cfg.kd.alpha_max, cfg.kd.warmup_frac, cfg.kd.decay_frac)
        else:
            sup, eff = build_superposed(method, ids, mask, cfg)
            labels, alpha = None, 1.0                 # superposed -> pure KD

        with torch.no_grad():
            t_logits = superposed_logits(teacher, sup)
        s_logits = superposed_logits(student, sup)

        if labels is not None:
            # shift for next-token prediction
            s_logits_ce = s_logits[:, :-1]
            labels = labels[:, 1:]
            loss, parts = kd_loss(s_logits[:, :-1], t_logits[:, :-1], cfg.kd.temperature,
                                  alpha=alpha, labels=labels, mask=sup.mask[:, :-1])
        else:
            loss, parts = kd_loss(s_logits, t_logits, cfg.kd.temperature, alpha=1.0,
                                  labels=None, mask=sup.mask)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.train.grad_clip)
        opt.step()

        T = sup.ids.shape[1]
        out_B = sup.ids.shape[0]
        fc.add_step(seq_len=T, batch=out_B, effective_sequences=eff * out_B)
        if step % cfg.train.log_every == 0:
            s = fc.summary()
            print(f"[{tag}] step {step:>5} T={T} B={out_B} alpha={alpha:.2f} "
                  f"loss={loss.item():.4f} kd={parts['kd']:.4f} ce={parts['ce']:.4f} "
                  f"flops={s['total_flops']:.3e} seq={s['sequences_seen']:.0f}")
            logf.append((step, loss.item(), s["total_flops"], s["sequences_seen"]))
    return logf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="cross_seq", choices=["none", "cross_seq", "token_merge"])
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--stage1_steps", type=int, default=None)
    ap.add_argument("--stage2_steps", type=int, default=None)
    ap.add_argument("--seq_len", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--synth_data", action="store_true", help="real models, synthetic tokens (offline path test)")
    ap.add_argument("--data_kind", default="pretrain", choices=["pretrain", "reasoning"])
    ap.add_argument("--teacher", default=None)
    ap.add_argument("--student", default=None)
    ap.add_argument("--fixed_lambda", type=float, default=None,
                    help="constant mixing weight; analysis shows ~0.7 keeps teacher coherent")
    ap.add_argument("--mix_alpha", type=float, default=None, help="Beta(a,a) for lambda")
    ap.add_argument("--merge_k", type=int, default=None)
    args = ap.parse_args()

    cfg = Config()
    cfg.debug = args.debug
    cfg.superpose.method = args.method
    if args.device: cfg.train.device = args.device
    if args.stage1_steps is not None: cfg.train.stage1_steps = args.stage1_steps
    if args.stage2_steps is not None: cfg.train.stage2_steps = args.stage2_steps
    if args.seq_len is not None: cfg.data.seq_len = args.seq_len
    if args.batch_size is not None: cfg.train.batch_size = args.batch_size
    if cfg.debug and args.device is None: cfg.train.device = "cpu"
    cfg.train.synth_data = args.synth_data
    if args.teacher: cfg.model.teacher = args.teacher
    if args.student: cfg.model.student = args.student
    if args.fixed_lambda is not None: cfg.superpose.fixed_lambda = args.fixed_lambda
    if args.mix_alpha is not None: cfg.superpose.mix_alpha = args.mix_alpha
    if args.merge_k is not None: cfg.superpose.merge_k = args.merge_k

    # per-method output dir (+ job id) so concurrent runs don't collide
    jid = os.environ.get("SLURM_JOB_ID", "local")
    cfg.train.output_dir = f"outputs/{args.data_kind}_{cfg.superpose.method}_{jid}"
    os.makedirs(cfg.train.output_dir, exist_ok=True)
    tokenizer = None if cfg.debug else load_tokenizer(cfg.model.student)
    teacher, student = build_models(cfg, tokenizer)
    opt = torch.optim.AdamW(student.parameters(), lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay)
    fc = FlopCounter(model_flops_from_config(student.config))

    kind = args.data_kind
    print(f"== Stage 1: superposed ({cfg.superpose.method}) on {kind} data ==")
    s1log = run_stage(cfg, teacher, student, opt, fc,
                      make_stream(cfg, tokenizer, kind),
                      method=cfg.superpose.method, steps=cfg.train.stage1_steps,
                      normal=False, tag="S1")

    print(f"== Stage 2: normal-data recovery on {kind} data ==")
    s2log = run_stage(cfg, teacher, student, opt, fc,
                      make_stream(cfg, tokenizer, kind),
                      method="none", steps=cfg.train.stage2_steps, normal=True, tag="S2")

    summary = fc.summary()
    print("== Done ==", summary)

    # persist the trained student + a results record for the iso-FLOP comparison
    if not cfg.debug:
        student.save_pretrained(cfg.train.output_dir)
        tokenizer.save_pretrained(cfg.train.output_dir)
    results = {
        "method": cfg.superpose.method, "data_kind": kind,
        "fixed_lambda": cfg.superpose.fixed_lambda, "merge_k": cfg.superpose.merge_k,
        "teacher": cfg.model.teacher, "student": cfg.model.student,
        "seq_len": cfg.data.seq_len, "batch_size": cfg.train.batch_size,
        "stage1_steps": cfg.train.stage1_steps, "stage2_steps": cfg.train.stage2_steps,
        "final_s1_loss": s1log[-1][1] if s1log else None,
        "final_s2_loss": s2log[-1][1] if s2log else None,
        "flops": summary,
    }
    with open(os.path.join(cfg.train.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("saved ->", cfg.train.output_dir)

    # auto-eval reasoning students on MATH-500 (best-effort; never breaks the run)
    if not cfg.debug and kind == "reasoning":
        try:
            from eval import eval_math
            ev = eval_math(student, tokenizer, cfg.train.device, which="math500", n=100)
            with open(os.path.join(cfg.train.output_dir, "eval_math500.json"), "w") as f:
                json.dump(ev, f, indent=2)
            print("eval MATH-500 ->", ev)
        except Exception as e:
            print("auto-eval skipped:", type(e).__name__, str(e)[:120])


if __name__ == "__main__":
    main()
