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
import math
import os
import signal

import torch
import torch.nn.functional as F

from flops import FlopCounter, measure_step_flops, model_flops_from_config
from kd_loss import chunked_ce_loss, chunked_distill_loss, wsd_alpha, wsd_lr_mult
from model import load_model, scaled_model, superposed_hidden, tiny_model
from nl_data import VOCAB_SIZE, eval_lm_loss, get_batch, load_split
from superpose import (superpose_cross_seq, superpose_none, superpose_token_merge,
                       superpose_cross_merge, superpose_input_noise)


def make_superposed(method, ids, mask, k, lam):
    if method == "none":
        return superpose_none(ids, mask), 1.0
    if method == "cross_seq":
        h = ids.shape[0] // 2
        return superpose_cross_seq(ids[:h], mask[:h], ids[h:2 * h], mask[h:2 * h], fixed=lam), 2.0
    if method == "token_merge":
        return superpose_token_merge(ids, mask, k=k, fixed=lam), 1.0
    if method == "cross_merge":                          # both axes: token_merge(k) x cross_seq
        h = ids.shape[0] // 2
        return superpose_cross_merge(ids[:h], mask[:h], ids[h:2 * h], mask[h:2 * h], k=k, fixed=lam), 2.0
    raise ValueError(method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--method", default="cross_seq", choices=["none", "cross_seq", "token_merge", "cross_merge"])
    ap.add_argument("--student_hidden", type=int, default=320)
    ap.add_argument("--student_layers", type=int, default=6)
    ap.add_argument("--student_ref", default="", help="if set (e.g. HuggingFaceTB/SmolLM2-135M): "
                    "build a RANDOM-INIT scaled-down version of this arch as the student "
                    "(same vocab/rope as a same-family finetuned teacher) instead of tiny_model")
    ap.add_argument("--student_heads", type=int, default=8)
    ap.add_argument("--student_kv_heads", type=int, default=4)
    ap.add_argument("--student_inter", type=int, default=0, help="0 -> 4*hidden (tiny) / ~2.67*hidden if --student_ref")
    ap.add_argument("--compile", type=int, default=1, help="torch.compile student+teacher (throughput; ON by default, --compile 0 to disable)")
    ap.add_argument("--grad_ckpt", type=int, default=0, help="gradient checkpointing on the student (fit bigger micro-batch / larger student)")
    ap.add_argument("--find_micro", type=int, default=0, help=">0: OOM-search the largest micro-batch up to this cap on the REAL fwd+bwd+step path, print MAX_MICRO, and exit (no training)")
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--stage1_steps", type=int, default=1500)
    ap.add_argument("--stage2_steps", type=int, default=3000)
    ap.add_argument("--batch_size", type=int, default=64, help="MICRO batch (per forward); eff batch = batch_size*grad_accum")
    ap.add_argument("--grad_accum", type=int, default=1, help="micro-batches per optimizer step (for a large eff batch)")
    ap.add_argument("--lr_sched", default="wsd", choices=["wsd", "cosine"], help="cosine = nanoGPT-style (warmup+cosine decay to 0.1x)")
    ap.add_argument("--warmup_frac", type=float, default=0.1, help="cosine warmup fraction of total steps")
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--fixed_lambda", type=float, default=0.7)
    ap.add_argument("--merge_k", type=int, default=2)
    ap.add_argument("--iso_token", type=int, default=0,
                    help="1 = hold RAW TOKENS/step constant (no bs scaling): the superposed "
                         "stage runs at 1/2 (cross_seq) or 1/k (token_merge) the positions/FLOPs "
                         "but sees the SAME data as baseline -> fair 'match loss at less compute' "
                         "test. 0 (default) = iso-FLOP/step (superposed stage eats 2x/kx tokens).")
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--loss_mode", default="kd_ce", choices=["kd_ce", "pure_kd", "ce_only", "mce"],
                    help="pure_kd: forward-KL only (no CE, no alpha) -> student matches teacher. "
                         "ce_only: NO teacher -> next-token CE (vanilla NTP / no-distillation floor). "
                         "mce: TST-style, NO teacher -> S1 superposed input + multi-hot CE on the "
                         "ground-truth next tokens of BOTH constituent sequences (lam,1-lam); S2 clean CE.")
    ap.add_argument("--noise_sigma", type=float, default=0.0,
                    help="Perturbation scale (relative to embedding/one-hot magnitude) applied to the "
                         "KD input during S1: perturbed-point distillation, train KD at x+sigma*u. 0=off.")
    ap.add_argument("--noise_mode", default="onehot", choices=["onehot", "embed"],
                    help="onehot (default): perturb the input ONE-HOT in shared vocab space (real token "
                         "+ k random tokens at small SHARED weights) -> teacher & student see the SAME "
                         "perturbation -> genuine shared-u gradient matching. embed: independent Gaussian "
                         "noise in each model's own embedding space (NOT shared -> smoothing only; ablation).")
    ap.add_argument("--noise_k", type=int, default=16,
                    help="onehot mode: number of random token components for the one-hot perturbation.")
    ap.add_argument("--anchor", type=int, default=0,
                    help="With noise_sigma>0: 1 = also include the clean-input KD term (loss = "
                         "avg(clean, noised)); 0 = noised-input KD only (no anchor).")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--ckpt_every", type=int, default=0,
                    help="save a resumable ckpt every N steps (0=off) -> reap-resilient resume "
                         "(restores model+opt+gstep+hist+FLOP counters; outputs/ is a stable PVC path)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch.distributed as dist            # 8-GPU data-parallel via torchrun; manual
    rank = int(os.environ.get("RANK", "0"))     # all-reduce (the custom superposed-embed forward
    world = int(os.environ.get("WORLD_SIZE", "1"))   # bypasses DDP's module-hook autograd sync).
    local_rank = int(os.environ.get("LOCAL_RANK", "0")); ddp = world > 1; is_main = rank == 0
    if ddp:
        dist.init_process_group("nccl"); torch.cuda.set_device(local_rank)
        args.device = f"cuda:{local_rank}"

    dev = args.device
    # --- throughput: TF32 matmul + cuDNN autotune (shapes are fixed per stage) ---
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)                # SAME model init on every rank (DDP requirement)
    jid = os.environ.get("SLURM_JOB_ID", "local")
    ce_only = args.loss_mode == "ce_only"
    mce = args.loss_mode == "mce"
    no_teacher = ce_only or mce                                    # both train without a teacher
    if ce_only:
        args.method = "none"                                       # no teacher -> no superposition
    if no_teacher: ttag = "noT"
    elif "smol" in args.teacher.lower(): ttag = "smol"
    elif "gpt2" in args.teacher.lower(): ttag = "gpt2"
    else: ttag = "ctrl"
    mtag = {"pure_kd": "kd", "kd_ce": "ce", "ce_only": "ceo", "mce": "mce"}[args.loss_mode]   # which loss
    # per-arm noise tag so noise-sweep arms get distinct dirs (else all arms collide on one
    # dir and clobber each other's ckpt/results). Empty when all-default -> non-noise runs and
    # the sigma=0 baseline keep their historical names (no dir rename / resume break).
    ntag = "" if (args.noise_sigma == 0 and args.anchor == 0 and args.noise_mode == "onehot") \
        else f"_nz{args.noise_sigma:g}_a{args.anchor}_{args.noise_mode}"
    if ntag and args.noise_k != 8:                        # K-sweep: distinct dir per noise_k (default 8 -> no suffix, preserves existing arm names)
        ntag += f"_k{args.noise_k}"
    out = f"outputs/lmdist_{ttag}_{mtag}_{args.method}_l{args.fixed_lambda}{ntag}_s1{args.stage1_steps}_seed{args.seed}_{jid}"
    if not args.find_micro:
        os.makedirs(out, exist_ok=True)

    train = load_split("train"); val = load_split("val")
    teacher = None if no_teacher else load_model(args.teacher, dtype=torch.bfloat16, device=dev, frozen=True)
    if args.student_ref:
        inter = args.student_inter or int(round(2.67 * args.student_hidden / 64) * 64)
        student = scaled_model(args.student_ref, hidden=args.student_hidden, layers=args.student_layers,
                               heads=args.student_heads, kv_heads=args.student_kv_heads, inter=inter,
                               dtype=torch.bfloat16, device=dev, tie=True, max_pos=args.seq_len)
    else:
        student = tiny_model(VOCAB_SIZE, hidden=args.student_hidden, layers=args.student_layers,
                             heads=8, inter=4 * args.student_hidden, dtype=torch.bfloat16,
                             device=dev, tie_embeddings=True, max_pos=args.seq_len)
    if args.grad_ckpt:
        student.gradient_checkpointing_enable(); student.config.use_cache = False
    if args.compile:
        student = torch.compile(student)
        if teacher is not None:
            teacher = torch.compile(teacher)          # frozen teacher fwd is inference-only -> compiling it is free speed
    np_s = sum(p.numel() for p in student.parameters())
    print(f"teacher={'(none/CE-only)' if ce_only else args.teacher} | student {np_s/1e6:.1f}M | method={args.method}")
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95),
                            fused=torch.cuda.is_available())   # fused CUDA AdamW step
    fc = FlopCounter(model_flops_from_config(student.config),
                     teacher_fm=None if no_teacher else model_flops_from_config(teacher.config),
                     opt_params=sum(p.numel() for p in student.parameters() if p.requires_grad))
    g = torch.Generator().manual_seed(args.seed + rank * 100003)   # distinct data stream per rank
    total_steps = args.stage1_steps + args.stage2_steps
    hist = []

    s_head = student.get_output_embeddings().weight
    t_head = None if no_teacher else teacher.get_output_embeddings().weight

    def lr_mult(gs):
        if args.lr_sched == "cosine":   # nanoGPT-style: linear warmup -> cosine decay to 0.1x
            w = max(1, int(args.warmup_frac * total_steps))
            if gs < w:
                return (gs + 1) / w
            p = min(1.0, (gs - w) / max(1, total_steps - w))
            return 0.1 + 0.5 * (1 - 0.1) * (1 + math.cos(math.pi * p))
        w = max(1, int(args.warmup_frac * total_steps))   # WSD: 10% warmup, full stable LR
        return wsd_lr_mult(gs, total_steps, warmup=w, decay=w)   # through S2, decay last 10%

    def micro(normal, alpha, measure=False):
        # ONE micro-batch: sample -> superpose -> forward -> (loss/grad_accum).backward().
        # No zero_grad/step here (the outer loop accumulates grad_accum micro-batches per
        # optimizer step -> large effective batch). Superposed-stage batch is scaled so the
        # forward stays B*L positions == baseline (cross_seq 2x, token_merge k x): equal
        # FLOPs/step AND tokens-per-update, so methods differ only by the packing.
        if not normal and args.method == "cross_seq": bs = args.batch_size * (1 if args.iso_token else 2)
        elif not normal and args.method == "token_merge": bs = args.batch_size * (1 if args.iso_token else args.merge_k)
        elif not normal and args.method == "cross_merge": bs = args.batch_size * (1 if args.iso_token else 2 * args.merge_k)
        else: bs = args.batch_size
        ids, mask = get_batch(train, bs, args.seq_len, dev, g)
        sup, eff = (superpose_none(ids, mask), 1.0) if (normal or ce_only) else \
                   make_superposed(args.method, ids, mask, args.merge_k, args.fixed_lambda)
        h = {}

        def fb():
            if ce_only:
                s_hidden = superposed_hidden(student, sup)
                labels = ids.clone(); labels[:, :-1] = ids[:, 1:]; labels[:, -1] = -100
                loss, parts = chunked_ce_loss(s_hidden, s_head, labels)
            elif mce:
                # TST-style, NO teacher. S2 (normal) = clean next-token CE. S1 (superposed)
                # = multi-hot CE: the student's output at each superposed position is scored
                # against the GROUND-TRUTH next tokens of BOTH constituent sequences, weighted
                # (lam, 1-lam) -- informative-by-construction targets, no OOD teacher.
                s_hidden = superposed_hidden(student, sup)
                if normal:
                    labels = ids.clone(); labels[:, :-1] = ids[:, 1:]; labels[:, -1] = -100
                    loss, parts = chunked_ce_loss(s_hidden, s_head, labels)
                else:
                    # multi-hot CE over the next BAG, generalized to K slots. slot i shifted
                    # one position = slot i of the NEXT bag/sequence. Slot weights are read
                    # from the ACTUAL superposition weights (mean over real positions), so the
                    # labels are weighted exactly as the inputs were mixed -- correct for
                    # cross_seq (K=2 -> [lam,1-lam]), token_merge (K=k tilt), AND cross_merge
                    # (K=2k product weights). With constant fixed_lambda and no padding this
                    # reproduces the prior [lam, 1-lam] / tilt scalars bit-for-bit.
                    K = sup.ids.shape[-1]
                    wsum = (sup.weights * sup.mask.unsqueeze(-1).float()).sum(dim=(0, 1))  # (K,)
                    ws = (wsum / wsum.sum().clamp_min(1e-6)).tolist()
                    loss = 0.0
                    for i in range(K):
                        Si = sup.ids[:, :, i]
                        li = Si.clone(); li[:, :-1] = Si[:, 1:]; li[:, -1] = -100
                        lossi, _ = chunked_ce_loss(s_hidden, s_head, li)
                        loss = loss + ws[i] * lossi
                    parts = {"kd": 0.0, "ce": float(loss.detach())}
            else:
                # perturbed-point KD: perturb the INPUT only during S1 (not S2 recovery).
                # onehot mode: build a SHARED perturbed one-hot (sup_n) fed to BOTH models
                #   (shared-u gradient matching); embed_ns stays 0.
                # embed mode: keep the base sup, add INDEPENDENT embedding noise per model (ablation).
                nsig = 0.0 if normal else args.noise_sigma
                def _kd(sup_in, embed_ns):
                    with torch.no_grad():
                        t_h = superposed_hidden(teacher, sup_in, embed_ns)
                    s_h = superposed_hidden(student, sup_in, embed_ns)
                    if normal and args.loss_mode == "kd_ce":
                        labels = ids.clone(); labels[:, :-1] = ids[:, 1:]; labels[:, -1] = -100
                        return chunked_distill_loss(s_h, s_head, t_h, t_head,
                                                    args.temperature, mask=sup_in.mask, labels=labels, alpha=alpha)
                    return chunked_distill_loss(s_h, s_head, t_h, t_head,
                                                args.temperature, mask=sup_in.mask)
                if nsig > 0:
                    if args.noise_mode == "onehot":
                        sup_n = superpose_input_noise(ids, mask, args.noise_k, nsig, VOCAB_SIZE)
                        l_n, p_n = _kd(sup_n, 0.0)
                    else:                                          # embed: independent per-model noise
                        l_n, p_n = _kd(sup, nsig)
                    if args.anchor:                                # loss = avg(clean, noised)
                        l_c, p_c = _kd(sup, 0.0)
                        loss = 0.5 * (l_n + l_c)
                        parts = {"kd": 0.5 * (float(p_n["kd"]) + float(p_c["kd"])),
                                 "ce": 0.5 * (float(p_n["ce"]) + float(p_c["ce"]))}
                    else:
                        loss, parts = l_n, p_n
                else:                                              # clean KD (sigma=0)
                    loss, parts = _kd(sup, 0.0)
            (loss / args.grad_accum).backward()
            h.update(loss=loss.item(), kd=float(parts["kd"]), ce=float(parts["ce"]))

        measured = measure_step_flops(fb) if measure else (fb() or None)
        return h, sup.ids.shape[1], sup.ids.shape[0], eff, measured

    # --- reap-resilient checkpoint/resume (ckpt.pt lives in the stable PVC out dir,
    # so it survives node reaps AND resubmits under a new job suffix) ---
    stages = [("S1", args.stage1_steps, False), ("S2", args.stage2_steps, True)]
    ckpt_path = os.path.join(out, "ckpt.pt")
    fc_keys = ["student_flops", "teacher_flops", "optimizer_flops", "measured_matmul",
               "sequences_seen", "tokens_processed", "_rate"]

    def save_ckpt(stage_idx, step_in_stage):
        tmp = ckpt_path + ".tmp"                       # atomic: survives a reap mid-write
        torch.save({"model": student.state_dict(), "opt": opt.state_dict(),
                    "gstep": gstep, "stage_idx": stage_idx, "step_in_stage": step_in_stage,
                    "hist": hist, "fc": {k: getattr(fc, k) for k in fc_keys}}, tmp)
        os.replace(tmp, ckpt_path)

    gstep = 0
    start_stage, start_step = 0, 0
    if args.ckpt_every and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=dev)
        student.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        gstep = ck["gstep"]; hist = ck["hist"]
        start_stage, start_step = ck["stage_idx"], ck["step_in_stage"] + 1
        for k in fc_keys:
            setattr(fc, k, ck["fc"][k])
        print(f"[resume] ckpt -> stage{start_stage} step{start_step} gstep{gstep} flops={fc.total_flops:.3e}")

    # Reap hook: on eviction the reaper `kubectl delete`s the pod -> SIGTERM, then the pod's
    # terminationGracePeriodSeconds (measured 30s for memento jobs) before SIGKILL. Save a
    # resumable ckpt inside that window so a reap loses ~0 steps instead of up to ckpt_every.
    # The breadcrumb is written FIRST and immediately: if it survives a reap the SIGTERM was
    # delivered in-window (graceful); if it's ever absent after a reap, the kill was force
    # (grace-period=0) or the signal didn't propagate past the wrapper -> fall back to a poller.
    _pos = {"si": start_stage, "step": start_step}
    _saving = {"busy": False}

    def _on_sigterm(signum, frame):
        try:
            with open(os.path.join(out, "REAP_SIGTERM.log"), "a") as f:
                f.write(f"SIGTERM rank={rank} gstep={gstep} stage={_pos['si']} step={_pos['step']}\n")
                f.flush(); os.fsync(f.fileno())
        except Exception:
            pass
        if is_main and not _saving["busy"]:
            _saving["busy"] = True
            try:
                save_ckpt(_pos["si"], _pos["step"])
            except Exception:
                pass
        os._exit(0)

    signal.signal(signal.SIGTERM, _on_sigterm)

    if args.find_micro:
        # OOM-search the largest per-rank micro-batch that fits the REAL fwd+bwd+opt.step path
        # (honors --compile / --grad_ckpt). Run single-process (no torchrun) for a per-GPU footprint.
        student.train()
        best = 0
        for cand in [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]:
            if cand > args.find_micro:
                break
            args.batch_size = cand
            try:
                opt.zero_grad(set_to_none=True)
                micro(False, 0.0)                       # S1 (superposed) path = the heavier forward
                opt.step(); torch.cuda.synchronize(); best = cand
                if is_main:
                    print(f"[find_micro] micro={cand} OK", flush=True)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if is_main:
                    print(f"[find_micro] micro={cand} STOP ({type(e).__name__})", flush=True)
                break
            finally:
                opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
        if is_main:
            print(f"MAX_MICRO={best}", flush=True)
        if ddp:
            dist.barrier(); dist.destroy_process_group()
        return

    student.train()
    for si, (tag, steps, normal) in enumerate(stages):
        if si < start_stage:
            continue                                   # whole stage already done
        s0 = start_step if si == start_stage else 0
        if s0 >= steps:
            continue
        print(f"== {tag} ({'normal' if normal else args.method}) | grad_accum={args.grad_accum} eff_batch={args.batch_size*args.grad_accum} ==")
        for step in range(s0, steps):
            _pos["si"], _pos["step"] = si, step        # keep the reap hook pointed at our position
            lr = args.lr * lr_mult(gstep)
            for grp in opt.param_groups:
                grp["lr"] = lr
            alpha = wsd_alpha(step, steps, alpha_max=0.9)
            opt.zero_grad()
            lasth = None
            for mi in range(args.grad_accum):
                h, T, outB, eff, measured = micro(normal, alpha, measure=(step == 0 and mi == 0))
                fc.add_step(seq_len=T, batch=outB * world, effective_sequences=eff * outB * world, measured_step=measured)
                lasth = h
            if ddp:                                        # average grads across ranks = the
                for p in student.parameters():             # full (world*grad_accum) effective batch
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM); p.grad.div_(world)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            gstep += 1
            if is_main and (step % args.eval_every == 0 or step == steps - 1):
                vloss = eval_lm_loss(student, val, dev, n_batches=30, batch=16, seq_len=args.seq_len)
                student.train()
                s = fc.summary()
                print(f"[{tag}] step {step:>5} loss={lasth['loss']:.3f} ce={lasth['ce']:.3f} val={vloss:.3f} "
                      f"est={s['total_flops']:.3e} rec={s['recorded_flops']:.3e}")
                hist.append({"stage": tag, "step": step, "val_loss": vloss,
                             "flops": s["total_flops"], "recorded_flops": s["recorded_flops"], "loss": lasth["loss"]})
            if is_main and args.ckpt_every and (step + 1) % args.ckpt_every == 0:
                save_ckpt(si, step)

    if is_main:                                        # only rank 0 evals/saves
        student.save_pretrained(out)
        json.dump({"task": "tinystories_distill", "teacher": args.teacher, "teacher_tag": ttag,
                   "loss_mode": args.loss_mode, "method": args.method, "fixed_lambda": args.fixed_lambda,
                   "merge_k": args.merge_k, "stage1_steps": args.stage1_steps, "stage2_steps": args.stage2_steps,
                   "seed": args.seed, "lr_sched": "wsd", "student_params": np_s, "world": world,
                   "final_val_loss": hist[-1]["val_loss"], "history": hist, "flops": fc.summary()},
                  open(os.path.join(out, "results.json"), "w"), indent=2)
        if args.ckpt_every and os.path.exists(ckpt_path):
            os.remove(ckpt_path)                       # done -> free the ~0.8GB resume ckpt
        print("saved ->", out, "| final val loss:", hist[-1]["val_loss"])
    if ddp:
        dist.barrier(); dist.destroy_process_group()   # non-main ranks wait for rank-0 save


if __name__ == "__main__":
    main()
