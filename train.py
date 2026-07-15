"""Two-stage superposition-distillation training (real-scale loop).

Stage 1: train the student on SUPERPOSED inputs (OOD for the frozen teacher) with
         pure forward-KL. This is where the compute-packing happens.
Stage 2: "recovery" on NORMAL single-sequence inputs.

The frozen teacher provides logits live (no_grad). Conditions share the same data
stream and are compared at equal TOTAL (student + teacher + optimizer) FLOPs.

Parallelism: single-node multi-GPU via torchrun. The forward is custom (transformer
body + external fused-head loss), which bypasses DDP's module-call bookkeeping, so
we use MANUAL gradient all-reduce once per optimizer step (cheap at 1.5B over
NVLink) + gradient accumulation for large effective batches:
    effective_batch = nproc * batch_size(micro, per-GPU) * grad_accum
Ranks shard the stream by example index; rank 0 owns logging/val/ckpt/wandb/save.

Loss modes: pure_kd = forward-KL only everywhere (no alpha, no CE — the cleanest
mechanism framing); kd_ce = alpha-scheduled KD+CE on normal-input steps only
(superposed steps are ALWAYS pure KL: a superposed position has no valid hard label).
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

from config import Config
from data import (batched, build_superposed, pretrain_stream, reasoning_stream,
                  synthetic_stream)
from flops import FlopCounter, measure_step_flops, model_flops_from_config
from kd_loss import chunked_ce_loss, chunked_distill_loss, wsd_alpha, wsd_lr_mult
from model import (load_model, load_tokenizer, superposed_hidden, tiny_model)
from superpose import superpose_none

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def setup_dist():
    """torchrun multi-GPU: returns (rank, world, device or None). Single-process
    when launched without torchrun. gloo on CPU enables local debug of this path."""
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        import datetime
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        # Long timeout: the capped-pool loader materializes+tokenizes its shard on the
        # first batch (~minutes), and cross-rank variance can push the first collective
        # well past NCCL's 600s default -> spurious DistBackendError. 60min absorbs it.
        dist.init_process_group(backend, timeout=datetime.timedelta(minutes=60))
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", rank))
        if torch.cuda.is_available():
            torch.cuda.set_device(local)
            dist.barrier()   # force NCCL init NOW (ranks in sync) before the slow pool build
            return rank, world, f"cuda:{local}"
        return rank, world, "cpu"
    return 0, 1, None


def all_reduce_grads(model, world):
    if world == 1:
        return
    for p in model.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad)
            p.grad.div_(world)


def build_models(cfg: Config, tokenizer, loss_mode="pure_kd"):
    dev = cfg.train.device
    need_teacher = loss_mode != "ce"               # CE-only baseline is teacher-free
    if cfg.debug:
        V = len(tokenizer) if tokenizer is not None else 256
        teacher = None
        if need_teacher:
            teacher = tiny_model(V, hidden=96, layers=3, device=dev)
            for p in teacher.parameters():
                p.requires_grad_(False)
            teacher.eval()
        student = tiny_model(V, hidden=48, layers=2, device=dev)
    else:
        dt = DTYPES[cfg.model.dtype]
        teacher = load_model(cfg.model.teacher, dtype=dt, device=dev, frozen=True) if need_teacher else None
        if getattr(cfg.model, "student_init", "pretrained") == "random":
            from model import random_model
            student = random_model(cfg.model.student, dtype=dt, device=dev,
                                   max_pos=cfg.model.student_max_pos,
                                   rope_theta=cfg.model.student_rope_theta)
        else:
            student = load_model(cfg.model.student, dtype=dt, device=dev, frozen=False,
                                 max_pos=cfg.model.student_max_pos,
                                 rope_theta=cfg.model.student_rope_theta)
    # MANDATORY at 16K: without checkpointing the student stores ~1.5MB/token of
    # backward activations (28 layers x 8960-wide MLP) -> ~100GB at 65K tokens in
    # flight -> OOM on 178GB B200s. ~30% recompute buys ~25x activation headroom.
    # (Analytic FLOPs keep the 3x-fwd model -> comparisons unaffected, absolute
    # totals undercount the recompute ~25%; same for every arm.)
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    print("[model] student gradient checkpointing ON")
    return teacher, student


def make_stream(cfg: Config, tokenizer, kind: str, rank=0, world=1):
    if cfg.debug or getattr(cfg.train, "synth_data", False):
        V = len(tokenizer) if tokenizer is not None else 256
        s = synthetic_stream(cfg, V)
    elif kind == "pretrain":
        s = pretrain_stream(cfg, tokenizer)
    else:
        # reasoning_stream shards across ranks BEFORE tokenization (no wasted work)
        return reasoning_stream(cfg, tokenizer, rank=rank, world=world)
    # shard by example index so ranks see disjoint data
    return itertools.islice(s, rank, None, world) if world > 1 else s


def build_val_set(cfg: Config, tokenizer, kind: str, n_batches: int):
    """A small FIXED set of NORMAL (single-seq) batches for periodic val CE (rank0)."""
    loader = batched(make_stream(cfg, tokenizer, kind), cfg.train.batch_size)
    val = []
    for _ in range(n_batches):
        try:
            val.append(next(loader))
        except StopIteration:
            break
    return val


@torch.no_grad()
def eval_val(student, val_set, device, vmin):
    """Mean next-token CE over the fixed val set (student normal forward)."""
    if not val_set:
        return float("nan")
    student.eval()
    tot, n = 0.0, 0
    for ids, mask in val_set:
        ids, mask = ids.to(device), mask.to(device)
        logits = student(input_ids=ids, attention_mask=mask).logits[..., :vmin]
        labels = ids[:, 1:].clone()
        labels[mask[:, 1:] == 0] = -100               # ignore padded targets
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, vmin), labels.reshape(-1),
                               ignore_index=-100)
        tot += loss.item(); n += 1
    student.train()
    return tot / max(n, 1)


def maybe_init_wandb(cfg, run_name, vocab_align, loss_mode, rank, world, grad_accum):
    """wandb (rank 0 only), keyed for iso-FLOP comparison: x-axis = total FLOPs."""
    if rank != 0:
        return None
    if cfg.debug and os.environ.get("SD_WANDB") != "1":
        return None
    if not (os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_MODE") == "offline"):
        return None
    try:
        import re
        import wandb
        # DETERMINISTIC id + resume="allow": requeued pods AND fresh RESUBMITS of the
        # same job name CONTINUE one wandb run. The id strips submit_job.sh's random
        # 5-char suffix (resubmit changes it), so 'supd-1bv7-s1240-<xxxxx>' always maps
        # to the same run; bump the version in the JOB_NAME for a fresh run.
        # SD_WANDB_RUN_ID overrides outright.
        run_id = os.environ.get("SD_WANDB_RUN_ID") or re.sub(
            r"-[a-z0-9]{5}$", "", re.sub(r"[^a-zA-Z0-9_-]", "-", run_name))[:120]
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "superposition-distillation"),
            id=run_id, resume="allow",
            name=run_name,
            config={"method": cfg.superpose.method, "loss_mode": loss_mode,
                    "teacher": cfg.model.teacher, "student": cfg.model.student,
                    "vocab_align": vocab_align, "seq_len": cfg.data.seq_len,
                    "micro_batch": cfg.train.batch_size, "grad_accum": grad_accum,
                    "world_size": world,
                    "effective_batch": cfg.train.batch_size * grad_accum * world,
                    "lr": cfg.train.lr,
                    "stage1_steps": cfg.train.stage1_steps, "stage2_steps": cfg.train.stage2_steps,
                    "fixed_lambda": cfg.superpose.fixed_lambda, "merge_k": cfg.superpose.merge_k,
                    "temperature": cfg.kd.temperature, "alpha_max": cfg.kd.alpha_max,
                    "student_max_pos": cfg.model.student_max_pos,
                    "student_rope_theta": cfg.model.student_rope_theta,
                    "reasoning_packed": getattr(cfg.data, "reasoning_packed", False)})
        wandb.define_metric("flops")
        wandb.define_metric("*", step_metric="flops")   # iso-FLOP x-axis by default
        return run
    except Exception as e:
        print(f"[wandb] disabled ({type(e).__name__}: {str(e)[:80]})")
        return None


def save_checkpoint(student, tokenizer, out_dir, gstep, stage, fc, vloss, manifest, opt=None):
    """Save a student checkpoint (HF format -> directly loadable by vLLM eval) plus
    optimizer state (resume) and a manifest entry tagging it with cumulative FLOPs."""
    cdir = os.path.join(out_dir, f"checkpoint-{gstep}")
    student.save_pretrained(cdir)
    if tokenizer is not None:
        tokenizer.save_pretrained(cdir)
    if opt is not None:
        torch.save({"opt": opt.state_dict(), "gstep": gstep}, os.path.join(cdir, "opt.pt"))
    s = fc.summary()
    manifest.append({"gstep": gstep, "stage": stage, "dir": cdir,
                     "total_flops": s["total_flops"], "teacher_flops": s["teacher_flops"],
                     "sequences_seen": s["sequences_seen"], "val_loss": vloss})
    with open(os.path.join(out_dir, "checkpoints.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[ckpt] saved {cdir} @ {s['total_flops']:.3e} FLOPs val={vloss:.4f}")


def run_stage(cfg, teacher, student, opt, fc, loader, *, method, steps, normal, tag,
              s_head, t_head, vmin, gstep0, total_steps, loss_mode, val_set,
              rank=0, world=1, grad_accum=1,
              measure_flops=False, ckpt_every=0, out_dir=None, tokenizer=None, manifest=None,
              wb=None):
    """One training stage. Each step = grad_accum micro fwd/bwd + all-reduce + opt.step."""
    student.train()
    dev = cfg.train.device
    warmup = min(cfg.train.warmup_steps, max(1, total_steps // 10))
    decay = min(300, max(5, total_steps // 5))
    logf = []
    real_tokens = 0
    for step in range(steps):
        gstep = gstep0 + step
        lr = cfg.train.lr * wsd_lr_mult(gstep, total_steps, warmup=warmup, decay=decay)
        for grp in opt.param_groups:
            grp["lr"] = lr

        opt.zero_grad()
        msum = {"loss": 0.0, "kd": 0.0, "ce": 0.0}
        exhausted = False
        for micro in range(grad_accum):
            try:
                ids, mask = next(loader)
            except StopIteration:
                print(f"[{tag}] r{rank} stream exhausted at step {step}.{micro}")
                exhausted = True
                break
            ids, mask = ids.to(dev), mask.to(dev)

            def ce_labels():
                lab = ids.clone()
                lab[:, :-1] = ids[:, 1:]
                lab[:, -1] = -100                     # next-token CE; last has no target
                sv = torch.zeros_like(mask)
                sv[:, :-1] = mask[:, 1:]
                lab[sv == 0] = -100                   # ignore padded targets
                return lab

            if normal:
                sup, eff = superpose_none(ids, mask), 1.0
                if loss_mode in ("kd_ce", "ce"):
                    labels = ce_labels()
                    alpha = (0.0 if loss_mode == "ce" else
                             wsd_alpha(step, steps, cfg.kd.alpha_max, cfg.kd.warmup_frac, cfg.kd.decay_frac))
                else:                                  # pure_kd: forward-KL only
                    labels, alpha = None, 1.0
            else:
                sup, eff = build_superposed(method, ids, mask, cfg)
                labels, alpha = None, 1.0             # superposed -> pure KD always

            h = {}

            def fwd_bwd():
                s_hidden = superposed_hidden(student, sup)
                if teacher is None:                    # CE-only: teacher-free baseline
                    loss, parts = chunked_ce_loss(s_hidden, s_head, labels)
                else:
                    with torch.no_grad():
                        t_hidden = superposed_hidden(teacher, sup)
                    loss, parts = chunked_distill_loss(s_hidden, s_head, t_hidden, t_head,
                                                       cfg.kd.temperature, mask=sup.mask,
                                                       labels=labels, alpha=alpha)
                (loss / grad_accum).backward()
                h.update(loss=loss.item(), kd=float(parts["kd"]), ce=float(parts["ce"]))

            # op-level FLOPs: optional cross-check (OOMs at scale w/ ckpting) — off by default
            if measure_flops and step == 0 and micro == 0:
                measured = measure_step_flops(fwd_bwd)
            else:
                fwd_bwd(); measured = None

            T = sup.ids.shape[1]
            out_B = sup.ids.shape[0]
            # global accounting: every rank runs a statistically identical micro
            fc.add_step(seq_len=T, batch=out_B * world, effective_sequences=eff * out_B * world,
                        measured_step=measured)
            real_tokens += int(mask.sum()) * world
            for k in msum:
                msum[k] += h[k] / grad_accum

        if exhausted:
            break
        all_reduce_grads(student, world)
        torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.train.grad_clip)
        opt.step()

        is_eval = (step % cfg.train.eval_every == 0) or (step == steps - 1)
        if rank == 0:
            s = fc.summary()
            vloss = eval_val(student, val_set, dev, vmin) if is_eval else float("nan")
            if is_eval or step % cfg.train.log_every == 0:
                print(f"[{tag}] step {step:>5} gstep={gstep} lr={lr:.2e} "
                      f"alpha={alpha:.2f} loss={msum['loss']:.4f} kd={msum['kd']:.4f} ce={msum['ce']:.4f} "
                      f"val={vloss:.4f} est={s['total_flops']:.3e} tok={real_tokens:.3e}")
            if is_eval:
                logf.append({"stage": tag, "step": step, "gstep": gstep, "loss": msum["loss"],
                             "kd": msum["kd"], "ce": msum["ce"], "val_loss": vloss,
                             "total_flops": s["total_flops"], "recorded_flops": s["recorded_flops"],
                             "teacher_flops": s["teacher_flops"], "sequences_seen": s["sequences_seen"],
                             "real_tokens": real_tokens})
            # wandb EVERY optimizer step: per-step loss curves, and the frequent
            # file-stream contact keeps the server's liveness window fed (runs were
            # flapping to "crashed" at the old once-per-5-min cadence).
            if wb:
                rec = {"flops": s["total_flops"], "gstep": gstep, "lr": lr, "alpha": alpha,
                       "stage": 1 if tag == "S1" else 2, "train/loss": msum["loss"],
                       "train/kd": msum["kd"], "train/ce": msum["ce"],
                       "teacher_flops": s["teacher_flops"], "sequences_seen": s["sequences_seen"],
                       "real_tokens": real_tokens}
                if is_eval:
                    rec["val/loss"] = vloss
                wb.log(rec)

        # periodic checkpoints (FLOP-tagged) for the accuracy-vs-FLOPs eval sweep
        if rank == 0 and ckpt_every and out_dir and gstep > 0 and gstep % ckpt_every == 0:
            cv = eval_val(student, val_set, dev, vmin)
            save_checkpoint(student, tokenizer, out_dir, gstep, tag, fc, cv, manifest, opt=opt)
    return logf, gstep0 + steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="cross_seq", choices=["none", "cross_seq", "token_merge"])
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--stage1_steps", type=int, default=None)
    ap.add_argument("--stage2_steps", type=int, default=None)
    ap.add_argument("--seq_len", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None, help="MICRO batch per GPU")
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--synth_data", action="store_true", help="real models, synthetic tokens (offline path test)")
    ap.add_argument("--data_kind", default="pretrain", choices=["pretrain", "reasoning"])
    ap.add_argument("--teacher", default=None)
    ap.add_argument("--student", default=None)
    ap.add_argument("--fixed_lambda", type=float, default=None,
                    help="constant mixing weight; analysis shows ~0.7 keeps teacher coherent")
    ap.add_argument("--mix_alpha", type=float, default=None, help="Beta(a,a) for lambda")
    ap.add_argument("--merge_k", type=int, default=None)
    ap.add_argument("--loss_mode", default="pure_kd", choices=["kd_ce", "pure_kd", "ce"],
                    help="pure_kd: forward-KL only (Minitron-canonical). kd_ce: alpha-mixed "
                         "KD+CE. ce: CE-only LM training, TEACHER-FREE (the 'is the teacher "
                         "worth its FLOPs' baseline for pretraining distillation).")
    ap.add_argument("--student_init", default=None, choices=["pretrained", "random"],
                    help="random = from-scratch student (pretraining distill, no prior)")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--temperature", type=float, default=None,
                    help="KD softmax temperature. Default cfg=2.0; Minitron/canonical "
                         "LLM logit-KD uses 1.0 (raw distribution, no high-T softening)")
    ap.add_argument("--val_batches", type=int, default=4)
    ap.add_argument("--student_max_pos", type=int, default=-1,
                    help="-1 = config default; 0 = disable RoPE ctx fix (native, e.g. "
                         "Qwen2.5-1.5B-Instruct is already 32K); >0 = that value")
    ap.add_argument("--student_rope_theta", type=float, default=-1,
                    help="-1 = config default; 0 = disable (use model's native theta)")
    ap.add_argument("--measure_flops", action="store_true",
                    help="op-level FLOP cross-check via FlopCounterMode (small batch only; "
                         "OOMs at scale due to checkpointing). Default off -> analytic only.")
    ap.add_argument("--ckpt_every", type=int, default=0,
                    help="save a FLOP-tagged checkpoint every N global steps (for the "
                         "vLLM accuracy-vs-FLOPs eval sweep)")
    args = ap.parse_args()

    rank, world, dist_dev = setup_dist()

    cfg = Config()
    cfg.debug = args.debug
    cfg.superpose.method = args.method
    if args.device:
        cfg.train.device = args.device
    elif dist_dev is not None:
        cfg.train.device = dist_dev
    if args.stage1_steps is not None: cfg.train.stage1_steps = args.stage1_steps
    if args.stage2_steps is not None: cfg.train.stage2_steps = args.stage2_steps
    if args.seq_len is not None: cfg.data.seq_len = args.seq_len
    if args.batch_size is not None: cfg.train.batch_size = args.batch_size
    if args.lr is not None: cfg.train.lr = args.lr
    if args.temperature is not None: cfg.kd.temperature = args.temperature
    if args.student_init is not None: cfg.model.student_init = args.student_init
    if cfg.debug and args.device is None and dist_dev is None: cfg.train.device = "cpu"
    cfg.train.synth_data = args.synth_data
    if args.teacher: cfg.model.teacher = args.teacher
    if args.student: cfg.model.student = args.student
    if args.student_max_pos != -1:
        cfg.model.student_max_pos = args.student_max_pos if args.student_max_pos > 0 else None
    if args.student_rope_theta != -1:
        cfg.model.student_rope_theta = args.student_rope_theta if args.student_rope_theta > 0 else None
    if args.fixed_lambda is not None: cfg.superpose.fixed_lambda = args.fixed_lambda
    if args.mix_alpha is not None: cfg.superpose.mix_alpha = args.mix_alpha
    if args.merge_k is not None: cfg.superpose.merge_k = args.merge_k

    # per-method output dir (+ job id) so concurrent runs don't collide. Write
    # DIRECTLY to PVC (SD_OUTPUT_BASE) on the cluster so periodic checkpoints survive
    # preemption and never fill the small ephemeral pod disk; local relative otherwise.
    jid = os.environ.get("SLURM_JOB_ID", os.environ.get("JOB_NAME", "local"))
    import re as _re
    jid = _re.sub(r"-[a-z0-9]{5}$", "", jid)   # STABLE across resubmits (drop random suffix)
    run_name = f"{args.data_kind}_{cfg.superpose.method}_{args.loss_mode}_s1{cfg.train.stage1_steps}_{jid}"
    base = os.environ.get("SD_OUTPUT_BASE")
    cfg.train.output_dir = os.path.join(base, run_name) if base else f"outputs/{run_name}"
    if rank == 0:
        os.makedirs(cfg.train.output_dir, exist_ok=True)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")   # TF32 for any fp32 matmuls
    tokenizer = None if cfg.debug else load_tokenizer(cfg.model.student)
    teacher, student = build_models(cfg, tokenizer, loss_mode=args.loss_mode)
    opt = torch.optim.AdamW(student.parameters(), lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay, betas=(0.9, 0.95),
                            fused=torch.cuda.is_available())   # fused CUDA AdamW kernel

    # COMPLETE FLOP accounting: student fwd+bwd, frozen-teacher forward (DOMINATES at
    # 7B-scale and is what cross_seq amortizes), AdamW update; op-level recorded too.
    fc = FlopCounter(model_flops_from_config(student.config),
                     teacher_fm=(model_flops_from_config(teacher.config) if teacher is not None else None),
                     opt_params=sum(p.numel() for p in student.parameters() if p.requires_grad))

    # vocab-align: shared Qwen2.5 tokenizer, padded to different widths -> truncate
    # both LM heads to the common real vocab (extra rows are unused padding).
    s_head = student.get_output_embeddings().weight
    if teacher is not None:
        t_head = teacher.get_output_embeddings().weight
        vmin = min(s_head.shape[0], t_head.shape[0])
        s_head, t_head = s_head[:vmin], t_head[:vmin]
    else:                                            # CE-only: no teacher head
        t_head, vmin = None, s_head.shape[0]

    kind = args.data_kind
    total_steps = cfg.train.stage1_steps + cfg.train.stage2_steps
    val_set = build_val_set(cfg, tokenizer, kind, args.val_batches) if rank == 0 else []
    manifest = []                                       # FLOP-tagged checkpoint records
    ckpt_dir = cfg.train.output_dir if (not cfg.debug or os.environ.get("SD_DEBUG_CKPT")) else None

    # ---- RESUME: latest checkpoint-N from SD_RESUME_FROM or our own (stable) run dir.
    # Restores weights (+ optimizer state when opt.pt exists), the FLOP counters
    # (exactly, from the manifest record the same accounting wrote), and later the
    # data loaders are fast-forwarded deterministically. SD_RESUME=0 disables.
    resume_gstep = 0
    if (not cfg.debug or os.environ.get("SD_DEBUG_CKPT")) and os.environ.get("SD_RESUME", "1") == "1":
        import glob as _glob
        # search BOTH the explicit seed dir and our own (stable) dir, take the newest:
        # a resubmit must prefer its own later checkpoints over the SD_RESUME_FROM seed
        rdirs = [d for d in [os.environ.get("SD_RESUME_FROM"), cfg.train.output_dir] if d]
        cks = [p for d in rdirs for p in _glob.glob(os.path.join(d, "checkpoint-*"))
               if p.rsplit("-", 1)[1].isdigit()]
        cks.sort(key=lambda p: int(p.rsplit("-", 1)[1]))
        rdir = os.path.dirname(cks[-1]) if cks else cfg.train.output_dir
        if cks:
            last = cks[-1]
            # checkpoint-N is saved AFTER completing gstep N -> steps 0..N are done,
            # training resumes at gstep N+1
            resume_gstep = int(last.rsplit("-", 1)[1]) + 1
            if rank == 0:
                print(f"[resume] loading {last} (completed {resume_gstep} steps)")
            tmp = load_model(last, dtype=DTYPES[cfg.model.dtype], device="cpu", frozen=False)
            student.load_state_dict({k: v.to(cfg.train.device) for k, v in tmp.state_dict().items()})
            del tmp
            optp = os.path.join(last, "opt.pt")
            if os.path.exists(optp):
                opt.load_state_dict(torch.load(optp, map_location=cfg.train.device)["opt"])
                if rank == 0:
                    print("[resume] optimizer state restored")
            elif rank == 0:
                print("[resume] WARN: no opt.pt -> fresh optimizer moments")
            mpath = os.path.join(rdir, "checkpoints.json")
            if os.path.exists(mpath):
                with open(mpath) as f:
                    manifest.extend(e for e in json.load(f)
                                    if e.get("stage") != "final" and e.get("gstep", 0) <= resume_gstep)
            if manifest:
                e = manifest[-1]
                fc.teacher_flops = float(e.get("teacher_flops") or 0.0)
                fc.student_flops = max(0.0, float(e["total_flops"]) - fc.teacher_flops)
                fc.sequences_seen = float(e.get("sequences_seen") or 0.0)
    wb = maybe_init_wandb(cfg, run_name, vmin, args.loss_mode, rank, world, args.grad_accum)
    if rank == 0:
        print(f"teacher={cfg.model.teacher} student={cfg.model.student} vocab_align={vmin} "
              f"method={cfg.superpose.method} loss_mode={args.loss_mode} world={world} "
              f"micro={cfg.train.batch_size} accum={args.grad_accum} "
              f"eff_batch={cfg.train.batch_size * args.grad_accum * world} "
              f"seq={cfg.data.seq_len} s1={cfg.train.stage1_steps} s2={cfg.train.stage2_steps} "
              f"val_batches={len(val_set)}")

    common = dict(s_head=s_head, t_head=t_head, vmin=vmin, total_steps=total_steps,
                  loss_mode=args.loss_mode, val_set=val_set, rank=rank, world=world,
                  grad_accum=args.grad_accum, measure_flops=args.measure_flops,
                  ckpt_every=args.ckpt_every, out_dir=ckpt_dir, tokenizer=tokenizer,
                  manifest=manifest, wb=wb)

    # Superposed S1 pulls take MORE data per step at ~equal forward size/FLOPs:
    #  - cross_seq: 2x sequences (pairs mixed position-wise; forward batch = B)
    #  - token_merge: k x sequences (each merged k->1 to L/k positions, so k*B
    #    merged sequences fill the same B*L positions per forward)
    s1_mult = {"cross_seq": 2, "token_merge": cfg.superpose.merge_k}.get(cfg.superpose.method, 1)
    s1_eff_batch = cfg.train.batch_size * s1_mult
    s1_done = min(resume_gstep, cfg.train.stage1_steps)
    s2_done = max(0, resume_gstep - cfg.train.stage1_steps)

    if cfg.train.stage1_steps - s1_done > 0:
        if rank == 0:
            print(f"== Stage 1: superposed ({cfg.superpose.method}) on {kind} data "
                  f"(resume at {s1_done}/{cfg.train.stage1_steps}) ==")
        s1_loader = batched(make_stream(cfg, tokenizer, kind, rank, world), s1_eff_batch)
        for _ in range(s1_done * args.grad_accum):     # deterministic fast-forward
            next(s1_loader)
        s1log, g = run_stage(cfg, teacher, student, opt, fc, s1_loader,
                             method=cfg.superpose.method,
                             steps=cfg.train.stage1_steps - s1_done,
                             normal=False, tag="S1", gstep0=s1_done, **common)
    else:
        s1log, g = [], cfg.train.stage1_steps

    if cfg.train.stage2_steps - s2_done > 0:
        if rank == 0:
            print(f"== Stage 2: normal-data on {kind} data "
                  f"(resume at {s2_done}/{cfg.train.stage2_steps}) ==")
        s2_loader = batched(make_stream(cfg, tokenizer, kind, rank, world), cfg.train.batch_size)
        for _ in range(s2_done * args.grad_accum):     # deterministic fast-forward
            next(s2_loader)
        s2log, g = run_stage(cfg, teacher, student, opt, fc, s2_loader,
                             method="none", steps=cfg.train.stage2_steps - s2_done,
                             normal=True, tag="S2",
                             gstep0=cfg.train.stage1_steps + s2_done, **common)
    else:
        s2log = []

    summary = fc.summary()
    if rank == 0:
        print("== Done ==", summary)

    if rank == 0:
        # persist the trained student + a results record for the iso-FLOP comparison
        if not cfg.debug:
            student.save_pretrained(cfg.train.output_dir)
            tokenizer.save_pretrained(cfg.train.output_dir)
        results = {
            "method": cfg.superpose.method, "data_kind": kind, "loss_mode": args.loss_mode,
            "fixed_lambda": cfg.superpose.fixed_lambda, "merge_k": cfg.superpose.merge_k,
            "teacher": cfg.model.teacher, "student": cfg.model.student, "vocab_align": vmin,
            "seq_len": cfg.data.seq_len, "micro_batch": cfg.train.batch_size,
            "grad_accum": args.grad_accum, "world_size": world,
            "effective_batch": cfg.train.batch_size * args.grad_accum * world,
            "lr": cfg.train.lr,
            "stage1_steps": cfg.train.stage1_steps, "stage2_steps": cfg.train.stage2_steps,
            "final_s1_loss": s1log[-1]["loss"] if s1log else None,
            "final_s2_loss": s2log[-1]["loss"] if s2log else None,
            "final_val_loss": (s2log[-1]["val_loss"] if s2log else
                               (s1log[-1]["val_loss"] if s1log else None)),
            "wandb_run_id": wb.id if wb else None,
            "history": s1log + s2log,
            "flops": summary,
        }
        with open(os.path.join(cfg.train.output_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
        # final model as the last point on the accuracy-vs-FLOPs curve
        manifest.append({"gstep": total_steps, "stage": "final", "dir": cfg.train.output_dir,
                         "total_flops": summary["total_flops"], "teacher_flops": summary["teacher_flops"],
                         "sequences_seen": summary["sequences_seen"],
                         "val_loss": results["final_val_loss"]})
        with open(os.path.join(cfg.train.output_dir, "checkpoints.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        print("saved ->", cfg.train.output_dir)

        if wb:
            wb.summary.update({"final/val_loss": results["final_val_loss"],
                               "final/total_flops": summary["total_flops"],
                               "final/teacher_flops": summary["teacher_flops"],
                               "final/sequences_seen": summary["sequences_seen"]})

        # auto-eval reasoning students on MATH-500 (best-effort; never breaks the run)
        if not cfg.debug and kind == "reasoning":
            try:
                from eval import eval_math
                eval_n = int(os.environ.get("EVAL_MATH_N", "100"))
                if eval_n > 0:
                    ev = eval_math(student, tokenizer, cfg.train.device, which="math500", n=eval_n)
                    with open(os.path.join(cfg.train.output_dir, "eval_math500.json"), "w") as f:
                        json.dump(ev, f, indent=2)
                    print("eval MATH-500 ->", ev)
                    if wb and ev.get("n", 0) > 0:
                        wb.summary.update({"final/math500_inline_acc": ev["accuracy"]})
            except Exception as e:
                print("auto-eval skipped:", type(e).__name__, str(e)[:120])
        if wb:
            wb.finish()

    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
