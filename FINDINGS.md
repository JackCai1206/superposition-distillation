# Findings & progress notes

Running log of what we've established, the methodology corrections along the way,
and what's open. (Setup/recipe is in `README.md`.)

## Question

Does feeding a **frozen teacher superposed inputs** make logit distillation more
**compute-efficient** (iso-FLOP)? Two superposition methods, both convex combos of
token embeddings:
- **cross_seq (parallel)** — mix two sequences position-wise (one forward pass
  carries two sequences; ~half the batch per step).
- **token_merge (sequential)** — mix `k` adjacent tokens of one sequence into one
  position (shorter sequence).

Two-stage: **Stage 1** superposed (the cheap pretraining), **Stage 2** normal-data
recovery. Metric: total (student+teacher) FLOPs to reach a target quality.

## Headline result

**Superposition-distillation gives a robust but modest ~1.2× iso-FLOP win**, on a
controlled addition toy *and* real language (TinyStories), consistent across:
- expert (controlled) **and** pretrained (GPT-2) teachers,
- KD+CE **and** pure-KD loss,
- estimated **and** op-level-recorded FLOPs.

Best/most-reliable cells ~1.2–1.3×. The optimal Stage-1 budget is an interior
sweet spot (too short = no head start, too long = wasted superposed compute).

## What was an artifact (caught by rigor)

Every flashy intermediate number turned out to be a measurement artifact. This is
the main lesson of the project:

| claimed | reality | what caught it |
|---|---|---|
| 1.75× (addition λ-grid) | single-seed noise; true ~1.3× | 5-seed error bars |
| 3× (GPT-2 NL) | unfair baseline: old `none` did a long pure-KD phase that doesn't reduce val loss for a generalist teacher, then too little CE → it was starved of the only phase that mattered | fair baseline (CE/​KD from step 0) + audit showing `none` crossed only in Stage 2 |
| any single-threshold win | threshold-sensitive at the easy/steep tail | win-vs-target **curve** instead of one threshold |
| est-only FLOPs | chunked-KD checkpointing adds ~29% recompute the analytic `3×fwd` misses | record op-level FLOPs too (cancels in the ratio) |

Sequential (token_merge) **loses** on addition (adjacent digits are each critical
for carries) but **wins** on language (adjacent tokens are redundant/mergeable) —
a clean interpretable contrast.

## Methodology that the result depends on

- **Multi-seed** (5) with error bars — single seed is misleading here.
- **Fair baseline**: `none` = normal training from step 0 (no artificial pure-KD
  prefix); both arms differ only by the superposed prefix.
- **Pure KD** (`--loss_mode pure_kd`) is the cleanest framing — removes the α/CE
  KD-vs-CE tug-of-war (which causes a val-loss *bounce* for generalist teachers),
  the generalist confound, and the stage-length α-schedule mismatch.
- **Win-vs-target curve**, anchored to the baseline's converged quality.
- **Complete, audited FLOP accounting** (see below).

## FLOP accounting (central to the claim — independently audited)

`audit_flops.py` re-derives the total from scratch (measure each stage's per-step
op-level cost, multiply by step counts) and matches the logged value to **ratio
1.000**. Confirmed counted: Stage 1 **and** Stage 2; cross_seq's half-batch
Stage-1 discount (ratio 0.50); the Stage-1 cost is **included** in FLOPs-to-target;
teacher forward (~35%/step); LM-head matmul; backward; checkpoint recompute;
and **optimizer** (AdamW, ~0.007% of total — including it leaves the win at
1.1966 → 1.1966). Memory: chunked fused-linear KD (`chunked_distill_loss`) never
materializes the `[B,T,V]` logits (bit-exact to plain forward-KL).

## File guide

- `superpose.py` collators · `kd_loss.py` (forward-KL, chunked KD, WSD schedules)
- `flops.py` (analytic + recorded + optimizer) · `audit_flops.py` (verification)
- Addition testbed: `addition.py`, `train_addition.py`, `distill_addition.py`
- TinyStories testbed: `nl_data.py`, `train_lm.py`, `distill_lm.py`, `prepare_tinystories.py`
- Analysis/plots: `nl_analysis.py`, `nl_plots.py`, `plot_alpha.py`, `*_analysis.py`
- Launchers: `scripts/*.slurm` (pli-c; `pack_*.slurm` pack N runs/GPU)

## Open threads

- Newer teacher family (Qwen2.5 152k-vocab — also stress-tests the memory path —
  or SmolLM2); GPT-2 kept for now as the diagnostic.
- Stable-crossing metric (vs first-touch) to remove the last easy-tail wiggle.
- Scale-up to the 7B reasoning setting (R1-Distill-Qwen → small Qwen2.5).
