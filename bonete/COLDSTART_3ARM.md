# 3-arm base-student cold-start (superposition's defensible home)

Tests whether a **superposed logit-KD cold-start** is a cheaper substitute for the
canonical CE (sequence-level KD) cold-start, in the *additive* regime (base student).

## Why this setting
- Post-training instruct distillation is net-destructive and dominated by on-policy KD
  (GKD/MiniLLM); superposition is structurally incompatible with on-policy (needs fixed
  superposable inputs). See research notes.
- The one phase where a full-vocab teacher forward on fixed inputs is on the critical
  path is the **off-policy logit-KD cold-start** — IF you use logit-KD rather than the
  default CE cold-start (CE has no teacher forward at train time). This experiment tests
  whether logit-KD cold-start is worth it, and whether superposition makes it cheaper.

## Arms (iso-step = iso-data; FLOPs differ → x-axis)
| arm | method | loss_mode | stages | teacher fwd/seq |
|-----|--------|-----------|--------|-----------------|
| CE       | none      | ce      | s1=0, s2=4500 | none (CE on teacher tokens) |
| p0       | none      | pure_kd | s1=0, s2=4500 | 1.0× (full-vocab fwd-KL) |
| cross_seq| cross_seq | pure_kd | s1=4500, s2=0 | ~0.5× (2 seqs/forward) |

## Shared config
- Student: `Qwen/Qwen2.5-Math-1.5B` (BASE) — RoPE-extended 4096→32768 (θ=500000),
  adapts during the run. Teacher: `nvidia/OpenMath-Nemotron-7B`.
- Data: OpenMathReasoning CoT, seq_len 16384, ~4B tokens/arm. CAPPED to a curated
  pool of SD_MAX_EXAMPLES=280K (data.py materializes + reshuffles/repeats across
  epochs; Muennighoff repetition is ~as good as fresh up to ~4 epochs). At 280K:
  p0/CE = 2 epochs, cross_seq = 4 epochs (it consumes 2× data/FLOP). All arms ≤4 ep.
- eff_batch 256 (GPUS 4 × micro 8 × accum 8), ~2250 steps, LR 3e-4, τ=2, ckpt/150.

## Measurement
- Plot GSM8K avg@1 + MATH-500 avg@4 vs total/teacher FLOPs (accuracy-vs-FLOPs curve).
- Claim to test: cross_seq reaches p0's accuracy at ~half the teacher FLOPs (and both
  vs the CE baseline). Eval at 16K via bonete/eval_cluster.sh on each checkpoint.

## Open design notes
- cross_seq runs PURE superposition throughout (no clean recovery tail). With a base
  student there is little clean behavior to disrupt, so OOD risk is lower than instruct.
  If transfer fails, v2 adds a clean recovery phase (s2>0).
- RoPE 8× extension on only ~4B tokens is a risk; teacher provides long-context targets.
- Same LR across arms (3e-4) despite CE vs KD gradient-scale differences — note as a
  potential confound; LR check is a cheap follow-up if results are close.

## Jobs
cs4b-ce-f67c8, cs4b-p0-972de, cs4b-cs-21d56  (PROJECT_NAME=supdistill, 4×B200 each)
