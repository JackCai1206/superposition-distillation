#!/usr/bin/env bash
# Overnight 4B-token FineWeb-Edu run: extend the matched-config experiment 4x
# (2000->8000 steps, ~1B->~4B tokens) to test whether the superposition deficit
# CLOSES with training (catch-up) or persists (residual harm). Seed 0, 4 arms,
# same FRAC=0.3 / lambda=0.5 / 0.5M-token batch as the 1B run -> directly comparable.
export SWEEP_TAG="${SWEEP_TAG:-fw4b}"   # override (e.g. fw4bwsd) for the WSD re-run; LR_SCHED passes through
export STEPS=8000          # 8000 * 524288 = ~4.19B tokens
export SEEDS="0"           # single seed: question is curve shape, not multi-seed sig
export FRAC=0.3; export LAMBDA="${LAMBDA:-0.5}"   # LAMBDA overridable for the ratio sweep
export MB=64 GA=8          # eff batch = 64*8*1024 = 524288 (~0.5M tokens), validated ~80% TDP
export EVAL=200            # ~40 eval points over the run
export CKPT_EVERY=200      # resumable ckpt every ~200 steps -> a reap loses <=200 steps
export LR=6e-4
export METHODS="${METHODS:-none cross_seq token_merge ce_only}"   # override per-arm for 1-GPU jobs
export MAXPAR=1
exec bash "$(dirname "$0")/run_fineweb.sh"
