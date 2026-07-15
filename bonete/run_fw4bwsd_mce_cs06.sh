#!/usr/bin/env bash
# TST-style superposed-label baseline (no teacher, multi-hot ground-truth CE), cross_seq λ=0.6, WSD.
export SWEEP_TAG=fw4bwsd LR_SCHED=wsd METHODS="cross_seq" LOSS_MODE=mce LAMBDA="0.6"
exec bash "$(dirname "$0")/run_fineweb_4b.sh"
