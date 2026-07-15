#!/usr/bin/env bash
export SWEEP_TAG=fw4bwsd LR_SCHED=wsd METHODS="cross_seq" LAMBDA="0.5"
exec bash "$(dirname "$0")/run_fineweb_4b.sh"
