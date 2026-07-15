#!/usr/bin/env bash
export SWEEP_TAG=fw4bwsd LR_SCHED=wsd METHODS="none"
exec bash "$(dirname "$0")/run_fineweb_4b.sh"
