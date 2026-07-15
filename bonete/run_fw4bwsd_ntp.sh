#!/usr/bin/env bash
# Vanilla NTP baseline: no teacher, no superposition, plain next-token CE (ce_only). WSD.
export SWEEP_TAG=fw4bwsd LR_SCHED=wsd METHODS="ce_only"
exec bash "$(dirname "$0")/run_fineweb_4b.sh"
