#!/usr/bin/env bash
# Multi-seed confirmation of the key WSD finals: none-KD, KD λ0.7 (winner),
# MCE λ0.5 (TST best), vanilla-NTP -- extra seeds to firm up the small-but-real effects.
export SWEEP_TAG=fw4bwsd LR_SCHED=wsd
export ARMS="pure_kd:none:0.5:1 pure_kd:none:0.5:2 pure_kd:cross_seq:0.7:1 pure_kd:cross_seq:0.7:2 mce:cross_seq:0.5:1 mce:cross_seq:0.5:2 ce_only:none:0.5:2 ce_only:none:0.5:3"
exec bash "$(dirname "$0")/run_fineweb_pack.sh"
