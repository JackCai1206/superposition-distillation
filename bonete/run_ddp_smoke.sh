#!/usr/bin/env bash
export SWEEP_TAG=fw4bddptest STEPS=24 EVAL=4 CKPT=12 LOSS_MODE=mce METHOD=cross_seq LAMBDA=0.5 SEED=9
exec bash "$(dirname "$0")/run_fineweb_ddp.sh"
