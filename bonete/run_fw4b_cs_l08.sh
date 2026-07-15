#!/usr/bin/env bash
# FineWeb 4B cross_seq at lambda=0.8 (fills the 0.7-0.9 gap in the ratio sweep). 1-GPU.
export METHODS="cross_seq" LAMBDA="0.8"
exec bash "$(dirname "$0")/run_fineweb_4b.sh"
