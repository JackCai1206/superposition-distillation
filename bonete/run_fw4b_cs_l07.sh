#!/usr/bin/env bash
# FineWeb 4B cross_seq at lambda=0.7 (milder superposition than the 0.5 main run).
# 1-GPU; writes lambda-distinct dirs (lmdist_smol_kd_cross_seq_l0.7_*_fw4b) + logs.
export METHODS="cross_seq" LAMBDA="0.7"
exec bash "$(dirname "$0")/run_fineweb_4b.sh"
