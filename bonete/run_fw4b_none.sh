#!/usr/bin/env bash
# 1-GPU single-arm (none) FineWeb 4B job: small gang schedules in any quota sliver,
# can't suffer a 4-GPU gang-abort. Writes to the shared lmdist_*_fw4b out dirs.
export METHODS="none"
exec bash "$(dirname "$0")/run_fineweb_4b.sh"
