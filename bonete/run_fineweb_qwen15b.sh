#!/usr/bin/env bash
# Qwen2.5-1.5B  <-  Qwen2.5-7B  noise-KD SCALE-UP (canonical recipe, small from-scratch ablation).
#
# Canonical grounding (Gemma-2 / Llama-3.2 / Minitron): forward-KL = CE vs teacher's full soft
# distribution, temperature 1, pure KD (no CE term), AdamW, peak LR 1e-4..3e-4, same-family
# bigger teacher sharing the tokenizer. Our loss_mode=pure_kd + temperature 1.0 already IS this.
#
# Student = RANDOM-INIT Qwen2.5-1.5B arch (h1536 / 28L / 12h / 2kv / inter8960, GQA, tie=true).
# Teacher = trained Qwen2.5-7B (frozen), shares the Qwen2.5 tokenizer -> the shared-one-hot noise
# path is valid with SD_VOCAB=151936 (= min of the two padded embeddings; token ids align).
# Data = Qwen-tokenized FineWeb-Edu (uint32) from bonete/prep_fineweb_qwen.sh (RUN THAT FIRST).
#
# Submit (holds until you say go):
#   RUN_SCRIPT=bonete/run_fineweb_qwen15b.sh \
#   bash /data/t-jackcai/supd/cjob_submit.sh qw15b-nz qw15bnz "<ARMS>" 8 21600 p1
set -uo pipefail

# --- model / recipe ---
export TEACHER="${TEACHER:-Qwen/Qwen2.5-7B}"
export STU="${STU:---student_ref Qwen/Qwen2.5-1.5B --student_hidden 1536 --student_layers 28 --student_heads 12 --student_kv_heads 2 --student_inter 8960}"
export SD_VOCAB="${SD_VOCAB:-151936}"     # valid random token ids in BOTH 1.5B(151936) & 7B(152064) embeddings
export SD_DTYPE="${SD_DTYPE:-uint32}"     # Qwen 152k vocab > uint16
export DATA_SUBDIR="${DATA_SUBDIR:-fineweb_edu_qwen}"
export LR="${LR:-2e-4}"                   # canonical 1e-4..3e-4 (down from the 135M run's 6e-4)
export LR_SCHED="${LR_SCHED:-cosine}"     # canonical cosine decay
export SEQ="${SEQ:-1024}"

# --- memory / throughput (1.5B student TRAIN + 7B teacher INFER co-resident per rank) ---
# AUTO_MICRO (on) runs the faithful find_micro OOM probe on a real B200 node before the arms and
# sets MB to the measured max; GA is then chosen to hold EFF_BATCH_SEQ. MB/GA below are FALLBACKS.
export AUTO_MICRO="${AUTO_MICRO:-1}"
export EFF_BATCH_SEQ="${EFF_BATCH_SEQ:-256}"   # target effective batch in sequences (~0.26M tok/step)
export MB="${MB:-2}"; export GA="${GA:-16}"    # fallback only (if the search is inconclusive)
export STEPS="${STEPS:-8000}"                   # small-ablation budget; the 1.5B student stays undertrained
export EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:---compile 1 --grad_ckpt 1}"
export SWEEP_TAG="${SWEEP_TAG:-qw15bnz}"

# Reuse the shared sequential launcher (arm loop, skip/resume, noise knobs, kernels fix all inherited).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[qw15b] teacher=$TEACHER  student=Qwen2.5-1.5B(random-init)  vocab=$SD_VOCAB dtype=$SD_DTYPE  LR=$LR  MB=$MB GA=$GA STEPS=$STEPS"
exec bash "${HERE}/run_fineweb_ddp_seq.sh"
