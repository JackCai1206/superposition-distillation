"""Pre-build the capped reasoning-pool cache to PVC, CPU-only (no GPU -> immune to the
GPU-budget reaper). The real 4-GPU training arms then skip the slow idle pool build and
start training within ~1 min (cache hit), minimizing their reap-exposure window.

Writes the SAME cache files data.reasoning_stream expects: keyed by
(dataset, split, cap, seq_len, min_len, world) under $HF_HOME/sd_pool. Run with
PREBUILD_WORLD == the training job's GPU count so the shard layout matches."""
import os

from transformers import AutoTokenizer

from config import Config
import data

cfg = Config()
cfg.data.seq_len = int(os.environ.get("SEQ_LEN", "16384"))
student = os.environ.get("STUDENT", "Qwen/Qwen2.5-Math-1.5B")
world = int(os.environ.get("PREBUILD_WORLD", "4"))
print(f"[prebuild] student={student} world={world} cap={os.environ.get('SD_MAX_EXAMPLES')} "
      f"seq={cfg.data.seq_len}", flush=True)
tok = AutoTokenizer.from_pretrained(student)
gen = data.reasoning_stream(cfg, tok, rank=0, world=world)
next(gen)   # triggers rank0 build of all `world` shards + the .done flag, then yields one
print("[prebuild] cache ready", flush=True)
