"""Iso-FLOP analysis for the TinyStories distillation runs.

Metric: total (student+teacher) FLOPs to reach val_loss <= THRESH (lower loss is
better, so we look for the first crossing DOWN through the threshold). Win-factor
= none_flops / method_flops, aggregated mean+/-std over seeds.

Usage: python nl_analysis.py [threshold]
"""

from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict

import numpy as np


def flops_to_loss(h, thresh):
    for d in h:
        if d["val_loss"] <= thresh:
            return d["flops"]
    return None


def main():
    thresh = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
    tag = sys.argv[2] if len(sys.argv) > 2 else None   # 'ctrl' or 'gpt2'
    runs = [json.load(open(p)) for p in glob.glob("outputs/lmdist_*/results.json")]
    if tag:
        runs = [r for r in runs if r.get("teacher_tag") == tag]
    if not runs:
        print(f"no TinyStories distill runs (tag={tag}) yet"); return
    print(f"teacher tag = {tag or 'ALL'}")

    # show the reachable val-loss range to help pick a threshold
    finals = sorted(min(d["val_loss"] for d in r["history"]) for r in runs)
    print(f"best val_loss reached across runs: min={finals[0]:.3f} max={finals[-1]:.3f}")
    print(f"threshold = {thresh}\n")

    none_f = [flops_to_loss(r["history"], thresh) for r in runs if r["method"] == "none"]
    none_f = [x for x in none_f if x]
    if not none_f:
        print(f"baseline never reaches val_loss<={thresh}; pick a higher threshold"); return
    base = float(np.mean(none_f))

    cells = defaultdict(list)
    for r in runs:
        if r["method"] in ("cross_seq", "token_merge"):
            f = flops_to_loss(r["history"], thresh)
            if f:
                cells[(r["method"], r["stage1_steps"])].append(base / f)

    print(f"baseline none: {base:.3e} FLOPs to val<= {thresh}  (n={len(none_f)})\n")
    print(f"{'method':>12} {'s1':>5} {'win mean':>9} {'std':>6} {'n':>3}")
    print("-" * 40)
    for (m, s1) in sorted(cells):
        w = np.array(cells[(m, s1)])
        print(f"{m:>12} {s1:>5} {w.mean():>9.2f} {w.std():>6.2f} {len(w):>3}")


if __name__ == "__main__":
    main()
