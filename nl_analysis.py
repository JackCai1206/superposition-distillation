"""Iso-FLOP analysis for TinyStories distillation, corrected.

Baseline = the FAIR `none` run (stage1_steps == 0, i.e. normal KD+CE from step 0,
no artificial pure-KD phase). Metric: total FLOPs to reach val_loss <= target.
Reports BOTH estimated (analytic) and recorded (op-level) FLOPs, and the win as a
curve over a range of targets (not one hand-picked threshold).

Usage: python nl_analysis.py <teacher_tag>   # 'ctrl' or 'gpt2'
"""

from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict

import numpy as np

FLOP_KEYS = [("flops", "est"), ("recorded_flops", "rec")]


def flops_to(h, target, key):
    for d in h:
        if d["val_loss"] <= target:
            return d.get(key)
    return None


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "ctrl"
    runs = [json.load(open(p)) for p in glob.glob("outputs/lmdist_*/results.json")]
    runs = [r for r in runs if r.get("teacher_tag") == tag and "recorded_flops" in r["history"][-1]]
    if not runs:
        print(f"no recorded-flops runs for tag={tag} yet"); return

    base = [r for r in runs if r["method"] == "none" and r.get("stage1_steps") == 0]
    sup = [r for r in runs if r["method"] != "none"]
    if not base:
        print(f"no FAIR baseline (none, s1=0) for {tag} yet"); return

    # reachable range + a principled target = baseline's converged val loss
    base_conv = float(np.mean([min(d["val_loss"] for d in r["history"]) for r in base]))
    print(f"teacher={tag}  |  fair-baseline converged val={base_conv:.3f}  (n_base={len(base)})")
    print("win-factor = baseline_flops / method_flops to reach target val_loss; mean±std over seeds\n")

    targets = [round(base_conv + d, 2) for d in (0.20, 0.10, 0.05)]  # easy->hard, all reachable
    for tgt in targets:
        print(f"=== target val_loss <= {tgt} ===")
        bf = {k: np.mean([f for f in (flops_to(r["history"], tgt, key) for r in base) if f])
              for key, k in FLOP_KEYS}
        cells = defaultdict(lambda: defaultdict(list))
        for r in sup:
            for key, k in FLOP_KEYS:
                f = flops_to(r["history"], tgt, key)
                if f:
                    cells[(r["method"], r["stage1_steps"])][k].append(bf[k] / f)
        print(f"  {'method':>11} {'s1':>5} | {'win(est)':>14} | {'win(rec)':>14}")
        for ms in sorted(cells):
            row = cells[ms]
            est = np.array(row["est"]); rec = np.array(row["rec"])
            print(f"  {ms[0]:>11} {ms[1]:>5} | {est.mean():>6.2f} ± {est.std():.2f}   | {rec.mean():>6.2f} ± {rec.std():.2f}")
        print()


if __name__ == "__main__":
    main()
