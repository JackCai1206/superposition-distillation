"""Aggregate the multi-seed runs: win-factor (mean +/- std over seeds) vs the
baseline, for parallel (cross_seq) and sequential (token_merge) superposition
across the stage-1 budget. Also checks the late-collapse is gone (final vs peak).

Reads runs that used the WSD-LR schedule (results.json has lr_sched == "wsd").
"""

from __future__ import annotations

import glob
import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TARGET = 0.98


def f2(h, t):
    for d in h:
        if d["exact_match"] >= t:
            return d["flops"]
    return None


def load():
    runs = []
    for p in glob.glob("outputs/distadd_*_seed*/results.json"):
        r = json.load(open(p))
        if r.get("lr_sched") == "wsd":
            runs.append(r)
    return runs


def main():
    runs = load()
    if not runs:
        print("no WSD-LR multi-seed runs yet"); return

    # baseline: mean FLOPs-to-target over none seeds
    none_f = [f2(r["history"], TARGET) for r in runs if r["method"] == "none"]
    none_f = [x for x in none_f if x]
    base = float(np.mean(none_f)) if none_f else None

    # group cross_seq/token_merge by (method, stage1)
    groups = defaultdict(list)
    for r in runs:
        if r["method"] in ("cross_seq", "token_merge"):
            f = f2(r["history"], TARGET)
            if f:
                groups[(r["method"], r["stage1_steps"])].append(base / f)

    print(f"baseline none: {base:.3e} FLOPs to {TARGET}  (n={len(none_f)} seeds)\n")
    print(f"{'method':>12} {'s1':>5} {'win mean':>9} {'std':>6} {'n':>3}")
    print("-" * 40)
    for (m, s1) in sorted(groups):
        w = np.array(groups[(m, s1)])
        print(f"{m:>12} {s1:>5} {w.mean():>9.2f} {w.std():>6.2f} {len(w):>3}")

    # collapse check: how many runs have final < peak - 0.05?
    collapsed = sum(1 for r in runs
                    if max(d["exact_match"] for d in r["history"]) - r["history"][-1]["exact_match"] > 0.05)
    print(f"\ncollapse check: {collapsed}/{len(runs)} runs end >0.05 below their peak")

    # plot win vs s1 with error bars, both methods
    fig, ax = plt.subplots(figsize=(7, 5))
    for m, col in [("cross_seq", "tab:blue"), ("token_merge", "tab:red")]:
        s1s = sorted({s for (mm, s) in groups if mm == m})
        means = [np.mean(groups[(m, s)]) for s in s1s]
        stds = [np.std(groups[(m, s)]) for s in s1s]
        ax.errorbar(s1s, means, yerr=stds, marker="o", capsize=4, color=col,
                    label=f"{m} ({'parallel' if m=='cross_seq' else 'sequential'})")
    ax.axhline(1.0, ls="--", color="k", alpha=0.4, label="baseline (none)")
    ax.set_xlabel("stage-1 (superposed) steps"); ax.set_ylabel(f"win-factor vs none (to {TARGET})")
    ax.set_title("Parallel vs sequential superposition (5 seeds, WSD-LR)")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig("outputs/multiseed_winfactor.png", dpi=130)
    print("\nsaved outputs/multiseed_winfactor.png")


if __name__ == "__main__":
    main()
