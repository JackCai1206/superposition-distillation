"""Seeded lambda x stage-1 heatmap: mean win-factor over seeds per cell, using
the WSD-LR runs (results.json lr_sched == "wsd"). Also writes a std heatmap so
the noise is visible. Refreshes the old single-seed grid_winfactor.png.
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


def main():
    runs = [json.load(open(p)) for p in glob.glob("outputs/distadd_*_seed*/results.json")]
    runs = [r for r in runs if r.get("lr_sched") == "wsd"]
    none_f = [f2(r["history"], TARGET) for r in runs if r["method"] == "none"]
    none_f = [x for x in none_f if x]
    base = float(np.mean(none_f))

    cells = defaultdict(list)
    for r in runs:
        if r["method"] == "cross_seq":
            f = f2(r["history"], TARGET)
            if f:
                cells[(r["fixed_lambda"], r["stage1_steps"])].append(base / f)

    lams = sorted({l for (l, _) in cells}); s1s = sorted({s for (_, s) in cells})
    M = np.full((len(lams), len(s1s)), np.nan)
    S = np.full((len(lams), len(s1s)), np.nan)
    N = np.zeros((len(lams), len(s1s)), int)
    for (l, s), v in cells.items():
        i, j = lams.index(l), s1s.index(s)
        M[i, j] = np.mean(v); S[i, j] = np.std(v); N[i, j] = len(v)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0.7, vmax=1.4, aspect="auto", origin="lower")
    ax.set_xticks(range(len(s1s))); ax.set_xticklabels(s1s)
    ax.set_yticks(range(len(lams))); ax.set_yticklabels(lams)
    ax.set_xlabel("stage-1 (superposed) steps"); ax.set_ylabel("lambda")
    ax.set_title(f"win-factor vs none (to {TARGET}), mean±std over seeds  [>1=win]")
    for i in range(len(lams)):
        for j in range(len(s1s)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i,j]:.2f}\n±{S[i,j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, label="win-factor"); fig.tight_layout()
    fig.savefig("outputs/grid_winfactor_seeded.png", dpi=130)
    print(f"baseline none {base:.3e} FLOPs (n={len(none_f)})")
    print("saved outputs/grid_winfactor_seeded.png")
    print(f"\n{'lam\\s1':>7}" + "".join(f"{s:>14}" for s in s1s))
    for i, l in enumerate(lams):
        print(f"{l:>7}" + "".join(f"  {M[i,j]:.2f}±{S[i,j]:.2f}(n{N[i,j]})" for j in range(len(s1s))))


if __name__ == "__main__":
    main()
