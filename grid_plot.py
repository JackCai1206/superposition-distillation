"""Plot the lambda x stage-1 grid:
  (1) heatmap of iso-FLOP win-factor (none_flops / config_flops to reach 0.98)
  (2) iso-FLOP accuracy-vs-FLOPs curve for the best cell vs the baseline

Reads the fine-eval grid runs (outputs/distadd_*_d10_9070*/results.json).
"""

from __future__ import annotations

import glob
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TARGET = 0.98


def flops_to(h, t):
    for d in h:
        if d["exact_match"] >= t:
            return d["flops"]
    return None


def load(pat="outputs/distadd_*_d10_l*/results.json"):   # new unique-dir naming (incl lambda)
    runs = []
    for p in sorted(glob.glob(pat)):
        runs.append(json.load(open(p)))
    return runs


def main():
    runs = load()
    if not runs:
        print("no grid runs yet"); return
    base = next((r for r in runs if r["method"] == "none"), None)
    base_f = flops_to(base["history"], TARGET) if base else None
    grid = [r for r in runs if r["method"] == "cross_seq"]

    lambdas = sorted({r["fixed_lambda"] for r in grid})
    s1s = sorted({r["stage1_steps"] for r in grid})
    W = np.full((len(lambdas), len(s1s)), np.nan)
    for r in grid:
        i, j = lambdas.index(r["fixed_lambda"]), s1s.index(r["stage1_steps"])
        f = flops_to(r["history"], TARGET)
        if f and base_f:
            W[i, j] = base_f / f

    # (1) heatmap
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(W, cmap="RdYlGn", vmin=0.5, vmax=1.6, aspect="auto", origin="lower")
    ax.set_xticks(range(len(s1s))); ax.set_xticklabels(s1s)
    ax.set_yticks(range(len(lambdas))); ax.set_yticklabels(lambdas)
    ax.set_xlabel("stage-1 (superposed) steps"); ax.set_ylabel("lambda")
    ax.set_title(f"iso-FLOP win-factor vs none (to {TARGET})  [>1 = win]")
    for i in range(len(lambdas)):
        for j in range(len(s1s)):
            if not np.isnan(W[i, j]):
                ax.text(j, i, f"{W[i,j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, label="win-factor"); fig.tight_layout()
    fig.savefig("outputs/grid_winfactor.png", dpi=130); print("saved outputs/grid_winfactor.png")

    # (2) best-cell iso-FLOP curve vs none
    best = max((r for r in grid if flops_to(r["history"], TARGET)),
               key=lambda r: base_f / flops_to(r["history"], TARGET))
    fig, ax = plt.subplots(figsize=(7, 5))
    for r, lab, col in [(base, "none (baseline)", "tab:gray"),
                        (best, f"cross_seq best (lam={best['fixed_lambda']}, s1={best['stage1_steps']})", "tab:blue")]:
        h = r["history"]
        ax.plot([d["flops"] for d in h], [d["exact_match"] for d in h], "-o", ms=3, color=col, label=lab)
    ax.axhline(TARGET, ls="--", color="k", alpha=0.3)
    bf, mf = base_f, flops_to(best["history"], TARGET)
    ax.set_xscale("log"); ax.set_xlabel("cumulative total FLOPs (student+teacher)")
    ax.set_ylabel("exact-match")
    ax.set_title(f"best iso-FLOP: {bf/mf:.2f}x fewer FLOPs to {TARGET}")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig("outputs/best_isoflops.png", dpi=130)
    print("saved outputs/best_isoflops.png")


if __name__ == "__main__":
    main()
