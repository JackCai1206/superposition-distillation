"""Plot exact-match vs cumulative student FLOPs for the addition distill runs.

The iso-FLOP question is visual here: for a given x (FLOPs), which method has the
higher curve? Saves outputs/addition_isoflops.png.
"""

from __future__ import annotations

import glob
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"none": "tab:gray", "cross_seq": "tab:blue", "token_merge": "tab:red"}


def main():
    fig, ax = plt.subplots(figsize=(7, 5))
    for p in sorted(glob.glob("outputs/distadd_*/results.json")):
        with open(p) as f:
            r = json.load(f)
        h = r["history"]
        xs = [d["flops"] for d in h]
        ys = [d["exact_match"] for d in h]
        m = r["method"]
        ax.plot(xs, ys, "-o", ms=3, label=f"{m} (final {r['final']['exact_match']:.3f})",
                color=COLORS.get(m))
        # mark the stage-1 -> stage-2 boundary
        s2 = [d for d in h if d.get("stage") == "S2"]
        if s2:
            ax.axvline(s2[0]["flops"], color=COLORS.get(m), ls=":", alpha=0.3)
    ax.set_xscale("log")
    ax.set_xlabel("cumulative student training FLOPs")
    ax.set_ylabel("exact-match accuracy")
    ax.set_title("Superposition distillation on LSB-first addition (iso-FLOP)")
    ax.grid(True, alpha=0.3); ax.legend()
    out = "outputs/addition_isoflops.png"
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print("saved", out)


if __name__ == "__main__":
    main()
