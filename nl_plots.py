"""Informative plots for the corrected TinyStories de-risk (fair baseline).

Fig A  outputs/nl_isoflops[_mode].png   : val-loss vs cumulative RECORDED FLOPs, fair-none
        vs superposition (mean +/- std over seeds), one panel per teacher. The
        leftward shift = the iso-FLOP win.
Fig B  outputs/nl_wincurve[_mode].png   : win-factor vs target val-loss (curve, not one
        threshold), both teachers x both methods. Flat ~1.2x = robust, not cherry-picked.
"""

from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FLOPKEY = "recorded_flops"   # honest, op-level
MODE = sys.argv[1] if len(sys.argv) > 1 else "kd_ce"   # kd_ce | pure_kd
SUFFIX = "" if MODE == "kd_ce" else "_" + MODE


def load(tag):
    runs = [json.load(open(p)) for p in glob.glob("outputs/lmdist_*/results.json")]
    runs = [r for r in runs if r.get("teacher_tag") == tag
            and r.get("loss_mode", "kd_ce") == MODE and FLOPKEY in r["history"][-1]]
    by = defaultdict(list)
    for r in runs:
        if r["method"] == "none":
            if r.get("stage1_steps") == 0 and r.get("stage2_steps", 0) >= 3000:
                by[("none", 0)].append(r)
        else:
            by[(r["method"], r["stage1_steps"])].append(r)
    return by


def mean_curve(runs):
    n = min(len(r["history"]) for r in runs)
    flops = np.array([d[FLOPKEY] for d in runs[0]["history"][:n]])
    vals = np.array([[d["val_loss"] for d in r["history"][:n]] for r in runs])
    return flops, vals.mean(0), vals.std(0)


def flops_to(h, target):
    for d in h:
        if d["val_loss"] <= target:
            return d[FLOPKEY]
    return None


# ---------- Fig A: iso-FLOP curves ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, tag in zip(axes, ["ctrl", "gpt2"]):
    by = load(tag)
    if ("none", 0) not in by:
        continue
    series = [(("none", 0), "none (fair baseline)", "tab:gray"),
              (("cross_seq", 1500), "cross_seq (parallel) s1=1500", "tab:blue"),
              (("token_merge", 1500), "token_merge (sequential) s1=1500", "tab:red")]
    for key, lab, c in series:
        if key not in by:
            continue
        runs = by[key]
        f, mu, sd = mean_curve(runs)
        ax.plot(f, mu, "-", color=c, label=lab)
        ax.fill_between(f, mu - sd, mu + sd, color=c, alpha=0.2)
        # mark the stage-1 -> stage-2 (superposed -> normal recovery) handoff
        if key[0] != "none":
            stages = [d["stage"] for d in runs[0]["history"][:len(f)]]
            ti = next((i for i in range(1, len(stages))
                       if stages[i] == "S2" and stages[i - 1] == "S1"), None)
            if ti is not None:
                ax.scatter([f[ti]], [mu[ti]], color=c, marker="*", s=200,
                           zorder=6, edgecolor="k", linewidth=0.6)
    ax.scatter([], [], color="k", marker="*", s=120, label="★ stage1→stage2 handoff")
    ax.set_xlabel("cumulative recorded FLOPs (student+teacher)")
    ax.set_title(f"{tag} teacher")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
axes[0].set_ylabel("val loss"); axes[0].set_ylim(1.5, 3.2)
fig.suptitle("TinyStories de-risk (fair baseline): iso-FLOP curves, 5 seeds")
fig.tight_layout(); fig.savefig(f"outputs/nl_isoflops{SUFFIX}.png", dpi=130)
print(f"saved outputs/nl_isoflops{SUFFIX}.png")

# ---------- Fig B: win-factor vs target ----------
fig, ax = plt.subplots(figsize=(8, 5))
styles = {("ctrl", "cross_seq"): ("tab:blue", "-"), ("ctrl", "token_merge"): ("tab:red", "-"),
          ("gpt2", "cross_seq"): ("tab:blue", "--"), ("gpt2", "token_merge"): ("tab:red", "--")}
for tag in ["ctrl", "gpt2"]:
    by = load(tag)
    if ("none", 0) not in by:
        continue
    base = by[("none", 0)]
    conv = np.mean([min(d["val_loss"] for d in r["history"]) for r in base])
    targets = np.round(np.arange(conv + 0.03, conv + 0.45, 0.03), 3)
    for method in ["cross_seq", "token_merge"]:
        s1 = 1500
        if (method, s1) not in by:
            continue
        ws = []
        for t in targets:
            bf = np.mean([f for f in (flops_to(r["history"], t) for r in base) if f])
            mf = [flops_to(r["history"], t) for r in by[(method, s1)]]
            mf = [x for x in mf if x]
            ws.append(bf / np.mean(mf) if mf else np.nan)
        c, ls = styles[(tag, method)]
        ax.plot(targets - conv, ws, ls, color=c, marker="o", ms=3,
                label=f"{tag}:{method} (s1=1500)")
ax.axhline(1.0, color="k", ls=":", alpha=0.5)
ax.set_xlabel("target val loss above baseline's converged loss"); ax.set_ylabel("win-factor vs none")
ax.set_title("Win-factor vs target (flat ~1.2x = robust, not threshold-picked)")
ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"outputs/nl_wincurve{SUFFIX}.png", dpi=130)
print(f"saved outputs/nl_wincurve{SUFFIX}.png")
