"""Addition 2-D sweep (pure-KL): cross_seq & token_merge over S1-fraction x lambda.
(1) heatmap of iso-FLOP advantage over `none` (=none_f99 / method_f99; >1 method wins)
(2) curve grid: per S1-fraction, the cross_seq curves at each lambda vs the baseline.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

d = json.load(open("/tmp/sweep.json"))
none_f99 = d["none"]["f99"]
TOTAL = 2250
S1S = [450, 900, 1350, 1800]                 # frac 0.2/0.4/0.6/0.8
FRACS = [s / TOTAL for s in S1S]
LAMS = [0.5, 0.7, 0.9]

# ---------- (1) advantage heatmap ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
for ax, m in zip(axes, ["cross_seq", "token_merge"]):
    M = np.full((len(S1S), len(LAMS)), np.nan)
    for i, s1 in enumerate(S1S):
        for j, lam in enumerate(LAMS):
            cell = d[m].get(f"{s1}_{lam}")
            if cell and cell["f99"]:
                M[i, j] = none_f99 / cell["f99"]      # >1 => method reaches 0.99 w/ fewer FLOPs
    im = ax.imshow(M, cmap="RdYlGn", vmin=0.5, vmax=1.5, aspect="auto", origin="lower")
    ax.set_xticks(range(len(LAMS))); ax.set_xticklabels([f"λ={l}" for l in LAMS])
    ax.set_yticks(range(len(S1S))); ax.set_yticklabels([f"frac={f:.1f}" for f in FRACS])
    for i in range(len(S1S)):
        for j in range(len(LAMS)):
            v = M[i, j]
            ax.text(j, i, "—" if np.isnan(v) else f"{v:.2f}", ha="center", va="center",
                    fontsize=9, color="black")
    ax.set_title(f"{m}: iso-FLOP advantage vs none\n(>1 = beats baseline to 0.99)")
fig.colorbar(im, ax=axes, shrink=0.8, label="none_f99 / method_f99")
fig.suptitle("Addition sweep — FLOPs-to-0.99 advantage over baseline (pure-KL, 5 seeds)", y=1.02)
fig.savefig("/home/t-jackcai/superposition-distillation/plots/addition_sweep_heatmap.png",
            dpi=140, bbox_inches="tight")
print("wrote addition_sweep_heatmap.png")

# ---------- (2) cross_seq curve grid by S1-fraction ----------
fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
ncur = np.array(d["none"]["curve"])
lam_c = {0.5: "#F4A582", 0.7: "#D6604D", 0.9: "#B2182B"}
for ax, s1, frac in zip(axes.ravel(), S1S, FRACS):
    ax.plot(ncur[:, 0], ncur[:, 1], "--", color="black", lw=1.6, label="none (baseline)")
    for lam in LAMS:
        cell = d["cross_seq"].get(f"{s1}_{lam}")
        if not cell:
            continue
        c = np.array(cell["curve"])
        ax.plot(c[:, 0], c[:, 1], "-", color=lam_c[lam], lw=1.6, label=f"cross_seq λ={lam}")
    ax.axhline(0.99, ls=":", color="grey", lw=1)
    ax.set_title(f"S1-fraction = {frac:.1f}  (S1={s1}/{TOTAL})")
    ax.set_ylim(0, 1.03); ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, loc="lower right")
for ax in axes[-1]:
    ax.set_xlabel("total FLOPs")
for ax in axes[:, 0]:
    ax.set_ylabel("exact-match")
fig.suptitle("Addition sweep — cross_seq vs baseline, by S1-fraction × λ (pure-KL, 5 seeds)")
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig("/home/t-jackcai/superposition-distillation/plots/addition_sweep_curves.png", dpi=140)
print("wrote addition_sweep_curves.png")
