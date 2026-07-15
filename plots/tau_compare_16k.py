"""tau=1 vs tau=2, p0 vs cross_seq, all at 16K rollout, iso-step (960). Binomial CIs."""
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# (gsm avg@1, gsm_n, math avg@4, math_nproblems)
data = {
    "tau=1 p0":       (0.3654, 1319, 0.3340, 500),
    "tau=1 cross_seq":(0.3525, 1319, 0.3380, 500),
    "tau=2 p0":       (0.4147, 1319, 0.4050, 500),
    "tau=2 cross_seq":(0.4678, 1319, 0.3950, 500),
}
# native (rope-native anchor) references
native = {"GSM8K": 0.774, "MATH-500": 0.558}

def se(p, n):  # binomial SE (MATH approximate: treats per-problem score ~bernoulli)
    return math.sqrt(max(p*(1-p), 1e-9)/n)

arms = list(data)
colors = {"p0": "#4C72B0", "cross_seq": "#C44E52"}
def col(a): return colors["cross_seq"] if "cross_seq" in a else colors["p0"]
hatch = lambda a: "" if "tau=2" in a else "////"

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, (bench, gi) in zip(axes, [("GSM8K", 0), ("MATH-500", 2)]):
    vals = [data[a][gi]*100 for a in arms]
    ns   = [data[a][gi+1] for a in arms]
    errs = [se(data[a][gi], data[a][gi+1])*100 for a in arms]
    x = np.arange(len(arms))
    bars = ax.bar(x, vals, yerr=errs, capsize=4,
                  color=[col(a) for a in arms],
                  hatch=[hatch(a) for a in arms], edgecolor="white")
    ax.axhline(native[bench]*100, ls="--", color="green", lw=1.4,
               label=f"native anchor ({native[bench]*100:.1f})")
    for xi, v, e in zip(x, vals, errs):
        ax.text(xi, v+e+0.8, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([a.replace("tau=", "τ=").replace(" ", "\n") for a in arms], fontsize=9)
    ax.set_ylabel("accuracy (%)")
    ax.set_title(f"{bench}  ({'avg@1, n=1319' if bench=='GSM8K' else 'avg@4, 500×4'})")
    ax.set_ylim(0, max(native[bench]*100, max(vals))*1.18)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.25)

fig.suptitle("Off-policy logit KD into Qwen2.5-Math-1.5B-Instruct  —  16K rollout, iso-step (~0.9B tok)",
             fontsize=11)
# legend for hatch (tau)
from matplotlib.patches import Patch
leg = [Patch(facecolor="grey", hatch="////", label="τ=1"),
       Patch(facecolor="grey", label="τ=2")]
fig.legend(handles=leg, loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.04, 1, 0.96))
out = "/home/t-jackcai/superposition-distillation/plots/tau_compare_16k.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)
