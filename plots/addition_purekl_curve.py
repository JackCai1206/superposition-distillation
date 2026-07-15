"""Addition toy (pure-KL, 5 seeds): exact-match vs FLOPs learning curves per method."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

d = json.load(open("/tmp/add_curve.json"))
colors = {"none": "#4C72B0", "cross_seq": "#C44E52", "token_merge": "#55A868"}
labels = {"none": "none (baseline)", "cross_seq": "cross_seq", "token_merge": "token_merge"}

fig, ax = plt.subplots(figsize=(8, 5.2))
for m in ["none", "cross_seq", "token_merge"]:
    if m not in d:
        continue
    cur = np.array(d[m]["curve"])           # [flops, mean_acc, std_acc]
    fl, mu, sd = cur[:, 0], cur[:, 1], cur[:, 2]
    ax.plot(fl, mu, "-o", ms=3, color=colors[m], label=f"{labels[m]} (n={d[m]['n']})")
    ax.fill_between(fl, mu - sd, mu + sd, color=colors[m], alpha=0.18)
    # mark the superposed->normal (S1->S2) transition. ONLY meaningful for the
    # superposition methods: for `none` both stages are identical normal-data KD,
    # so its "S2 boundary" is a no-op and would mislead.
    s2 = d[m].get("s2_flops")
    if s2 and m != "none":
        ax.axvline(s2, ls=":", color=colors[m], lw=1.4, alpha=0.8)
        ax.text(s2, 0.03, " S2", color=colors[m], fontsize=8, rotation=90, va="bottom", ha="left")

ax.set_xlabel("total FLOPs (student + teacher)")
ax.set_ylabel("exact-match accuracy")
ax.set_title("Addition toy — clean pure-KL distillation (no α), 5 seeds\nexact-match vs iso-FLOP (linear x)")
ax.axhline(0.99, ls="--", color="grey", lw=1, alpha=0.7)
ax.text(0.01, 0.992, "0.99", transform=ax.get_yaxis_transform(), fontsize=8, color="grey", va="bottom")
ax.set_ylim(0, 1.03)
ax.grid(True, which="both", alpha=0.25)
ax.legend(loc="lower right")
fig.tight_layout()
out = "/home/t-jackcai/superposition-distillation/plots/addition_purekl_curve.png"
fig.savefig(out, dpi=140)
print("wrote", out)
