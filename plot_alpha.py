"""Plot the Stage-2 WSD-alpha (KD/CE mix) schedule overlaid on the val-loss
trajectory, per teacher. Shows the GPT-2 bounce coincides with the alpha-warmup
(rising KD weight pulls val loss toward the generalist teacher's distribution),
while the expert (ctrl) teacher shows no bounce. -> outputs/nl_alpha.png
"""

from __future__ import annotations

import glob
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kd_loss import wsd_alpha

S2_STEPS = 2000


def s2_traj(tag, method="cross_seq", s1=1500):
    ps = [x for x in glob.glob(f"outputs/lmdist_{tag}_{method}_l0.7_s1{s1}_seed0_*/results.json")
          if "recorded_flops" in json.load(open(x))["history"][-1]]
    h = json.load(open(ps[0]))["history"]
    s2 = [d for d in h if d["stage"] == "S2"]
    return [d["step"] for d in s2], [d["val_loss"] for d in s2]


def main():
    steps = np.arange(0, S2_STEPS, 5)
    alpha = np.array([wsd_alpha(int(s), S2_STEPS, 0.9, 0.1, 0.1) for s in steps])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, tag in zip(axes, ["ctrl", "gpt2"]):
        ax.plot(steps, alpha, color="tab:green", lw=2, label="α (KD weight)")
        ax.plot(steps, 1 - alpha, color="tab:orange", lw=2, ls="--", label="1−α (CE weight)")
        ax.set_xlabel("stage-2 step"); ax.set_ylabel("loss weight"); ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"{tag} teacher"); ax.grid(True, alpha=0.3)
        ax2 = ax.twinx()
        sx, sv = s2_traj(tag)
        ax2.plot(sx, sv, color="tab:blue", marker="o", ms=3, label="val loss (cross_seq)")
        ax2.set_ylabel("val loss", color="tab:blue"); ax2.tick_params(axis="y", labelcolor="tab:blue")
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
    fig.suptitle("WSD-α (Stage-2 KD/CE mix) vs val loss — α-warmup drives the GPT-2 bounce")
    fig.tight_layout(); fig.savefig("outputs/nl_alpha.png", dpi=130)
    print("saved outputs/nl_alpha.png")


if __name__ == "__main__":
    main()
