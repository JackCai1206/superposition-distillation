"""Is the TRUE next token still TOP-RANKED in the superposed output, despite low prob?
(Low prob is fine if ordering is preserved; what matters is rank, not magnitude.)
Compute rank of the true next token under super vs clean, across lambda."""
from __future__ import annotations
import os, torch, base64
import torch.nn.functional as F
from model import load_model, superposed_logits
from superpose import superpose_none, superpose_cross_seq
from nl_data import load_split, get_batch

dev = "cuda"; T = 1.0; B, L = 16, 256
teacher = load_model(os.environ.get("TEACHER", "HuggingFaceTB/SmolLM2-1.7B"), dtype=torch.bfloat16, device=dev, frozen=True)
train = load_split("train"); g = torch.Generator().manual_seed(0)
A, mA = get_batch(train, B, L, dev, g); Bx, mB = get_batch(train, B, L, dev, g)
nA = A[:, 1:]; mm = (mA.bool() & mB.bool())[:, :-1]
@torch.no_grad()
def logits(sup): return superposed_logits(teacher, sup).float()[:, :-1]   # [B,L-1,V]
def rank_of(lg, idx):
    tp = lg.gather(-1, idx.unsqueeze(-1))               # [B,L-1,1] logit of true token
    return ((lg > tp).sum(-1) + 1)[mm]                  # strict rank (1=top), valid positions
def stats(lg, idx):
    r = rank_of(lg, idx).float()
    return dict(top1=(r == 1).float().mean().item(), top5=(r <= 5).float().mean().item(),
                top10=(r <= 10).float().mean().item(), top100=(r <= 100).float().mean().item(),
                median=r.median().item(), mean=r.mean().item())

clean = stats(logits(superpose_none(A, mA)), nA)
print("VOCAB=49152  (random-chance median rank ~24576)")
print("clean: top1=%.3f top5=%.3f top10=%.3f top100=%.3f | median rank=%.0f" %
      (clean["top1"], clean["top5"], clean["top10"], clean["top100"], clean["median"]))
lams = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]
rows = []
print(f"\n{'lam':>5} {'top1':>6} {'top5':>6} {'top10':>6} {'top100':>7} {'medianRank':>11}")
for lam in lams:
    s = stats(logits(superpose_cross_seq(A, mA, Bx, mB, fixed=lam)), nA)
    rows.append((lam, s["top1"], s["top5"], s["top10"], s["top100"], s["median"]))
    print(f"{lam:>5.2f} {s['top1']:>6.3f} {s['top5']:>6.3f} {s['top10']:>6.3f} {s['top100']:>7.3f} {s['median']:>11.0f}")

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
r = np.array(rows); lam = r[:, 0]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5.3))
for i, (k, lab, c) in enumerate([(1, "top-1 (argmax)", "tab:red"), (2, "top-5", "tab:orange"), (3, "top-10", "tab:green"), (4, "top-100", "tab:blue")]):
    a1.plot(lam, r[:, k], "o-", color=c, lw=2, label=lab)
    a1.axhline(clean[["top1", "top5", "top10", "top100"][i]], color=c, ls=":", lw=1.3, alpha=.6)
a1.set_xlabel("λ (0.5=even blend → 1.0=clean)"); a1.set_ylabel("fraction of positions")
a1.set_title("Is the TRUE next token in the top-k of the superposed output?\n(dotted = clean)")
a1.grid(alpha=.3); a1.legend(fontsize=8); a1.set_ylim(0, 1)
a2.plot(lam, r[:, 5], "o-", color="tab:purple", lw=2, label="superposed median rank")
a2.axhline(clean["median"], color="tab:purple", ls=":", lw=1.3, alpha=.7, label=f"clean median={clean['median']:.0f}")
a2.axhline(24576, color="k", ls="--", lw=1, alpha=.5, label="random chance (~24576)")
a2.set_xlabel("λ (0.5=even blend → 1.0=clean)"); a2.set_ylabel("median rank of true token (lower=better)")
a2.set_title("Median rank of the true next token"); a2.set_yscale("log"); a2.grid(alpha=.3, which="both"); a2.legend(fontsize=8)
fig.suptitle("RANK (not probability) of the true next token under superposition — SmolLM2-1.7B", fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.93]); plt.savefig("/mnt/pvc/t-jackcai/toys/ood_rank.png", dpi=120)
print("B64:" + base64.b64encode(open("/mnt/pvc/t-jackcai/toys/ood_rank.png", "rb").read()).decode())
