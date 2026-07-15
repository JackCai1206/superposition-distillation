"""Lambda-WEIGHTED superposed-NTP metrics: scale each sequence's error/KL/capture
by its lambda-share, so at uneven lambda the minority sequence's mismatch is
automatically down-weighted (judging the teacher fairly for what's actually present).

For weight lam on A and (1-lam) on B:
  capture_weighted = (lam*P(next_A) + (1-lam)*P(next_B)) / clean_P        (ideal=1)
  KL_weighted      = lam*KL(super||clean_A) + (1-lam)*KL(super||clean_B)
  KL_weighted_ideal= lam*KL(Pmix||A) + (1-lam)*KL(Pmix||B)  (floor a linear teacher hits)
Compare to the unweighted (symmetric 0.5/0.5) versions.
"""
from __future__ import annotations
import os, torch, base64
import torch.nn.functional as F
from model import load_model
from superpose import superpose_none, superpose_cross_seq
from model import superposed_logits
from nl_data import load_split, get_batch

dev = "cuda"; T = 1.0; B, L = 16, 256
teacher = load_model(os.environ.get("TEACHER", "HuggingFaceTB/SmolLM2-1.7B"), dtype=torch.bfloat16, device=dev, frozen=True)
train = load_split("train"); g = torch.Generator().manual_seed(0)
A, mA = get_batch(train, B, L, dev, g); Bx, mB = get_batch(train, B, L, dev, g)
nA, nB = A[:, 1:], Bx[:, 1:]; m = (mA.bool() & mB.bool()); mm = m[:, :-1]
def a(x): return x[m].mean().item()
def KL(p, q): return (p * (p.clamp_min(1e-9).log() - q.clamp_min(1e-9).log())).sum(-1)
def pget(P, idx): return P[:, :-1].gather(-1, idx.unsqueeze(-1)).squeeze(-1)[mm].mean().item()
@torch.no_grad()
def probs(sup): return F.softmax(superposed_logits(teacher, sup).float() / T, -1)

PA = probs(superpose_none(A, mA)); PB = probs(superpose_none(Bx, mB))
cleanP = pget(PA, nA)
lams = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]
rows = []
print(f"clean P(true_next)={cleanP:.4f}")
print(f"\n{'lam':>5} | {'cap_unw':>8} {'cap_wtd':>8} | {'KL_unw':>7} {'KL_wtd':>7} {'KL_wtd_ideal':>12} {'excess':>7}")
for lam in lams:
    PS = probs(superpose_cross_seq(A, mA, Bx, mB, fixed=lam))
    Pmix = lam * PA + (1 - lam) * PB
    pa, pb = pget(PS, nA), pget(PS, nB)
    cap_u = (0.5 * pa + 0.5 * pb) / cleanP
    cap_w = (lam * pa + (1 - lam) * pb) / cleanP
    klA, klB = a(KL(PS, PA)), a(KL(PS, PB))
    kl_u = 0.5 * klA + 0.5 * klB
    kl_w = lam * klA + (1 - lam) * klB
    kl_w_ideal = lam * a(KL(Pmix, PA)) + (1 - lam) * a(KL(Pmix, PB))
    rows.append((lam, cap_u, cap_w, kl_u, kl_w, kl_w_ideal, kl_w - kl_w_ideal))
    print(f"{lam:>5.2f} | {cap_u:>8.3f} {cap_w:>8.3f} | {kl_u:>7.3f} {kl_w:>7.3f} {kl_w_ideal:>12.3f} {kl_w-kl_w_ideal:>7.3f}")

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
r = np.array(rows); lam = r[:, 0]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.4))
ax1.plot(lam, r[:, 2], "o-", color="tab:blue", label="λ-weighted (minority down-weighted)")
ax1.plot(lam, r[:, 1], "s--", color="tab:gray", label="unweighted (0.5/0.5)")
ax1.axhline(1.0, color="k", ls=":", alpha=.6, label="ideal (=1)")
ax1.set_title("Capture fraction: (λ·P(next_A)+(1−λ)·P(next_B)) / clean"); ax1.set_ylabel("capture fraction")
ax2.plot(lam, r[:, 4], "o-", color="tab:red", label="λ-weighted KL(super‖clean)")
ax2.plot(lam, r[:, 5], "^--", color="tab:green", label="ideal-blend floor (Pmix)")
ax2.plot(lam, r[:, 3], "s:", color="tab:gray", label="unweighted KL")
ax2.set_title("λ-weighted KL = λ·KL(super‖A)+(1−λ)·KL(super‖B)"); ax2.set_ylabel("KL (nats)")
for ax in (ax1, ax2): ax.set_xlabel("λ (0.5=even → 1.0=clean A)"); ax.grid(alpha=.3); ax.legend(fontsize=8)
fig.suptitle("λ-WEIGHTED metrics (minority sequence's error scaled by its share) — SmolLM2-1.7B", fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.94]); plt.savefig("/mnt/pvc/t-jackcai/toys/ood_weighted.png", dpi=115)
print("B64:" + base64.b64encode(open("/mnt/pvc/t-jackcai/toys/ood_weighted.png", "rb").read()).decode())
