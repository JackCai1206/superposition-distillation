"""Two reference targets for the superposed teacher output (lambda only in the targets):

  ref1 = lam*onehot(true_next_A) + (1-lam)*onehot(true_next_B)   GROUND TRUTH
  ref2 = lam*teacher(A) + (1-lam)*teacher(B) = Pmix              TEACHER-DIST BLEND

D1 = KL(ref1 || super)   -> is super a lincomb of the one-hot true next tokens?
D2 = KL(ref2 || super)   -> is super the lincomb of the teacher's clean distributions?
D1_floor = KL(ref1 || Pmix) -> best a perfectly-LINEAR teacher (super=Pmix) does on D1.
(No extra per-sequence lambda weighting; lambda is inside each mixture target.)
"""
from __future__ import annotations
import os, torch, base64, math
import torch.nn.functional as F
from model import load_model, superposed_logits
from superpose import superpose_none, superpose_cross_seq
from nl_data import load_split, get_batch

dev = "cuda"; T = 1.0; B, L = 16, 256
teacher = load_model(os.environ.get("TEACHER", "HuggingFaceTB/SmolLM2-1.7B"), dtype=torch.bfloat16, device=dev, frozen=True)
train = load_split("train"); g = torch.Generator().manual_seed(0)
A, mA = get_batch(train, B, L, dev, g); Bx, mB = get_batch(train, B, L, dev, g)
nA, nB = A[:, 1:], Bx[:, 1:]; m = (mA.bool() & mB.bool()); mm = m[:, :-1]
def a_all(x): return x[m].mean().item()
def a_nx(x): return x[mm].mean().item()
def KLfull(p, q): return (p * (p.clamp_min(1e-9).log() - q.clamp_min(1e-9).log())).sum(-1)
def gat(P, idx): return P[:, :-1].gather(-1, idx.unsqueeze(-1)).squeeze(-1)  # [B,L-1]
@torch.no_grad()
def probs(sup): return F.softmax(superposed_logits(teacher, sup).float() / T, -1)

PA = probs(superpose_none(A, mA)); PB = probs(superpose_none(Bx, mB))
cleanP = a_nx(gat(PA, nA))
lams = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]
rows = []
print(f"clean P(true_next)={cleanP:.4f}")
print(f"\n{'lam':>5} | {'D1 super':>9} {'D1 floor':>9} (gap) | {'D2 super':>9} | super mass on true: {'A':>6} {'B':>6}")
def D_onehot(P, lam):  # KL(lam*onehot(nA)+(1-lam)*onehot(nB) || P), per-position
    pa = gat(P, nA).clamp_min(1e-9); pb = gat(P, nB).clamp_min(1e-9)
    d = lam * (math.log(max(lam, 1e-9)) - pa.log()) + (1 - lam) * (math.log(max(1 - lam, 1e-9)) - pb.log())
    return a_nx(d)
for lam in lams:
    PS = probs(superpose_cross_seq(A, mA, Bx, mB, fixed=lam))
    Pmix = lam * PA + (1 - lam) * PB
    d1 = D_onehot(PS, lam); d1_floor = D_onehot(Pmix, lam)
    d2 = a_all(KLfull(Pmix, PS))
    msA, msB = a_nx(gat(PS, nA)), a_nx(gat(PS, nB))
    rows.append((lam, d1, d1_floor, d2, msA, msB))
    print(f"{lam:>5.2f} | {d1:>9.3f} {d1_floor:>9.3f} ({d1-d1_floor:>+5.2f}) | {d2:>9.3f} | {'':>17} {msA:>6.3f} {msB:>6.3f}")

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
r = np.array(rows); lam = r[:, 0]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.4))
ax1.plot(lam, r[:, 1], "o-", color="tab:blue", label="D1: KL(one-hot truth ‖ super)")
ax1.plot(lam, r[:, 2], "^--", color="tab:green", label="D1 floor: KL(one-hot truth ‖ Pmix) [linear teacher]")
ax1.set_title("(1) vs ONE-HOT ground truth  λ·δ(next_A)+(1−λ)·δ(next_B)"); ax1.set_ylabel("KL (nats)")
ax2.plot(lam, r[:, 3], "o-", color="tab:red", label="D2: KL(teacher blend Pmix ‖ super)")
ax2.axhline(0, color="k", ls=":", alpha=.5, label="0 = super is exactly the blend")
ax2.set_title("(2) vs TEACHER-DIST blend  λ·teacher(A)+(1−λ)·teacher(B)"); ax2.set_ylabel("KL (nats)")
for ax in (ax1, ax2): ax.set_xlabel("λ (0.5=even → 1.0=clean A)"); ax.grid(alpha=.3); ax.legend(fontsize=8)
fig.suptitle("Two reference targets for the superposed teacher output (SmolLM2-1.7B)", fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.94]); plt.savefig("/mnt/pvc/t-jackcai/toys/ood_tworef.png", dpi=115)
print("B64:" + base64.b64encode(open("/mnt/pvc/t-jackcai/toys/ood_tworef.png", "rb").read()).decode())
