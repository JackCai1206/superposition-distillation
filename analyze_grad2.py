"""Dimension-robust version of the gradient-alignment test (addresses 'high-D => everything
orthogonal'). Beyond cosine, compute:
  R^2 = fraction of g_sup that lies in span{g_A, g_B}  (least-squares projection; cosine-free)
  recovered coefficients (a,b) in g_sup ~ a*g_A + b*g_B  (ideal: a=lam, b=1-lam)
  calibration cosines: cos(g_A,lincomb) and cos(g_B,lincomb) = "if g_sup captured ONLY one seq"
A useful '2-in-1' gradient would have R^2 ~ 1 and cos near the 'captured-one-seq' line (~0.75),
NOT near the random-orthogonal 0.  Baseline cos(g_A,g_B) shows real grads are far from random.
"""
from __future__ import annotations
import os, torch, base64
import numpy as np
import torch.nn.functional as F
from model import load_model, superposed_logits
from superpose import superpose_none, superpose_cross_seq
from nl_data import load_split, get_batch

dev = "cuda"; T = 1.0; B, L = 8, 256
teacher = load_model("HuggingFaceTB/SmolLM2-1.7B", dtype=torch.bfloat16, device=dev, frozen=True)
student = load_model("HuggingFaceTB/SmolLM2-135M", dtype=torch.bfloat16, device=dev, frozen=False)
for p in student.parameters(): p.requires_grad_(True)
train = load_split("train"); g = torch.Generator().manual_seed(0)
A, mA = get_batch(train, B, L, dev, g); Bx, mB = get_batch(train, B, L, dev, g)
m = (mA.bool() & mB.bool())

def tprobs(sup):
    with torch.no_grad(): return F.softmax(superposed_logits(teacher, sup).float() / T, -1)
def kl_loss(sup, tp, mask):
    logq = F.log_softmax(superposed_logits(student, sup).float() / T, -1)
    return (tp * (tp.clamp_min(1e-9).log() - logq)).sum(-1)[mask].mean()
def flatgrad(loss):
    student.zero_grad(set_to_none=True); loss.backward()
    return torch.cat([p.grad.detach().float().flatten() for p in student.parameters() if p.grad is not None])

supA = superpose_none(A, mA); supB = superpose_none(Bx, mB); PA = tprobs(supA); PB = tprobs(supB)
gA = flatgrad(kl_loss(supA, PA, mA.bool())); gB = flatgrad(kl_loss(supB, PB, mB.bool()))
AA = (gA@gA).item(); BB = (gB@gB).item(); AB = (gA@gB).item()
base = AB / (AA**.5 * BB**.5)
print(f"#p={gA.numel()/1e6:.1f}M | cos(gA,gB)={base:.4f} (random≈{1/gA.numel()**.5:.2e})")

def analyze(gsup, lam):
    SA = (gsup@gA).item(); SB = (gsup@gB).item(); SS = (gsup@gsup).item()
    lin_dot = lam*SA + (1-lam)*SB
    lin_n2 = lam*lam*AA + (1-lam)**2*BB + 2*lam*(1-lam)*AB
    cos_lin = lin_dot/(SS**.5 * lin_n2**.5)
    Gm = np.array([[AA, AB], [AB, BB]]); bb = np.array([SA, SB])
    coef = np.linalg.solve(Gm, bb); R2 = float(coef @ Gm @ coef) / SS   # ||proj||^2/||g_sup||^2
    cosA = (lam*AA + (1-lam)*AB)/(AA**.5 * lin_n2**.5)                   # cos(gA, lincomb)
    return cos_lin, R2, coef[0], coef[1], cosA

lams = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
rows = []
print(f"\n{'lam':>5} | {'cosACT':>7} {'R2_ACT':>7} {'a':>6} {'b':>6} | {'cosIDL':>7} {'R2_IDL':>7} | {'cos(gA,lin)ref':>14}")
for lam in lams:
    sup = superpose_cross_seq(A, mA, Bx, mB, fixed=lam)
    g_act = flatgrad(kl_loss(sup, tprobs(sup), m))
    g_idl = flatgrad(kl_loss(sup, lam*PA + (1-lam)*PB, m))
    ca, R2a, a, b, cosref = analyze(g_act, lam)
    ci, R2i, _, _, _ = analyze(g_idl, lam)
    rows.append((lam, ca, R2a, a, b, ci, R2i, cosref))
    print(f"{lam:>5.2f} | {ca:>7.3f} {R2a:>7.3f} {a:>6.2f} {b:>6.2f} | {ci:>7.3f} {R2i:>7.3f} | {cosref:>14.3f}")

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
r = np.array(rows); lam = r[:, 0]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.5))
a1.plot(lam, r[:, 1], "o-", color="tab:red", lw=2, label="cos(g_sup, lincomb) — ACTUAL")
a1.plot(lam, r[:, 5], "s-", color="tab:green", lw=2, label="cos(g_sup, lincomb) — IDEAL")
a1.plot(lam, r[:, 7], "--", color="tab:blue", lw=1.6, label="cos(g_A, lincomb): 'captured ONE seq'")
a1.axhline(base, color="gray", ls="--", lw=1.2, label=f"cos(g_A,g_B) unrelated = {base:.3f}")
a1.axhline(1.0, color="k", ls=":", lw=1, alpha=.5, label="1.0 = perfect 2-in-1")
a1.set_title("Cosine, calibrated"); a1.set_ylabel("cosine"); a1.legend(fontsize=8); a1.set_ylim(-0.1, 1.03)
a2.plot(lam, r[:, 2], "o-", color="tab:red", lw=2, label="ACTUAL target")
a2.plot(lam, r[:, 6], "s-", color="tab:green", lw=2, label="IDEAL target")
a2.axhline(1.0, color="k", ls=":", lw=1, alpha=.5, label="1.0 = g_sup fully in span{g_A,g_B}")
a2.set_title("R²: fraction of g_sup explained by span{g_A, g_B}  (cosine-free)")
a2.set_ylabel("explained variance R²"); a2.legend(fontsize=8); a2.set_ylim(-0.02, 1.02)
for ax in (a1, a2): ax.set_xlabel("λ (0.5=even → 1.0=clean A)"); ax.grid(alpha=.3)
fig.suptitle("Is the superposition gradient in the span of the two clean gradients? (dimension-robust)", fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.94]); plt.savefig("/mnt/pvc/t-jackcai/toys/ood_grad2.png", dpi=120)
print("B64:" + base64.b64encode(open("/mnt/pvc/t-jackcai/toys/ood_grad2.png", "rb").read()).decode())
