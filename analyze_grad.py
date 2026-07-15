"""Gradient alignment: does the superposition gradient = the linear combination of the
two clean-sequence gradients? (the "2 gradients in 1 pass" claim)

g_A   = grad of KL(teacher(A)   || student(A))     [normal clean-A training gradient]
g_B   = grad of KL(teacher(B)   || student(B))
g_sup = grad of KL(target       || student(superpose(A,B,lam)))
lincomb = lam*g_A + (1-lam)*g_B
Report cos(g_sup, lincomb) for two targets:
  (a) ACTUAL  target = teacher(superpose)        -> full effect (OOD target + nonlinearity)
  (b) IDEAL   target = lam*teacher(A)+(1-lam)*teacher(B) (=Pmix) -> isolates model nonlinearity
Baseline cos(g_A,g_B) = alignment of two UNRELATED gradients (chance level).
Full-parameter gradients of a real student (SmolLM2-135M), teacher SmolLM2-1.7B.
"""
from __future__ import annotations
import os, torch, base64
import torch.nn.functional as F
from model import load_model, superposed_logits
from superpose import superpose_none, superpose_cross_seq
from nl_data import load_split, get_batch

dev = "cuda"; T = 1.0; B, L = 8, 256
teacher = load_model("HuggingFaceTB/SmolLM2-1.7B", dtype=torch.bfloat16, device=dev, frozen=True)
student = load_model("HuggingFaceTB/SmolLM2-135M", dtype=torch.bfloat16, device=dev, frozen=False)  # bf16 for flash-attn; grads cast to fp32 below
for p in student.parameters(): p.requires_grad_(True)
train = load_split("train"); g = torch.Generator().manual_seed(0)
A, mA = get_batch(train, B, L, dev, g); Bx, mB = get_batch(train, B, L, dev, g)
m = (mA.bool() & mB.bool())

def tprobs(sup):
    with torch.no_grad(): return F.softmax(superposed_logits(teacher, sup).float() / T, -1)
def kl_loss(sup, target_p, mask):
    sl = superposed_logits(student, sup).float()
    logq = F.log_softmax(sl / T, -1)
    kl = (target_p * (target_p.clamp_min(1e-9).log() - logq)).sum(-1)
    return kl[mask].mean()
def flatgrad(loss):
    student.zero_grad(set_to_none=True); loss.backward()
    return torch.cat([p.grad.detach().float().flatten() for p in student.parameters() if p.grad is not None])  # fp32 for stable cosine
def cos(a, b): return F.cosine_similarity(a, b, dim=0).item()

supA = superpose_none(A, mA); supB = superpose_none(Bx, mB)
PA = tprobs(supA); PB = tprobs(supB)
gA = flatgrad(kl_loss(supA, PA, mA.bool()))
gB = flatgrad(kl_loss(supB, PB, mB.bool()))
base = cos(gA, gB)
print(f"#params(student)={gA.numel()/1e6:.1f}M | cos(g_A,g_B) UNRELATED baseline = {base:.4f}")
lams = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
rows = []
print(f"\n{'lam':>5} {'cos(actual,lincomb)':>20} {'cos(ideal,lincomb)':>19} {'||g_sup||/||lincomb||':>21}")
for lam in lams:
    sup = superpose_cross_seq(A, mA, Bx, mB, fixed=lam)
    g_act = flatgrad(kl_loss(sup, tprobs(sup), m))
    g_idl = flatgrad(kl_loss(sup, lam * PA + (1 - lam) * PB, m))
    lincomb = lam * gA + (1 - lam) * gB
    ca, ci = cos(g_act, lincomb), cos(g_idl, lincomb)
    mag = (g_act.norm() / lincomb.norm()).item()
    rows.append((lam, ca, ci, mag))
    print(f"{lam:>5.2f} {ca:>20.4f} {ci:>19.4f} {mag:>21.3f}")

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
r = np.array(rows); lam = r[:, 0]
fig, ax = plt.subplots(figsize=(8.5, 5.6))
ax.plot(lam, r[:, 2], "s-", color="tab:green", lw=2, label="cos(g_sup, lincomb) — IDEAL target (Pmix)")
ax.plot(lam, r[:, 1], "o-", color="tab:red", lw=2, label="cos(g_sup, lincomb) — ACTUAL teacher target")
ax.axhline(base, color="gray", ls="--", lw=1.3, label=f"cos(g_A,g_B) unrelated baseline = {base:.3f}")
ax.axhline(1.0, color="k", ls=":", lw=1, alpha=.5, label="1.0 = perfect '2 gradients in 1 pass'")
ax.set_xlabel("λ (0.5 = even blend → 1.0 = clean A)"); ax.set_ylabel("gradient cosine similarity")
ax.set_title("Does the superposition gradient = λ·g_A + (1−λ)·g_B?\n(SmolLM2-135M student, 1.7B teacher)")
ax.grid(alpha=.3); ax.legend(fontsize=9); ax.set_ylim(min(-0.05, base - 0.05), 1.02)
plt.tight_layout(); plt.savefig("/mnt/pvc/t-jackcai/toys/ood_grad.png", dpi=120)
print("B64:" + base64.b64encode(open("/mnt/pvc/t-jackcai/toys/ood_grad.png", "rb").read()).decode())
