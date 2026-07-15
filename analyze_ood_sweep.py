"""Fine-grained lambda sweep of teacher superposed-NTP quality, across SmolLM2 sizes.

cross_seq: emb = lam*A + (1-lam)*B. lam=0.5 = max superposition, lam->1 = clean A.
For each teacher size and each lam, measure OOD-ness (entropy, KL to the informative
blend) and informativeness (prob on each sequence's TRUE next token).
All SmolLM2 sizes share vocab/tokenizer, so the token sequences are reused.
"""
from __future__ import annotations
import os, torch, base64
import torch.nn.functional as F
from model import load_model, superposed_logits
from superpose import superpose_none, superpose_cross_seq
from nl_data import load_split, get_batch

dev = "cuda"; T = 1.0; B, L = 16, 256
train = load_split("train"); g = torch.Generator().manual_seed(0)
A, mA = get_batch(train, B, L, dev, g); Bx, mB = get_batch(train, B, L, dev, g)
nA, nB = A[:, 1:], Bx[:, 1:]
m = (mA.bool() & mB.bool()); mm = m[:, :-1]
def a(x): return x[m].mean().item()
def Hn(p): return -(p * p.clamp_min(1e-9).log()).sum(-1)
def KL(p, q): return (p * (p.clamp_min(1e-9).log() - q.clamp_min(1e-9).log())).sum(-1)
def pget(P, idx): return P[:, :-1].gather(-1, idx.unsqueeze(-1)).squeeze(-1)[mm].mean().item()

TEACHERS = [("135M", "HuggingFaceTB/SmolLM2-135M"),
            ("360M", "HuggingFaceTB/SmolLM2-360M"),
            ("1.7B", "HuggingFaceTB/SmolLM2-1.7B")]
lams = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]
results = {}
for sz, name in TEACHERS:
    teacher = load_model(name, dtype=torch.bfloat16, device=dev, frozen=True)
    @torch.no_grad()
    def probs(sup): return F.softmax(superposed_logits(teacher, sup).float() / T, dim=-1)
    PA = probs(superpose_none(A, mA)); PB = probs(superpose_none(Bx, mB))
    cleanH, cleanP = a(Hn(PA)), pget(PA, nA); klAB = a(KL(PA, PB))
    rows = []
    for lam in lams:
        PS = probs(superpose_cross_seq(A, mA, Bx, mB, fixed=lam))
        Pmix = lam * PA + (1 - lam) * PB
        rows.append((lam, a(Hn(PS)), a(PS.max(-1).values), pget(PS, nA), pget(PS, nB),
                     a(KL(PS, Pmix)) / max(klAB, 1e-6)))
    results[sz] = dict(clean_H=cleanH, clean_P=cleanP, rows=rows)
    print(f"\n=== {sz} ({name}) clean: entropy={cleanH:.3f} P(true_next)={cleanP:.4f} ===")
    print(f"{'lam':>5} {'entropy':>8} {'conf':>6} {'P(nextA)':>9} {'P(nextB)':>9} {'KL/KL(A||B)':>11}")
    for r in rows:
        print(f"{r[0]:>5.2f} {r[1]:>8.3f} {r[2]:>6.3f} {r[3]:>9.4f} {r[4]:>9.4f} {r[5]:>11.3f}")
    del teacher; torch.cuda.empty_cache()

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
COL = {"135M": "tab:green", "360M": "tab:orange", "1.7B": "tab:purple"}
fig, axs = plt.subplots(1, 3, figsize=(17, 5.4))
for sz in results:
    r = np.array(results[sz]["rows"]); lam = r[:, 0]; c = COL[sz]
    axs[0].plot(lam, r[:, 1], "o-", color=c, label=f"{sz}")
    axs[0].axhline(results[sz]["clean_H"], color=c, ls=":", alpha=.5)
    axs[1].plot(lam, r[:, 3], "o-", color=c, label=f"{sz} P(next_A)")
    axs[1].axhline(results[sz]["clean_P"], color=c, ls=":", alpha=.5)
    axs[2].plot(lam, r[:, 5], "o-", color=c, label=f"{sz}")
axs[0].set_title("Teacher confusion: entropy(super) vs λ\n(dotted = clean entropy)"); axs[0].set_ylabel("entropy (nats)")
axs[1].set_title("Informativeness: P(true next_A | super) vs λ\n(dotted = clean P)"); axs[1].set_ylabel("P(true next token)")
axs[2].set_title("OOD-ness: KL(super‖blend) / KL(A‖B) vs λ\n(0=linear/informative, ~1=unrelated)"); axs[2].set_ylabel("KL ratio")
for ax in axs: ax.set_xlabel("λ (0.5=max superpose → 1.0=clean A)"); ax.grid(alpha=.3); ax.legend(fontsize=8)
fig.suptitle("Teacher superposed-NTP quality vs λ across SmolLM2 sizes (FineWeb, cross_seq)", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig("/mnt/pvc/t-jackcai/toys/ood_lambda_size_sweep.png", dpi=115)
print("B64:" + base64.b64encode(open("/mnt/pvc/t-jackcai/toys/ood_lambda_size_sweep.png", "rb").read()).decode())
