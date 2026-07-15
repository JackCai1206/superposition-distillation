"""Does norm-renormalization remove the superposed-input OOD-ness?

The convex blend e = lam*E(A) + (1-lam)*E(B) has ||e|| < a real token embedding's
norm (partial cancellation), and transformers are norm-sensitive -> that magnitude
deficit alone is OOD. Here we rescale e per-position to the "expected" norm
target = lam*||E(A)|| + (1-lam)*||E(B)||  (i.e. renormalize the lam weights by a
common factor c = target/||e||, keeping the lam:1-lam RATIO and sweeping lam).
Compare metrics for the original blend vs the renormalized blend.
"""
from __future__ import annotations
import os, torch, base64
import torch.nn.functional as F
from model import load_model
from nl_data import load_split, get_batch

dev = "cuda"; T = 1.0; B, L = 16, 256
teacher = load_model(os.environ.get("TEACHER", "HuggingFaceTB/SmolLM2-1.7B"),
                     dtype=torch.bfloat16, device=dev, frozen=True)
E = teacher.get_input_embeddings()
train = load_split("train"); g = torch.Generator().manual_seed(0)
A, mA = get_batch(train, B, L, dev, g); Bx, mB = get_batch(train, B, L, dev, g)
nA_idx, nB_idx = A[:, 1:], Bx[:, 1:]
m = (mA.bool() & mB.bool()); mm = m[:, :-1]
def a(x): return x[m].mean().item()
def Hn(p): return -(p * p.clamp_min(1e-9).log()).sum(-1)
def KL(p, q): return (p * (p.clamp_min(1e-9).log() - q.clamp_min(1e-9).log())).sum(-1)
def pget(P, idx): return P[:, :-1].gather(-1, idx.unsqueeze(-1)).squeeze(-1)[mm].mean().item()

@torch.no_grad()
def P_of(emb): return F.softmax(teacher(inputs_embeds=emb.to(torch.bfloat16)).logits.float() / T, -1)

eA = E(A).float(); eB = E(Bx).float()
normA = eA.norm(dim=-1); normB = eB.norm(dim=-1)
PA = P_of(eA); PB = P_of(eB)
klAB = a(KL(PA, PB)); cleanP = pget(PA, nA_idx); cleanH = a(Hn(PA))
print(f"clean: entropy={cleanH:.3f} P(true_next)={cleanP:.4f} | mean token-emb norm={normA.mean().item():.3f}")

lams = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]
rows = []
print(f"\n{'lam':>5} | {'||blend||/target':>15} || {'ORIGINAL blend':^34} || {'NORM-RENORMALIZED':^34}")
print(f"{'':>5} | {'(norm deficit)':>15} || {'H':>9} {'P(nextA)':>10} {'KL/KLab':>9} || {'H':>9} {'P(nextA)':>10} {'KL/KLab':>9}")
for lam in lams:
    blend = lam * eA + (1 - lam) * eB
    bn = blend.norm(dim=-1)
    target = lam * normA + (1 - lam) * normB
    blend_rn = blend * (target / bn.clamp_min(1e-6)).unsqueeze(-1)
    Pmix = lam * PA + (1 - lam) * PB
    PS = P_of(blend); PSr = P_of(blend_rn)
    o = (a(Hn(PS)), pget(PS, nA_idx), a(KL(PS, Pmix)) / max(klAB, 1e-6))
    r = (a(Hn(PSr)), pget(PSr, nA_idx), a(KL(PSr, Pmix)) / max(klAB, 1e-6))
    ndef = a(bn / target.clamp_min(1e-6))
    rows.append((lam, ndef, *o, *r))
    print(f"{lam:>5.2f} | {ndef:>15.3f} || {o[0]:>9.3f} {o[1]:>10.4f} {o[2]:>9.3f} || {r[0]:>9.3f} {r[1]:>10.4f} {r[2]:>9.3f}")

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
r = np.array(rows); lam = r[:, 0]
fig, axs = plt.subplots(1, 3, figsize=(17, 5.4))
def panel(ax, oi, ri, title, ylab, clean=None):
    ax.plot(lam, r[:, oi], "o-", color="tab:red", label="original blend")
    ax.plot(lam, r[:, ri], "s--", color="tab:blue", label="norm-renormalized")
    if clean is not None: ax.axhline(clean, color="k", ls=":", alpha=.6, label="clean")
    ax.set_title(title); ax.set_ylabel(ylab); ax.set_xlabel("λ (0.5=max superpose → 1.0=clean A)")
    ax.grid(alpha=.3); ax.legend(fontsize=8)
panel(axs[0], 2, 5, "entropy(super) vs λ", "entropy (nats)", cleanH)
panel(axs[1], 3, 6, "P(true next_A | super) vs λ", "P(true next token)", cleanP)
panel(axs[2], 4, 7, "OOD-ness KL(super‖blend)/KL(A‖B) vs λ", "KL ratio")
fig.suptitle("Original vs norm-RENORMALIZED superposed input — is the OOD-ness just a magnitude artifact? (SmolLM2-1.7B)", fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig("/mnt/pvc/t-jackcai/toys/ood_renorm.png", dpi=115)
print("B64:" + base64.b64encode(open("/mnt/pvc/t-jackcai/toys/ood_renorm.png", "rb").read()).decode())
