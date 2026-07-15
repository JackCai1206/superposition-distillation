"""Is the teacher's NTP distribution on SUPERPOSED inputs OOD / uninformative?

Test (cross_seq, lambda=0.5): compare teacher(0.5A+0.5B) against:
  - the linear blend 0.5*teacher(A)+0.5*teacher(B)  (what an "informative" superposed
    output would be -> predicting BOTH continuations, TST-like)
  - the true next tokens of A and B (does it retain per-sequence info?)
High KL(super || blend) ~ KL(A||B) and near-zero P(true next) => OOD & uninformative.
"""
from __future__ import annotations
import os, torch
import torch.nn.functional as F
from model import load_model, superposed_logits
from superpose import superpose_none, superpose_cross_seq, superpose_token_merge
from nl_data import load_split, get_batch

dev = "cuda"
T = float(os.environ.get("KD_T", "1.0"))   # FineWeb KD used T=1
B, L = 16, 256
teacher = load_model(os.environ.get("TEACHER", "HuggingFaceTB/SmolLM2-1.7B"),
                     dtype=torch.bfloat16, device=dev, frozen=True)
train = load_split("train")
g = torch.Generator().manual_seed(0)
A, mA = get_batch(train, B, L, dev, g)
Bx, mB = get_batch(train, B, L, dev, g)

@torch.no_grad()
def probs(sup):
    lg = superposed_logits(teacher, sup).float()
    return F.softmax(lg / T, dim=-1)

PA = probs(superpose_none(A, mA))
PB = probs(superpose_none(Bx, mB))
PS = probs(superpose_cross_seq(A, mA, Bx, mB, fixed=0.5))
Pmix = 0.5 * PA + 0.5 * PB

def Hn(p): return -(p * p.clamp_min(1e-9).log()).sum(-1)
def KL(p, q): return (p * (p.clamp_min(1e-9).log() - q.clamp_min(1e-9).log())).sum(-1)
m = (mA.bool() & mB.bool())
def a(x): return x[m].mean().item()

print("=== cross_seq lambda=0.5, teacher=SmolLM2-1.7B, T=%g ===" % T)
print("ENTROPY (nats, higher=more confused):")
print("  clean_A=%.3f  clean_B=%.3f  SUPERPOSED=%.3f   (uniform-vocab max=%.2f)"
      % (a(Hn(PA)), a(Hn(PB)), a(Hn(PS)), torch.log(torch.tensor(PA.shape[-1]).float())))
print("CONFIDENCE (max prob):  clean_A=%.3f  SUPERPOSED=%.3f" % (a(PA.max(-1).values), a(PS.max(-1).values)))
print("\nIS THE SUPERPOSED OUTPUT THE BLEND OF THE CLEAN OUTPUTS?")
print("  KL(super || 0.5A+0.5B) = %.3f   <- small => informative (teacher ~linear)" % a(KL(PS, Pmix)))
print("  reference KL(cleanA || cleanB) = %.3f   <- 'unrelated' scale" % a(KL(PA, PB)))
print("  reference KL(cleanA || 0.5A+0.5B) = %.3f   <- 'self vs blend' scale" % a(KL(PA, Pmix)))
print("  ratio KL(super||blend) / KL(A||B) = %.2f   (~0 linear/informative, ~1 OOD/unrelated)"
      % (a(KL(PS, Pmix)) / max(a(KL(PA, PB)), 1e-6)))

print("\nDOES THE SUPERPOSED OUTPUT RETAIN EACH SEQUENCE'S TRUE NEXT TOKEN?")
nA, nB = A[:, 1:], Bx[:, 1:]; mm = m[:, :-1]
def pget(P, idx): return P[:, :-1].gather(-1, idx.unsqueeze(-1)).squeeze(-1)[mm].mean().item()
print("  P(true next_A):  clean=%.4f   superposed=%.4f" % (pget(PA, nA), pget(PS, nA)))
print("  P(true next_B):  clean=%.4f   superposed=%.4f" % (pget(PB, nB), pget(PS, nB)))
aS, aA, aB = PS.argmax(-1), PA.argmax(-1), PB.argmax(-1)
print("  argmax(super)==argmax(cleanA): %.3f  ==argmax(cleanB): %.3f  ==either: %.3f"
      % ((aS == aA)[m].float().mean(), (aS == aB)[m].float().mean(), ((aS == aA) | (aS == aB))[m].float().mean()))

# token_merge (k=2) bonus: entropy + does it predict the true next-bag tokens?
PStm = probs(superpose_token_merge(A, mA, k=2, fixed=0.5))   # [B, L/2, V]
print("\n=== token_merge k=2 (bonus) ===")
print("  ENTROPY: clean_A=%.3f  merged=%.3f   CONFIDENCE merged=%.3f"
      % (a(Hn(PA)), PStm.mul(-PStm.clamp_min(1e-9).log()).sum(-1).mean().item(), PStm.max(-1).values.mean().item()))
# merged position j (tokens 2j,2j+1) should predict next bag tokens 2j+2, 2j+3
Lh = PStm.shape[1]
nb1 = A[:, 2::2][:, :Lh]; nb2 = A[:, 3::2][:, :Lh]
def pg2(P, idx):
    k = min(P.shape[1], idx.shape[1]); return P[:, :k].gather(-1, idx[:, :k].unsqueeze(-1)).squeeze(-1).mean().item()
print("  P(next-bag tok1)=%.4f  P(next-bag tok2)=%.4f   (clean P(true next)=%.4f)"
      % (pg2(PStm, nb1), pg2(PStm, nb2), pget(PA, nA)))
