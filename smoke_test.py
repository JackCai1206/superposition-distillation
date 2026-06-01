"""CPU smoke test: validates the superposition + KD core end-to-end with tiny
random models (no downloads, no GPU). Checks all three input methods train a
step, losses are finite, and FLOP accounting reflects the packing/shortening.
"""

from __future__ import annotations

import torch

from flops import FlopCounter, model_flops_from_config
from kd_loss import kd_loss, wsd_alpha
from model import superposed_logits, tiny_model
from superpose import (superpose_cross_seq, superpose_none, superpose_token_merge)

torch.manual_seed(0)
V, L, B = 256, 32, 4
device = "cpu"

# Same vocab for teacher & student (white-box KD requirement); different hidden.
teacher = tiny_model(V, hidden=96, layers=3, device=device)
student = tiny_model(V, hidden=48, layers=2, device=device)
for p in teacher.parameters():
    p.requires_grad_(False)
teacher.eval()

opt = torch.optim.AdamW(student.parameters(), lr=1e-3)
fc = FlopCounter(model_flops_from_config(student.config))


def fake_batch():
    return torch.randint(0, V, (B, L)), torch.ones(B, L, dtype=torch.long)


def run_method(name, build_sup, effective_seqs, steps=5):
    print(f"\n--- method: {name} ---")
    last = None
    for step in range(steps):
        sup = build_sup()
        with torch.no_grad():
            t_logits = superposed_logits(teacher, sup)
        s_logits = superposed_logits(student, sup)
        assert s_logits.shape[1] == t_logits.shape[1], "T mismatch teacher/student"
        a = wsd_alpha(step, steps, alpha_max=0.9)
        # superposed inputs -> pure KD (labels=None)
        loss, parts = kd_loss(s_logits, t_logits, temperature=2.0, alpha=a,
                              labels=None, mask=sup.mask)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        T = sup.ids.shape[1]
        fc.add_step(seq_len=T, batch=B, effective_sequences=effective_seqs * B)
        assert torch.isfinite(loss), f"non-finite loss in {name}"
        last = loss.item()
        print(f"  step {step}  T={T:>2}  alpha={a:.2f}  loss={loss.item():.4f}  kd={parts['kd']:.4f}")
    return last


# Baseline: one sequence per pass -> 1 effective sequence/example.
run_method("none (baseline)", lambda: superpose_none(*fake_batch()), effective_seqs=1.0)

# Method 1: two sequences packed -> 2 effective sequences/example, same T.
def cross():
    a_ids, a_m = fake_batch(); b_ids, b_m = fake_batch()
    return superpose_cross_seq(a_ids, a_m, b_ids, b_m, mix_alpha=1.0)
run_method("cross_seq (pack 2)", cross, effective_seqs=2.0)

# Method 2: merge k=2 adjacent tokens -> shorter T (L//2), 1 sequence/example.
run_method("token_merge k=2", lambda: superpose_token_merge(*fake_batch(), k=2),
           effective_seqs=1.0)

print("\n=== FLOP accounting ===")
for k, v in fc.summary().items():
    print(f"  {k}: {v:,.3e}" if isinstance(v, float) else f"  {k}: {v}")
print("\nSMOKE TEST PASSED")
