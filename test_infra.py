"""Unit tests for the critical infrastructure: superposition math, the iso-FLOP
batch fix, the superposed forward, the MCE/TST loss, and the chunked KD/CE losses.
Runs on CPU with tiny models (fast). Each check prints PASS/FAIL; exits non-zero on any FAIL.
"""
import sys, torch, torch.nn.functional as F
torch.manual_seed(0)
from superpose import build_inputs_embeds, superpose_none, superpose_cross_seq, superpose_token_merge
from model import tiny_model, superposed_hidden
from kd_loss import chunked_ce_loss, chunked_distill_loss

P = []; Fl = []
def check(name, cond, detail=""):
    ok = bool(cond); (P if ok else Fl).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))

V, H = 64, 32
dev = "cpu"

print("== T1: build_inputs_embeds is the convex combination λ·E(A)+(1-λ)·E(B) ==")
emb = torch.nn.Embedding(V, H)
B, T = 3, 5
ids = torch.randint(0, V, (B, T, 2)); lam = 0.7
w = torch.stack([torch.full((B, T), lam), torch.full((B, T), 1 - lam)], -1)
got = build_inputs_embeds(emb, ids, w)
exp = lam * emb(ids[..., 0]) + (1 - lam) * emb(ids[..., 1])
check("blended embedding == λ·E(A)+(1-λ)·E(B)", torch.allclose(got, exp, atol=1e-6),
      f"maxdiff={(got-exp).abs().max():.1e}")

print("== T2: superpose_cross_seq carries both sequences + correct weights/mask ==")
A = torch.randint(0, V, (B, T)); Bx = torch.randint(0, V, (B, T))
o = torch.ones(B, T, dtype=torch.long)
sup = superpose_cross_seq(A, o, Bx, o, fixed=0.7)
check("ids[...,0]==A and ids[...,1]==B", torch.equal(sup.ids[..., 0], A) and torch.equal(sup.ids[..., 1], Bx))
check("weights == [0.7, 0.3] where both real",
      torch.allclose(sup.weights[..., 0], torch.tensor(0.7)) and torch.allclose(sup.weights[..., 1], torch.tensor(0.3)))
check("mask == (maskA OR maskB)", torch.equal(sup.mask, ((o + o) > 0).long()))
mB = o.clone(); mB[:, 3:] = 0          # B padded after pos 2
sup2 = superpose_cross_seq(A, o, Bx, mB, fixed=0.5)
check("tail-keep: weight renorms to [1,0] where B is padding",
      torch.allclose(sup2.weights[:, 3:, 0], torch.tensor(1.0)) and torch.allclose(sup2.weights[:, 3:, 1], torch.tensor(0.0)))

print("== T3: superpose_token_merge bags k adjacent tokens, output length L/k ==")
ids2 = torch.randint(0, V, (B, 2 * T)); o2 = torch.ones(B, 2 * T, dtype=torch.long)
sm = superpose_token_merge(ids2, o2, k=2, fixed=0.6)
check("output length == L/k", sm.ids.shape[1] == T, f"{sm.ids.shape[1]} vs {T}")
check("ids[b,t] == [tok 2t, tok 2t+1]", torch.equal(sm.ids[..., 0], ids2[:, 0::2]) and torch.equal(sm.ids[..., 1], ids2[:, 1::2]))
check("weights == [0.6, 0.4]", torch.allclose(sm.weights[..., 0], torch.tensor(0.6)))

print("== T4: iso-FLOP batch fix — every method forwards B*L positions ==")
bs, L = 8, 16
none_pos = superpose_none(torch.randint(0, V, (bs, L)), torch.ones(bs, L, dtype=torch.long)).ids[..., 0].numel()
# cross_seq: micro samples 2*bs -> pairs to bs superposed sequences of length L
cs = superpose_cross_seq(torch.randint(0, V, (bs, L)), torch.ones(bs, L, dtype=torch.long),
                         torch.randint(0, V, (bs, L)), torch.ones(bs, L, dtype=torch.long), fixed=0.5)
cs_pos = cs.ids.shape[0] * cs.ids.shape[1]
# token_merge: micro samples k*bs -> merge k -> (k*bs, L/k)
tm = superpose_token_merge(torch.randint(0, V, (2 * bs, L)), torch.ones(2 * bs, L, dtype=torch.long), k=2, fixed=0.5)
tm_pos = tm.ids.shape[0] * tm.ids.shape[1]
check("none forward positions == bs*L", none_pos == bs * L, f"{none_pos}")
check("cross_seq(2*bs sampled) forward positions == bs*L", cs_pos == bs * L, f"{cs_pos}")
check("token_merge(k*bs sampled) forward positions == bs*L", tm_pos == bs * L, f"{tm_pos}")

print("== T5: superposed forward applies the weights (λ=1 -> clean A, λ=0 -> clean B) ==")
m = tiny_model(V, hidden=H, layers=2, heads=4, inter=4 * H, dtype=torch.float32,
               device=dev, tie_embeddings=True, max_pos=64)
m.eval()
A5 = torch.randint(0, V, (2, 8)); B5 = torch.randint(0, V, (2, 8)); on = torch.ones(2, 8, dtype=torch.long)
with torch.no_grad():
    hA = superposed_hidden(m, superpose_none(A5, on)); hB = superposed_hidden(m, superpose_none(B5, on))
    h1 = superposed_hidden(m, superpose_cross_seq(A5, on, B5, on, fixed=1.0))
    h0 = superposed_hidden(m, superpose_cross_seq(A5, on, B5, on, fixed=0.0))
    h5 = superposed_hidden(m, superpose_cross_seq(A5, on, B5, on, fixed=0.5))
check("cross_seq λ=1 forward == clean A", torch.allclose(h1, hA, atol=1e-4), f"maxdiff={(h1-hA).abs().max():.1e}")
check("cross_seq λ=0 forward == clean B", torch.allclose(h0, hB, atol=1e-4), f"maxdiff={(h0-hB).abs().max():.1e}")
check("cross_seq λ=0.5 forward differs from both (genuine blend)",
      not torch.allclose(h5, hA, atol=1e-3) and not torch.allclose(h5, hB, atol=1e-3))

print("== T6: MCE/TST loss == λ·CE(next_A) + (1-λ)·CE(next_B) (vs full-logit reference) ==")
s_head = m.get_output_embeddings().weight
sup_cs = superpose_cross_seq(A5, on, B5, on, fixed=0.7)
s_hidden = superposed_hidden(m, sup_cs)
Ai, Bi = sup_cs.ids[:, :, 0], sup_cs.ids[:, :, 1]
lA = Ai.clone(); lA[:, :-1] = Ai[:, 1:]; lA[:, -1] = -100
lB = Bi.clone(); lB[:, :-1] = Bi[:, 1:]; lB[:, -1] = -100
lossA, _ = chunked_ce_loss(s_hidden, s_head, lA); lossB, _ = chunked_ce_loss(s_hidden, s_head, lB)
mce_code = 0.7 * lossA + 0.3 * lossB
logits = s_hidden @ s_head.T
refA = F.cross_entropy(logits[:, :-1].reshape(-1, V), Ai[:, 1:].reshape(-1))
refB = F.cross_entropy(logits[:, :-1].reshape(-1, V), Bi[:, 1:].reshape(-1))
mce_ref = 0.7 * refA + 0.3 * refB
check("MCE matches full-logit reference", torch.allclose(mce_code, mce_ref, atol=1e-4),
      f"code={mce_code.item():.5f} ref={mce_ref.item():.5f}")

print("== T7: chunked losses equal their full-logit references ==")
# CE
lab = A5.clone(); lab[:, :-1] = A5[:, 1:]; lab[:, -1] = -100
ce_chunk, _ = chunked_ce_loss(hA, s_head, lab)
ce_ref = F.cross_entropy((hA @ s_head.T)[:, :-1].reshape(-1, V), A5[:, 1:].reshape(-1))
check("chunked_ce_loss == F.cross_entropy", torch.allclose(ce_chunk, ce_ref, atol=1e-4),
      f"{ce_chunk.item():.5f} vs {ce_ref.item():.5f}")
# forward-KL (teacher = a 2nd random model, same V); T=1
mt = tiny_model(V, hidden=H, layers=2, heads=4, inter=4 * H, dtype=torch.float32, device=dev, tie_embeddings=True, max_pos=64)
mt.eval(); t_head = mt.get_output_embeddings().weight
with torch.no_grad(): t_hidden = superposed_hidden(mt, superpose_none(A5, on))
kd_chunk, _ = chunked_distill_loss(hA, s_head, t_hidden, t_head, 1.0, mask=on)
sl = hA @ s_head.T; tl = t_hidden @ t_head.T
tp = F.softmax(tl, -1); fkl = (tp * (tp.clamp_min(1e-9).log() - F.log_softmax(sl, -1))).sum(-1).mean()
check("chunked_distill_loss == forward-KL KL(teacher‖student)", torch.allclose(kd_chunk, fkl, atol=1e-3),
      f"chunk={kd_chunk.item():.5f} ref={fkl.item():.5f}")

print(f"\n==== {len(P)} passed, {len(Fl)} failed ====")
if Fl:
    print("FAILED:", Fl); sys.exit(1)
print("ALL INFRASTRUCTURE TESTS PASSED")
