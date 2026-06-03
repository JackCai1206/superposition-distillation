"""Independent audit of the FLOP accounting for a real run.

Re-derives the total recorded FLOPs from scratch (measure per-step cost of each
stage's actual config, multiply by stage step counts) and checks it equals what
distill_lm logged. Also breaks the per-step cost into teacher / student-fwd /
backward+recompute, and confirms stage-1 is counted and cheaper.
"""

from __future__ import annotations

import glob
import json
import sys

import torch
from torch.utils.flop_counter import FlopCounterMode

from kd_loss import chunked_distill_loss
from model import load_model, superposed_hidden, tiny_model
from nl_data import VOCAB_SIZE
from superpose import superpose_cross_seq, superpose_none

DEV = "cuda"
TEACHER = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
LAM, B, L, T = 0.7, 32, 512, 2.0


def measure(fn):
    fcm = FlopCounterMode(display=False)
    with fcm:
        fn()
    return float(fcm.get_total_flops())


def main():
    teacher = load_model(TEACHER, dtype=torch.bfloat16, device=DEV, frozen=True)
    student = tiny_model(VOCAB_SIZE, hidden=320, layers=6, heads=8, inter=1280,
                         dtype=torch.bfloat16, device=DEV, tie_embeddings=True, max_pos=L)
    opt = torch.optim.AdamW(student.parameters(), lr=1e-4)
    s_head = student.get_output_embeddings().weight
    t_head = teacher.get_output_embeddings().weight
    g = torch.Generator().manual_seed(0)

    def make(kind):
        ids = torch.randint(0, VOCAB_SIZE, (B, L), device=DEV)
        m = torch.ones(B, L, dtype=torch.long, device=DEV)
        if kind == "normal":
            return superpose_none(ids, m), B
        # cross_seq stage-1: pairs -> B/2 examples
        h = B // 2
        return superpose_cross_seq(ids[:h], m[:h], ids[h:], m[h:], fixed=LAM), h

    def full_step(sup, labels=None):
        with torch.no_grad():
            th = superposed_hidden(teacher, sup)
        sh = superposed_hidden(student, sup)
        loss, _ = chunked_distill_loss(sh, s_head, th, t_head, T, mask=sup.mask, labels=labels)
        opt.zero_grad(); loss.backward()

    rates = {}
    print(f"teacher={TEACHER}  student=320/6 tied  vocab={VOCAB_SIZE}  B={B} L={L}\n")
    for kind, tag in [("cross_seq", "stage-1 (superposed, B/2)"), ("normal", "stage-2 (normal, B)")]:
        sup, outB = make(kind)
        # warm up (cudnn/autotune) then measure
        full_step(sup); torch.cuda.synchronize()
        rec = measure(lambda: full_step(sup))
        # teacher-only forward cost (no_grad)
        t_only = measure(lambda: superposed_hidden(teacher, sup))
        rates[kind] = rec
        print(f"{tag}: outB={outB}  recorded/step = {rec:.3e}   (of which teacher-fwd = {t_only:.3e}, {100*t_only/rec:.0f}%)")
    print(f"\nper-step ratio stage1/stage2 = {rates['cross_seq']/rates['normal']:.2f}  "
          f"(expect <1: cross_seq stage-1 uses half the batch)\n")

    # ---- check against a real logged run ----
    for tag in ["cross_seq", "none"]:
        pat = f"outputs/lmdist_{'gpt2' if TEACHER=='gpt2' else 'ctrl'}_kd_{tag}_*/results.json"
        ps = [p for p in glob.glob(pat) if "recorded_flops" in json.load(open(p))["history"][-1]]
        if not ps:
            continue
        r = json.load(open(ps[0]))
        s1, s2 = r["stage1_steps"], r["stage2_steps"]
        logged = r["flops"]["recorded_flops"]
        derived = s1 * rates["cross_seq"] + s2 * rates["normal"] if tag != "none" else s2 * rates["normal"]
        print(f"[{tag}] s1={s1} s2={s2}  logged_recorded={logged:.3e}  "
              f"re-derived(s1·r1+s2·r2)={derived:.3e}  ratio={logged/derived:.3f}")
        # confirm stage-1 portion is present: flops at the S1->S2 handoff
        s2first = next((d for d in r["history"] if d["stage"] == "S2"), None)
        if s2first and tag != "none":
            print(f"        cumulative recorded at S1->S2 handoff = {s2first['recorded_flops']:.3e}  "
                  f"(should ≈ s1·r1 = {s1*rates['cross_seq']:.3e})")


if __name__ == "__main__":
    main()
