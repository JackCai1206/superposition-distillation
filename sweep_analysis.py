"""Corrected iso-FLOP analysis: total FLOPs (stage1 superposed + stage2 normal
recovery) to reach an accuracy threshold, as a function of the stage-1 budget.

This is the test the earlier runs got wrong: previously stage 1 ran so long that
cross_seq crossed the threshold during stage 1, so stage 2 never demonstrated the
"cheap superposed pretrain + short normal recovery" value. Here we sweep short
stage-1 budgets and let stage 2 do the recovery.
"""

from __future__ import annotations

import glob
import json

THRESHOLDS = (0.9, 0.95, 0.98)


def flops_to(h, t):
    for d in h:
        if d["exact_match"] >= t:
            return d["flops"]
    return None


def load(pattern="outputs/distadd_*_d10_*/results.json"):
    """Only teacher-accounted runs (flops include teacher_flops)."""
    runs = []
    for p in sorted(glob.glob(pattern)):
        r = json.load(open(p))
        if r.get("flops", {}).get("teacher_flops", 0) > 0:
            runs.append(r)
    return runs


def main():
    runs = load()
    if not runs:
        print("no sweep results yet"); return
    base = next((r for r in runs if r["method"] == "none"), None)
    base_f98 = flops_to(base["history"], 0.98) if base else None

    print(f"{'method':>10} {'s1':>5} {'peak':>6} " + " ".join(f"F>={t}" for t in THRESHOLDS) + "   vs_none@0.98")
    print("-" * 70)
    for r in sorted(runs, key=lambda x: (x["method"], x.get("stage1_steps", 0))):
        h = r["history"]
        peak = max(d["exact_match"] for d in h)
        cells = []
        for t in THRESHOLDS:
            f = flops_to(h, t)
            cells.append(f"{f:.2e}" if f else "  never ")
        f98 = flops_to(h, 0.98)
        ratio = ""
        if r["method"] != "none" and base_f98 and f98:
            sp = base_f98 / f98
            ratio = f"{sp:.2f}x ({'WIN' if sp > 1 else 'lose'})"
        elif r["method"] != "none" and base_f98 and not f98:
            ratio = "never reaches 0.98"
        print(f"{r['method']:>10} {r.get('stage1_steps','?'):>5} {peak:>6.3f} "
              + " ".join(f"{c:>9}" for c in cells) + f"   {ratio}")
    if base_f98:
        print(f"\nbaseline none reaches 0.98 at {base_f98:.3e} FLOPs")
        print("WIN = cross_seq (short stage1 + recovery) reaches 0.98 in fewer total FLOPs than none")


if __name__ == "__main__":
    main()
