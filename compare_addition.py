"""Iso-FLOP comparison for the addition distillation runs.

Reads outputs/distadd_*/results.json (each has history of {step, exact_match,
flops}). Reports:
  - final exact-match and total FLOPs per method
  - accuracy at a common FLOP budget (the headline iso-FLOP question)
  - FLOPs needed to first reach a target accuracy (compute-to-threshold)
"""

from __future__ import annotations

import glob
import json
import os


def load():
    runs = {}
    for p in sorted(glob.glob("outputs/distadd_*/results.json")):
        with open(p) as f:
            r = json.load(f)
        runs[r["method"]] = r
    return runs


def acc_at_flops(hist, budget):
    """Last recorded exact_match at or below the FLOP budget."""
    best = 0.0
    for h in hist:
        if h["flops"] <= budget:
            best = h["exact_match"]
    return best


def flops_to_acc(hist, target):
    for h in hist:
        if h["exact_match"] >= target:
            return h["flops"]
    return None


def main():
    runs = load()
    if not runs:
        print("no outputs/distadd_*/results.json yet"); return

    print(f"{'method':>12} {'final_acc':>10} {'total_flops':>12} {'student_M':>10}")
    print("-" * 48)
    for m, r in runs.items():
        print(f"{m:>12} {r['final']['exact_match']:>10.3f} "
              f"{r['flops']['total_flops']:>12.3e} {r['student_params']/1e6:>10.2f}")

    # iso-FLOP: accuracy at the smallest of the methods' total budgets
    budget = min(r["flops"]["total_flops"] for r in runs.values())
    print(f"\n=== accuracy at common budget {budget:.3e} FLOPs ===")
    for m, r in runs.items():
        print(f"  {m:>12}: {acc_at_flops(r['history'], budget):.3f}")

    print("\n=== FLOPs to first reach accuracy threshold ===")
    base = runs.get("none")
    for target in (0.5, 0.9, 0.99):
        row = []
        for m, r in runs.items():
            f = flops_to_acc(r["history"], target)
            row.append(f"{m}={'%.2e' % f if f else '—'}")
        line = f"  acc>={target}: " + "  ".join(row)
        if base:
            bf = flops_to_acc(base["history"], target)
            for m, r in runs.items():
                if m != "none":
                    mf = flops_to_acc(r["history"], target)
                    if bf and mf:
                        line += f"   [{m} {bf/mf:.2f}x vs none]"
        print(line)


if __name__ == "__main__":
    main()
