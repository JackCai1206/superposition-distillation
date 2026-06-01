"""Aggregate outputs/*/results.json into the iso-FLOP comparison table.

Each training run writes results.json (method, losses, FLOP accounting). Optionally
merges eval accuracy if an eval_<which>.json sits next to it. The headline question:
at equal total_flops, does a superposition method reach lower loss / higher accuracy
than the `none` baseline?
"""

from __future__ import annotations

import glob
import json
import os


def load_runs(root="outputs"):
    runs = []
    for p in sorted(glob.glob(os.path.join(root, "*", "results.json"))):
        with open(p) as f:
            r = json.load(f)
        r["_dir"] = os.path.dirname(p)
        # fold in any eval_*.json (e.g. eval_math500.json -> {"accuracy":..})
        for ep in glob.glob(os.path.join(r["_dir"], "eval_*.json")):
            with open(ep) as f:
                ev = json.load(f)
            r[os.path.basename(ep)[5:-5]] = ev.get("accuracy")
        runs.append(r)
    return runs


def main():
    runs = load_runs()
    if not runs:
        print("no results.json found under outputs/"); return
    hdr = f"{'method':>12} {'lam':>4} {'s1_loss':>8} {'s2_loss':>8} {'total_flops':>12} {'flops/seq':>11} {'math500':>8} {'gsm8k':>7}"
    print(hdr); print("-" * len(hdr))
    base = next((r for r in runs if r["method"] == "none"), None)
    for r in runs:
        fl = r["flops"]
        print(f"{r['method']:>12} {str(r.get('fixed_lambda')):>4} "
              f"{(r.get('final_s1_loss') or 0):>8.3f} {(r.get('final_s2_loss') or 0):>8.3f} "
              f"{fl['total_flops']:>12.3e} {fl['flops_per_sequence']:>11.3e} "
              f"{_fmt(r.get('math500')):>8} {_fmt(r.get('gsm8k')):>7}")
    if base:
        print(f"\nbaseline(none) flops/seq = {base['flops']['flops_per_sequence']:.3e}")
        for r in runs:
            if r["method"] != "none":
                ratio = base["flops"]["flops_per_sequence"] / r["flops"]["flops_per_sequence"]
                print(f"  {r['method']:>12}: {ratio:.2f}x cheaper per effective sequence")


def _fmt(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) else "-"


if __name__ == "__main__":
    main()
