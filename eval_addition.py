"""Re-evaluate a saved addition checkpoint with the (fixed) exact-match decoder."""

from __future__ import annotations

import argparse

import torch

from addition import exact_match
from model import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n_digits", type=int, default=4)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    model = load_model(args.ckpt, dtype=torch.bfloat16, device=args.device, frozen=True)
    acc = exact_match(model, args.n_digits, args.device, n=args.n)
    print(f"{args.ckpt}  n_digits={args.n_digits}  n={args.n}  exact_match={acc:.4f}")


if __name__ == "__main__":
    main()
