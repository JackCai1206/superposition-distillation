"""Superposition collators.

Unified representation
----------------------
Every superposed example is a tensor of *component token ids* plus *mixing
weights*. For an output sequence of length T where each position is a convex
combination of M token ids:

    ids:     LongTensor (B, T, M)   component token ids per output position
    weights: FloatTensor (B, T, M)  convex weights (sum to 1 over M)
    mask:    LongTensor (B, T)       1 for real positions, 0 for padding

The model-side embedding is then

    inputs_embeds[b, t] = sum_m weights[b, t, m] * E(ids[b, t, m])

where E is *that model's own* embedding matrix (teacher and student differ in
hidden size but share the vocab / tokenizer).

Methods
-------
- none        : M=1, weight 1 -> standard single-sequence input (baseline).
- cross_seq   : M=2, mix two sequences position-wise (MixKD-style, but here as a
                compute-packing input: one forward pass carries two sequences).
- token_merge : M=k, mix k adjacent tokens of ONE sequence into one input
                position -> output length T = L // k (shorter sequence).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def build_inputs_embeds(embed: torch.nn.Module, ids: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Convex combination of token embeddings.

    embed:   nn.Embedding (a single model's input embedding)
    ids:     (B, T, M) long
    weights: (B, T, M) float
    returns: (B, T, H) float
    """
    comp = embed(ids)                       # (B, T, M, H)
    return (comp * weights.unsqueeze(-1)).sum(dim=2)  # (B, T, H)


def _sample_lambda(batch: int, mix_alpha: float, fixed: float | None,
                   device=None) -> torch.Tensor:
    """One mixing coefficient per example. fixed overrides the Beta sample."""
    if fixed is not None:
        return torch.full((batch,), float(fixed), device=device)
    if mix_alpha <= 0:                       # degenerate -> 0.5
        return torch.full((batch,), 0.5, device=device)
    beta = torch.distributions.Beta(mix_alpha, mix_alpha)
    return beta.sample((batch,)).to(device)


@dataclass
class Superposed:
    ids: torch.Tensor          # (B, T, M)
    weights: torch.Tensor      # (B, T, M)
    mask: torch.Tensor         # (B, T)

    def to(self, device):
        return Superposed(self.ids.to(device), self.weights.to(device), self.mask.to(device))


def superpose_none(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Superposed:
    """Baseline: (B, L) -> (B, L, 1)."""
    return Superposed(input_ids.unsqueeze(-1), torch.ones_like(input_ids, dtype=torch.float).unsqueeze(-1), attention_mask)


def superpose_cross_seq(
    seq_a: torch.Tensor, mask_a: torch.Tensor,
    seq_b: torch.Tensor, mask_b: torch.Tensor,
    mix_alpha: float = 1.0, fixed: float | None = None,
) -> Superposed:
    """Method 1: position-wise mix of two equal-length sequences.

    seq_a, seq_b: (B, L) already padded to a common length L.
    Returns (B, L, 2). A position is "real" only where BOTH sequences are real
    (so the mix is well-defined); elsewhere masked out.
    """
    B, L = seq_a.shape
    lam = _sample_lambda(B, mix_alpha, fixed, seq_a.device).view(B, 1)  # (B,1)
    ids = torch.stack([seq_a, seq_b], dim=-1)                          # (B,L,2)
    w = torch.stack([lam.expand(B, L), (1 - lam).expand(B, L)], dim=-1)  # (B,L,2)
    mask = mask_a * mask_b
    return Superposed(ids, w, mask)


def superpose_token_merge(
    input_ids: torch.Tensor, attention_mask: torch.Tensor,
    k: int = 2, mix_alpha: float = 1.0, fixed: float | None = None,
) -> Superposed:
    """Method 2: merge k adjacent tokens of one sequence into one position.

    input_ids: (B, L) with L divisible by k (trim outside if needed).
    Returns (B, L//k, k). Weights are a per-example convex vector over the k
    slots: for k=2, [lam, 1-lam]; for k>2, a normalized lam-tilted vector that
    still emphasizes the first (current) token.
    """
    B, L = input_ids.shape
    L = (L // k) * k
    input_ids = input_ids[:, :L]
    attention_mask = attention_mask[:, :L]
    T = L // k
    ids = input_ids.view(B, T, k)                                     # (B,T,k)
    m = attention_mask.view(B, T, k)
    lam = _sample_lambda(B, mix_alpha, fixed, input_ids.device).view(B, 1, 1)  # (B,1,1)
    if k == 2:
        w = torch.cat([lam, 1 - lam], dim=-1).expand(B, T, k).contiguous()
    else:
        # lam-tilt: first slot gets lam, the rest split (1-lam) uniformly
        rest = (1 - lam) / (k - 1)
        w = torch.cat([lam, rest.expand(B, 1, k - 1)], dim=-1).expand(B, T, k).contiguous()
    # zero out weights on padded slots, renormalize over real slots
    w = w * m.float()
    denom = w.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    w = w / denom
    mask = (m.sum(dim=-1) > 0).long()                                 # (B,T) real if any slot real
    return Superposed(ids, w, mask)
