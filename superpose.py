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
    # Loop-accumulate over the M component axis instead of materializing the full
    # (B,T,M,H) tensor. Mathematically identical to (embed(ids)*w).sum(2), but peak
    # memory stays at the M=1 level -- critical for the one-hot noise perturbation
    # (M=1+k_noise), where (B,T,M,H) on the 2048-dim teacher is multi-GB and OOMs.
    w = weights.unsqueeze(-1)               # (B, T, M, 1)
    out = embed(ids[..., 0]) * w[..., 0, :]         # (B, T, H)
    for j in range(1, ids.shape[-1]):
        out = out + embed(ids[..., j]) * w[..., j, :]
    return out


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


def superpose_input_noise(input_ids: torch.Tensor, attention_mask: torch.Tensor,
                          k_noise: int = 16, sigma: float = 0.1,
                          vocab_size: int = 49152) -> Superposed:
    """Perturb the input ONE-HOT in (shared) vocab space: o -> o + sigma*u.

    Realized as superposition: the real token (weight 1) plus k_noise RANDOM tokens
    with small shared Gaussian weights ~ N(0, sigma^2 / k). Because the returned
    (ids, weights) are handed to BOTH teacher and student, the perturbation is SHARED
    across models despite their different hidden sizes -- each maps the same perturbed
    one-hot through its own embedding matrix (E^T(o+sigma*u)). This is the shared-u
    that makes 1-point gradient matching well-defined on models of different width.
    Normalizing the noise weights by sqrt(k) makes sigma ~ the relative perturbation
    magnitude (comparable to the embedding-space noise mode). Weights do NOT sum to 1
    (it is o + sigma*u, a perturbation, not a convex mixture). T is unchanged (= L)."""
    B, L = input_ids.shape
    dev = input_ids.device
    noise_ids = torch.randint(0, vocab_size, (B, L, k_noise), device=dev)
    ids = torch.cat([input_ids.unsqueeze(-1), noise_ids], dim=-1)              # (B,L,1+k)
    real_w = torch.ones(B, L, 1, device=dev)
    noise_w = (sigma / (k_noise ** 0.5)) * torch.randn(B, L, k_noise, device=dev)
    weights = torch.cat([real_w, noise_w], dim=-1)                            # (B,L,1+k)
    return Superposed(ids, weights, attention_mask)


def superpose_cross_seq(
    seq_a: torch.Tensor, mask_a: torch.Tensor,
    seq_b: torch.Tensor, mask_b: torch.Tensor,
    mix_alpha: float = 1.0, fixed: float | None = None,
) -> Superposed:
    """Method 1: position-wise mix of two equal-length sequences.

    seq_a, seq_b: (B, L) already padded to a common length L.
    Returns (B, L, 2). A position is real where EITHER sequence is real: where both
    are real it's the lam-blend; past the shorter sequence's end the weights
    renormalize onto the surviving token ([1, 0]) so the longer sequence's tail
    trains as clean single tokens instead of being discarded (same pattern as
    token_merge's padded-slot renormalization). Positions real in neither are masked.
    """
    B, L = seq_a.shape
    lam = _sample_lambda(B, mix_alpha, fixed, seq_a.device).view(B, 1)  # (B,1)
    ids = torch.stack([seq_a, seq_b], dim=-1)                          # (B,L,2)
    w = torch.stack([lam.expand(B, L) * mask_a.float(),
                     (1 - lam).expand(B, L) * mask_b.float()], dim=-1)  # zero weight on padding
    denom = w.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    w = w / denom                                                       # renormalize -> [1,0] on tails
    mask = ((mask_a + mask_b) > 0).long()                               # real if ANY component real
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


def superpose_cross_merge(
    seq_a: torch.Tensor, mask_a: torch.Tensor,
    seq_b: torch.Tensor, mask_b: torch.Tensor,
    k: int = 2, mix_alpha: float = 1.0, fixed: float | None = None,
) -> Superposed:
    """Method 3: BOTH axes at once -- token_merge (k adjacent, sequential) x cross_seq
    (two sequences, parallel). Each output position packs 2k raw tokens: k adjacent from
    seq_a's bag + k adjacent from seq_b's bag. seq_a/seq_b: (B, L). Returns (B, L//k, 2k).
    Weights are the PRODUCT of the two convex structures: seq_a's k slots scaled by the
    cross-blend lam, seq_b's k slots by (1-lam), each already tilt-weighted within its bag.
    At lam=0.5, k=2 -> four equal 0.25 weights. So at iso-FLOP this packs 2k x the raw
    tokens of the baseline into the same forward position count."""
    supA = superpose_token_merge(seq_a, mask_a, k=k, fixed=fixed)      # (B,T,k)
    supB = superpose_token_merge(seq_b, mask_b, k=k, fixed=fixed)
    B = seq_a.shape[0]
    clam = _sample_lambda(B, mix_alpha, fixed, seq_a.device).view(B, 1, 1)  # cross blend
    ids = torch.cat([supA.ids, supB.ids], dim=-1)                     # (B,T,2k)
    wA = supA.weights * clam * supA.mask.unsqueeze(-1).float()        # A slots -> sum clam
    wB = supB.weights * (1 - clam) * supB.mask.unsqueeze(-1).float()  # B slots -> sum 1-clam
    w = torch.cat([wA, wB], dim=-1)
    denom = w.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    w = w / denom                                                     # renormalize -> convex
    mask = ((supA.mask + supB.mask) > 0).long()
    return Superposed(ids, w, mask)
