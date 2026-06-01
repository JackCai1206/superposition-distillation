"""Analytic transformer FLOP accounting for the iso-FLOP comparison.

We compare conditions at equal *student* training FLOPs. The teacher forward
cost is identical across conditions (baseline vs superposed both need teacher
logits on the same number of positions) so it cancels in the comparison; we
still report it for completeness.

Per forward pass over a sequence of length T (one example):
  dense (embeddings + MLP + attn projections):  ~ 2 * N_params * T
  attention score/!value matmuls (quadratic):    ~ 2 * 2 * n_layer * d_model * T^2
A training step (fwd+bwd) is ~3x the forward FLOPs.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FlopModel:
    n_params: int        # non-embedding parameter count (approx ok)
    n_layer: int
    d_model: int

    def forward_flops(self, seq_len: int, batch: int = 1) -> float:
        dense = 2.0 * self.n_params * seq_len
        attn = 2.0 * 2.0 * self.n_layer * self.d_model * (seq_len ** 2)
        return (dense + attn) * batch

    def train_step_flops(self, seq_len: int, batch: int = 1) -> float:
        return 3.0 * self.forward_flops(seq_len, batch)


def model_flops_from_config(config) -> FlopModel:
    """Build a FlopModel from a HF config (Qwen2/Llama-like)."""
    n_layer = getattr(config, "num_hidden_layers")
    d_model = getattr(config, "hidden_size")
    # crude non-embedding param estimate: 12 * n_layer * d_model^2 (attn+mlp)
    inter = getattr(config, "intermediate_size", 4 * d_model)
    per_layer = 4 * d_model * d_model + 3 * d_model * inter   # attn qkvo + gate/up/down
    n_params = n_layer * per_layer
    return FlopModel(n_params=n_params, n_layer=n_layer, d_model=d_model)


class FlopCounter:
    """Accumulates student training FLOPs and effective data (sequences) seen."""

    def __init__(self, fm: FlopModel):
        self.fm = fm
        self.total_flops = 0.0
        self.sequences_seen = 0.0   # effective #source-sequences of learning signal
        self.tokens_processed = 0.0 # student forward positions

    def add_step(self, seq_len: int, batch: int, effective_sequences: float):
        self.total_flops += self.fm.train_step_flops(seq_len, batch)
        self.tokens_processed += seq_len * batch
        self.sequences_seen += effective_sequences

    def summary(self) -> dict:
        return {
            "total_flops": self.total_flops,
            "sequences_seen": self.sequences_seen,
            "tokens_processed": self.tokens_processed,
            "flops_per_sequence": self.total_flops / max(self.sequences_seen, 1.0),
        }
