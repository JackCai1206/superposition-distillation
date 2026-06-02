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
    n_params: int        # non-embedding (transformer-body) parameter count
    n_layer: int
    d_model: int
    vocab_size: int = 0  # for the LM-head matmul (negligible at tiny vocab, big at 50k)

    def forward_flops(self, seq_len: int, batch: int = 1) -> float:
        dense = 2.0 * self.n_params * seq_len
        attn = 2.0 * 2.0 * self.n_layer * self.d_model * (seq_len ** 2)
        head = 2.0 * self.vocab_size * self.d_model * seq_len   # logits projection
        return (dense + attn + head) * batch

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
    vocab = getattr(config, "vocab_size", 0)
    return FlopModel(n_params=n_params, n_layer=n_layer, d_model=d_model, vocab_size=vocab)


class FlopCounter:
    """Accumulates FLOPs and effective data seen.

    Student: fwd+bwd (3x forward) each step. Teacher (optional): one frozen
    forward each step on the SAME batch the student sees. Because cross_seq packs
    two sequences per example, its per-step batch is halved -> the teacher does
    one forward per two sequences, which is the amortization that matters when the
    teacher is much larger than the student.
    """

    def __init__(self, fm: FlopModel, teacher_fm: FlopModel | None = None):
        self.fm = fm
        self.teacher_fm = teacher_fm
        self.student_flops = 0.0
        self.teacher_flops = 0.0
        self.measured_flops = 0.0    # actual op-level FLOPs (recorded), incl. any recompute
        self._rate = {}              # (seq_len,batch) -> measured per-step FLOPs
        self.sequences_seen = 0.0    # effective #source-sequences of learning signal
        self.tokens_processed = 0.0  # student forward positions

    def add_step(self, seq_len: int, batch: int, effective_sequences: float,
                 measured_step: float | None = None):
        self.student_flops += self.fm.train_step_flops(seq_len, batch)
        if self.teacher_fm is not None:
            self.teacher_flops += self.teacher_fm.forward_flops(seq_len, batch)
        # recorded: cache the measured per-step rate per (seq_len,batch); accumulate it
        key = (seq_len, batch)
        if measured_step is not None:
            self._rate[key] = measured_step
        if key in self._rate:
            self.measured_flops += self._rate[key]
        self.tokens_processed += seq_len * batch
        self.sequences_seen += effective_sequences

    @property
    def total_flops(self) -> float:    # estimated (analytic), student+teacher
        return self.student_flops + self.teacher_flops

    def summary(self) -> dict:
        est = self.total_flops
        return {
            "total_flops": est,                  # estimated (analytic) — kept as the default key
            "estimated_flops": est,
            "recorded_flops": self.measured_flops,  # actual op-level (FlopCounterMode)
            "student_flops": self.student_flops,
            "teacher_flops": self.teacher_flops,
            "sequences_seen": self.sequences_seen,
            "tokens_processed": self.tokens_processed,
            "flops_per_sequence": est / max(self.sequences_seen, 1.0),
        }


def measure_step_flops(fwd_bwd_fn) -> float:
    """Actual op-level FLOPs of one training step (fwd+bwd) via torch FlopCounterMode."""
    from torch.utils.flop_counter import FlopCounterMode
    fcm = FlopCounterMode(display=False)
    with fcm:
        fwd_bwd_fn()
    return float(fcm.get_total_flops())
