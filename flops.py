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

    ADAM_FLOPS_PER_PARAM = 18   # AdamW elementwise ops/param/step (m,v,bias-correct,update,wd)

    def __init__(self, fm: FlopModel, teacher_fm: FlopModel | None = None,
                 opt_params: int = 0):
        self.fm = fm
        self.teacher_fm = teacher_fm
        self.opt_params = opt_params   # trainable params -> AdamW update cost (no token mult)
        self.student_flops = 0.0
        self.teacher_flops = 0.0
        self.optimizer_flops = 0.0
        self.measured_matmul = 0.0   # op-level matmul FLOPs (recorded), incl. recompute
        self._rate = {}              # (seq_len,batch) -> measured per-step matmul FLOPs
        self.sequences_seen = 0.0
        self.tokens_processed = 0.0

    def add_step(self, seq_len: int, batch: int, effective_sequences: float,
                 measured_step: float | None = None):
        self.student_flops += self.fm.train_step_flops(seq_len, batch)
        if self.teacher_fm is not None:
            self.teacher_flops += self.teacher_fm.forward_flops(seq_len, batch)
        self.optimizer_flops += self.ADAM_FLOPS_PER_PARAM * self.opt_params
        key = (seq_len, batch)
        if measured_step is not None:
            self._rate[key] = measured_step
        if key in self._rate:
            self.measured_matmul += self._rate[key]
        self.tokens_processed += seq_len * batch
        self.sequences_seen += effective_sequences

    @property
    def total_flops(self) -> float:    # COMPLETE estimated: student+teacher matmul + optimizer
        return self.student_flops + self.teacher_flops + self.optimizer_flops

    def summary(self) -> dict:
        est = self.total_flops
        rec = self.measured_matmul + self.optimizer_flops   # complete recorded
        return {
            "total_flops": est,                  # complete estimated (matmul + optimizer)
            "estimated_flops": est,
            "recorded_flops": rec,               # complete recorded (op-level matmul + optimizer)
            "student_flops": self.student_flops,
            "teacher_flops": self.teacher_flops,
            "optimizer_flops": self.optimizer_flops,
            "measured_matmul_flops": self.measured_matmul,
            "sequences_seen": self.sequences_seen,
            "tokens_processed": self.tokens_processed,
            "flops_per_sequence": est / max(self.sequences_seen, 1.0),
        }


def _gqa_sdpa_flop(query_shape, key_shape, value_shape, *args, out_shape=None, **kwargs) -> int:
    """GQA-correct scaled-dot-product-attention FLOPs.

    torch's builtin sdpa_flop_count (in older torch, e.g. the NVIDIA 25.08 image)
    asserts query-heads == key-heads, which is FALSE for grouped-query attention
    -> every real Qwen2.5 model uses GQA, so the builtin crashes. The kernel
    broadcasts the h_kv key/value heads up to the h_q query heads, so both matmuls
    cost h_q heads: 2*b*h_q*s_q*s_k*(d_q + d_v).
    """
    b, h_q, s_q, d_q = query_shape
    s_k = key_shape[2]
    d_v = value_shape[-1]
    return 2 * b * h_q * s_q * s_k * (d_q + d_v)


def measure_step_flops(fwd_bwd_fn):
    """Actual op-level FLOPs of one training step (fwd+bwd) via torch FlopCounterMode.

    Overrides the SDPA flop formula with a GQA-correct one (see _gqa_sdpa_flop).
    Returns None if op-level counting is unavailable on this torch build -- the
    analytic accounting is the primary, audited metric; op-level is a cross-check.
    On failure the step is re-run cleanly so gradients are still populated for the
    optimizer step.
    """
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    custom = {}
    for nm in ("_scaled_dot_product_efficient_attention",
               "_scaled_dot_product_flash_attention",
               "_scaled_dot_product_cudnn_attention"):
        op = getattr(torch.ops.aten, nm, None)
        if op is not None:
            custom[op] = _gqa_sdpa_flop
    try:
        fcm = FlopCounterMode(display=False, custom_mapping=custom)
        with fcm:
            fwd_bwd_fn()
        return float(fcm.get_total_flops())
    except Exception as e:
        msg = f"{type(e).__name__}: {str(e)[:80]}"
        e = None   # drop the exception/traceback -> releases the frame locals (the
                   # failed graph + activations) so empty_cache can actually free them
        print(f"[flops] op-level measurement unavailable ({msg}); analytic FLOPs only")
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        fwd_bwd_fn()   # clean re-run so grads exist for the subsequent opt.step()
        return None
