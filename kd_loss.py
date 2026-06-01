"""Knowledge-distillation loss + WSD weight schedule.

Canonical logit-KD recipe confirmed by Peng et al. ACL 2025
(arXiv 2410.16215, "Pre-training Distillation Design Space"):
  - distillation term = forward KL (KL(teacher || student)), beats NLL/MSE
  - temperature tau = 2.0, scale KD term by tau^2
  - mix with the LM cross-entropy term, weighting the KD term by alpha
  - schedule alpha WSD-style: warm 0 -> alpha_max, hold, decay -> 0

For SUPERPOSED inputs there is no valid hard label, so pass labels=None and the
loss reduces to pure forward KL (alpha is effectively 1).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def forward_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
               temperature: float, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Mean forward KL = KL(p_teacher || p_student), tau^2-scaled.

    *_logits: (B, T, V). mask: (B, T) of 1/0 (optional).
    """
    t = temperature
    # Teacher/student may share the base Qwen2.5 vocab but differ in padded width
    # (e.g. R1-Distill 152064 vs Qwen2.5-0.5B 151936). The extra rows are unused
    # pad slots -> truncate both to the common real vocab before the KL.
    V = min(student_logits.size(-1), teacher_logits.size(-1))
    if student_logits.size(-1) != V:
        student_logits = student_logits[..., :V]
    if teacher_logits.size(-1) != V:
        teacher_logits = teacher_logits[..., :V]
    log_p_student = F.log_softmax(student_logits / t, dim=-1)
    p_teacher = F.softmax(teacher_logits / t, dim=-1)
    log_p_teacher = F.log_softmax(teacher_logits / t, dim=-1)
    # KL(P||Q) = sum P (logP - logQ)
    kl_tok = (p_teacher * (log_p_teacher - log_p_student)).sum(dim=-1)   # (B,T)
    kl_tok = kl_tok * (t * t)
    if mask is not None:
        m = mask.float()
        return (kl_tok * m).sum() / m.sum().clamp_min(1.0)
    return kl_tok.mean()


def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
            temperature: float = 2.0, alpha: float = 1.0,
            labels: torch.Tensor | None = None,
            mask: torch.Tensor | None = None,
            ignore_index: int = -100):
    """total = alpha * KD + (1 - alpha) * CE.

    Returns (total, dict(kd=, ce=)). If labels is None, CE is skipped and the
    total is pure KD regardless of alpha (the superposed-input case).
    """
    kd = forward_kl(student_logits, teacher_logits, temperature, mask)
    if labels is None or alpha >= 1.0:
        return kd, {"kd": kd.detach(), "ce": torch.zeros((), device=kd.device)}
    ce = F.cross_entropy(
        student_logits.reshape(-1, student_logits.size(-1)),
        labels.reshape(-1), ignore_index=ignore_index,
    )
    total = alpha * kd + (1.0 - alpha) * ce
    return total, {"kd": kd.detach(), "ce": ce.detach()}


def wsd_alpha(step: int, total_steps: int, alpha_max: float = 0.9,
              warmup_frac: float = 0.1, decay_frac: float = 0.1) -> float:
    """Warmup-Stable-Decay schedule for the KD weight alpha (Peng et al. 2025)."""
    if total_steps <= 1:
        return alpha_max
    p = step / total_steps
    if p < warmup_frac:
        return alpha_max * (p / warmup_frac)
    if p > 1.0 - decay_frac:
        return alpha_max * max(0.0, (1.0 - p) / decay_frac)
    return alpha_max
