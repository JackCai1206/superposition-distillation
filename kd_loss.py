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


def chunked_distill_loss(s_hidden, s_head_w, t_hidden, t_head_w, temperature=2.0,
                         mask=None, labels=None, alpha=1.0, chunk=2048,
                         ignore_index=-100):
    """Fused-linear forward-KL (+ optional CE), chunked over tokens with gradient
    checkpointing so the full [B,T,V] logits are never materialized.

    s_hidden/t_hidden: (B,T,H*) last hidden states. s_head_w/t_head_w: (V,H*) LM-head
    weights. Returns (total, {kd, ce}). KD uses teacher logits (no_grad). If labels
    given and alpha<1, adds (1-alpha)*CE on `labels` (already shifted; ignore_index
    masks the last position). Same math as forward_kl, O(chunk*V) peak memory.
    """
    import torch.utils.checkpoint as ckpt
    B, T, _ = s_hidden.shape
    sh = s_hidden.reshape(B * T, -1)
    th = t_hidden.reshape(B * T, -1)
    m = (mask.reshape(B * T).float() if mask is not None else sh.new_ones(B * T))
    lab = labels.reshape(B * T) if labels is not None else None
    t = temperature
    denom = m.sum().clamp_min(1.0)
    kd_sum = sh.new_zeros(())
    ce_sum = sh.new_zeros(())
    ce_n = sh.new_zeros(())

    def piece(sh_c, th_c, m_c, lab_c):
        s_logits = sh_c @ s_head_w.t()
        with torch.no_grad():
            t_logits = th_c @ t_head_w.t()
        logp_s = F.log_softmax(s_logits / t, dim=-1)
        p_t = F.softmax(t_logits / t, dim=-1)
        logp_t = F.log_softmax(t_logits / t, dim=-1)
        kl = (p_t * (logp_t - logp_s)).sum(-1) * (t * t)            # (c,)
        kd = (kl * m_c).sum()
        if lab_c is not None and alpha < 1.0:
            ce_tok = F.cross_entropy(s_logits, lab_c, ignore_index=ignore_index,
                                     reduction="none")               # (c,)
            valid = (lab_c != ignore_index).float()
            return kd, (ce_tok * valid).sum(), valid.sum()
        return kd, sh.new_zeros(()), sh.new_zeros(())

    for i in range(0, B * T, chunk):
        sl = slice(i, i + chunk)
        kd_c, ce_c, n_c = ckpt.checkpoint(
            piece, sh[sl], th[sl], m[sl],
            (lab[sl] if lab is not None else None), use_reentrant=False)
        kd_sum = kd_sum + kd_c
        ce_sum = ce_sum + ce_c
        ce_n = ce_n + n_c

    kd = kd_sum / denom
    if labels is not None and alpha < 1.0:
        ce = ce_sum / ce_n.clamp_min(1.0)
        return alpha * kd + (1 - alpha) * ce, {"kd": kd.detach(), "ce": ce.detach()}
    return kd, {"kd": kd.detach(), "ce": kd.new_zeros(())}


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


def wsd_lr_mult(global_step: int, total_steps: int, warmup: int = 50,
                decay: int = 300) -> float:
    """WSD learning-rate multiplier over the WHOLE two-stage run.

    One warmup at the very start (Stage 1), a length-agnostic stable phase
    spanning the stage boundary, and a fixed terminal decay window (in steps, not
    a fraction of total) so configs with different Stage-1 budgets stay comparable
    and the cooldown lands after the threshold crossing.
    """
    if global_step < warmup:
        return (global_step + 1) / warmup
    decay_start = total_steps - decay
    if global_step >= decay_start:
        return max(0.0, (total_steps - global_step) / decay)
    return 1.0
