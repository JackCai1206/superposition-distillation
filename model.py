"""Teacher/student loading and superposition-aware forward.

Teacher is FROZEN (eval, no grad) — central to the compute-savings argument and
the recommended offline-logit setting (Peng et al. 2025). Teacher and student
must share a tokenizer/vocab so white-box forward-KL on logits is valid.
"""

from __future__ import annotations

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from superpose import Superposed, build_inputs_embeds


def load_tokenizer(name: str):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(name: str, dtype=torch.bfloat16, device="cuda", frozen=False,
               max_pos: int | None = None, rope_theta: float | None = None):
    """Load a causal LM. Optional RoPE context extension (max_pos / rope_theta):
    the OpenMath-Nemotron recipe extends Qwen2.5-Math (4096 ctx, theta 10000) to long
    context by base-frequency scaling (theta 500000, max_pos 131072) -- needed so a
    Math student can train/eval on long CoT. Baked into the saved config -> vLLM honors it."""
    kwargs = {}
    # FlashAttention-2 when available: with PADDED batches transformers/SDPA cannot
    # take the flash/causal fast path and materializes [B,H,T,T]-scale attention
    # buffers (~25GB/layer transients at T=16K -> OOM). FA2's varlen path handles
    # padding masks in linear memory. (Training pods keep the image's flash-attn.)
    if torch.cuda.is_available():
        try:
            import flash_attn  # noqa: F401
            kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            pass
    if max_pos is not None or rope_theta is not None:
        cfg = AutoConfig.from_pretrained(name)
        if max_pos is not None:
            cfg.max_position_embeddings = max_pos
        if rope_theta is not None:
            cfg.rope_theta = rope_theta
            rs = getattr(cfg, "rope_scaling", None)
            if isinstance(rs, dict):
                cfg.rope_scaling = {**rs, "rope_theta": rope_theta}
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, config=cfg, **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, **kwargs)
    model.to(device)
    print(f"[model] {name}: attn={getattr(model.config, '_attn_implementation', '?')} dtype={dtype}")
    if frozen:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    return model


def random_model(name: str, dtype=torch.bfloat16, device="cuda",
                 max_pos: int | None = None, rope_theta: float | None = None):
    """A RANDOM-INIT model with the architecture of `name` (e.g. Qwen/Qwen2.5-0.5B).
    For pretraining distillation from scratch: the student has the target arch but no
    learned weights -> distillation can only ADD (no prior to destroy)."""
    cfg = AutoConfig.from_pretrained(name)
    if max_pos is not None:
        cfg.max_position_embeddings = max_pos
    if rope_theta is not None:
        cfg.rope_theta = rope_theta
        rs = getattr(cfg, "rope_scaling", None)
        if isinstance(rs, dict):
            cfg.rope_scaling = {**rs, "rope_theta": rope_theta}
    kwargs = {}
    if torch.cuda.is_available():
        try:
            import flash_attn  # noqa: F401
            kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            pass
    model = AutoModelForCausalLM.from_config(cfg, **kwargs).to(dtype).to(device)
    print(f"[model] {name}: RANDOM INIT ({sum(p.numel() for p in model.parameters())/1e6:.0f}M) "
          f"attn={getattr(model.config, '_attn_implementation', '?')}")
    return model


def scaled_model(ref: str, hidden: int, layers: int, heads: int, kv_heads: int,
                 inter: int, dtype=torch.bfloat16, device="cuda", tie=True,
                 max_pos: int | None = None):
    """RANDOM-INIT student with the ARCHITECTURE of `ref` (e.g. SmolLM2-135M) but
    SCALED DOWN (smaller hidden/layers/inter). Keeps ref's vocab/tokenizer + rope/
    norm/activation so KD logits align with a same-family finetuned teacher. The
    canonical 'distill a big Llama into a small Llama' student."""
    cfg = AutoConfig.from_pretrained(ref)
    cfg.hidden_size = hidden
    cfg.num_hidden_layers = layers
    cfg.num_attention_heads = heads
    cfg.num_key_value_heads = kv_heads
    cfg.intermediate_size = inter
    cfg.tie_word_embeddings = tie
    if max_pos is not None:
        cfg.max_position_embeddings = max_pos
    if getattr(cfg, "head_dim", None):
        cfg.head_dim = hidden // heads
    kwargs = {}
    if torch.cuda.is_available():
        try:
            import flash_attn  # noqa: F401
            kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            pass
    model = AutoModelForCausalLM.from_config(cfg, **kwargs).to(dtype).to(device)
    print(f"[model] scaled {ref}: {sum(p.numel() for p in model.parameters())/1e6:.1f}M "
          f"(h{hidden} L{layers} kv{kv_heads}) attn={getattr(model.config,'_attn_implementation','?')}")
    return model


def tiny_model(vocab_size: int, hidden=64, layers=2, heads=4, inter=128,
               dtype=torch.float32, device="cpu", tie_embeddings=False,
               max_pos=1024):
    """A small random Qwen2-style model (from-scratch). tie_embeddings halves the
    embedding cost for large-vocab NL models."""
    cfg = AutoConfig.for_model(
        "qwen2", vocab_size=vocab_size, hidden_size=hidden,
        num_hidden_layers=layers, num_attention_heads=heads,
        num_key_value_heads=heads, intermediate_size=inter,
        max_position_embeddings=max_pos, tie_word_embeddings=tie_embeddings,
    )
    model = AutoModelForCausalLM.from_config(cfg).to(dtype).to(device)
    return model


def _body(model):
    """The transformer body (before the LM head), model-agnostic."""
    fn = getattr(model, "get_decoder", None)
    if callable(fn):
        d = fn()
        if d is not None:
            return d
    for attr in ("model", "transformer", "gpt_neox", "decoder"):
        b = getattr(model, attr, None)
        if b is not None:
            return b
    raise ValueError("cannot locate transformer body")


def superposed_hidden(model: torch.nn.Module, sup: Superposed,
                      noise_sigma: float = 0.0) -> torch.Tensor:
    """Last hidden state on a Superposed input (no LM head). Pair with the head
    weight from model.get_output_embeddings() for a fused-linear (chunked) loss
    that never materializes the full [B,T,V] logits.

    noise_sigma > 0 adds Gaussian noise to the input embeddings, scaled by the
    batch embedding RMS (so sigma is a *relative* perturbation magnitude). This is
    the perturbed-point / smoothed-teacher query: train KD at x+sigma*u instead of x.
    Note: teacher and student have different hidden sizes, so each is perturbed in
    its OWN embedding space with an INDEPENDENT draw (no shared u across models)."""
    embed = model.get_input_embeddings()
    ids = sup.ids.to(embed.weight.device)
    weights = sup.weights.to(dtype=embed.weight.dtype, device=embed.weight.device)
    inputs_embeds = build_inputs_embeds(embed, ids, weights)
    if noise_sigma and noise_sigma > 0:
        scale = inputs_embeds.detach().pow(2).mean().sqrt()          # embedding RMS (scalar)
        inputs_embeds = inputs_embeds + noise_sigma * scale * torch.randn_like(inputs_embeds)
    attn = sup.mask.to(embed.weight.device)
    out = _body(model)(inputs_embeds=inputs_embeds, attention_mask=attn)
    return out.last_hidden_state


def superposed_logits(model: torch.nn.Module, sup: Superposed) -> torch.Tensor:
    """Run a model on a Superposed input, returning logits (B, T, V).

    Builds inputs_embeds from the model's own embedding so teacher and student
    consume the *same* superposed input despite different hidden sizes.
    """
    embed = model.get_input_embeddings()
    ids = sup.ids.to(embed.weight.device)
    weights = sup.weights.to(dtype=embed.weight.dtype, device=embed.weight.device)
    inputs_embeds = build_inputs_embeds(embed, ids, weights)            # (B,T,H)
    attn = sup.mask.to(embed.weight.device)
    out = model(inputs_embeds=inputs_embeds, attention_mask=attn)
    return out.logits
