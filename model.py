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


def load_model(name: str, dtype=torch.bfloat16, device="cuda", frozen=False):
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
    model.to(device)
    if frozen:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
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


def superposed_hidden(model: torch.nn.Module, sup: Superposed) -> torch.Tensor:
    """Last hidden state on a Superposed input (no LM head). Pair with the head
    weight from model.get_output_embeddings() for a fused-linear (chunked) loss
    that never materializes the full [B,T,V] logits."""
    embed = model.get_input_embeddings()
    ids = sup.ids.to(embed.weight.device)
    weights = sup.weights.to(dtype=embed.weight.dtype, device=embed.weight.device)
    inputs_embeds = build_inputs_embeds(embed, ids, weights)
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
