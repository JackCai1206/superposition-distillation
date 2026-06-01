"""Evaluation: (1) held-out LM loss/perplexity on normal data, and (2) downstream
math-reasoning accuracy on GSM8K / MATH500 via math_verify.

Used to fill the iso-FLOP comparison: for each condition, plot loss / accuracy
against FlopCounter.total_flops.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from config import Config
from data import batched, pretrain_stream, synthetic_stream
from model import load_model, load_tokenizer, tiny_model


@torch.no_grad()
def eval_lm_loss(student, stream, device, n_batches=20, batch_size=8, seq_len=1024):
    """Mean next-token CE (normal single-sequence inputs)."""
    student.eval()
    loader = batched(stream, batch_size)
    tot, n = 0.0, 0
    for _ in range(n_batches):
        try:
            ids, mask = next(loader)
        except StopIteration:
            break
        ids = ids.to(device)
        logits = student(input_ids=ids, attention_mask=mask.to(device)).logits
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                               ids[:, 1:].reshape(-1))
        tot += loss.item(); n += 1
    mean = tot / max(n, 1)
    return {"lm_loss": mean, "perplexity": float(torch.tensor(mean).exp())}


EVAL_SETS = {   # name -> (hf_id, config, split)
    "math500": ("HuggingFaceH4/MATH-500", None, "test"),
    "gsm8k": ("gsm8k", "main", "test"),
}


def _gold(ex):
    g = ex.get("answer") or ex.get("solution") or ""
    return g.split("####")[-1].strip() if "####" in g else g   # gsm8k -> final number


@torch.no_grad()
def eval_math(student, tokenizer, device, which="math500", n=100, max_new_tokens=512):
    """Generate solutions and score with math_verify. Returns accuracy."""
    from datasets import load_dataset
    from math_verify import parse, verify

    hf_id, cfg_name, split = EVAL_SETS[which]
    ds = load_dataset(hf_id, cfg_name, split=split)
    n = min(n, len(ds)); ds = ds.select(range(n))
    student.eval()
    correct = 0
    for ex in ds:
        problem = ex.get("problem") or ex.get("question")
        prompt = f"{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        out = student.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                               pad_token_id=tokenizer.pad_token_id)
        text = tokenizer.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
        try:
            if verify(parse(_gold(ex)), parse(text)):
                correct += 1
        except Exception:
            pass
    return {"dataset": which, "n": n, "accuracy": correct / max(n, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None, help="student model path/name")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--math", action="store_true", help="also run MATH-500 accuracy")
    ap.add_argument("--gsm8k", action="store_true", help="also run GSM8K accuracy")
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()

    cfg = Config(); cfg.debug = args.debug; cfg.train.device = args.device
    if args.debug:
        cfg.train.device = "cpu"
        tok = None
        student = tiny_model(256, hidden=48, layers=2, device="cpu")
        loss = eval_lm_loss(student, synthetic_stream(cfg, 256), "cpu",
                            n_batches=3, batch_size=4, seq_len=cfg.data.seq_len)
        print("LM:", loss); return

    tok = load_tokenizer(args.checkpoint or cfg.model.student)
    student = load_model(args.checkpoint or cfg.model.student,
                         dtype=torch.bfloat16, device=args.device, frozen=True)
    if args.math:
        print("MATH:", eval_math(student, tok, args.device, which="math500", n=args.n))
    if args.gsm8k:
        print("GSM8K:", eval_math(student, tok, args.device, which="gsm8k", n=args.n))
    if not (args.math or args.gsm8k):
        print("LM:", eval_lm_loss(student, pretrain_stream(cfg, tok), args.device,
                                  seq_len=cfg.data.seq_len))


if __name__ == "__main__":
    main()
