"""vLLM math-reasoning eval for distilled students (GSM8K + MATH-500).

Why vLLM (not eval.py's HF generate): reasoning students emit long CoT -- a faithful
eval needs max_new_tokens ~16K, and HF greedy generate over hundreds of problems at
that length is hopelessly slow. vLLM batches it. Recipe (proven on B200 in the
memento-jacobi work): `pip install vllm==0.13.0`, uninstall the image's standalone
flash_attn (vLLM swaps torch and breaks its ABI), PVC-pinned caches.

ONE checkpoint per process -> the sweep launcher (bonete/eval_cluster.sh) calls this
per checkpoint so each gets a clean GPU/parallel state; results are merged with the
FLOP-tagged checkpoints.json into an accuracy-vs-FLOPs curve.
"""

from __future__ import annotations

import argparse
import json
import os

EVAL_SETS = {  # name -> (hf_id, config, split)
    "math500": ("HuggingFaceH4/MATH-500", None, "test"),
    "gsm8k": ("gsm8k", "main", "test"),
}


def _gold(ex):
    g = ex.get("answer") or ex.get("solution") or ""
    return g.split("####")[-1].strip() if "####" in g else g   # gsm8k -> final number


def load_problems(which, n):
    from datasets import load_dataset
    hf_id, cfg_name, split = EVAL_SETS[which]
    ds = load_dataset(hf_id, cfg_name, split=split)
    if n is not None and 0 < n < len(ds):     # n<=0 / None / >len -> full set
        ds = ds.select(range(n))
    probs = [(ex.get("problem") or ex.get("question")) for ex in ds]
    golds = [_gold(ex) for ex in ds]
    return probs, golds


def parse_spec(item, default_n):
    """'gsm8k' | 'gsm8k:full:1' | 'math500:500:4' -> (name, n, k).
    n='full'/'-1'/'0' -> full set; k = samples/rollouts per problem."""
    parts = item.split(":")
    name = parts[0]
    n = default_n
    k = 1
    if len(parts) > 1:
        n = None if parts[1] in ("full", "-1", "0", "") else int(parts[1])
    if len(parts) > 2:
        k = int(parts[2])
    return name, n, k


def make_prompts(probs, style):
    if style == "instruct":
        return [f"{p}\nPlease reason step by step, and put your final answer within \\boxed{{}}." for p in probs]
    return [p.rstrip() + "\n" for p in probs]   # training-matched continuation (data.py)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="HF checkpoint dir (student)")
    ap.add_argument("--benchmarks", default="gsm8k,math500")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max_new_tokens", type=int, default=16384)
    ap.add_argument("--max_model_len", type=int, default=20000)
    ap.add_argument("--gpu_mem", type=float, default=0.9)
    ap.add_argument("--prompt_style", default="train", choices=["train", "instruct"])
    ap.add_argument("--temperature", type=float, default=0.7, help="used when k>1 samples")
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--out", default=None, help="default: <checkpoint>/eval_vllm.json")
    ap.add_argument("--tokenizer", default=None,
                    help="tokenizer source override. Default: the checkpoint config's "
                         "_name_or_path (hub id) -- checkpoints saved by transformers 5.x "
                         "write tokenizer_config.json that transformers<5 (vLLM's pin) "
                         "cannot read ('list' object has no attribute 'keys').")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from math_verify import parse, verify

    tok = args.tokenizer
    if tok is None:
        try:
            with open(os.path.join(args.checkpoint, "config.json")) as f:
                tok = json.load(f).get("_name_or_path") or None
        except Exception:
            tok = None
    print(f"[eval_vllm] model={args.checkpoint} tokenizer={tok or args.checkpoint}")
    llm = LLM(model=args.checkpoint, tokenizer=tok or args.checkpoint, dtype="bfloat16",
              max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem, trust_remote_code=True)

    results = {}
    for item in [b for b in args.benchmarks.split(",") if b]:
        which, n, k = parse_spec(item, args.n)
        probs, golds = load_problems(which, n)
        # k>1 rollouts/problem -> sample (temperature>0); k==1 -> greedy. SamplingParams
        # n=k returns k generations per prompt; accuracy = avg correctness over all k
        # (pass@1 averaged over k, e.g. avg@4). Also report pass@k (any correct).
        sp = SamplingParams(n=k, max_tokens=args.max_new_tokens,
                            temperature=(0.0 if k == 1 else args.temperature),
                            top_p=(1.0 if k == 1 else args.top_p))
        outs = llm.generate(make_prompts(probs, args.prompt_style), sp)
        ncorrect = 0; passk = 0; ntot = 0
        for o, g in zip(outs, golds):
            try: gp = parse(g)
            except Exception: gp = None
            hits = 0
            for cand in o.outputs:                     # k generations
                ntot += 1
                try:
                    if gp is not None and verify(gp, parse(cand.text)):
                        hits += 1
                except Exception:
                    pass
            ncorrect += hits
            passk += 1 if hits > 0 else 0
        acc = ncorrect / max(ntot, 1)                  # avg@k pass@1
        pk = passk / max(len(golds), 1)                # pass@k
        results[which] = {"n": len(golds), "k": k, "accuracy": acc, "pass_at_k": pk}
        print(f"[eval_vllm] {which}: avg@{k}={acc:.4f} pass@{k}={pk:.4f} "
              f"(problems={len(golds)}, rollouts={ntot})")

    rec = {"checkpoint": args.checkpoint, "max_new_tokens": args.max_new_tokens,
           "prompt_style": args.prompt_style, "results": results}
    # if checkpoint is a hub id (not a local dir), fall back to cwd
    default_out = (os.path.join(args.checkpoint, "eval_vllm.json")
                   if os.path.isdir(args.checkpoint) else "eval_vllm.json")
    out = args.out or default_out
    with open(out, "w") as f:
        json.dump(rec, f, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
