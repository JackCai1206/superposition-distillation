# Superposition Distillation

Does feeding a **frozen teacher superposed inputs** make logit distillation more
compute-efficient? Measured **iso-FLOP** to equal loss / downstream accuracy.

Two superposition methods (both = convex combos of token embeddings):
- **cross_seq** — mix two sequences position-wise (one forward pass carries two
  sequences). Closest prior art: MixKD (Liang et al. 2021), but for GLUE
  classification / data-aug, not autoregressive compute-packing.
- **token_merge** — mix `k` adjacent tokens of one sequence into one position
  (shorter sequence → fewer FLOPs via quadratic attention).

Superposed inputs are OOD for the frozen teacher, so training is two-stage:
**Stage 1** superposed (pure forward-KL), **Stage 2** normal-data recovery
(CE-mixed, WSD-scheduled KD).

## Recipe (confirmed canonical)
Forward-KL, temperature τ=2.0, WSD α schedule (Peng et al., ACL 2025,
arXiv 2410.16215). Pretrain: Qwen2.5-1.5B → 0.5B on FineWeb-Edu. Reasoning:
DeepSeek-R1-Distill-Qwen-7B → Qwen2.5-0.5B (shared tokenizer ⇒ white-box KD
valid) on OpenR1-Math; eval GSM8K / MATH-500.

## Layout
| file | role |
|---|---|
| `superpose.py` | superposition collators (unified ids×weights representation) |
| `kd_loss.py` | forward-KL + WSD α schedule |
| `flops.py` | analytic FLOP accounting for the iso-FLOP comparison |
| `model.py` | frozen teacher / student load + `inputs_embeds` plumbing |
| `data.py` | FineWeb-Edu / OpenR1 streams + superposed dispatcher |
| `train.py` | two-stage training loop |
| `eval.py` | LM loss + MATH-500 accuracy (math_verify) |
| `smoke_test.py` | CPU end-to-end check, no downloads |
| `scripts/train.slurm` | GPU job (della, partition `gpu`, account `arora`) |

## Run
```bash
.venv/bin/python smoke_test.py                              # CPU sanity
.venv/bin/python train.py --debug --method cross_seq        # CPU tiny end-to-end
sbatch scripts/train.slurm cross_seq                        # GPU run
```
Compare `none` vs `cross_seq` vs `token_merge` at equal `total_flops`.
