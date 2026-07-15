"""Dataclass configs for the two-stage superposition-distillation runs."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    # reasoning teacher -> math student, shared Qwen2.5 tokenizer (white-box logit KD
    # valid). OpenMath-Nemotron-7B = Qwen2.5-Math-7B SFT'd on exactly OpenMathReasoning.
    # Student Qwen2.5-Math-1.5B-Instruct matches the teacher's math lineage; it ships
    # at 4096 ctx so we apply the Nemotron RoPE fix (theta 500000, 32K) for long CoT.
    teacher: str = "nvidia/OpenMath-Nemotron-7B"
    student: str = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    dtype: str = "bfloat16"
    student_max_pos: int | None = 32768       # RoPE context extension for the student
    student_rope_theta: float | None = 500000.0
    student_init: str = "pretrained"          # 'pretrained' | 'random' (from-scratch distill)


@dataclass
class DataConfig:
    # stage 1 (pretraining KD) source; stage 2 (reasoning KD) source
    pretrain_dataset: str = "HuggingFaceFW/fineweb-edu"
    pretrain_subset: str = "sample-10BT"
    reasoning_dataset: str = "nvidia/OpenMathReasoning"
    reasoning_split: str = "cot"     # R1-generated CoT traces (problem + generated_solution)
    seq_len: int = 1024
    max_examples: int | None = None
    # reasoning data: keep each (problem + full CoT solution) as ONE sequence
    # (truncate to seq_len, pad in the batch) instead of greedily packing -> the
    # student learns coherent reasoning instead of fragments split across blocks.
    reasoning_packed: bool = False


@dataclass
class SuperposeConfig:
    method: str = "cross_seq"        # none | cross_seq | token_merge
    mix_alpha: float = 1.0           # Beta(a,a); 1.0 ~ uniform lambda in (0,1)
    fixed_lambda: float | None = None  # if set, use a constant mixing weight
    merge_k: int = 2                 # token_merge factor


@dataclass
class KDConfig:
    temperature: float = 2.0
    alpha_max: float = 0.9           # WSD peak KD weight (normal-data stage)
    warmup_frac: float = 0.1
    decay_frac: float = 0.1


@dataclass
class TrainConfig:
    # stage 1 = superposed (OOD) ; stage 2 = normal data (recovery)
    stage1_steps: int = 2000
    stage2_steps: int = 500
    batch_size: int = 8
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 50
    grad_clip: float = 1.0
    log_every: int = 20
    eval_every: int = 500
    seed: int = 0
    output_dir: str = "outputs/run"
    device: str = "cuda"
    wandb: bool = False


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    superpose: SuperposeConfig = field(default_factory=SuperposeConfig)
    kd: KDConfig = field(default_factory=KDConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    debug: bool = False
