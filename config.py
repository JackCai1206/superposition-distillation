"""Dataclass configs for the two-stage superposition-distillation runs."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    teacher: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    student: str = "Qwen/Qwen2.5-0.5B"
    dtype: str = "bfloat16"


@dataclass
class DataConfig:
    # stage 1 (pretraining KD) source; stage 2 (reasoning KD) source
    pretrain_dataset: str = "HuggingFaceFW/fineweb-edu"
    pretrain_subset: str = "sample-10BT"
    reasoning_dataset: str = "nvidia/OpenMathReasoning"
    reasoning_split: str = "cot"     # R1-generated CoT traces (problem + generated_solution)
    seq_len: int = 1024
    max_examples: int | None = None


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
