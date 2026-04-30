from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .remi import REMI_TOKEN_END, REMI_TOKEN_PAD, REMI_VOCAB_SIZE


@dataclass
class ModelConfig:
    d_model: int = 128
    num_heads: int = 4
    num_kv_heads: int | None = None
    num_layers: int = 4
    ffn_dim: int = 512
    dropout: float = 0.1
    rope_theta: float = 10000.0


@dataclass
class SamplingConfig:
    temperature: float = 1.1
    top_k: int = 40
    top_p: float = 0.92
    prompt: list[int] = field(default_factory=lambda: [60, 281])
    repetition_penalty: float = 1.15
    constrained: bool = True


@dataclass
class TrainConfig:
    seed: int = 21
    data_dir: Path = Path("data")
    processed_dir: Path = Path("data/processed_maestro")
    models_dir: Path = Path("models")
    output_dir: Path = Path("outputs")
    maestro_url: str = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"
    seq_len: int = 512
    generation_len: int = 512
    batch_size: int = 8
    epochs: int = 1
    learning_rate: float | None = 3e-4
    optimizer: str = "adamw"
    muon_learning_rate: float = 0.02
    muon_momentum: float = 0.95
    weight_decay: float = 0.01
    warmup_steps: int = 200
    lr_schedule: str = "none"
    min_lr_ratio: float = 0.1
    early_stopping_patience: int | None = None
    grad_accum_steps: int = 1
    num_workers: int = 2
    mixed_precision: bool = True
    label_smoothing: float = 0.05
    resume_from: Path | None = None
    model: ModelConfig = field(default_factory=ModelConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    vocab_size: int = REMI_VOCAB_SIZE
    token_pad: int = REMI_TOKEN_PAD
    token_end: int = REMI_TOKEN_END

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)


def _coerce_path(value: str | Path) -> Path:
    return value if isinstance(value, Path) else Path(value)


def load_config(path: str | Path) -> TrainConfig:
    raw = json.loads(Path(path).read_text())
    model = ModelConfig(**raw.pop("model", {}))
    sampling = SamplingConfig(**raw.pop("sampling", {}))
    cfg = TrainConfig(**raw, model=model, sampling=sampling)
    for key in ("data_dir", "processed_dir", "models_dir", "output_dir"):
        setattr(cfg, key, _coerce_path(getattr(cfg, key)))
    cfg.vocab_size = REMI_VOCAB_SIZE
    cfg.token_pad = REMI_TOKEN_PAD
    cfg.token_end = REMI_TOKEN_END
    return cfg


def save_config(config: TrainConfig, path: str | Path) -> None:
    def convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if hasattr(value, "__dataclass_fields__"):
            return {k: convert(getattr(value, k)) for k in value.__dataclass_fields__}
        return value

    Path(path).write_text(json.dumps(convert(config), indent=2) + "\n")


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
