"""Training configuration shared by every training algorithm (no architecture settings: see models/config.py)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainConfig:
    epochs: int = 100
    batch: int = 256
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup: float = 0.05         # fraction of all optimizer steps with linear warmup, then cosine to 0
    grad_clip: float = 1.0       # max global grad norm
    patience: int = 15           # early stopping: epochs without validation improvement
    seed: int = 0
