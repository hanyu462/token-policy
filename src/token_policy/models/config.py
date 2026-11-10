"""Hyper-parameters shared by every network component (the only thing components import from each other)."""
from __future__ import annotations

from dataclasses import dataclass

from ..common.fsq import LEVELS, TOKEN_DIM


@dataclass
class ModelConfig:
    d_model: int = 512
    num_heads: int = 8
    ffn_dim: int = 2048
    num_skill_queries: int = 16
    num_skill_layers: int = 2
    num_state_layers: int = 2
    num_motion_layers: int = 4
    horizon: int = 40
    history: int = 10
    feature_dim: int = 150         # per-frame state input size; feature extractors must match it
    lang_dim: int = 1024
    dropout: float = 0.1
    activation: str = "relu"      # FFN of every Transformer layer (PyTorch default; the existing checkpoints use it)

    @classmethod
    def small(cls) -> "ModelConfig":
        return cls(d_model=256, num_heads=4, ffn_dim=1024, num_skill_layers=1, num_state_layers=1, num_motion_layers=2)

    # FSQ output grid (fixed by the frozen SONIC decoder, not a tunable hyper-parameter)
    token_dim = TOKEN_DIM
    n_classes = LEVELS
