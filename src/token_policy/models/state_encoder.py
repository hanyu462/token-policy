"""State Encoder: state-feature history X -> state tokens S.

    X (B,T_s,150) -> frame MLP + temporal pos -> Transformer encoder -> S (B,T_s,d)
"""
from __future__ import annotations

import torch
from torch import nn

from .config import ModelConfig


class StateEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        self.feature_dim = cfg.feature_dim
        self.mlp = nn.Sequential(nn.Linear(cfg.feature_dim, d), nn.LayerNorm(d), nn.GELU(), nn.Linear(d, d))
        self.pos = nn.Parameter(torch.randn(cfg.history, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, cfg.num_heads, cfg.ffn_dim, cfg.dropout, batch_first=True, norm_first=True,
                                           activation=cfg.activation)
        self.enc = nn.TransformerEncoder(layer, cfg.num_state_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """X (B,T_s,feature_dim) -> S (B,T_s,d).  T_s is fixed (= cfg.history); no variable-length history."""
        if X.dim() != 3 or X.shape[1] != self.pos.shape[0] or X.shape[2] != self.feature_dim:
            raise ValueError(f"StateEncoder expects X (B, {self.pos.shape[0]}, {self.feature_dim}), got {tuple(X.shape)}")
        return self.norm(self.enc(self.mlp(X) + self.pos))
