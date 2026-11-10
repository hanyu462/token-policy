"""Skill Program Decoder: language hidden states H_L -> skill tokens C.

    H_L (B,N,1024) -> proj -> K learnable skill queries (self-attn among slots + cross-attn to H_L) -> C (B,K,d)
"""
from __future__ import annotations

import torch
from torch import nn

from .config import ModelConfig


class SkillProgramDecoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        self.proj = nn.Sequential(nn.Linear(cfg.lang_dim, d), nn.LayerNorm(d), nn.Dropout(cfg.dropout))
        self.queries = nn.Parameter(torch.randn(cfg.num_skill_queries, d) * 0.02)
        layer = nn.TransformerDecoderLayer(d, cfg.num_heads, cfg.ffn_dim, cfg.dropout, batch_first=True, norm_first=True,
                                           activation=cfg.activation)
        self.dec = nn.TransformerDecoder(layer, cfg.num_skill_layers)
        self.norm = nn.LayerNorm(d)

    def forward(self, H: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """H (B,N,lang_dim), mask (B,N) True = valid -> C (B,K,d)."""
        mem = self.proj(H)
        q = self.queries.unsqueeze(0).expand(H.shape[0], -1, -1)
        return self.norm(self.dec(q, mem, memory_key_padding_mask=~mask))   # mask True = valid; PyTorch True = ignore
