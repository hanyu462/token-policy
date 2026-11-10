"""Motion Transformer: (C, S) -> future motion latents M.

    [C;S] + type emb -> memory (B,K+T_s,d);  Q_future (H,d) -> Transformer decoder -> M (B,H,d)

Non-causal: the H future queries attend to each other (one-shot action chunk, not autoregressive); no target token
is fed back, so there is no future-target leakage.  The queries double as learned future-step positions.
"""
from __future__ import annotations

import torch
from torch import nn

from .config import ModelConfig


class MotionTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        self.queries = nn.Parameter(torch.randn(cfg.horizon, d) * 0.02)
        self.type_emb = nn.Parameter(torch.zeros(2, d))           # 0 = skill token, 1 = state token
        layer = nn.TransformerDecoderLayer(d, cfg.num_heads, cfg.ffn_dim, cfg.dropout, batch_first=True, norm_first=True,
                                           activation=cfg.activation)
        self.dec = nn.TransformerDecoder(layer, cfg.num_motion_layers)
        self.norm = nn.LayerNorm(d)

    def forward(self, C: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        """C (B,K,d), S (B,T_s,d) -> M (B,H,d)."""
        mem = torch.cat([C + self.type_emb[0], S + self.type_emb[1]], dim=1)
        q = self.queries.unsqueeze(0).expand(C.shape[0], -1, -1)
        return self.norm(self.dec(q, mem))
