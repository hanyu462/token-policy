"""Action Decoder: motion latents M -> FSQ token logits A, and A -> (k, Z) for the frozen SONIC decoder.

    M (B,H,d) -> Linear(d, 64*32) -> A (B,H,64,32) logits over FSQ classes y = k + 16
    decode: k = argmax(A) - 16 in [-16, 15],  Z = k / 16
"""
from __future__ import annotations

import torch
from torch import nn

from ..common.fsq import HALF
from .config import ModelConfig


class ActionDecoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.token_dim, self.n_classes = cfg.token_dim, cfg.n_classes
        self.out = nn.Linear(cfg.d_model, cfg.token_dim * cfg.n_classes)

    def forward(self, M: torch.Tensor) -> torch.Tensor:
        """M (B,H,d) -> logits A (B,H,64,32)."""
        B, H, _ = M.shape
        return self.out(M).view(B, H, self.token_dim, self.n_classes)

    @staticmethod
    def decode(logits: torch.Tensor) -> tuple:
        """A (...,64,32) -> (k (...,64) int64, Z (...,64) float = k / 16)."""
        k = logits.argmax(dim=-1) - HALF
        return k, k.float() / HALF
