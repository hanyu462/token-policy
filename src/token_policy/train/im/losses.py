"""Imitation losses (used for gradients).

    L_action = CE(logits.reshape(-1, 32), y.reshape(-1)), mean over B x H x 64 cells (label smoothing default 0)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def action_ce(logits: torch.Tensor, y: torch.Tensor, label_smoothing: float = 0.0) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), label_smoothing=label_smoothing)
