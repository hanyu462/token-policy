"""Optimizer and learning-rate schedule.

AdamW(lr, weight_decay); per optimizer step: linear warmup (s + 1) / warm for s < warm, then cosine decay
0.5 * (1 + cos(pi * (s - warm) / (total - warm))) with warm = int(warmup * epochs * steps_per_epoch).
"""
from __future__ import annotations

import math

import torch

from .config import TrainConfig


def make_optimizer(params, cfg: TrainConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)


def make_scheduler(opt: torch.optim.Optimizer, cfg: TrainConfig, steps_per_epoch: int) -> torch.optim.lr_scheduler.LambdaLR:
    total, warm = cfg.epochs * steps_per_epoch, int(cfg.warmup * cfg.epochs * steps_per_epoch)
    return torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(warm, 1) if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(total - warm, 1))))
