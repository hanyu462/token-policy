"""Student losses and offline metrics.

    L = L_action + w_mag * L_mag  (+ w_ord * L_ord + w_para * L_para, default weights 0)
    L_action = CE(logits.reshape(-1, 32), y.reshape(-1)), cell mean over B x H x 64  (label smoothing default 0)
    L_mag    = Huber(d_hat, d_cmd / 0.1), delta 0.5   (decimetres so the term is O(1))
    L_ord    = Huber(E_softmax[k] / 16, k / 16)       (optional ordinal term)
    L_para   = 1 - cos(c_pool_a, c_pool_b) over pairs with equal d_cmd in the batch (optional)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

K_VALUES = torch.arange(-16, 16, dtype=torch.float32)


@dataclass
class LossConfig:
    w_mag: float = 0.1
    w_ord: float = 0.0
    w_para: float = 0.0
    label_smoothing: float = 0.0
    mag_scale: float = 0.1       # d_cmd unit for the magnitude head (m)


def action_ce(logits: torch.Tensor, y: torch.Tensor, label_smoothing: float = 0.0) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), label_smoothing=label_smoothing)


def ordinal_huber(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    p = logits.softmax(-1)
    ek = (p * K_VALUES.to(logits.device)).sum(-1) / 16.0
    return F.huber_loss(ek, (y.float() - 16.0) / 16.0, delta=0.1)


def paraphrase_cos(c_pool: torch.Tensor, d_cmd: torch.Tensor) -> torch.Tensor:
    same = (d_cmd[:, None] - d_cmd[None, :]).abs() < 1e-6
    same.fill_diagonal_(False)
    if not same.any():
        return c_pool.new_zeros(())
    c = F.normalize(c_pool, dim=-1)
    cos = c @ c.T
    return (1.0 - cos[same]).mean()


def compute_losses(out: dict, batch: dict, cfg: LossConfig) -> dict:
    y = batch["y"]
    l_action = action_ce(out["logits"], y, cfg.label_smoothing)
    l_mag = F.huber_loss(out["d_hat"], batch["d_cmd"] / cfg.mag_scale, delta=0.5)
    total = l_action + cfg.w_mag * l_mag
    terms = {"action": l_action, "mag": l_mag}
    if cfg.w_ord > 0:
        terms["ord"] = ordinal_huber(out["logits"], y); total = total + cfg.w_ord * terms["ord"]
    if cfg.w_para > 0:
        terms["para"] = paraphrase_cos(out["c_pool"], batch["d_cmd"]); total = total + cfg.w_para * terms["para"]
    terms["total"] = total
    return terms


@torch.no_grad()
def offline_metrics(logits: torch.Tensor, y: torch.Tensor, is_motion: torch.Tensor | None = None) -> dict:
    """exact cell accuracy, mean |k_pred - k_gt|, CE split into motion / standing samples."""
    pred = logits.argmax(-1)
    m = {"acc": (pred == y).float().mean().item(), "mean_abs_dk": (pred - y).abs().float().mean().item(),
         "ce": action_ce(logits, y).item()}
    if is_motion is not None and is_motion.any():
        m["ce_motion"] = action_ce(logits[is_motion], y[is_motion]).item()
        m["acc_motion"] = (pred[is_motion] == y[is_motion]).float().mean().item()
        if (~is_motion).any():
            m["ce_standing"] = action_ce(logits[~is_motion], y[~is_motion]).item()
            m["acc_standing"] = (pred[~is_motion] == y[~is_motion]).float().mean().item()
    return m
