"""Task-agnostic offline token metrics for evaluation / logging only (no gradients).

Task-specific views (motion / hold, STEP_FOOT foot / dx / dy) belong to the recipe that knows the task (student/);
closed-loop simulator metrics (touchdown error, clearance, drift) need the simulator and are not here.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def per_sample_metrics(logits: torch.Tensor, y: torch.Tensor) -> tuple:
    """Per sample (B,): CE (cell mean), exact FSQ-cell accuracy, mean |k_pred - k_gt|."""
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="none").view(y.shape).mean((1, 2))
    p = logits.argmax(-1)
    return ce, (p == y).float().mean((1, 2)), (p - y).abs().float().mean((1, 2))


def summarize(ev: dict) -> dict:
    """Means over all samples of an evaluate() result."""
    return {"ce": float(ev["ce"].mean()), "acc": float(ev["acc"].mean()), "mean_abs_dk": float(ev["dk"].mean())}
