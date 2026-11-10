"""Checkpoint save / load for TokenPolicyModel.

Format: {"model": state_dict, "config": asdict(ModelConfig), "epoch", "val", ...} plus, when given,
"optimizer" / "scheduler" state dicts and "global_step".  Checkpoints saved before the auxiliary magnitude head
left the core model also carry mag.* weights; load_model drops them.
"""
from __future__ import annotations

import torch

from ..models.config import ModelConfig
from ..models.model import TokenPolicyModel

LEGACY_AUX_PREFIXES = ("mag.",)


def save_checkpoint(path: str, model, epoch: int, val: dict | None = None, optimizer=None, scheduler=None,
                    global_step: int | None = None, **extra):
    ck = {"model": model.state_dict(), "config": model.config_dict(), "epoch": epoch, "val": val, **extra}
    if optimizer is not None:
        ck["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        ck["scheduler"] = scheduler.state_dict()
    if global_step is not None:
        ck["global_step"] = global_step
    torch.save(ck, path)


def load_model(path: str, device="cpu") -> tuple:
    """(TokenPolicyModel in eval mode, checkpoint dict).  Strict except for legacy auxiliary-head weights."""
    ck = torch.load(path, map_location=device)
    model = TokenPolicyModel(ModelConfig(**ck["config"])).to(device)
    state = {k: v for k, v in ck["model"].items() if not k.startswith(LEGACY_AUX_PREFIXES)}
    model.load_state_dict(state, strict=True)
    return model.eval(), ck
