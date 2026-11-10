"""TokenPolicyModel: language-conditioned SONIC token policy (H_L, X) -> A -> Z.

The language backbone is NOT inside this model: its input is H_L, the frozen Qwen hidden states produced by
language_backbone.py (normally read from a TextCache).  This module only assembles

    H_L (B,N,1024)  -> skill_program_decoder -> C (B,K,d)
    X   (B,T_s,150) -> state_encoder         -> S (B,T_s,d)
    (C,S)           -> motion_transformer    -> M (B,H,d)
    M               -> action_decoder        -> A (B,H,64,32) logits;  ActionDecoder.decode(A) -> (k, Z = k/16)

"Student" is a training role (behaviour cloning of teacher data), not part of the architecture.
Module attribute names (skill, state, motion, head) are the checkpoint keys; do not rename them.  Task-specific
auxiliary heads are training-only and live outside models/; checkpoint I/O is in token_policy.train.
"""
from __future__ import annotations

from dataclasses import asdict

import torch
from torch import nn

from .action_decoder import ActionDecoder
from .config import ModelConfig
from .motion_transformer import MotionTransformer
from .skill_program_decoder import SkillProgramDecoder
from .state_encoder import StateEncoder


class TokenPolicyModel(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        self.skill = SkillProgramDecoder(self.cfg)
        self.state = StateEncoder(self.cfg)
        self.motion = MotionTransformer(self.cfg)
        self.head = ActionDecoder(self.cfg)

    def forward(self, H: torch.Tensor, mask: torch.Tensor, X: torch.Tensor) -> dict:
        C = self.skill(H, mask)
        S = self.state(X)
        M = self.motion(C, S)
        return {"logits": self.head(M), "C": C, "S": S, "M": M}

    @torch.no_grad()
    def predict_k(self, H, mask, X) -> torch.Tensor:
        """(B,H,64) integer FSQ codes k."""
        return ActionDecoder.decode(self.forward(H, mask, X)["logits"])[0]

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def config_dict(self) -> dict:
        return asdict(self.cfg)

