"""Student S0 modules.

    H_L (B,N,1024) -> proj -> Skill Decoder (K learnable queries, cross-attn) -> C (B,K,d)
    X   (B,T_s,150) -> frame MLP + temporal pos -> Transformer encoder -> S (B,T_s,d)
    [C;S] + type emb -> memory (B,K+T_s,d);  Q_future (H,d) -> Transformer decoder -> M (B,H,d)
    M -> Linear(d, 64*32) -> A (B,H,64,32) logits over FSQ classes y = k + 16
    C -> attention pool -> c_pool (B,d) -> MLP -> d_hat (B,)   (auxiliary magnitude head)
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
from torch import nn

from .features import FEATURE_DIM

TOKEN_DIM, N_CLASSES = 64, 32


@dataclass
class StudentConfig:
    d_model: int = 512
    num_heads: int = 8
    ffn_dim: int = 2048
    num_skill_queries: int = 16
    num_skill_layers: int = 2
    num_state_layers: int = 2
    num_motion_layers: int = 4
    horizon: int = 40
    history: int = 10
    feature_dim: int = FEATURE_DIM
    lang_dim: int = 1024
    dropout: float = 0.1

    @classmethod
    def small(cls) -> "StudentConfig":
        return cls(d_model=256, num_heads=4, ffn_dim=1024, num_skill_layers=1, num_state_layers=1, num_motion_layers=2)


class SkillDecoder(nn.Module):
    def __init__(self, cfg: StudentConfig):
        super().__init__()
        d = cfg.d_model
        self.proj = nn.Sequential(nn.Linear(cfg.lang_dim, d), nn.LayerNorm(d), nn.Dropout(cfg.dropout))
        self.queries = nn.Parameter(torch.randn(cfg.num_skill_queries, d) * 0.02)
        layer = nn.TransformerDecoderLayer(d, cfg.num_heads, cfg.ffn_dim, cfg.dropout, batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(layer, cfg.num_skill_layers)
        self.norm = nn.LayerNorm(d)

    def forward(self, H: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """H (B,N,lang_dim), mask (B,N) True = valid -> C (B,K,d)."""
        mem = self.proj(H)
        q = self.queries.unsqueeze(0).expand(H.shape[0], -1, -1)
        return self.norm(self.dec(q, mem, memory_key_padding_mask=~mask))


class StateEncoder(nn.Module):
    def __init__(self, cfg: StudentConfig):
        super().__init__()
        d = cfg.d_model
        self.mlp = nn.Sequential(nn.Linear(cfg.feature_dim, d), nn.LayerNorm(d), nn.GELU(), nn.Linear(d, d))
        self.pos = nn.Parameter(torch.randn(cfg.history, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, cfg.num_heads, cfg.ffn_dim, cfg.dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, cfg.num_state_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """X (B,T_s,feature_dim) -> S (B,T_s,d)."""
        return self.norm(self.enc(self.mlp(X) + self.pos))


class MotionTransformer(nn.Module):
    def __init__(self, cfg: StudentConfig):
        super().__init__()
        d = cfg.d_model
        self.queries = nn.Parameter(torch.randn(cfg.horizon, d) * 0.02)
        self.type_emb = nn.Parameter(torch.zeros(2, d))           # 0 = skill token, 1 = state token
        layer = nn.TransformerDecoderLayer(d, cfg.num_heads, cfg.ffn_dim, cfg.dropout, batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(layer, cfg.num_motion_layers)
        self.norm = nn.LayerNorm(d)

    def forward(self, C: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        """C (B,K,d), S (B,T_s,d) -> M (B,H,d)."""
        mem = torch.cat([C + self.type_emb[0], S + self.type_emb[1]], dim=1)
        q = self.queries.unsqueeze(0).expand(C.shape[0], -1, -1)
        return self.norm(self.dec(q, mem))


class ActionHead(nn.Module):
    def __init__(self, cfg: StudentConfig):
        super().__init__()
        self.out = nn.Linear(cfg.d_model, TOKEN_DIM * N_CLASSES)

    def forward(self, M: torch.Tensor) -> torch.Tensor:
        """M (B,H,d) -> logits (B,H,64,32)."""
        B, H, _ = M.shape
        return self.out(M).view(B, H, TOKEN_DIM, N_CLASSES)


class MagnitudeHead(nn.Module):
    """Attention pooling of C with one learnable query, then an MLP -> d_hat (in units of 0.1 m)."""

    def __init__(self, cfg: StudentConfig):
        super().__init__()
        d = cfg.d_model
        self.query = nn.Parameter(torch.randn(1, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, cfg.num_heads, dropout=cfg.dropout, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 1))

    def forward(self, C: torch.Tensor) -> tuple:
        q = self.query.unsqueeze(0).expand(C.shape[0], -1, -1)
        pooled, _ = self.attn(q, C, C)
        pooled = pooled[:, 0]
        return self.mlp(pooled)[:, 0], pooled


class StudentS0(nn.Module):
    def __init__(self, cfg: StudentConfig | None = None):
        super().__init__()
        self.cfg = cfg or StudentConfig()
        self.skill = SkillDecoder(self.cfg)
        self.state = StateEncoder(self.cfg)
        self.motion = MotionTransformer(self.cfg)
        self.head = ActionHead(self.cfg)
        self.mag = MagnitudeHead(self.cfg)

    def forward(self, H: torch.Tensor, mask: torch.Tensor, X: torch.Tensor) -> dict:
        C = self.skill(H, mask)
        S = self.state(X)
        M = self.motion(C, S)
        d_hat, c_pool = self.mag(C)
        return {"logits": self.head(M), "d_hat": d_hat, "c_pool": c_pool, "C": C, "S": S, "M": M}

    @torch.no_grad()
    def predict_k(self, H, mask, X) -> torch.Tensor:
        """(B,H,64) integer FSQ codes k = argmax - 16."""
        return self.forward(H, mask, X)["logits"].argmax(-1) - 16

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def config_dict(self) -> dict:
        return asdict(self.cfg)
