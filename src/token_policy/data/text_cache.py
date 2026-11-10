"""Disk cache of frozen language-backbone hidden states: text -> (H_L (N x hidden), mask (N,)).

Training never runs Qwen: every text is encoded once (fp32, one text per forward) and looked up by exact string.
The cache stores its encoder settings and refuses to load if they differ from the expected ones (no silent stale cache).
"""
from __future__ import annotations

import os

import torch

from ..models.language_backbone import CACHE_BATCH, CACHE_DTYPE, MAX_TOKENS, MODEL_ID, QwenTextEncoder


def expected_meta(hidden: int = 1024) -> dict:
    return {"model_id": MODEL_ID, "max_tokens": MAX_TOKENS, "hidden_size": hidden, "dtype": CACHE_DTYPE, "encode_batch": CACHE_BATCH}


class TextCache:
    """text -> (H_L (N x hidden), mask (N,)) stored in one .pt file; lookup by exact string.

    The per-entry mask is all True (padding is stripped when stored); it is kept in case tokenization changes.
    """

    def __init__(self, path: str, expect: dict | None = None):
        self.path = path
        self.H, self.mask, self.meta = {}, {}, None
        if os.path.exists(path):
            d = torch.load(path, map_location="cpu")
            self.H, self.mask, self.meta = d["H"], d["mask"], d.get("meta")
            exp = expect or expected_meta()
            if self.meta is None:
                raise ValueError(f"{path}: text cache has no encoder metadata (legacy cache); rebuild it")
            bad = {k: (self.meta.get(k), v) for k, v in exp.items() if self.meta.get(k) != v}
            if bad:
                raise ValueError(f"{path}: text cache built with different settings (cached, expected): {bad}")
        self.hidden = self.meta["hidden_size"] if self.meta else None

    def build(self, texts: list, encoder: QwenTextEncoder):
        meta = {"model_id": encoder.model_id, "max_tokens": MAX_TOKENS, "hidden_size": encoder.hidden,
                "dtype": str(encoder.dtype).replace("torch.", ""), "encode_batch": CACHE_BATCH}
        if meta != expected_meta(encoder.hidden):
            raise ValueError(f"encoder settings {meta} differ from the validated cache settings {expected_meta(encoder.hidden)}")
        if self.meta is not None and self.meta != meta:
            raise ValueError(f"cannot extend a cache built with {self.meta} using {meta}")
        for t in dict.fromkeys(texts):
            if t in self.H:
                continue
            H, M = encoder.encode([t])                     # one text per forward (see module docstring)
            n = int(M[0].sum())
            self.H[t], self.mask[t] = H[0, :n].clone(), M[0, :n].clone()
        self.meta, self.hidden = meta, encoder.hidden
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        torch.save({"H": self.H, "mask": self.mask, "meta": self.meta}, self.path)

    def __contains__(self, text: str) -> bool:
        return text in self.H

    def get(self, text: str) -> tuple:
        return self.H[text], self.mask[text]

    def batch(self, texts: list) -> tuple:
        """Padded (B, N_max, hidden) and (B, N_max) bool mask."""
        hs = [self.H[t] for t in texts]
        n = max(h.shape[0] for h in hs)
        H = torch.zeros(len(hs), n, hs[0].shape[-1]); M = torch.zeros(len(hs), n, dtype=torch.bool)
        for i, h in enumerate(hs):
            H[i, :h.shape[0]] = h; M[i, :h.shape[0]] = self.mask[texts[i]]
        return H, M
