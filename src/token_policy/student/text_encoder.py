"""Frozen language encoder (Qwen3-0.6B) and a disk cache of its hidden states.

H_L = last hidden state (N x 1024) of the raw command text; nothing structured (d_cmd, lambda, cut) is ever added
to the text.  Since the texts are few, the cache is computed once and training never loads Qwen.
"""
from __future__ import annotations

import os

import numpy as np
import torch

MODEL_ID = "Qwen/Qwen3-0.6B"
MAX_TOKENS = 32


class QwenTextEncoder:
    def __init__(self, model_id: str = MODEL_ID, device: str = "cuda", dtype=torch.bfloat16):
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device, self.hidden = device, int(self.model.config.hidden_size)

    @torch.no_grad()
    def encode(self, texts: list) -> tuple:
        """(H (B, N, hidden) float32, mask (B, N) bool, N <= MAX_TOKENS)."""
        texts = [t if t.strip() else self.tok.eos_token for t in texts]   # blank text -> a single EOS token
        enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_TOKENS).to(self.device)
        out = self.model(**enc).last_hidden_state.float()
        return out.cpu(), enc["attention_mask"].bool().cpu()


class TextCache:
    """text -> (H_L (N x hidden), mask (N,)) stored in one .pt file; lookup by exact string."""

    def __init__(self, path: str):
        self.path = path
        self.H, self.mask, self.index = {}, {}, {}
        if os.path.exists(path):
            d = torch.load(path)
            self.H, self.mask = d["H"], d["mask"]
        self.hidden = next(iter(self.H.values())).shape[-1] if self.H else None

    def build(self, texts: list, encoder: QwenTextEncoder, batch: int = 16):
        texts = [t for t in dict.fromkeys(texts) if t not in self.H]
        for i in range(0, len(texts), batch):
            chunk = texts[i:i + batch]
            H, M = encoder.encode(chunk)
            for t, h, m in zip(chunk, H, M):
                n = int(m.sum())
                self.H[t], self.mask[t] = h[:n].clone(), m[:n].clone()
        self.hidden = encoder.hidden
        torch.save({"H": self.H, "mask": self.mask, "model": encoder.model.config._name_or_path}, self.path)

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
            H[i, :h.shape[0]] = h; M[i, :h.shape[0]] = True
        return H, M
