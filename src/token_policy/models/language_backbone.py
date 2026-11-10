"""Language backbone: frozen Qwen3-0.6B text encoder and a disk cache of its hidden states.

H_L = last hidden state (N x 1024) of the raw command text; nothing structured (d_cmd, lambda, cut) is ever added
to the text.  Since the texts are few, the cache is computed once and training never loads Qwen.

Numerics: bf16 with padded batches changes H_L of the same sentence by up to ~1.5 versus a single fp32 pass, so the
encoder runs in fp32 and the cache is built one text per forward (exact; the cache is built only once).  The cache
stores its encoder settings and refuses to load if they differ from the expected ones (no silent stale cache).
"""
from __future__ import annotations

import os

import torch

MODEL_ID = "Qwen/Qwen3-0.6B"
MAX_TOKENS = 32
CACHE_DTYPE = "float32"
CACHE_BATCH = 1


class QwenTextEncoder:
    def __init__(self, model_id: str = MODEL_ID, device: str = "cuda", dtype=torch.float32):
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id, dtype=dtype).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model_id, self.device, self.dtype = model_id, device, dtype
        self.hidden = int(self.model.config.hidden_size)

    @torch.no_grad()
    def encode(self, texts: list) -> tuple:
        """(H (B, N, hidden) float32, mask (B, N) bool, N <= MAX_TOKENS)."""
        texts = [t if t.strip() else self.tok.eos_token for t in texts]   # blank text -> a single EOS token
        enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_TOKENS).to(self.device)
        out = self.model(**enc).last_hidden_state.float()
        return out.cpu(), enc["attention_mask"].bool().cpu()


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
