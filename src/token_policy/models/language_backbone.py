"""Language backbone: frozen Qwen3-0.6B text encoder.

H_L = last hidden state (N x 1024) of the raw command text; nothing structured (d_cmd, lambda, cut) is ever added
to the text.  Since the texts are few, the cache is computed once and training never loads Qwen.

Numerics: bf16 with padded batches changes H_L of the same sentence by up to ~1.5 versus a single fp32 pass, so the
encoder runs in fp32 and caches are built one text per forward (data/text_cache.py; exact, built only once).
"""
from __future__ import annotations

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
