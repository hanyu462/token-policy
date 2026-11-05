"""StudentTokenSource: the trained Student as a SONIC token hook (receding horizon, one token per tick).

Every control tick: state -> 150-D features in the primitive-start frame -> 10-frame history -> model ->
k_hat (40 x 64) -> execute k_hat[0] / 16 only; the executed code becomes next tick's prev_token.
"""
from __future__ import annotations

import numpy as np
import torch

from ..policy.fsq import k_to_z
from .features import StartFrame, frame_features, history_window
from .text_encoder import TextCache


class StudentTokenSource:
    needs_base = False

    def __init__(self, model, norm, cache: TextCache, text: str, k_stand: np.ndarray, device: str = "cuda", T_s: int = 10):
        self.model, self.norm, self.cache, self.device, self.T_s = model.eval(), norm, cache, device, T_s
        self.k_stand = np.asarray(k_stand, int)
        self.set_text(text)
        self.reset()

    def set_text(self, text: str):
        self.text = text
        H, M = self.cache.batch([text])
        self.H, self.M = H.to(self.device), M.to(self.device)

    def reset(self):
        self.start, self.feats, self.prev_k, self.k_pred = None, [], self.k_stand.copy(), []

    def __call__(self, state: dict) -> np.ndarray:          # source(state) under policy.ReplaceToken
        if self.start is None:
            self.start = StartFrame.from_state(state)
        self.feats.append(frame_features(state, self.start, self.prev_k))
        F = np.stack(self.feats)
        X = torch.from_numpy(self.norm.apply(history_window(F, len(F) - 1, self.T_s)).astype(np.float32))[None].to(self.device)
        with torch.no_grad():
            k = self.model.predict_k(self.H, self.M, X)[0].cpu().numpy()      # (40, 64)
        self.prev_k = k[0].astype(int)
        self.k_pred.append(k[0].copy())
        return k_to_z(self.prev_k)
