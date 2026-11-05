"""SONIC decoder re-implemented in torch for batched inference (the ONNX export is batch-1).

The exported decoder is a plain MLP: 994 -> 4096 -> 4096 -> 2048 -> 2048 -> 1024 -> 1024 -> 512 -> 512 -> 29
with SiLU (x * sigmoid(x)) between layers; weights are read straight from the ONNX initializers.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class TorchDecoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        mods = []
        for i, (w, b) in enumerate(layers):
            lin = nn.Linear(w.shape[0], w.shape[1])
            with torch.no_grad():
                lin.weight.copy_(torch.from_numpy(np.ascontiguousarray(w.T)))
                lin.bias.copy_(torch.from_numpy(np.array(b)))
            mods.append(lin)
            if i < len(layers) - 1:
                mods.append(nn.SiLU())
        self.net = nn.Sequential(*mods)
        self.in_dim, self.out_dim = layers[0][0].shape[0], layers[-1][0].shape[1]

    @classmethod
    def from_onnx(cls, path: str, device="cpu", dtype=torch.float32) -> "TorchDecoder":
        import onnx
        from onnx import numpy_helper
        m = onnx.load(path)
        inits = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
        layers = []
        for n in m.graph.node:
            if n.op_type == "MatMul":
                w = inits[n.input[1]]
                layers.append([w, None])
            elif n.op_type == "Add" and layers and layers[-1][1] is None:
                b = inits[n.input[0]] if n.input[0] in inits else inits[n.input[1]]
                layers[-1][1] = b
        return cls([(w, b) for w, b in layers]).to(device=device, dtype=dtype).eval()

    @torch.no_grad()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    @torch.no_grad()
    def decode_numpy(self, obs: np.ndarray) -> np.ndarray:
        """(N, 994) float -> (N, 29) float32, on the module's device."""
        p = next(self.parameters())
        x = torch.as_tensor(np.asarray(obs, np.float32), device=p.device, dtype=p.dtype)
        return self.net(x).float().cpu().numpy()
