import torch

from token_policy.models.action_decoder import ActionDecoder
from token_policy.models.config import ModelConfig
from token_policy.common import fsq


def test_shapes_and_decode():
    cfg = ModelConfig.small()
    dec = ActionDecoder(cfg)
    A = dec(torch.randn(3, cfg.horizon, cfg.d_model))
    assert A.shape == (3, cfg.horizon, fsq.TOKEN_DIM, fsq.LEVELS)
    k, z = ActionDecoder.decode(A)
    assert k.dtype == torch.int64 and k.min() >= fsq.K_MIN and k.max() <= fsq.K_MAX
    assert torch.equal(z, k.float() / 16)
    assert fsq.on_grid(z.numpy())


def test_decode_class_mapping():
    logits = torch.full((1, 1, 64, 32), -10.0)
    logits[..., 0] = 10.0                     # class 0 -> k = -16
    logits[0, 0, 5, 31] = 20.0                # class 31 -> k = 15
    k, z = ActionDecoder.decode(logits)
    assert k[0, 0, 0] == -16 and k[0, 0, 5] == 15 and z[0, 0, 5] == 15 / 16
