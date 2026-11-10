import pytest
import torch

from token_policy.models.config import ModelConfig
from token_policy.models.state_encoder import StateEncoder


def test_fixed_history_shape():
    cfg = ModelConfig.small()
    enc = StateEncoder(cfg)
    assert enc(torch.randn(2, cfg.history, cfg.feature_dim)).shape == (2, cfg.history, cfg.d_model)
    for bad in [(2, cfg.history - 1, cfg.feature_dim), (2, cfg.history, cfg.feature_dim + 1), (cfg.history, cfg.feature_dim)]:
        with pytest.raises(ValueError):
            enc(torch.randn(*bad))


def test_activation_is_explicit():
    cfg = ModelConfig.small()
    layer = StateEncoder(cfg).enc.layers[0]
    assert layer.activation is torch.nn.functional.relu       # existing checkpoints were trained with ReLU FFNs
