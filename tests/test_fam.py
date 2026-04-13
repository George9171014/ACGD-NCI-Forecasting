"""Tests for the Tier 2 feature attention mechanism."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.models.attention_fam import FeatureAttentionMechanism


def test_fam_forward_shapes_and_softmax_weights() -> None:
    model = FeatureAttentionMechanism(input_dim=7, hidden_dim=16, dropout=0.0)
    x = torch.randn(4, 72, 7)

    output = model(x)

    assert output.weighted_features.shape == (4, 72, 7)
    assert output.weights.shape == (4, 7)
    assert output.logits.shape == (4, 7)
    assert torch.allclose(output.weights.sum(dim=-1), torch.ones(4), atol=1e-5)
