"""Feature Attention Mechanism (FAM) for Tier 2 ACGD forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass
class FeatureAttentionOutput:
    """FAM output with weighted features and interpretable feature weights."""

    weighted_features: torch.Tensor
    weights: torch.Tensor
    logits: torch.Tensor


class FeatureAttentionMechanism(nn.Module):
    """Learn feature-importance weights for each sample in a temporal window.

    FAM summarizes each feature over the lookback window, scores feature
    relevance with a small trainable MLP, and applies the resulting gates back
    to every time step. A softmax over features gives interpretable weights that
    sum to one per sample; optional internal rescaling preserves the rough
    magnitude of the original input sequence for the downstream CNN-GRU
    backbone without changing the returned attention distribution.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        *,
        normalize: str = "softmax",
        preserve_scale: bool = True,
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if normalize not in {"softmax", "sigmoid"}:
            raise ValueError("normalize must be either 'softmax' or 'sigmoid'.")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.normalize = normalize
        self.preserve_scale = bool(preserve_scale)

        self.scorer = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.input_dim),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Apply Xavier initialization to linear layers and zero biases."""

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, *, return_logits: bool = True) -> FeatureAttentionOutput:
        """Apply feature attention to a temporal feature tensor.

        Args:
            x: Input tensor with shape ``[batch, time, features]``.
            return_logits: Kept explicit for downstream diagnostics.

        Returns:
            FeatureAttentionOutput with:
            - weighted_features: ``[batch, time, features]``
            - weights: ``[batch, features]``
            - logits: ``[batch, features]``
        """

        if x.ndim != 3:
            raise ValueError(f"FAM expects x with shape [batch, time, features], got {tuple(x.shape)}")
        if x.size(-1) != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} input features, got {x.size(-1)}")

        # x: [B, T, F]
        feature_summary = x.mean(dim=1)
        # feature_summary: [B, F]

        logits = self.scorer(feature_summary)
        # logits: [B, F]

        if self.normalize == "softmax":
            weights = torch.softmax(logits, dim=-1)
            # weights: [B, F], sum over F equals 1.
            gates = weights * self.input_dim if self.preserve_scale else weights
            # gates: [B, F], optionally scaled so equal attention gives gate value near 1.
        else:
            weights = torch.sigmoid(logits)
            # weights: [B, F], independent feature gates in [0, 1].
            gates = weights
            # gates: [B, F], same values as weights for sigmoid gating.

        weighted_features = x * gates.unsqueeze(1)
        # gates.unsqueeze(1): [B, 1, F]
        # weighted_features: [B, T, F]

        if not return_logits:
            logits = torch.empty(0, device=x.device, dtype=x.dtype)
        return FeatureAttentionOutput(weighted_features=weighted_features, weights=weights, logits=logits)
