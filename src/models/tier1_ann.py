"""Tier 1 lightweight ANN models for renewable generation forecasting."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import nn


class Tier1ANNForecaster(nn.Module):
    """Lightweight MLP for 24-hour PV or WT generation forecasting."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (256, 128, 64),
        *,
        output_horizon: int = 24,
        target_dim: int = 1,
        dropout: float = 0.2,
        negative_slope: float = 0.01,
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if output_horizon <= 0:
            raise ValueError("output_horizon must be positive.")
        if target_dim <= 0:
            raise ValueError("target_dim must be positive.")

        self.input_dim = int(input_dim)
        self.output_horizon = int(output_horizon)
        self.target_dim = int(target_dim)
        self.output_size = self.output_horizon * self.target_dim

        layers: list[nn.Module] = []
        previous_dim = self.input_dim
        for hidden_dim in hidden_dims:
            hidden_dim = int(hidden_dim)
            if hidden_dim <= 0:
                raise ValueError("All hidden dimensions must be positive.")
            layers.extend(
                [
                    nn.Linear(previous_dim, hidden_dim),
                    nn.LeakyReLU(negative_slope=negative_slope),
                    nn.Dropout(dropout),
                ]
            )
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, self.output_size))

        self.network = nn.Sequential(*layers)
        self.reset_parameters()

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, input_dim: int) -> "Tier1ANNForecaster":
        """Build a Tier 1 ANN from YAML config."""

        model_config = config.get("model", {})
        if not isinstance(model_config, Mapping):
            raise ValueError("Expected config['model'] to be a mapping.")

        return cls(
            input_dim=input_dim,
            hidden_dims=tuple(model_config.get("hidden_dims", (256, 128, 64))),
            output_horizon=int(model_config.get("output_horizon", 24)),
            target_dim=int(model_config.get("target_dim", 1)),
            dropout=float(model_config.get("dropout", 0.2)),
            negative_slope=float(model_config.get("negative_slope", 0.01)),
        )

    def reset_parameters(self) -> None:
        """Apply Xavier initialization to linear layers and zero biases."""

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forecast renewable generation from a lookback window.

        Args:
            x: Tensor with shape ``[batch, lookback, features]`` or already
                flattened ``[batch, lookback * features]``.

        Returns:
            ``[batch, horizon]`` for a single renewable source.
        """

        if x.ndim == 3:
            # x: [B, T, F]
            flat = x.reshape(x.size(0), -1)
            # flat: [B, T * F]
        elif x.ndim == 2:
            # x: [B, T * F]
            flat = x
        else:
            raise ValueError(f"Tier1ANNForecaster expects [batch, time, features] or [batch, features], got {tuple(x.shape)}")

        if flat.size(-1) != self.input_dim:
            raise ValueError(f"Expected flattened input_dim={self.input_dim}, got {flat.size(-1)}")

        predictions_flat = self.network(flat)
        # predictions_flat: [B, horizon * target_dim]

        if self.target_dim == 1:
            predictions = predictions_flat.view(flat.size(0), self.output_horizon)
            # predictions: [B, horizon]
        else:
            predictions = predictions_flat.view(flat.size(0), self.output_horizon, self.target_dim)
            # predictions: [B, horizon, target_dim]
        return predictions

