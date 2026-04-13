"""Flexible regression losses for ACGD forecasting models."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch
from torch import nn


class FlexibleForecastLoss(nn.Module):
    """Weighted combination of MAE, MSE, and optional period-sensitive terms."""

    def __init__(
        self,
        *,
        mae_weight: float = 0.5,
        mse_weight: float = 0.5,
        peak_enabled: bool = False,
        peak_threshold_quantile: float = 0.9,
        peak_weight: float = 1.5,
        renewable_sensitive_enabled: bool = False,
        renewable_sensitive_weight: float = 1.2,
    ):
        super().__init__()
        self.mae_weight = float(mae_weight)
        self.mse_weight = float(mse_weight)
        self.peak_enabled = bool(peak_enabled)
        self.peak_threshold_quantile = float(peak_threshold_quantile)
        self.peak_weight = float(peak_weight)
        self.renewable_sensitive_enabled = bool(renewable_sensitive_enabled)
        self.renewable_sensitive_weight = float(renewable_sensitive_weight)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "FlexibleForecastLoss":
        """Build the loss from the YAML ``loss`` section."""

        loss_config = config.get("loss", {})
        components = loss_config.get("components", {}) if isinstance(loss_config, Mapping) else {}
        peak = loss_config.get("peak_weight", {}) if isinstance(loss_config, Mapping) else {}
        renewable = loss_config.get("renewable_sensitive_weight", {}) if isinstance(loss_config, Mapping) else {}
        return cls(
            mae_weight=float(components.get("mae", 0.5)),
            mse_weight=float(components.get("mse", 0.5)),
            peak_enabled=bool(peak.get("enabled", False)),
            peak_threshold_quantile=float(peak.get("threshold_quantile", 0.9)),
            peak_weight=float(peak.get("weight", 1.5)),
            renewable_sensitive_enabled=bool(renewable.get("enabled", False)),
            renewable_sensitive_weight=float(renewable.get("weight", 1.2)),
        )

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        *,
        renewable_signal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the configured regression loss."""

        if predictions.shape != targets.shape:
            raise ValueError(f"Prediction shape {tuple(predictions.shape)} does not match target shape {tuple(targets.shape)}")

        # predictions, targets: [B, horizon] or [B, horizon, target_dim]
        absolute_error = torch.abs(predictions - targets)
        squared_error = (predictions - targets) ** 2
        weights = torch.ones_like(targets)
        # weights: same shape as targets.

        if self.peak_enabled:
            threshold = torch.quantile(targets.detach().flatten(), self.peak_threshold_quantile)
            weights = torch.where(targets >= threshold, weights * self.peak_weight, weights)
            # weights: larger on high-NCI or high-generation target periods.

        if self.renewable_sensitive_enabled and renewable_signal is not None:
            renewable_threshold = torch.quantile(renewable_signal.detach().flatten(), 0.75)
            renewable_mask = renewable_signal >= renewable_threshold
            while renewable_mask.ndim < weights.ndim:
                renewable_mask = renewable_mask.unsqueeze(-1)
            weights = torch.where(renewable_mask, weights * self.renewable_sensitive_weight, weights)
            # weights: larger for high-renewable hours if a compatible signal is passed.

        loss = torch.zeros((), dtype=predictions.dtype, device=predictions.device)
        if self.mae_weight:
            loss = loss + self.mae_weight * (weights * absolute_error).mean()
        if self.mse_weight:
            loss = loss + self.mse_weight * (weights * squared_error).mean()
        return loss

