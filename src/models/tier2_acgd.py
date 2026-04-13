"""Tier 2 ACGD CNN-GRU-DSAM model for day-ahead NCI forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import torch
from torch import nn

from src.models.attention_fam import FeatureAttentionMechanism
from src.models.attention_tam import TemporalAttentionMechanism
from src.models.cnn_blocks import TemporalCNNEncoder
from src.models.gru_blocks import GRUEncoder


@dataclass
class ACGDAttentionDiagnostics:
    """Attention diagnostics that can be saved for interpretability analysis."""

    feature_weights: torch.Tensor
    feature_logits: torch.Tensor
    temporal_weights: torch.Tensor
    temporal_scores: torch.Tensor


@dataclass
class ACGDModelOutput:
    """Tier 2 model output with optional attention diagnostics."""

    predictions: torch.Tensor
    diagnostics: Optional[ACGDAttentionDiagnostics] = None


class ForecastHead(nn.Module):
    """Fully connected multi-step regression head."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        *,
        output_horizon: int = 24,
        target_dim: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if output_horizon <= 0:
            raise ValueError("output_horizon must be positive.")
        if target_dim <= 0:
            raise ValueError("target_dim must be positive.")

        self.output_horizon = int(output_horizon)
        self.target_dim = int(target_dim)
        self.output_size = self.output_horizon * self.target_dim

        layers: list[nn.Module] = []
        previous_dim = int(input_dim)
        for hidden_dim in hidden_dims:
            hidden_dim = int(hidden_dim)
            if hidden_dim <= 0:
                raise ValueError("All head hidden dimensions must be positive.")
            layers.extend(
                [
                    nn.Linear(previous_dim, hidden_dim),
                    nn.LeakyReLU(negative_slope=0.01),
                    nn.Dropout(dropout),
                ]
            )
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, self.output_size))

        self.head = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Apply Xavier initialization to head linear layers."""

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Map the TAM context vector to a day-ahead forecast.

        Args:
            context: Tensor with shape ``[batch, hidden]``.

        Returns:
            ``[batch, horizon]`` for single-target forecasting, or
            ``[batch, horizon, target_dim]`` for future multi-node/multi-target
            forecasting.
        """

        if context.ndim != 2:
            raise ValueError(f"ForecastHead expects [batch, hidden], got {tuple(context.shape)}")

        # context: [B, H]
        flat_predictions = self.head(context)
        # flat_predictions: [B, horizon * target_dim]

        if self.target_dim == 1:
            predictions = flat_predictions.view(context.size(0), self.output_horizon)
            # predictions: [B, horizon]
        else:
            predictions = flat_predictions.view(context.size(0), self.output_horizon, self.target_dim)
            # predictions: [B, horizon, target_dim]
        return predictions


class Tier2ACGDForecaster(nn.Module):
    """CNN-GRU-DSAM forecaster for Tier 2 day-ahead NCI prediction.

    Pipeline:
    FAM weights raw Tier 2 features, the CNN extracts local temporal patterns,
    the GRU models sequential dynamics, and TAM aggregates hidden states into a
    context vector for direct 24-step NCI regression.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        output_horizon: int = 24,
        target_dim: int = 1,
        cnn_channels: Sequence[int] = (32, 64),
        cnn_kernel_size: int = 3,
        cnn_dropout: float = 0.1,
        cnn_batch_norm: bool = True,
        fam_hidden_dim: int = 64,
        fam_dropout: float = 0.1,
        fam_normalize: str = "softmax",
        gru_hidden_dim: int = 128,
        gru_num_layers: int = 2,
        gru_dropout: float = 0.2,
        gru_bidirectional: bool = False,
        tam_scoring: str = "additive",
        tam_hidden_dim: int = 64,
        tam_dropout: float = 0.1,
        head_hidden_dims: Sequence[int] = (128,),
        head_dropout: float = 0.2,
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")

        self.input_dim = int(input_dim)
        self.output_horizon = int(output_horizon)
        self.target_dim = int(target_dim)

        self.fam = FeatureAttentionMechanism(
            input_dim=self.input_dim,
            hidden_dim=fam_hidden_dim,
            dropout=fam_dropout,
            normalize=fam_normalize,
            preserve_scale=True,
        )
        self.cnn = TemporalCNNEncoder(
            input_dim=self.input_dim,
            channels=cnn_channels,
            kernel_size=cnn_kernel_size,
            dropout=cnn_dropout,
            batch_norm=cnn_batch_norm,
        )
        self.gru = GRUEncoder(
            input_dim=self.cnn.output_dim,
            hidden_dim=gru_hidden_dim,
            num_layers=gru_num_layers,
            dropout=gru_dropout,
            bidirectional=gru_bidirectional,
        )
        self.tam = TemporalAttentionMechanism(
            hidden_dim=self.gru.output_dim,
            attention_dim=tam_hidden_dim,
            dropout=tam_dropout,
            scoring=tam_scoring,
        )
        self.head = ForecastHead(
            input_dim=self.gru.output_dim,
            hidden_dims=head_hidden_dims,
            output_horizon=output_horizon,
            target_dim=target_dim,
            dropout=head_dropout,
        )

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        input_dim: Optional[int] = None,
        target_dim: int = 1,
    ) -> "Tier2ACGDForecaster":
        """Build the Tier 2 model from a loaded YAML config."""

        model_config = config.get("model", {})
        if not isinstance(model_config, Mapping):
            raise ValueError("Expected config['model'] to be a mapping.")

        data_config = config.get("data", {})
        if input_dim is None:
            configured_input = model_config.get("input_channels")
            if configured_input is not None:
                input_dim = int(configured_input)
            else:
                input_dim = len(data_config.get("feature_columns", []))
        if input_dim is None or input_dim <= 0:
            raise ValueError("input_dim could not be inferred. Set model.input_channels or data.feature_columns.")

        cnn_config = model_config.get("cnn", {})
        fam_config = model_config.get("fam", {})
        gru_config = model_config.get("gru", {})
        tam_config = model_config.get("tam", {})
        head_config = model_config.get("head", {})

        return cls(
            input_dim=int(input_dim),
            output_horizon=int(head_config.get("output_horizon", 24)),
            target_dim=int(head_config.get("target_dim", target_dim)),
            cnn_channels=tuple(cnn_config.get("channels", (32, 64))),
            cnn_kernel_size=int(cnn_config.get("kernel_size", 3)),
            cnn_dropout=float(cnn_config.get("dropout", 0.1)),
            cnn_batch_norm=bool(cnn_config.get("batch_norm", True)),
            fam_hidden_dim=int(fam_config.get("hidden_dim", 64)),
            fam_dropout=float(fam_config.get("dropout", 0.1)),
            fam_normalize=str(fam_config.get("normalize", "softmax")),
            gru_hidden_dim=int(gru_config.get("hidden_dim", 128)),
            gru_num_layers=int(gru_config.get("num_layers", 2)),
            gru_dropout=float(gru_config.get("dropout", 0.2)),
            gru_bidirectional=bool(gru_config.get("bidirectional", False)),
            tam_scoring=str(tam_config.get("scoring", "additive")),
            tam_hidden_dim=int(tam_config.get("hidden_dim", 64)),
            tam_dropout=float(tam_config.get("dropout", 0.1)),
            head_hidden_dims=tuple(head_config.get("hidden_dims", (128,))),
            head_dropout=float(head_config.get("dropout", 0.2)),
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        temporal_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> torch.Tensor | ACGDModelOutput:
        """Run the Tier 2 ACGD forecast model.

        Args:
            x: Tier 2 input tensor with shape ``[batch, lookback, features]``.
            temporal_mask: Optional valid-time mask with shape ``[batch, lookback]``.
            return_attention: When true, return predictions plus FAM/TAM weights.

        Returns:
            Predictions ``[batch, horizon]`` for single-target NCI forecasting,
            or ``ACGDModelOutput`` when ``return_attention=True``.
        """

        if x.ndim != 3:
            raise ValueError(f"Tier2ACGDForecaster expects [batch, lookback, features], got {tuple(x.shape)}")
        if x.size(-1) != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {x.size(-1)}")

        # x: [B, T, F]
        fam_output = self.fam(x)
        # fam_output.weighted_features: [B, T, F]
        # fam_output.weights: [B, F]

        cnn_features = self.cnn(fam_output.weighted_features)
        # cnn_features: [B, T, C]

        gru_output = self.gru(cnn_features)
        # gru_output.sequence: [B, T, H]
        # gru_output.final_hidden: [L, B, H_base] or [L * 2, B, H_base]

        tam_output = self.tam(gru_output.sequence, mask=temporal_mask)
        # tam_output.context: [B, H]
        # tam_output.weights: [B, T]

        predictions = self.head(tam_output.context)
        # predictions: [B, horizon] for target_dim=1
        # predictions: [B, horizon, target_dim] for target_dim>1

        if not return_attention:
            return predictions

        diagnostics = ACGDAttentionDiagnostics(
            feature_weights=fam_output.weights,
            feature_logits=fam_output.logits,
            temporal_weights=tam_output.weights,
            temporal_scores=tam_output.scores,
        )
        return ACGDModelOutput(predictions=predictions, diagnostics=diagnostics)

