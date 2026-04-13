"""Temporal Attention Mechanism (TAM) for Tier 2 ACGD forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass
class TemporalAttentionOutput:
    """TAM output with a context vector and temporal attention diagnostics."""

    context: torch.Tensor
    weights: torch.Tensor
    scores: torch.Tensor


class TemporalScorer(nn.Module):
    """Base class for trainable TAM scoring modules."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class AdditiveTemporalScorer(TemporalScorer):
    """Additive temporal scoring, similar to Bahdanau-style attention."""

    def __init__(self, hidden_dim: int, attention_dim: int, dropout: float = 0.1):
        super().__init__()
        self.projection = nn.Linear(hidden_dim, attention_dim)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        self.score_vector = nn.Linear(attention_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Apply Xavier initialization to scorer layers."""

        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        nn.init.xavier_uniform_(self.score_vector.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Score each temporal hidden state.

        Args:
            hidden_states: GRU output tensor with shape ``[batch, time, hidden]``.

        Returns:
            scores with shape ``[batch, time]``.
        """

        # hidden_states: [B, T, H]
        projected = self.projection(hidden_states)
        # projected: [B, T, A]

        activated = self.activation(projected)
        # activated: [B, T, A]

        dropped = self.dropout(activated)
        # dropped: [B, T, A]

        scores = self.score_vector(dropped).squeeze(-1)
        # scores before squeeze: [B, T, 1]
        # scores after squeeze: [B, T]
        return scores


class LinearTemporalScorer(TemporalScorer):
    """Simple linear temporal scoring baseline."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.score.weight)
        nn.init.zeros_(self.score.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Score hidden states with a single learned linear projection."""

        # hidden_states: [B, T, H]
        scores = self.score(hidden_states).squeeze(-1)
        # scores before squeeze: [B, T, 1]
        # scores after squeeze: [B, T]
        return scores


class TemporalAttentionMechanism(nn.Module):
    """Weighted aggregation over GRU hidden states.

    The accepted paper may describe Spearman-style temporal relevance. Exact
    Spearman ranking is not ideal as a default training objective here because
    hard rank/sort operations are non-smooth and target-dependent. The default
    implementation uses additive trainable attention as a differentiable
    approximation: it learns which hourly hidden states are most relevant for
    the day-ahead forecast while keeping the scoring module swappable for future
    rank-aware or LLM-assisted variants.
    """

    def __init__(
        self,
        hidden_dim: int,
        attention_dim: int = 64,
        dropout: float = 0.1,
        *,
        scoring: str = "additive",
    ):
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if attention_dim <= 0:
            raise ValueError("attention_dim must be positive.")
        if scoring == "additive":
            scorer: TemporalScorer = AdditiveTemporalScorer(hidden_dim, attention_dim, dropout)
        elif scoring == "linear":
            scorer = LinearTemporalScorer(hidden_dim)
        else:
            raise ValueError(f"Unsupported TAM scoring method: {scoring}")

        self.hidden_dim = int(hidden_dim)
        self.attention_dim = int(attention_dim)
        self.scoring = scoring
        self.scorer = scorer
        self.attention_dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
    ) -> TemporalAttentionOutput:
        """Apply temporal attention over GRU hidden states.

        Args:
            hidden_states: Tensor with shape ``[batch, time, hidden]``.
            mask: Optional boolean tensor with shape ``[batch, time]`` where
                true entries are valid and false entries are ignored.

        Returns:
            TemporalAttentionOutput with:
            - context: ``[batch, hidden]``
            - weights: ``[batch, time]``
            - scores: ``[batch, time]``
        """

        if hidden_states.ndim != 3:
            raise ValueError(
                "TAM expects hidden_states with shape [batch, time, hidden], "
                f"got {tuple(hidden_states.shape)}"
            )
        if hidden_states.size(-1) != self.hidden_dim:
            raise ValueError(f"Expected hidden_dim={self.hidden_dim}, got {hidden_states.size(-1)}")

        # hidden_states: [B, T, H]
        scores = self.scorer(hidden_states)
        # scores: [B, T]

        if mask is not None:
            if mask.shape != scores.shape:
                raise ValueError(f"mask shape {tuple(mask.shape)} must match scores shape {tuple(scores.shape)}")
            if not mask.bool().any(dim=1).all():
                raise ValueError("Each temporal attention mask row must contain at least one valid time step.")
            scores = scores.masked_fill(~mask.bool(), torch.finfo(scores.dtype).min)
            # scores: [B, T], invalid time steps receive a very small logit.

        weights = torch.softmax(scores, dim=1)
        # weights: [B, T], sum over T equals 1.

        dropped_weights = self.attention_dropout(weights)
        # dropped_weights: [B, T]

        context = torch.bmm(dropped_weights.unsqueeze(1), hidden_states).squeeze(1)
        # dropped_weights.unsqueeze(1): [B, 1, T]
        # hidden_states: [B, T, H]
        # context before squeeze: [B, 1, H]
        # context after squeeze: [B, H]

        return TemporalAttentionOutput(context=context, weights=weights, scores=scores)
