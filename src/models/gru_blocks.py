"""GRU sequence modeling blocks for Tier 2 ACGD."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class GRUEncoderOutput:
    """GRU output container for downstream temporal attention."""

    sequence: torch.Tensor
    final_hidden: torch.Tensor


class GRUEncoder(nn.Module):
    """Multi-layer GRU encoder with explicit output shapes."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        *,
        bidirectional: bool = False,
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.bidirectional = bool(bidirectional)
        self.output_dim = self.hidden_dim * (2 if self.bidirectional else 1)

        self.gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=dropout if self.num_layers > 1 else 0.0,
            bidirectional=self.bidirectional,
            batch_first=True,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize GRU weights with Xavier/orthogonal matrices and zero biases."""

        for name, parameter in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(parameter)
            elif "weight_hh" in name:
                nn.init.orthogonal_(parameter)
            elif "bias" in name:
                nn.init.zeros_(parameter)

    def forward(self, x: torch.Tensor) -> GRUEncoderOutput:
        """Encode a temporal sequence.

        Args:
            x: Tensor with shape ``[batch, time, input_dim]``.

        Returns:
            GRUEncoderOutput with:
            - sequence: ``[batch, time, hidden_dim * directions]``
            - final_hidden: ``[layers * directions, batch, hidden_dim]``
        """

        if x.ndim != 3:
            raise ValueError(f"GRUEncoder expects [batch, time, input_dim], got {tuple(x.shape)}")
        if x.size(-1) != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {x.size(-1)}")

        # x: [B, T, C]
        sequence, final_hidden = self.gru(x)
        # sequence: [B, T, H * D]
        # final_hidden: [L * D, B, H]
        return GRUEncoderOutput(sequence=sequence, final_hidden=final_hidden)

