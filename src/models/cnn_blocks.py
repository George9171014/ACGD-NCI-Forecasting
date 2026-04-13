"""1D CNN blocks for local temporal feature extraction in Tier 2 ACGD."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class TemporalConvBlock(nn.Module):
    """Single Conv1d block that preserves sequence length for odd kernels."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        dropout: float = 0.1,
        batch_norm: bool = True,
    ):
        super().__init__()
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive.")
        if kernel_size % 2 == 0:
            raise ValueError("TemporalConvBlock uses same-length padding; use an odd kernel_size.")

        padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
        ]
        if batch_norm:
            layers.append(nn.BatchNorm1d(out_channels))
        layers.extend(
            [
                nn.LeakyReLU(negative_slope=0.01),
                nn.Dropout(dropout),
            ]
        )
        self.block = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Apply Xavier initialization to convolution layers and zero biases."""

        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the convolutional block.

        Args:
            x: Tensor with shape ``[batch, channels, time]``.

        Returns:
            Tensor with shape ``[batch, out_channels, time]``.
        """

        # x: [B, C_in, T]
        out = self.block(x)
        # out: [B, C_out, T]
        return out


class TemporalCNNEncoder(nn.Module):
    """Stack of Conv1d blocks for the CNN front-end."""

    def __init__(
        self,
        input_dim: int,
        channels: Sequence[int],
        *,
        kernel_size: int = 3,
        dropout: float = 0.1,
        batch_norm: bool = True,
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if not channels:
            raise ValueError("channels must contain at least one output channel.")

        channel_list = [int(channel) for channel in channels]
        if any(channel <= 0 for channel in channel_list):
            raise ValueError("All CNN channel sizes must be positive.")

        blocks: list[nn.Module] = []
        in_channels = int(input_dim)
        for out_channels in channel_list:
            blocks.append(
                TemporalConvBlock(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    batch_norm=batch_norm,
                )
            )
            in_channels = out_channels

        self.input_dim = int(input_dim)
        self.channels = channel_list
        self.output_dim = channel_list[-1]
        self.encoder = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode temporal features with Conv1d layers.

        Args:
            x: Tensor with shape ``[batch, time, features]``.

        Returns:
            Tensor with shape ``[batch, time, cnn_channels]``.
        """

        if x.ndim != 3:
            raise ValueError(f"TemporalCNNEncoder expects [batch, time, features], got {tuple(x.shape)}")
        if x.size(-1) != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {x.size(-1)}")

        # x: [B, T, F]
        channels_first = x.transpose(1, 2)
        # channels_first: [B, F, T]

        encoded = self.encoder(channels_first)
        # encoded: [B, C, T]

        time_first = encoded.transpose(1, 2)
        # time_first: [B, T, C]
        return time_first

