"""Checkpoint helpers for PyTorch training runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import torch

from src.utils.io import ensure_dir, utc_timestamp


def save_checkpoint(
    *,
    model: torch.nn.Module,
    path: str | Path,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int,
    metrics: Mapping[str, float] | None = None,
    config: Mapping[str, Any] | None = None,
) -> Path:
    """Save a model checkpoint and return the output path."""

    output_path = Path(path).expanduser().resolve()
    ensure_dir(output_path.parent)
    payload: dict[str, Any] = {
        "epoch": int(epoch),
        "created_at_utc": utc_timestamp(),
        "model_state_dict": model.state_dict(),
        "metrics": dict(metrics or {}),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if config is not None:
        payload["config"] = dict(config)
    torch.save(payload, output_path)
    return output_path


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint into a model and optionally optimizer/scheduler."""

    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in payload:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    return payload

