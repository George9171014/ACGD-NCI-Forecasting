"""Shared PyTorch training/evaluation engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch
from torch import nn

from src.evaluation.metrics import compute_regression_metrics
from src.training.checkpointing import save_checkpoint
from src.training.early_stopping import EarlyStopping
from src.utils.io import ensure_dir


@dataclass
class TrainingResult:
    """Structured output from a training run."""

    history: dict[str, list[float]]
    best_epoch: int
    best_val_loss: float
    best_checkpoint_path: Optional[Path]
    test_metrics: dict[str, float] = field(default_factory=dict)
    predictions: Optional[np.ndarray] = None
    targets: Optional[np.ndarray] = None


def train_model(
    *,
    model: nn.Module,
    train_loader: Any,
    val_loader: Any,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    checkpoint_dir: str | Path,
    run_name: str,
    scheduler: Optional[Any] = None,
    early_stopping: Optional[EarlyStopping] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> TrainingResult:
    """Train a PyTorch model and save the best validation checkpoint."""

    model.to(device)
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_epoch = 0
    best_checkpoint_path: Optional[Path] = None
    checkpoint_dir = ensure_dir(checkpoint_dir)

    for epoch in range(1, int(epochs) + 1):
        train_loss = run_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )
        val_loss = run_epoch(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        improved = val_loss < best_val_loss
        if early_stopping is not None:
            improved = early_stopping.step(val_loss)

        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            best_checkpoint_path = checkpoint_dir / f"{run_name}_best.pt"
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                path=best_checkpoint_path,
                epoch=epoch,
                metrics={"val_loss": val_loss, "train_loss": train_loss},
                config=config,
            )

        if early_stopping is not None and early_stopping.should_stop:
            break

    return TrainingResult(
        history=history,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        best_checkpoint_path=best_checkpoint_path,
    )


def run_epoch(
    *,
    model: nn.Module,
    data_loader: Any,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
) -> float:
    """Run one training or evaluation epoch."""

    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_samples = 0

    for features, targets in data_loader:
        features = features.to(device)
        targets = targets.to(device)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            output = model(features)
            predictions = output.predictions if hasattr(output, "predictions") else output
            loss = criterion(predictions, targets)

        if is_training:
            loss.backward()
            optimizer.step()

        batch_size = int(features.size(0))
        total_loss += float(loss.detach().cpu()) * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


def predict_model(
    *,
    model: nn.Module,
    data_loader: Any,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return model predictions and targets for a DataLoader."""

    model.to(device)
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for features, batch_targets in data_loader:
            features = features.to(device)
            output = model(features)
            batch_predictions = output.predictions if hasattr(output, "predictions") else output
            predictions.append(batch_predictions.detach().cpu().numpy())
            targets.append(batch_targets.detach().cpu().numpy())
    return np.concatenate(predictions, axis=0), np.concatenate(targets, axis=0)


def evaluate_model(
    *,
    model: nn.Module,
    data_loader: Any,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Compute standard metrics for a trained model."""

    predictions, targets = predict_model(model=model, data_loader=data_loader, device=device)
    return compute_regression_metrics(targets, predictions), predictions, targets

