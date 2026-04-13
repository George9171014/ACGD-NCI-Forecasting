"""Trainer adapter for Tier 1 renewable ANN forecasters."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from src.agents.evaluation_agent import EvaluationAgent
from src.data.datasets import make_dataloader
from src.evaluation.plots import plot_error_distribution, plot_prediction_vs_actual, plot_training_curves
from src.models.losses import FlexibleForecastLoss
from src.models.tier1_ann import Tier1ANNForecaster
from src.training.checkpointing import load_checkpoint
from src.training.early_stopping import EarlyStopping
from src.training.engine import TrainingResult, evaluate_model, train_model
from src.utils.config import get_config_value, resolve_config_path
from src.utils.io import timestamped_path, write_json
from src.utils.seed import get_torch_device


def train_tier1_from_windows(
    *,
    config: Mapping[str, Any],
    windows_path: str | Path,
    run_name: str,
) -> TrainingResult:
    """Train a Tier 1 ANN from a window NPZ artifact."""

    windows = _load_window_npz(windows_path)
    X_train, y_train = windows["X_train"], windows["y_train"]
    X_val, y_val = windows["X_val"], windows["y_val"]
    X_test, y_test = windows["X_test"], windows["y_test"]

    input_dim = int(X_train.shape[1] * X_train.shape[2])
    model = Tier1ANNForecaster.from_config(config, input_dim=input_dim)
    result = _train_model_bundle(
        config=config,
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        run_name=run_name,
    )
    return result


def _train_model_bundle(
    *,
    config: Mapping[str, Any],
    model: torch.nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    run_name: str,
) -> TrainingResult:
    device = get_torch_device(str(config.get("device", "auto")))
    training = config.get("training", {})
    batch_size = int(training.get("batch_size", 64))
    train_loader = make_dataloader(X_train, y_train, batch_size=batch_size, shuffle=True, generator_seed=int(config.get("seed", 42)))
    val_loader = make_dataloader(X_val, y_val, batch_size=batch_size, shuffle=False)
    test_loader = make_dataloader(X_test, y_test, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training.get("learning_rate", 0.001)),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    scheduler = _build_scheduler(optimizer, training)
    early_stopping = _build_early_stopping(training)
    criterion = FlexibleForecastLoss.from_config(config)
    checkpoints_dir = resolve_config_path(config, "paths.checkpoints_dir", create=True)

    result = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stopping=early_stopping,
        device=device,
        epochs=int(training.get("epochs", 100)),
        checkpoint_dir=checkpoints_dir,
        run_name=run_name,
        config=config,
    )

    if result.best_checkpoint_path is not None:
        load_checkpoint(result.best_checkpoint_path, model=model, map_location=device)

    metrics, predictions, targets = evaluate_model(model=model, data_loader=test_loader, device=device)
    result.test_metrics = metrics
    result.predictions = predictions
    result.targets = targets

    reports_dir = resolve_config_path(config, "paths.reports_dir", create=True)
    figures_dir = resolve_config_path(config, "paths.figures_dir", create=True)
    prediction_path = timestamped_path(reports_dir, f"{run_name}_test_predictions", ".npz")
    np.savez_compressed(prediction_path, predictions=predictions, targets=targets)

    evaluation_report = EvaluationAgent().evaluate(
        targets,
        predictions,
        split="test",
        report_dir=reports_dir,
        run_name=run_name,
    )
    plot_paths = {
        "training_curves": str(plot_training_curves(result.history, figures_dir, run_name)),
        "prediction_vs_actual": str(plot_prediction_vs_actual(targets, predictions, figures_dir, run_name)),
        "error_distribution": str(plot_error_distribution(targets, predictions, figures_dir, run_name)),
    }
    summary_path = timestamped_path(reports_dir, f"{run_name}_training_summary", ".json")
    write_json(
        {
            "run_name": run_name,
            "best_epoch": result.best_epoch,
            "best_val_loss": result.best_val_loss,
            "best_checkpoint_path": str(result.best_checkpoint_path) if result.best_checkpoint_path else None,
            "test_metrics": metrics,
            "prediction_path": str(prediction_path),
            "evaluation_report_path": evaluation_report.get("path"),
            "figures": plot_paths,
        },
        summary_path,
    )
    return result


def _build_scheduler(optimizer: torch.optim.Optimizer, training: Mapping[str, Any]) -> Any | None:
    scheduler_config = training.get("scheduler", {})
    if not scheduler_config or not bool(scheduler_config.get("enabled", False)):
        return None
    name = str(scheduler_config.get("name", "reduce_on_plateau"))
    if name == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=float(scheduler_config.get("factor", 0.5)),
            patience=int(scheduler_config.get("patience", 5)),
        )
    raise ValueError(f"Unsupported scheduler: {name}")


def _build_early_stopping(training: Mapping[str, Any]) -> EarlyStopping | None:
    early = training.get("early_stopping", {})
    if not early or not bool(early.get("enabled", False)):
        return None
    return EarlyStopping(
        patience=int(early.get("patience", 15)),
        min_delta=float(early.get("min_delta", 0.0)),
        mode="min",
    )


def _load_window_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path).expanduser().resolve()) as data:
        return {key: data[key] for key in data.files}

