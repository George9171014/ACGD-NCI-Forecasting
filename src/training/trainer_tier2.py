"""Trainer adapter for the Tier 2 ACGD CNN-GRU-DSAM forecaster."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.models.tier2_acgd import Tier2ACGDForecaster
from src.training.engine import TrainingResult
from src.training.trainer_tier1 import _train_model_bundle


def train_tier2_from_windows(
    *,
    config: Mapping[str, Any],
    windows_path: str | Path,
    run_name: str,
) -> TrainingResult:
    """Train the Tier 2 ACGD model from a window NPZ artifact."""

    windows = _load_window_npz(windows_path)
    X_train, y_train = windows["X_train"], windows["y_train"]
    X_val, y_val = windows["X_val"], windows["y_val"]
    X_test, y_test = windows["X_test"], windows["y_test"]

    model = Tier2ACGDForecaster.from_config(config, input_dim=int(X_train.shape[2]))
    return _train_model_bundle(
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


def _load_window_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path).expanduser().resolve()) as data:
        return {key: data[key] for key in data.files}

