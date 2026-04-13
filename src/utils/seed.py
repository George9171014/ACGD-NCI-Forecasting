"""Reproducibility and device helpers."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np


def set_global_seed(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch when PyTorch is available."""

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def get_torch_device(preferred: str = "auto", *, require_torch: bool = True) -> Any:
    """Return a PyTorch device, falling back to CPU when CUDA is unavailable.

    When ``require_torch`` is false, the function returns the string ``"cpu"``
    if PyTorch is not installed. This keeps lightweight configuration commands
    usable before the project environment has been fully installed.
    """

    try:
        import torch
    except ImportError as exc:
        if not require_torch:
            return "cpu"
        raise RuntimeError("PyTorch is required for device selection.") from exc

    normalized = preferred.lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(normalized)
