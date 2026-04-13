"""Early stopping helper for validation-loss monitoring."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStopping:
    """Track validation improvements and decide when training should stop."""

    patience: int = 15
    min_delta: float = 0.0
    mode: str = "min"

    def __post_init__(self) -> None:
        if self.patience <= 0:
            raise ValueError("patience must be positive.")
        if self.mode not in {"min", "max"}:
            raise ValueError("mode must be either 'min' or 'max'.")
        self.best_score: float | None = None
        self.num_bad_epochs = 0
        self.should_stop = False

    def step(self, score: float) -> bool:
        """Update state and return true when the score improves."""

        if self.best_score is None or self._is_improvement(score):
            self.best_score = float(score)
            self.num_bad_epochs = 0
            self.should_stop = False
            return True

        self.num_bad_epochs += 1
        self.should_stop = self.num_bad_epochs >= self.patience
        return False

    def _is_improvement(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "min":
            return score < self.best_score - self.min_delta
        return score > self.best_score + self.min_delta

