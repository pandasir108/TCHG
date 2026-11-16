from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStopping:


    patience: int
    mode: str = "min"
    min_delta: float = 0.0
    best: float | None = None
    wait: int = 0

    def is_improvement(self, value: float) -> bool:

        if self.best is None:
            return True
        if self.mode == "max":
            return value > self.best + self.min_delta
        return value < self.best - self.min_delta

    def step(self, value: float) -> bool:

        if self.is_improvement(value):
            self.best = value
            self.wait = 0
            return False
        self.wait += 1
        return self.wait >= self.patience
