"""Shared result types for RL algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class TabularResult:
    algorithm: str
    environment: str
    policy: np.ndarray
    iterations: int
    gamma: float
    metadata: dict | None = None

    def __str__(self) -> str:
        return (
            f"{self.algorithm} on {self.environment}: "
            f"converged in {self.iterations} iterations (gamma={self.gamma})"
        )


@dataclass(frozen=True)
class QLearningResult:
    algorithm: str
    environment: str
    q_table: np.ndarray
    policy: np.ndarray
    episodes: int
    gamma: float
    epsilon: float
    learning_curve: tuple[float, ...] = ()
    eval_returns: tuple[float, ...] = ()

    def __str__(self) -> str:
        final = self.eval_returns[-1] if self.eval_returns else float("nan")
        return (
            f"{self.algorithm} on {self.environment}: "
            f"{self.episodes} episodes (gamma={self.gamma}, eps={self.epsilon}, "
            f"final_eval={final:.3f})"
        )


@dataclass(frozen=True)
class MonteCarloPiResult:
    samples: int
    estimate: float
    error: float

    def __str__(self) -> str:
        return f"pi estimate={self.estimate:.6f} error={self.error:.6f} ({self.samples} samples)"


@dataclass(frozen=True)
class BanditResult:
    algorithm: str
    arms: int
    steps: int
    cumulative_regret: tuple[float, ...]
    action_counts: tuple[int, ...]

    def __str__(self) -> str:
        return (
            f"{self.algorithm} bandit: regret={self.cumulative_regret[-1]:.2f} "
            f"over {self.steps} steps"
        )


@dataclass(frozen=True)
class DQNResult:
    algorithm: str
    environment: str
    episodes: int
    eval_returns: tuple[float, ...]
    best_return: float

    def __str__(self) -> str:
        return (
            f"{self.algorithm} on {self.environment}: "
            f"best={self.best_return:.1f} over {self.episodes} episodes"
        )


@dataclass
class ExperimentReport:
    name: str
    environment: str
    results: list[dict] = field(default_factory=list)

    def add(self, **metrics):
        self.results.append(metrics)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "environment": self.environment,
            "results": self.results,
        }

    def rank_by(self, key: str) -> list[dict]:
        return sorted(self.results, key=lambda item: item.get(key, 0.0), reverse=True)
