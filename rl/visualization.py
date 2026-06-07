"""Visualization helpers for RL experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rl.envs import make_env
from rl.models import BanditResult, QLearningResult
from rl.td_control import q_learning


ACTION_LABELS = ["Left", "Down", "Right", "Up"]


def plot_learning_curve(
    result: QLearningResult,
    output_path: str | Path,
    title: str | None = None,
) -> Path:
    if not result.eval_returns:
        raise ValueError("Result has no evaluation checkpoints to plot.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.plot(result.eval_returns, marker="o")
    plt.xlabel("Evaluation checkpoint")
    plt.ylabel("Average return")
    plt.title(title or f"{result.algorithm} learning curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def plot_bandit_regret(
    results: list[BanditResult],
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4))
    for result in results:
        plt.plot(result.cumulative_regret, label=result.algorithm)
    plt.xlabel("Step")
    plt.ylabel("Cumulative regret")
    plt.title("Bandit strategy comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def plot_frozenlake_policy(
    policy: np.ndarray,
    output_path: str | Path,
    map_name: str = "4x4",
) -> Path:
    """Render a FrozenLake policy as a directional grid."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    size = 4 if map_name == "4x4" else 8
    grid = policy.reshape(size, size)
    arrows = {0: "\u2190", 1: "\u2193", 2: "\u2192", 3: "\u2191"}

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(size - 0.5, -0.5)
    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.grid(True)

    for row in range(size):
        for col in range(size):
            action = int(grid[row, col])
            ax.text(col, row, arrows.get(action, "?"), ha="center", va="center", fontsize=16)

    ax.set_title("FrozenLake policy")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def generate_demo_plots(output_dir: str | Path = "outputs/plots") -> list[Path]:
    """Generate example plots for documentation and smoke testing."""
    output_dir = Path(output_dir)
    env = make_env("frozenlake")
    result = q_learning(
        env,
        episodes=4000,
        alpha=0.5,
        gamma=0.99,
        epsilon=0.2,
        eval_every=500,
        eval_episodes=50,
        seed=7,
    )

    paths = [
        plot_learning_curve(result, output_dir / "q_learning_curve.png"),
        plot_frozenlake_policy(result.policy, output_dir / "frozenlake_policy.png"),
    ]
    return paths
