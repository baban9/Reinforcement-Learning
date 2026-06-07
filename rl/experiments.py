"""Benchmark experiments and JSON export."""

from __future__ import annotations

import json
from pathlib import Path

from rl.analysis import compare_td_algorithms
from rl.bandits import epsilon_greedy_bandit, thompson_sampling_bandit, ucb1_bandit
from rl.dqn import DQNConfig, train_dqn
from rl.envs import make_env
from rl.models import ExperimentReport
from rl.monte_carlo import estimate_pi
from rl.policy_iteration import policy_iteration
from rl.td_control import double_q_learning, expected_sarsa, q_learning, sarsa
from rl.tabular import evaluate_policy
from rl.value_iteration import value_iteration


def run_tabular_benchmark(
    env_name: str = "frozenlake",
    episodes: int = 8000,
    seed: int = 42,
) -> ExperimentReport:
    env = make_env(env_name)
    report = ExperimentReport(name="tabular_benchmark", environment=getattr(env.spec, "id", env_name))

    pi = policy_iteration(env, gamma=1.0)
    vi = value_iteration(env, gamma=1.0)
    report.add(
        algorithm="policy_iteration",
        iterations=pi.iterations,
        avg_return=evaluate_policy(env, pi.policy, seed=seed),
    )
    report.add(
        algorithm="value_iteration",
        iterations=vi.iterations,
        avg_return=evaluate_policy(env, vi.policy, seed=seed),
    )

    for algo, fn in (
        ("q_learning", q_learning),
        ("sarsa", sarsa),
        ("expected_sarsa", expected_sarsa),
        ("double_q_learning", double_q_learning),
    ):
        result = fn(
            env,
            episodes=episodes,
            alpha=0.5,
            gamma=0.99,
            epsilon=0.2,
            epsilon_decay=0.002,
            eval_every=1000,
            eval_episodes=100,
            seed=seed,
        )
        report.add(
            algorithm=algo,
            episodes=result.episodes,
            avg_return=evaluate_policy(env, result.policy, seed=seed),
            final_eval=result.eval_returns[-1] if result.eval_returns else None,
        )
    return report


def run_bandit_benchmark(
    arm_means: list[float] | None = None,
    steps: int = 5000,
    seed: int = 42,
) -> ExperimentReport:
    arm_means = arm_means or [0.2, 0.5, 0.3, 0.8, 0.1]
    report = ExperimentReport(name="bandit_benchmark", environment="gaussian_bandit")

    for fn in (epsilon_greedy_bandit, ucb1_bandit, thompson_sampling_bandit):
        result = fn(arm_means, steps=steps, seed=seed)
        report.add(
            algorithm=result.algorithm,
            steps=result.steps,
            final_regret=result.cumulative_regret[-1],
            action_counts=list(result.action_counts),
        )
    return report


def run_dqn_benchmark(
    episodes: int = 120,
    seed: int = 42,
) -> ExperimentReport:
    env = make_env("cartpole")
    config = DQNConfig(episodes=episodes, eval_every=20, seed=seed)
    result = train_dqn(env, config)
    report = ExperimentReport(name="dqn_benchmark", environment=result.environment)
    report.add(
        algorithm=result.algorithm,
        episodes=result.episodes,
        best_return=result.best_return,
        eval_points=list(result.eval_returns),
    )
    return report


def run_full_benchmark(
    output_dir: str | Path = "outputs",
    seed: int = 42,
) -> dict:
    """Run all benchmark suites and write JSON artifacts."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    tabular = run_tabular_benchmark(seed=seed)
    bandits = run_bandit_benchmark(seed=seed)
    td_compare = compare_td_algorithms(seed=seed)

    reports = {
        "tabular": tabular.to_dict(),
        "bandits": bandits.to_dict(),
        "td_compare": td_compare.to_dict(),
    }

    for name, payload in reports.items():
        path = output / f"{name}_benchmark.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return reports
