"""Algorithm comparison and reporting."""

from __future__ import annotations

from rl.envs import make_env
from rl.models import ExperimentReport
from rl.td_control import double_q_learning, expected_sarsa, q_learning, sarsa
from rl.tabular import evaluate_policy


def compare_td_algorithms(
    env_name: str = "frozenlake",
    episodes: int = 6000,
    seed: int = 42,
) -> ExperimentReport:
    """Compare on-policy vs off-policy TD methods on the same environment."""
    env = make_env(env_name)
    report = ExperimentReport(
        name="td_algorithm_comparison",
        environment=getattr(env.spec, "id", env_name),
    )

    configs = [
        ("q_learning", q_learning, "off-policy"),
        ("sarsa", sarsa, "on-policy"),
        ("expected_sarsa", expected_sarsa, "on-policy"),
        ("double_q_learning", double_q_learning, "off-policy"),
    ]

    for name, fn, policy_type in configs:
        result = fn(
            env,
            episodes=episodes,
            alpha=0.5,
            gamma=0.99,
            epsilon=0.15,
            epsilon_decay=0.002,
            eval_every=1000,
            eval_episodes=100,
            seed=seed,
        )
        report.add(
            algorithm=name,
            policy_type=policy_type,
            avg_return=evaluate_policy(env, result.policy, seed=seed),
            final_eval=result.eval_returns[-1] if result.eval_returns else None,
            learning_curve=list(result.eval_returns),
        )

    return report


def print_leaderboard(report: ExperimentReport, metric: str = "avg_return") -> None:
    print(f"Leaderboard: {report.name} ({report.environment})")
    for index, row in enumerate(report.rank_by(metric), start=1):
        value = row.get(metric, "n/a")
        algo = row.get("algorithm", "unknown")
        print(f"{index}. {algo}: {value}")
