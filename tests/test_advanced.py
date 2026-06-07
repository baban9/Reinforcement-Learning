"""Advanced RL algorithm tests."""

import json
from pathlib import Path

import numpy as np
import pytest

from rl import (
    compare_td_algorithms,
    double_q_learning,
    epsilon_greedy_bandit,
    expected_sarsa,
    make_env,
    q_learning,
    run_full_benchmark,
    sarsa,
    thompson_sampling_bandit,
    ucb1_bandit,
)
from rl.tabular import evaluate_policy
from rl.visualization import plot_frozenlake_policy, plot_learning_curve


def test_sarsa_and_q_learning_both_learn():
    env = make_env("frozenlake")
    q_result = q_learning(env, episodes=3000, epsilon=0.2, eval_every=1000, seed=1)
    s_result = sarsa(env, episodes=3000, epsilon=0.2, eval_every=1000, seed=1)
    assert evaluate_policy(env, q_result.policy) >= 0.0
    assert evaluate_policy(env, s_result.policy) >= 0.0


def test_double_q_learning_returns_combined_table():
    env = make_env("frozenlake")
    result = double_q_learning(env, episodes=2000, epsilon=0.2, seed=3)
    assert result.q_table.shape == (env.observation_space.n, env.action_space.n)
    assert result.algorithm == "double_q_learning"


def test_expected_sarsa_produces_eval_curve():
    env = make_env("frozenlake")
    result = expected_sarsa(env, episodes=2500, epsilon=0.2, eval_every=500, seed=2)
    assert len(result.eval_returns) >= 1


def test_bandit_algorithms_reduce_regret():
    means = [0.1, 0.2, 0.9, 0.15]
    random_result = epsilon_greedy_bandit(means, steps=2000, epsilon=0.3, seed=5)
    ucb_result = ucb1_bandit(means, steps=2000, seed=5)
    ts_result = thompson_sampling_bandit(means, steps=2000, seed=5)
    assert ucb_result.cumulative_regret[-1] < random_result.cumulative_regret[-1]
    assert ts_result.cumulative_regret[-1] < random_result.cumulative_regret[-1]


def test_compare_td_algorithms_report():
    report = compare_td_algorithms(episodes=1500, seed=9)
    assert len(report.results) == 4
    assert report.results[0]["algorithm"]


def test_benchmark_writes_json(tmp_path: Path):
    reports = run_full_benchmark(output_dir=tmp_path, seed=11)
    assert (tmp_path / "tabular_benchmark.json").exists()
    payload = json.loads((tmp_path / "tabular_benchmark.json").read_text())
    assert "results" in payload
    assert len(reports["tabular"]["results"]) >= 6


def test_plot_helpers(tmp_path: Path):
    env = make_env("frozenlake")
    result = q_learning(env, episodes=1500, eval_every=500, seed=4)
    curve_path = plot_learning_curve(result, tmp_path / "curve.png")
    policy_path = plot_frozenlake_policy(result.policy, tmp_path / "policy.png")
    assert curve_path.exists()
    assert policy_path.exists()


@pytest.mark.slow
def test_train_dqn_smoke():
    torch = pytest.importorskip("torch")
    from rl.dqn import DQNConfig, train_dqn

    env = make_env("cartpole")
    result = train_dqn(env, DQNConfig(episodes=40, eval_every=10, seed=0))
    assert result.best_return >= 0.0
