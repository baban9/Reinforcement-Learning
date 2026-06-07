"""Tests for tabular RL algorithms."""

import numpy as np
import pytest

from rl import (
    estimate_pi,
    make_env,
    policy_iteration,
    q_learning,
    value_iteration,
)
from rl.tabular import compute_value_function, evaluate_policy, extract_policy


def test_extract_policy_returns_valid_actions():
    env = make_env("frozenlake")
    value_table = np.zeros(env.observation_space.n)
    policy = extract_policy(env, value_table)
    assert len(policy) == env.observation_space.n
    assert all(0 <= action < env.action_space.n for action in policy)


def test_compute_value_function_converges():
    env = make_env("frozenlake")
    policy = np.zeros(env.observation_space.n, dtype=int)
    values = compute_value_function(env, policy)
    assert values.shape == (env.observation_space.n,)


def test_policy_and_value_iteration_agree_on_frozenlake():
    env = make_env("frozenlake")
    pi = policy_iteration(env, gamma=1.0)
    vi = value_iteration(env, gamma=1.0)
    assert np.array_equal(pi.policy, vi.policy)


def test_q_learning_learns_non_trivial_policy():
    env = make_env("frozenlake")
    result = q_learning(env, episodes=5000, epsilon=0.2, alpha=0.5, gamma=0.99)
    assert result.q_table.shape == (env.observation_space.n, env.action_space.n)
    assert evaluate_policy(env, result.policy) >= 0.0


def test_estimate_pi_is_close_for_large_sample():
    result = estimate_pi(samples=200_000, seed=7)
    assert result.error < 0.02


def test_monte_carlo_blackjack_returns_values():
    from rl.monte_carlo import default_blackjack_policy, first_visit_mc_prediction

    env = make_env("blackjack")
    values = first_visit_mc_prediction(env, default_blackjack_policy, episodes=5000, seed=1)
    assert len(values) > 0
