"""Tests for tabular RL utilities."""

import numpy as np
import gymnasium as gym

from rl_utils import compute_value_function, extract_policy


def test_extract_policy_returns_valid_actions():
    env = gym.make("FrozenLake-v1")
    value_table = np.zeros(env.observation_space.n)
    policy = extract_policy(env, value_table)
    assert len(policy) == env.observation_space.n
    assert all(0 <= action < env.action_space.n for action in policy)


def test_compute_value_function_converges():
    env = gym.make("FrozenLake-v1")
    policy = np.zeros(env.observation_space.n)
    values = compute_value_function(env, policy)
    assert values.shape == (env.observation_space.n,)
