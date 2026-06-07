"""Shared tabular RL utilities."""

from __future__ import annotations

import numpy as np


def tabular_env(env):
    """Return the underlying env for algorithms that require env.P."""
    base = env.unwrapped
    if not hasattr(base, "P"):
        raise TypeError("Environment does not expose tabular dynamics via env.P")
    return base


def compute_value_function(env, policy, gamma: float = 1.0, threshold: float = 1e-10) -> np.ndarray:
    """Policy evaluation for tabular environments with env.P."""
    env = tabular_env(env)
    value_table = np.zeros(env.observation_space.n)
    while True:
        updated_value_table = np.copy(value_table)
        for state in range(env.observation_space.n):
            action = int(policy[state])
            value_table[state] = sum(
                trans_prob * (reward + gamma * updated_value_table[next_state])
                for trans_prob, next_state, reward, _ in env.P[state][action]
            )
        if np.sum(np.abs(updated_value_table - value_table)) <= threshold:
            break
    return value_table


def extract_policy(env, value_table, gamma: float = 1.0) -> np.ndarray:
    """Greedy policy from a value table."""
    env = tabular_env(env)
    policy = np.zeros(env.observation_space.n, dtype=int)
    for state in range(env.observation_space.n):
        q_values = np.zeros(env.action_space.n)
        for action in range(env.action_space.n):
            for trans_prob, next_state, reward, _ in env.P[state][action]:
                q_values[action] += trans_prob * (reward + gamma * value_table[next_state])
        policy[state] = int(np.argmax(q_values))
    return policy


def evaluate_policy(env, policy, episodes: int = 1000, seed: int = 42) -> float:
    """Estimate average return for a deterministic tabular policy."""
    returns = []
    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        total_reward = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            action = int(policy[state])
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
        returns.append(total_reward)
    return float(np.mean(returns))
