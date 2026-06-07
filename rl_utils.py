"""Shared helpers for tabular Gymnasium environments."""

from __future__ import annotations

import numpy as np


def compute_value_function(env, policy, gamma: float = 1.0) -> np.ndarray:
    value_table = np.zeros(env.observation_space.n)
    threshold = 1e-10
    while True:
        updated_value_table = np.copy(value_table)
        for state in range(env.observation_space.n):
            action = int(policy[state])
            value_table[state] = sum(
                trans_prob * (reward_prob + gamma * updated_value_table[next_state])
                for trans_prob, next_state, reward_prob, _ in env.P[state][action]
            )
        if np.sum(np.fabs(updated_value_table - value_table)) <= threshold:
            break
    return value_table


def extract_policy(env, value_table, gamma: float = 1.0) -> np.ndarray:
    policy = np.zeros(env.observation_space.n)
    for state in range(env.observation_space.n):
        q_table = np.zeros(env.action_space.n)
        for action in range(env.action_space.n):
            for trans_prob, next_state, reward_prob, _ in env.P[state][action]:
                q_table[action] += trans_prob * (reward_prob + gamma * value_table[next_state])
        policy[state] = np.argmax(q_table)
    return policy
