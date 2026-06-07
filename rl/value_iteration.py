"""Value iteration for tabular MDPs."""

from __future__ import annotations

import numpy as np

from rl.models import TabularResult
from rl.tabular import extract_policy


def value_iteration(
    env,
    gamma: float = 1.0,
    threshold: float = 1e-12,
    max_iterations: int = 200000,
) -> TabularResult:
    value_table = np.zeros(env.observation_space.n)
    iterations = 0
    base_env = env.unwrapped

    for iteration in range(max_iterations):
        iterations = iteration + 1
        updated_value_table = np.copy(value_table)
        for state in range(env.observation_space.n):
            q_values = []
            for action in range(env.action_space.n):
                q_value = sum(
                    trans_prob * (reward + gamma * value_table[next_state])
                    for trans_prob, next_state, reward, _ in base_env.P[state][action]
                )
                q_values.append(q_value)
            value_table[state] = max(q_values)
        if np.sum(np.abs(updated_value_table - value_table)) <= threshold:
            break
    else:
        raise RuntimeError("Value iteration did not converge.")

    policy = extract_policy(env, value_table, gamma)
    return TabularResult(
        algorithm="value_iteration",
        environment=getattr(env.spec, "id", "unknown"),
        policy=policy,
        iterations=iterations,
        gamma=gamma,
        metadata={"value_table": value_table.tolist()},
    )
