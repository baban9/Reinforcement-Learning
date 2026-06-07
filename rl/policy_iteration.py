"""Policy iteration for tabular MDPs."""

from __future__ import annotations

import numpy as np

from rl.models import TabularResult
from rl.tabular import compute_value_function, extract_policy


def policy_iteration(
    env,
    gamma: float = 1.0,
    max_iterations: int = 200000,
) -> TabularResult:
    policy = np.zeros(env.observation_space.n, dtype=int)
    iterations = 0

    for iteration in range(max_iterations):
        iterations = iteration + 1
        value_table = compute_value_function(env, policy, gamma)
        new_policy = extract_policy(env, value_table, gamma)
        if np.array_equal(policy, new_policy):
            return TabularResult(
                algorithm="policy_iteration",
                environment=getattr(env.spec, "id", "unknown"),
                policy=new_policy,
                iterations=iterations,
                gamma=gamma,
                metadata={"value_table": value_table.tolist()},
            )
        policy = new_policy

    raise RuntimeError("Policy iteration did not converge.")
