"""Policy iteration on FrozenLake."""

import gymnasium as gym
import numpy as np

from rl_utils import compute_value_function, extract_policy


def policy_iteration(env, gamma: float = 1.0, max_iterations: int = 200000) -> np.ndarray:
    policy = np.zeros(env.observation_space.n)
    for iteration in range(max_iterations):
        value_table = compute_value_function(env, policy, gamma)
        new_policy = extract_policy(env, value_table, gamma)
        if np.all(policy == new_policy):
            print(f"Policy iteration converged at step {iteration + 1}.")
            return new_policy
        policy = new_policy
    raise RuntimeError("Policy iteration did not converge.")


if __name__ == "__main__":
    env = gym.make("FrozenLake-v1")
    print(policy_iteration(env))
