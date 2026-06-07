"""Value iteration on FrozenLake."""

import gymnasium as gym
import numpy as np

from rl_utils import extract_policy


def value_iteration(env, gamma: float = 1.0, threshold: float = 1e-20) -> np.ndarray:
    value_table = np.zeros(env.observation_space.n)
    while True:
        updated_value_table = np.copy(value_table)
        for state in range(env.observation_space.n):
            q_values = []
            for action in range(env.action_space.n):
                q_value = sum(
                    trans_prob * (reward_prob + gamma * value_table[next_state])
                    for trans_prob, next_state, reward_prob, _ in env.P[state][action]
                )
                q_values.append(q_value)
            value_table[state] = max(q_values)
        if np.sum(np.fabs(updated_value_table - value_table)) <= threshold:
            break
    return extract_policy(env, value_table, gamma)


if __name__ == "__main__":
    env = gym.make("FrozenLake-v1")
    print(value_iteration(env))
