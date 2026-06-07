"""Backward-compatible entry point."""

from rl.policy_iteration import policy_iteration

if __name__ == "__main__":
    import gymnasium as gym

    env = gym.make("FrozenLake-v1")
    result = policy_iteration(env)
    print(result.policy)
