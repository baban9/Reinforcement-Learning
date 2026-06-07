"""Backward-compatible entry point."""

from rl.value_iteration import value_iteration

if __name__ == "__main__":
    import gymnasium as gym

    env = gym.make("FrozenLake-v1")
    result = value_iteration(env)
    print(result.policy)
