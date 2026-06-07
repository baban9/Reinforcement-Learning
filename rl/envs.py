"""Environment helpers."""

from __future__ import annotations

import gymnasium as gym

ENV_IDS = {
    "frozenlake": "FrozenLake-v1",
    "blackjack": "Blackjack-v1",
    "taxi": "Taxi-v3",
    "cartpole": "CartPole-v1",
}


def make_env(name: str, **kwargs):
    """Create a Gymnasium environment by short name or full ID."""
    env_id = ENV_IDS.get(name.lower(), name)
    return gym.make(env_id, **kwargs)
