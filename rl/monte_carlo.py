"""Monte Carlo methods for Blackjack and pi estimation."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Callable

import numpy as np

from rl.models import MonteCarloPiResult


def estimate_pi(samples: int = 100_000, seed: int = 42) -> MonteCarloPiResult:
    """Estimate pi with Monte Carlo sampling inside a unit square."""
    rng = np.random.default_rng(seed)
    points = rng.random((samples, 2))
    inside = np.sum(np.sum(points**2, axis=1) <= 1.0)
    estimate = 4.0 * inside / samples
    return MonteCarloPiResult(samples=samples, estimate=estimate, error=abs(math.pi - estimate))


def default_blackjack_policy(observation) -> int:
    """Simple baseline: stand on 20+, hit otherwise."""
    player_sum, _, _ = observation
    return 0 if player_sum >= 20 else 1


def generate_episode(env, policy: Callable) -> tuple[list, list, list]:
    states, actions, rewards = [], [], []
    observation, _ = env.reset()
    while True:
        states.append(observation)
        action = policy(observation)
        actions.append(action)
        observation, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        if terminated or truncated:
            break
    return states, actions, rewards


def first_visit_mc_prediction(
    env,
    policy: Callable,
    episodes: int = 100_000,
    seed: int = 42,
) -> dict:
    """First-visit Monte Carlo state-value prediction."""
    value_table: dict = defaultdict(float)
    visits: dict = defaultdict(int)

    for episode in range(episodes):
        states, _, rewards = generate_episode(env, policy)
        returns = 0.0
        visited = set()
        for step in range(len(states) - 1, -1, -1):
            returns += rewards[step]
            state = states[step]
            if state in visited:
                continue
            visited.add(state)
            visits[state] += 1
            value_table[state] += (returns - value_table[state]) / visits[state]

    return dict(value_table)
