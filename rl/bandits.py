"""Multi-armed bandit algorithms and regret analysis."""

from __future__ import annotations

import math

import numpy as np

from rl.models import BanditResult


def _sample_reward(mean: float, rng: np.random.Generator) -> float:
    return float(rng.normal(mean, 1.0))


def epsilon_greedy_bandit(
    arm_means: list[float],
    steps: int = 10000,
    epsilon: float = 0.1,
    seed: int = 42,
) -> BanditResult:
    rng = np.random.default_rng(seed)
    n_arms = len(arm_means)
    q = np.zeros(n_arms)
    counts = np.zeros(n_arms, dtype=int)
    regret = 0.0
    regrets: list[float] = []
    best_mean = max(arm_means)

    for step in range(steps):
        if rng.random() < epsilon:
            arm = int(rng.integers(0, n_arms))
        else:
            arm = int(np.argmax(q))
        reward = _sample_reward(arm_means[arm], rng)
        counts[arm] += 1
        q[arm] += (reward - q[arm]) / counts[arm]
        regret += best_mean - arm_means[arm]
        regrets.append(regret)

    return BanditResult(
        algorithm="epsilon_greedy",
        arms=n_arms,
        steps=steps,
        cumulative_regret=tuple(regrets),
        action_counts=tuple(int(c) for c in counts),
    )


def ucb1_bandit(
    arm_means: list[float],
    steps: int = 10000,
    c: float = 2.0,
    seed: int = 42,
) -> BanditResult:
    rng = np.random.default_rng(seed)
    n_arms = len(arm_means)
    counts = np.zeros(n_arms, dtype=int)
    q = np.zeros(n_arms)
    regret = 0.0
    regrets: list[float] = []
    best_mean = max(arm_means)

    for arm in range(n_arms):
        reward = _sample_reward(arm_means[arm], rng)
        q[arm] = reward
        counts[arm] = 1

    for step in range(n_arms, steps):
        total = step + 1
        ucb_values = q + c * np.sqrt(np.log(total) / counts)
        arm = int(np.argmax(ucb_values))
        reward = _sample_reward(arm_means[arm], rng)
        counts[arm] += 1
        q[arm] += (reward - q[arm]) / counts[arm]
        regret += best_mean - arm_means[arm]
        regrets.append(regret)

    return BanditResult(
        algorithm="ucb1",
        arms=n_arms,
        steps=steps,
        cumulative_regret=tuple(regrets),
        action_counts=tuple(int(c) for c in counts),
    )


def thompson_sampling_bandit(
    arm_means: list[float],
    steps: int = 10000,
    seed: int = 42,
) -> BanditResult:
    rng = np.random.default_rng(seed)
    n_arms = len(arm_means)
    counts = np.zeros(n_arms, dtype=int)
    sums = np.zeros(n_arms)
    regret = 0.0
    regrets: list[float] = []
    best_mean = max(arm_means)

    for step in range(steps):
        samples = []
        for arm in range(n_arms):
            if counts[arm] == 0:
                samples.append(rng.normal(0, 1))
            else:
                mean = sums[arm] / counts[arm]
                std = math.sqrt(1.0 / counts[arm])
                samples.append(rng.normal(mean, std))
        arm = int(np.argmax(samples))
        reward = _sample_reward(arm_means[arm], rng)
        counts[arm] += 1
        sums[arm] += reward
        regret += best_mean - arm_means[arm]
        regrets.append(regret)

    return BanditResult(
        algorithm="thompson_sampling",
        arms=n_arms,
        steps=steps,
        cumulative_regret=tuple(regrets),
        action_counts=tuple(int(c) for c in counts),
    )
