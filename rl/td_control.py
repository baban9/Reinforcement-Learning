"""Temporal-difference control: Q-learning, SARSA, Expected SARSA, Double Q-learning."""

from __future__ import annotations

import numpy as np

from rl.models import QLearningResult
from rl.schedules import exponential_decay
from rl.tabular import evaluate_policy


def _select_action(q_table, state, epsilon, action_space, rng) -> int:
    if rng.random() < epsilon:
        return int(action_space.sample())
    return int(np.argmax(q_table[state]))


def _expected_sarsa_target(q_row, epsilon: float, gamma: float, reward: float) -> float:
    greedy = int(np.argmax(q_row))
    mean_q = float(np.mean(q_row))
    expected = epsilon * mean_q + (1.0 - epsilon) * q_row[greedy]
    return reward + gamma * expected


def td_control(
    env,
    algorithm: str = "q_learning",
    episodes: int = 10000,
    alpha: float = 0.5,
    gamma: float = 0.99,
    epsilon: float = 0.1,
    epsilon_decay: float = 0.001,
    min_epsilon: float = 0.01,
    eval_every: int = 500,
    eval_episodes: int = 200,
    seed: int = 42,
) -> QLearningResult:
    """Run tabular TD control with optional epsilon decay and evaluation checkpoints."""
    state_count = env.observation_space.n
    action_count = env.action_space.n
    q_a = np.zeros((state_count, action_count))
    q_b = np.zeros((state_count, action_count)) if algorithm == "double_q_learning" else None
    rng = np.random.default_rng(seed)
    eps_schedule = exponential_decay(epsilon, epsilon_decay, min_epsilon)

    learning_curve: list[float] = []
    eval_returns: list[float] = []

    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        eps = eps_schedule(episode)
        terminated = truncated = False

        if algorithm == "sarsa":
            action = _select_action(q_a, state, eps, env.action_space, rng)

        while not (terminated or truncated):
            if algorithm != "sarsa":
                action = _select_action(q_a, state, eps, env.action_space, rng)

            next_state, reward, terminated, truncated, _ = env.step(action)

            if algorithm == "sarsa":
                next_action = _select_action(q_a, next_state, eps, env.action_space, rng)
                target = reward if (terminated or truncated) else reward + gamma * q_a[next_state, next_action]
                q_a[state, action] += alpha * (target - q_a[state, action])
            elif algorithm == "expected_sarsa":
                if terminated or truncated:
                    target = reward
                else:
                    target = _expected_sarsa_target(q_a[next_state], eps, gamma, reward)
                q_a[state, action] += alpha * (target - q_a[state, action])
            elif algorithm == "double_q_learning":
                if rng.random() < 0.5:
                    if terminated or truncated:
                        target = reward
                    else:
                        best = int(np.argmax(q_a[next_state]))
                        target = reward + gamma * q_b[next_state, best]
                    q_a[state, action] += alpha * (target - q_a[state, action])
                else:
                    if terminated or truncated:
                        target = reward
                    else:
                        best = int(np.argmax(q_b[next_state]))
                        target = reward + gamma * q_a[next_state, best]
                    q_b[state, action] += alpha * (target - q_b[state, action])
            else:  # q_learning
                if terminated or truncated:
                    target = reward
                else:
                    target = reward + gamma * np.max(q_a[next_state])
                q_a[state, action] += alpha * (target - q_a[state, action])

            state = next_state
            if algorithm == "sarsa":
                action = next_action

        if eval_every and (episode + 1) % eval_every == 0:
            policy = np.argmax(q_a, axis=1)
            avg = evaluate_policy(env, policy, episodes=eval_episodes, seed=seed)
            eval_returns.append(avg)
            learning_curve.append(float(np.mean(q_a)))

    policy = np.argmax(q_a, axis=1)
    if algorithm == "double_q_learning":
        combined = q_a + q_b
        policy = np.argmax(combined, axis=1)

    return QLearningResult(
        algorithm=algorithm,
        environment=getattr(env.spec, "id", "unknown"),
        q_table=q_a if q_b is None else q_a + q_b,
        policy=policy,
        episodes=episodes,
        gamma=gamma,
        epsilon=epsilon,
        learning_curve=tuple(learning_curve),
        eval_returns=tuple(eval_returns),
    )


def q_learning(env, **kwargs) -> QLearningResult:
    return td_control(env, algorithm="q_learning", **kwargs)


def sarsa(env, **kwargs) -> QLearningResult:
    return td_control(env, algorithm="sarsa", **kwargs)


def expected_sarsa(env, **kwargs) -> QLearningResult:
    return td_control(env, algorithm="expected_sarsa", **kwargs)


def double_q_learning(env, **kwargs) -> QLearningResult:
    return td_control(env, algorithm="double_q_learning", **kwargs)
