"""Deep Q-Network for CartPole using PyTorch."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random

import numpy as np

from rl.models import DQNResult

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    nn = None
    optim = None


@dataclass
class DQNConfig:
    episodes: int = 300
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    buffer_size: int = 10000
    target_update: int = 10
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    eval_every: int = 25
    seed: int = 42


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


def _build_network(state_dim: int, action_dim: int):
    if nn is None:
        raise ImportError("PyTorch is required for DQN. Install torch>=2.4.1")
    return nn.Sequential(
        nn.Linear(state_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, action_dim),
    )


def train_dqn(env, config: DQNConfig | None = None) -> DQNResult:
    if torch is None:
        raise ImportError("PyTorch is required for DQN. Install torch>=2.4.1")

    config = config or DQNConfig()
    rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    policy_net = _build_network(state_dim, action_dim)
    target_net = _build_network(state_dim, action_dim)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=config.lr)
    buffer = ReplayBuffer(config.buffer_size)

    eval_returns: list[float] = []
    best_return = -float("inf")
    epsilon = config.epsilon_start

    for episode in range(config.episodes):
        state, _ = env.reset(seed=config.seed + episode)
        total_reward = 0.0
        terminated = truncated = False

        while not (terminated or truncated):
            if rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32)
                    action = int(policy_net(state_tensor).argmax().item())

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(buffer) >= config.batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(config.batch_size)
                states_t = torch.tensor(states)
                actions_t = torch.tensor(actions).unsqueeze(1)
                rewards_t = torch.tensor(rewards).unsqueeze(1)
                next_states_t = torch.tensor(next_states)
                dones_t = torch.tensor(dones).unsqueeze(1)

                q_values = policy_net(states_t).gather(1, actions_t)
                with torch.no_grad():
                    next_q = target_net(next_states_t).max(1, keepdim=True)[0]
                    target = rewards_t + config.gamma * next_q * (1 - dones_t)
                loss = nn.MSELoss()(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(config.epsilon_end, epsilon * config.epsilon_decay)
        if (episode + 1) % config.target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if (episode + 1) % config.eval_every == 0:
            avg = _evaluate_dqn(env, policy_net, episodes=10, seed=config.seed)
            eval_returns.append(avg)
            best_return = max(best_return, avg)

    return DQNResult(
        algorithm="dqn",
        environment=getattr(env.spec, "id", "unknown"),
        episodes=config.episodes,
        eval_returns=tuple(eval_returns),
        best_return=best_return,
    )


def _evaluate_dqn(env, policy_net, episodes: int = 10, seed: int = 0) -> float:
    returns = []
    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        total = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32)
                action = int(policy_net(state_tensor).argmax().item())
            state, reward, terminated, truncated, _ = env.step(action)
            total += reward
        returns.append(total)
    return float(np.mean(returns))
