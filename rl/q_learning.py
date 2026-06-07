"""Backward-compatible Q-learning entry point."""

from rl.td_control import double_q_learning, expected_sarsa, q_learning, sarsa, td_control

__all__ = [
    "double_q_learning",
    "expected_sarsa",
    "q_learning",
    "sarsa",
    "td_control",
]
