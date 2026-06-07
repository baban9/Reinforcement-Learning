"""Training schedules for exploration and learning rate."""

from __future__ import annotations

import math


def constant(value: float):
    return lambda step: value


def linear_decay(start: float, end: float, duration: int):
    def schedule(step: int) -> float:
        if duration <= 0:
            return end
        progress = min(step / duration, 1.0)
        return start + (end - start) * progress

    return schedule


def exponential_decay(start: float, decay: float, minimum: float = 0.01):
    def schedule(step: int) -> float:
        return max(minimum, start * math.exp(-decay * step))

    return schedule
