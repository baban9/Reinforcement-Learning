"""Legacy import compatibility."""

import numpy as np

from rl import make_env
from rl_utils import compute_value_function, extract_policy


def test_legacy_rl_utils_exports():
    env = make_env("frozenlake")
    policy = np.zeros(env.observation_space.n, dtype=int)
    values = compute_value_function(env, policy)
    greedy = extract_policy(env, values)
    assert len(greedy) == env.observation_space.n
