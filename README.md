# Reinforcement Learning

Tabular reinforcement learning implementations with shared utilities and Gymnasium-compatible environments.

## Problem

Implement and compare classic RL algorithms on small, interpretable MDPs before scaling to function approximation.

## Approach

| Module | Algorithm | Environment |
|--------|-----------|-------------|
| `policy_iteration.py` | Policy iteration | FrozenLake-v1 |
| `value_iteration.py` | Value iteration | FrozenLake-v1 |
| `Monte-Carlo Blackjack.py` | Monte Carlo control | Blackjack |
| `Monte-Carlo pi calculation.py` | Monte Carlo estimation | Synthetic |

Shared logic in `rl_utils.py` keeps policy evaluation DRY across algorithms.

## Reproducibility

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
make setup
python policy_iteration.py
python value_iteration.py
make test
```

## Tech stack

Python 3, Gymnasium, NumPy

## Design principles

- Pin environment IDs explicitly (`FrozenLake-v1`, not deprecated `-v0`)
- Separate algorithm logic from environment instantiation
- Legacy `gym` scripts retained for reference only

## Limitations and next steps

- Add pytest coverage for convergence on toy MDPs
- Extend to Q-learning with epsilon-greedy exploration
- Log episode returns for learning curve visualization
