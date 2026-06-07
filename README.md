# Advanced Reinforcement Learning

Production-style RL library spanning dynamic programming, temporal-difference control, multi-armed bandits, deep Q-networks, experiment benchmarks, and visualization.

## Algorithm coverage

| Category | Algorithms |
|----------|------------|
| Dynamic programming | Policy iteration, value iteration |
| TD control | Q-learning, SARSA, Expected SARSA, Double Q-learning |
| Monte Carlo | Blackjack prediction, pi estimation |
| Bandits | Epsilon-greedy, UCB1, Thompson Sampling |
| Deep RL | DQN with replay buffer and target network (CartPole) |

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
make setup
make test
make leaderboard
make benchmark
```

## CLI

```bash
python run.py policy-iteration
python run.py q-learning --episodes 10000 --epsilon-decay 0.002
python run.py sarsa
python run.py expected-sarsa
python run.py double-q
python run.py bandits --steps 5000
python run.py train-dqn --episodes 200
python run.py benchmark --output outputs
python run.py plot --output outputs/plots
python run.py leaderboard
python run.py compare
```

## Python API

```python
from rl import (
    make_env,
    q_learning,
    sarsa,
    double_q_learning,
    compare_td_algorithms,
    run_full_benchmark,
    train_dqn,
    DQNConfig,
)

env = make_env("frozenlake")
result = q_learning(env, episodes=8000, epsilon_decay=0.002)
print(result.eval_returns)

report = compare_td_algorithms()
dqn = train_dqn(make_env("cartpole"), DQNConfig(episodes=200))
```

## Project layout

```
rl/
  tabular.py          Policy evaluation helpers
  policy_iteration.py Dynamic programming
  value_iteration.py  Dynamic programming
  td_control.py       Q-learning, SARSA, Expected SARSA, Double Q
  bandits.py          UCB1, Thompson Sampling, epsilon-greedy
  dqn.py              Deep Q-network for CartPole
  monte_carlo.py      MC prediction and pi estimation
  schedules.py        Epsilon decay schedules
  experiments.py      Benchmark suites + JSON export
  analysis.py         Algorithm comparison reports
  visualization.py    Learning curves and policy plots
  cli.py              Command-line interface
legacy/               Original tutorial scripts (deprecated gym API)
outputs/              Generated benchmark JSON and plots (gitignored)
```

## Advanced features

- **Learning curves**: TD algorithms log evaluation returns during training
- **On-policy vs off-policy**: Compare SARSA, Expected SARSA, and Q-learning side by side
- **Regret analysis**: Bandit strategies with cumulative regret tracking
- **Experiment export**: `run_full_benchmark()` writes JSON artifacts to `outputs/`
- **Visualization**: Policy heatmaps and learning curve plots via matplotlib

## Tech stack

Python 3.9+, Gymnasium, NumPy, PyTorch, matplotlib, pytest

## Legacy scripts

Original coursework files live under `legacy/` and use the deprecated `gym` package. Use the `rl` package for all new work.
