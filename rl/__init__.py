"""Advanced reinforcement learning library."""

from rl.analysis import compare_td_algorithms, print_leaderboard
from rl.bandits import epsilon_greedy_bandit, thompson_sampling_bandit, ucb1_bandit
from rl.dqn import DQNConfig, train_dqn
from rl.envs import make_env
from rl.experiments import run_bandit_benchmark, run_full_benchmark, run_tabular_benchmark
from rl.models import BanditResult, DQNResult, ExperimentReport, MonteCarloPiResult, QLearningResult, TabularResult
from rl.monte_carlo import (
    default_blackjack_policy,
    estimate_pi,
    first_visit_mc_prediction,
    generate_episode,
)
from rl.policy_iteration import policy_iteration
from rl.td_control import double_q_learning, expected_sarsa, q_learning, sarsa, td_control
from rl.tabular import compute_value_function, evaluate_policy, extract_policy
from rl.value_iteration import value_iteration
from rl.visualization import generate_demo_plots, plot_bandit_regret, plot_frozenlake_policy, plot_learning_curve

__all__ = [
    "BanditResult",
    "DQNConfig",
    "DQNResult",
    "ExperimentReport",
    "MonteCarloPiResult",
    "QLearningResult",
    "TabularResult",
    "compare_td_algorithms",
    "compute_value_function",
    "default_blackjack_policy",
    "double_q_learning",
    "epsilon_greedy_bandit",
    "estimate_pi",
    "evaluate_policy",
    "expected_sarsa",
    "extract_policy",
    "first_visit_mc_prediction",
    "generate_demo_plots",
    "generate_episode",
    "make_env",
    "plot_bandit_regret",
    "plot_frozenlake_policy",
    "plot_learning_curve",
    "policy_iteration",
    "print_leaderboard",
    "q_learning",
    "run_bandit_benchmark",
    "run_full_benchmark",
    "run_tabular_benchmark",
    "sarsa",
    "td_control",
    "thompson_sampling_bandit",
    "train_dqn",
    "ucb1_bandit",
    "value_iteration",
]
