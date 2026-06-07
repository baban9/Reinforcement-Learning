"""Advanced reinforcement learning CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rl.analysis import compare_td_algorithms, print_leaderboard
from rl.bandits import epsilon_greedy_bandit, thompson_sampling_bandit, ucb1_bandit
from rl.dqn import DQNConfig, train_dqn
from rl.envs import make_env
from rl.experiments import run_full_benchmark
from rl.monte_carlo import default_blackjack_policy, estimate_pi, first_visit_mc_prediction
from rl.policy_iteration import policy_iteration
from rl.td_control import double_q_learning, expected_sarsa, q_learning, sarsa
from rl.tabular import evaluate_policy
from rl.value_iteration import value_iteration
from rl.visualization import generate_demo_plots, plot_bandit_regret, plot_learning_curve


def _add_td_args(cmd):
    cmd.add_argument("--episodes", type=int, default=8000)
    cmd.add_argument("--alpha", type=float, default=0.5)
    cmd.add_argument("--gamma", type=float, default=0.99)
    cmd.add_argument("--epsilon", type=float, default=0.15)
    cmd.add_argument("--epsilon-decay", type=float, default=0.002)
    cmd.add_argument("--seed", type=int, default=42)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Advanced reinforcement learning toolkit.")
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("policy-iteration", "value-iteration"):
        cmd = sub.add_parser(name.replace("-", "_") if False else name)
        cmd.add_argument("--gamma", type=float, default=1.0)

    for name, help_text in (
        ("q-learning", "Off-policy Q-learning"),
        ("sarsa", "On-policy SARSA"),
        ("expected-sarsa", "Expected SARSA"),
        ("double-q", "Double Q-learning"),
    ):
        cmd = sub.add_parser(name, help=help_text)
        _add_td_args(cmd)

    mc = sub.add_parser("monte-carlo-blackjack", help="Monte Carlo prediction on Blackjack")
    mc.add_argument("--episodes", type=int, default=100000)

    pi = sub.add_parser("estimate-pi", help="Estimate pi with Monte Carlo sampling")
    pi.add_argument("--samples", type=int, default=100000)

    bandits = sub.add_parser("bandits", help="Compare multi-armed bandit strategies")
    bandits.add_argument("--steps", type=int, default=5000)
    bandits.add_argument("--seed", type=int, default=42)
    bandits.add_argument("--output", default="outputs/plots")

    benchmark = sub.add_parser("benchmark", help="Run full benchmark suite and export JSON")
    benchmark.add_argument("--output", default="outputs")

    dqn = sub.add_parser("train-dqn", help="Train DQN on CartPole")
    dqn.add_argument("--episodes", type=int, default=150)
    dqn.add_argument("--seed", type=int, default=42)

    plot = sub.add_parser("plot", help="Generate demo plots")
    plot.add_argument("--output", default="outputs/plots")

    sub.add_parser("compare", help="Compare classic tabular algorithms")
    sub.add_parser("leaderboard", help="Compare TD algorithms and print leaderboard")
    return parser


def _run_td(fn, args):
    env = make_env("frozenlake")
    result = fn(
        env,
        episodes=args.episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        eval_every=1000,
        eval_episodes=100,
        seed=args.seed,
    )
    avg = evaluate_policy(env, result.policy, seed=args.seed)
    print(result)
    print(f"Average return over 1000 episodes: {avg:.3f}")
    if result.eval_returns:
        print(f"Eval checkpoints: {[round(x, 3) for x in result.eval_returns]}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        if args.command == "policy-iteration":
            env = make_env("frozenlake")
            result = policy_iteration(env, gamma=args.gamma)
            print(result)
            print(f"Average return: {evaluate_policy(env, result.policy):.3f}")
            return 0

        if args.command == "value-iteration":
            env = make_env("frozenlake")
            result = value_iteration(env, gamma=args.gamma)
            print(result)
            print(f"Average return: {evaluate_policy(env, result.policy):.3f}")
            return 0

        td_map = {
            "q-learning": q_learning,
            "sarsa": sarsa,
            "expected-sarsa": expected_sarsa,
            "double-q": double_q_learning,
        }
        if args.command in td_map:
            _run_td(td_map[args.command], args)
            return 0

        if args.command == "monte-carlo-blackjack":
            env = make_env("blackjack")
            values = first_visit_mc_prediction(env, default_blackjack_policy, episodes=args.episodes)
            print(f"Learned values for {len(values)} Blackjack states")
            return 0

        if args.command == "estimate-pi":
            print(estimate_pi(samples=args.samples))
            return 0

        if args.command == "bandits":
            means = [0.2, 0.5, 0.3, 0.8, 0.1]
            results = [
                epsilon_greedy_bandit(means, steps=args.steps, seed=args.seed),
                ucb1_bandit(means, steps=args.steps, seed=args.seed),
                thompson_sampling_bandit(means, steps=args.steps, seed=args.seed),
            ]
            for result in results:
                print(result)
            plot_bandit_regret(results, Path(args.output) / "bandit_regret.png")
            return 0

        if args.command == "benchmark":
            reports = run_full_benchmark(output_dir=args.output, seed=42)
            print(f"Wrote benchmarks to {args.output}/")
            print(f"Tabular algorithms tested: {len(reports['tabular']['results'])}")
            return 0

        if args.command == "train-dqn":
            env = make_env("cartpole")
            result = train_dqn(env, DQNConfig(episodes=args.episodes, seed=args.seed, eval_every=25))
            print(result)
            return 0

        if args.command == "plot":
            paths = generate_demo_plots(args.output)
            for path in paths:
                print(f"Saved {path}")
            return 0

        if args.command == "leaderboard":
            report = compare_td_algorithms()
            print_leaderboard(report)
            return 0

        env = make_env("frozenlake")
        pi_result = policy_iteration(env, gamma=1.0)
        vi_result = value_iteration(env, gamma=1.0)
        ql_result = q_learning(env, episodes=6000, gamma=0.99, epsilon=0.15, eval_every=1000)
        print(pi_result)
        print(vi_result)
        print(ql_result)
        print(f"Policy iteration return: {evaluate_policy(env, pi_result.policy):.3f}")
        print(f"Value iteration return:  {evaluate_policy(env, vi_result.policy):.3f}")
        print(f"Q-learning return:       {evaluate_policy(env, ql_result.policy):.3f}")
        return 0
    except ImportError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
