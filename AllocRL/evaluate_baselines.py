"""Evaluate the required heuristic baselines on fixed holdout scenarios."""

from __future__ import annotations

import argparse

import numpy as np

from baseline_policies import GreedyImmediateAreaPolicy, RandomValidPolicy
from evaluation_runner import evaluate_scenarios, write_evaluation_metrics
from evaluation_scenarios import read_scenarios
from train import DEFAULT_ACTIVE_WORKSPACE_CODES, parse_workspace_codes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate heuristic policies on fixed holdout scenarios"
    )
    parser.add_argument(
        "--scenarios", default="./data/fixed_eval_scenarios.json"
    )
    parser.add_argument(
        "--output",
        default="./output_ablation/baselines/evaluation_scenarios.csv",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--n-future-blocks", type=int, default=4)
    parser.add_argument(
        "--active-workspace-codes",
        default=DEFAULT_ACTIVE_WORKSPACE_CODES,
    )
    args = parser.parse_args()

    scenarios = read_scenarios(args.scenarios)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("limit must be positive")
        scenarios = scenarios[:args.limit]

    workspace_codes = parse_workspace_codes(args.active_workspace_codes)
    factories = (
        lambda seed: RandomValidPolicy(seed),
        lambda _seed: GreedyImmediateAreaPolicy(),
    )
    rows = []
    for factory in factories:
        rows.extend(
            evaluate_scenarios(
                factory,
                scenarios,
                grid_size=args.grid_size,
                n_future_blocks=args.n_future_blocks,
                workspace_codes=workspace_codes,
            )
        )

    write_evaluation_metrics(args.output, rows)
    for policy_name in sorted({row["policy"] for row in rows}):
        selected = [row for row in rows if row["policy"] == policy_name]
        print(
            policy_name,
            "score=",
            np.mean([
                float(row["mean_terminal_score"]) for row in selected
            ]),
            "dropout=",
            np.mean([
                float(row["mean_dropout_rate"]) for row in selected
            ]),
            "delay=",
            np.mean([
                float(row["mean_delay_days"]) for row in selected
            ]),
        )


if __name__ == "__main__":
    main()
