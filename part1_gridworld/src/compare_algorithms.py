"""Creativity hook (d): run Q-learning, SARSA, and Expected SARSA on the
SAME hazard-containing level (level4 or level5) with identical
hyperparameters and seeds where possible, then plot all three learning
curves together.

This is intentionally separate from plot_results.py because it's comparing
ALGORITHMS on one level, not the same algorithm across levels/intrinsic
settings -- keeping it a distinct script makes the creativity contribution
easy for a marker to find and run standalone.
"""

from __future__ import annotations

import pathlib

from src.plot_results import plot_training_curve
from src.trainer import train

COMPARISON_LEVEL_ID = 4  # reuse the level4 hazard layout; change if level5 fits better
FIGURES_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "report" / "figures"
LOGS_DIR = pathlib.Path(__file__).resolve().parent.parent / "logs"


def run_comparison(seed: int = 0) -> dict[str, pathlib.Path]:
    """Train all three algorithms on COMPARISON_LEVEL_ID with the same
    seed/hyperparameters, logging each to its own CSV, then produce one
    combined plot via plot_results.plot_training_curve.
    """
    csv_paths = {}
    for algorithm in ("q_learning", "sarsa", "expected_sarsa"):
        csv_path = LOGS_DIR / f"compare_level{COMPARISON_LEVEL_ID}_{algorithm}.csv"
        train(
            level_id=COMPARISON_LEVEL_ID,
            algorithm=algorithm,
            seed=seed,
            render=False,
            csv_log_path=csv_path,
        )
        csv_paths[algorithm] = csv_path

    figure_path = plot_training_curve(
        csv_paths,
        title=f"Q-learning vs SARSA vs Expected SARSA (level {COMPARISON_LEVEL_ID})",
        output_name=f"compare_algorithms_level{COMPARISON_LEVEL_ID}.png",
    )

    return {**csv_paths, "figure": figure_path}


if __name__ == "__main__":
    run_comparison()
