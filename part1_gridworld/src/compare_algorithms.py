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

from src.trainer import train

COMPARISON_LEVEL_ID = 4  # reuse the level4 hazard layout; change if level5 fits better
FIGURES_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "report" / "figures"


def run_comparison(seed: int = 0) -> dict[str, pathlib.Path]:
    """Train all three algorithms on COMPARISON_LEVEL_ID with the same
    seed/hyperparameters, logging each to its own CSV, then produce one
    combined plot via plot_results.plot_training_curve.

    TODO: implement -- call train() three times with algorithm=
    "q_learning" / "sarsa" / "expected_sarsa", then reuse
    plot_results.plot_training_curve on the resulting CSVs.
    """
    raise NotImplementedError


if __name__ == "__main__":
    run_comparison()
