"""Task 2 (rubric row Part I-C) REQUIRED comparison: Q-learning vs SARSA on
the SAME hazard level (level1), same exploration schedule and seed, plotted
together to show SARSA's more conservative behaviour around the fire gap.

This is deliberately separate from compare_algorithms.py, which is the
creativity(d) THREE-algorithm comparison (adds Expected SARSA) on level4.
Keeping the required Task-2 evidence in its own single-purpose script makes
it easy for a marker to find and run, and keeps its level fixed at level1
(the layout whose _design_note sets up the conservative-vs-greedy contrast).

Usage:
    python src/compare_q_vs_sarsa.py
"""

from __future__ import annotations

import pathlib

from src.trainer import train

COMPARISON_LEVEL_ID = 1  # fixed: level1 is the Task-2 hazard-shortcut layout
FIGURES_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "report" / "figures"


def run_comparison(seed: int = 0) -> dict[str, pathlib.Path]:
    """Train Q-learning and SARSA on COMPARISON_LEVEL_ID with identical
    hyperparameters and `seed`, logging each to its own CSV, then produce
    one combined learning-curve plot.

    TODO: implement -- call train(level_id=COMPARISON_LEVEL_ID,
    algorithm="q_learning", seed=seed, csv_log_path=...) and again with
    algorithm="sarsa", then reuse plot_results.plot_training_curve on the
    two resulting CSVs (output_name e.g. "task2_q_vs_sarsa_level1.png").
    Return {"q_learning": csv_path, "sarsa": csv_path, "figure": fig_path}.

    The report (section 6) also needs a qualitative artefact alongside the
    curves: a greedy (epsilon=0) rollout of each learned policy on level1,
    showing SARSA routing around the fire gap where Q-learning cuts through
    it. Produce that via trainer.evaluate_policy with render=True, or a
    saved screenshot -- not part of this function's return, but note it here
    so it is not forgotten.
    """
    raise NotImplementedError


if __name__ == "__main__":
    run_comparison()
