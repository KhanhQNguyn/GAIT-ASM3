"""Task 4 (Monster Levels 4-5) REQUIRED evidence: "training curves showing
learning behavior on Levels 4 and 5". There was no dedicated script for
this -- plot_results.py has the plotting functions (plot_training_curve,
plot_death_rate) but nothing called them for level4/level5. This script
fills that gap without touching any existing file, mirroring the same
pattern as compare_q_vs_sarsa.py (Task 2) and compare_intrinsic_reward.py
(Task 5).

For each monster level (4 and 5), trains Q-learning and SARSA with
identical hyperparameters/seed, then produces two figures:
  - a return-vs-episode training curve (evidence of learning), and
  - a rolling death-rate-vs-episode curve (evidence the agent learns to
    avoid the monsters specifically, per plot_death_rate's docstring).

Usage:
    python -m src.compare_monster_levels
"""

from __future__ import annotations

import pathlib

from src.plot_results import plot_death_rate, plot_training_curve
from src.trainer import train

MONSTER_LEVEL_IDS = [4, 5]
LOGS_DIR = pathlib.Path(__file__).resolve().parent.parent / "logs" / "task4"


def run_comparison(level_id: int, seed: int = 0) -> dict[str, pathlib.Path]:
    """Train Q-learning and SARSA on `level_id` with an identical seed,
    logging each to its own CSV, then produce one return-curve figure and
    one death-rate figure for that level.
    """
    csv_paths: dict[str, pathlib.Path] = {}
    for algorithm in ("q_learning", "sarsa"):
        csv_path = LOGS_DIR / f"task4_level{level_id}_{algorithm}.csv"
        train(
            level_id=level_id,
            algorithm=algorithm,
            seed=seed,
            render=False,
            csv_log_path=csv_path,
        )
        csv_paths[algorithm] = csv_path

    return_figure = plot_training_curve(
        {k: str(v) for k, v in csv_paths.items()},
        title=f"Task 4: Q-learning vs SARSA return (level {level_id})",
        output_name=f"task4_level{level_id}_return.png",
    )
    death_rate_figure = plot_death_rate(
        {k: str(v) for k, v in csv_paths.items()},
        title=f"Task 4: rolling death rate (level {level_id})",
        output_name=f"task4_level{level_id}_death_rate.png",
    )

    return {**csv_paths, "return_figure": return_figure, "death_rate_figure": death_rate_figure}


if __name__ == "__main__":
    for level_id in MONSTER_LEVEL_IDS:
        paths = run_comparison(level_id)
        for key, p in paths.items():
            print(f"level{level_id} {key}: {p}")