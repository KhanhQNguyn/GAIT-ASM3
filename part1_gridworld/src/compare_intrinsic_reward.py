"""Creativity/Task 5 evidence: train level 6 with intrinsic reward ON vs OFF
(identical seed), plot both curves together, and persist both Q-tables.

Regenerates report/figures/task5_intrinsic_comparison_level6.png reproducibly
(that figure previously existed with no producing script).
"""

from __future__ import annotations

import pathlib

from src.algorithms import qtable_path, save_qtable
from src.plot_results import plot_training_curve
from src.trainer import train

COMPARISON_LEVEL_ID = 6
LOGS_DIR = pathlib.Path(__file__).resolve().parent.parent / "logs" / "task5"


def run_comparison(seed: int = 0) -> dict[str, pathlib.Path]:
    """Train q_learning on level 6 with and without intrinsic reward using an
    identical seed, plot both curves together, save both Q-tables, and return
    the produced paths.

    The no-intrinsic Q-table is persisted at the canonical
    qtable_path(6, "q_learning"); the with-intrinsic one at the same path with
    an "_intrinsic" suffix, so both are loadable without overwriting.

    Returns:
        {"with_intrinsic": Path, "without_intrinsic": Path, "figure": Path}
    """
    csv_paths: dict[str, pathlib.Path] = {}
    q_paths: dict[str, pathlib.Path] = {}
    for label, use_intrinsic in (("with_intrinsic", True), ("without_intrinsic", False)):
        csv_path = LOGS_DIR / f"task5_level{COMPARISON_LEVEL_ID}_{label}.csv"
        q_table = train(
            level_id=COMPARISON_LEVEL_ID,
            algorithm="q_learning",
            seed=seed,
            render=False,
            use_intrinsic_reward=use_intrinsic,
            csv_log_path=csv_path,
        )
        base = qtable_path(COMPARISON_LEVEL_ID, "q_learning")
        q_path = (
            base
            if not use_intrinsic
            else base.with_name(base.stem + "_intrinsic" + base.suffix)
        )
        save_qtable(q_table, q_path)
        csv_paths[label] = csv_path
        q_paths[label] = q_path

    figure_path = plot_training_curve(
        {label: str(path) for label, path in csv_paths.items()},
        title=f"Task 5: Intrinsic reward on vs off (level {COMPARISON_LEVEL_ID})",
        output_name=f"task5_intrinsic_comparison_level{COMPARISON_LEVEL_ID}.png",
    )
    return {**csv_paths, "figure": figure_path}


if __name__ == "__main__":
    paths = run_comparison()
    for key, p in paths.items():
        print(f"{key}: {p}")
