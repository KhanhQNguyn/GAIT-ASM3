"""Task 3 (Key/Chest Levels 2-3) REQUIRED evidence: training curves and a
greedy batch-evaluation table showing the agent learns the multi-step
key -> chest dependency on both level2 (open room) and level3 (maze).

There was no dedicated script for this section -- `report/figures/part1/
task_level2_3_batch_eval.csv` existed as an orphan artefact with nothing
that regenerates it, and there was no figure at all. This script fills that
gap without touching any existing file, mirroring the exact pattern of
compare_q_vs_sarsa.py (Task 2), compare_monster_levels.py (Task 4), and
compare_intrinsic_reward.py (Task 5).

For each key/chest level (2 and 3) it trains Q-learning and SARSA with the
config-driven hyperparameters and an identical seed, then produces:
  - one 2-panel return-vs-episode figure (evidence of learning; each panel
    also draws the "all apples + chest opened" return as a reference line,
    computed from the level layout and config/rewards_constants.py -- not
    hardcoded), and
  - one CSV table of 200-episode greedy (epsilon=0) evaluation stats
    (success rate, mean/std steps, key-before-chest fraction) per
    level x algorithm.

Note on `key_before_chest_fraction`: it is 1.0 by construction (env.py only
opens the chest while `_has_key` is True), so it is a consistency check, NOT
independent proof the dependency was learned. The number that speaks to
"learned, not stumbled into" is `success_rate` under a purely greedy policy
-- see trainer.evaluate_policy_batch's docstring.

Side effects: writes logs/task3/*.csv (new folder) and rewrites the two
figure artefacts above; also overwrites models/level/level{2,3}/level{2,3}_
{q_learning,sarsa}.json with the freshly trained tables (same reproducible-
retrain pattern compare_q_vs_sarsa.py uses for level1). The batch-eval CSV
gains an `algorithm` column versus the previous orphan version; nothing else
in the repo reads that file.

Usage:
    python -m src.compare_key_chest
"""

from __future__ import annotations

import csv
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config.rewards_constants import REWARD_APPLE, REWARD_CHEST, REWARD_KEY
from src.algorithms import load_qtable, qtable_path, save_qtable
from src.plot_results import load_episode_csv
from src.trainer import evaluate_policy_batch, make_env, train

KEY_CHEST_LEVEL_IDS = (2, 3)
ALGORITHMS = ("q_learning", "sarsa")
EVAL_EPISODES = 200
SMOOTH_WINDOW = 20  # matches plot_results.plot_training_curve

LOGS_DIR = pathlib.Path(__file__).resolve().parent.parent / "logs" / "task3"
FIGURES_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "report" / "figures" / "part1"
CURVE_FIGURE_NAME = "task_level2_3_key_chest_curve.png"
EVAL_TABLE_NAME = "task_level2_3_batch_eval.csv"

_ALGO_COLORS = {"q_learning": "#42c8ff", "sarsa": "#ffb020"}


def _solved_return(level_id: int) -> float:
    """Return the episode return of a perfect run on `level_id`: every apple
    collected plus the chest opened, using config/rewards_constants.py so the
    reference line tracks any future reward-constant change (REWARD_KEY is 0
    per spec but included for completeness).
    """
    env = make_env(level_id)
    n_apples = len(env._apple_positions)
    has_chest = env.level.get("chest") is not None
    has_key = env.level.get("key") is not None
    return (
        n_apples * REWARD_APPLE
        + (REWARD_KEY if has_key else 0.0)
        + (REWARD_CHEST if has_chest else 0.0)
    )


def _plot_key_chest_curves(curve_csv_map: dict[int, dict[str, pathlib.Path]]) -> pathlib.Path:
    """Draw one panel per key/chest level, each overlaying the Q-learning and
    SARSA return curves (raw + rolling mean) with a dashed reference line at
    the fully-solved return. Mirrors plot_results.plot_training_curve's
    smoothing (window=SMOOTH_WINDOW, np.convolve 'valid').
    """
    n = len(curve_csv_map)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 5), squeeze=False)
    for ax, (level_id, algo_paths) in zip(axes[0], sorted(curve_csv_map.items())):
        env = make_env(level_id)
        solved = _solved_return(level_id)
        # Reference line first and *behind* everything: once a run converges,
        # its raw and rolling-mean curves sit exactly on `solved` for every
        # remaining episode, so a reference line drawn on top would hide the
        # (correct, non-truncated) flat tail of the curves.
        ax.axhline(solved, color="#3ba55d", linestyle="--", linewidth=1.3,
                   alpha=0.55, zorder=1, label=f"all apples + chest ({solved:g})")
        for algo, path in algo_paths.items():
            data = load_episode_csv(path)
            eps = data["episode"]
            ret = np.asarray(data["return"], dtype=float)
            color = _ALGO_COLORS.get(algo)
            ax.plot(eps, ret, alpha=0.15, linewidth=0.8, color=color, zorder=2)
            if len(ret) >= SMOOTH_WINDOW:
                kernel = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW
                smooth = np.convolve(ret, kernel, mode="valid")
                ax.plot(eps[SMOOTH_WINDOW - 1:], smooth, label=algo, color=color,
                        linewidth=2, alpha=0.85, zorder=3)
            else:
                ax.plot(eps, ret, label=algo, color=color, linewidth=2,
                        alpha=0.85, zorder=3)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Return (environment reward only)")
        ax.set_ylim(-0.4, solved + 0.7)
        ax.set_title(f"Level {level_id} - {env.level.get('name', '')}")
        ax.legend(loc="lower right")
    fig.suptitle("Task 3: learning the key -> chest dependency (levels 2-3)")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / CURVE_FIGURE_NAME
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def write_batch_eval_table(seed: int = 0) -> pathlib.Path:
    """Run a 200-episode greedy evaluation for each level x algorithm using
    the freshly saved Q-tables and write the aggregate stats to
    FIGURES_DIR / EVAL_TABLE_NAME. Returns the CSV path.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / EVAL_TABLE_NAME
    fieldnames = [
        "level", "algorithm", "success_rate", "mean_steps", "std_steps",
        "key_before_chest_fraction",
    ]
    rows: list[dict] = []
    for level_id in KEY_CHEST_LEVEL_IDS:
        env = make_env(level_id)
        for algorithm in ALGORITHMS:
            q_table = load_qtable(
                qtable_path(level_id, algorithm), n_actions=env.action_space_n
            )
            stats = evaluate_policy_batch(
                env, q_table, n_episodes=EVAL_EPISODES, seed=seed
            )
            rows.append({
                "level": level_id,
                "algorithm": algorithm,
                "success_rate": stats["success_rate"],
                "mean_steps": stats["mean_steps"],
                "std_steps": stats["std_steps"],
                "key_before_chest_fraction": stats["key_before_chest_fraction"],
            })
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out


def run_comparison(seed: int = 0) -> dict[str, object]:
    """Train Q-learning and SARSA on levels 2 and 3 with an identical seed,
    logging each run to its own CSV under logs/task3/, persist the freshly
    trained Q-tables to their canonical paths, then produce the combined
    learning-curve figure and the greedy batch-evaluation table.

    Returns {"figure": Path, "eval_table": Path,
             "curves": {level_id: {algorithm: csv_path}}}.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    curve_csv_map: dict[int, dict[str, pathlib.Path]] = {}
    for level_id in KEY_CHEST_LEVEL_IDS:
        curve_csv_map[level_id] = {}
        for algorithm in ALGORITHMS:
            csv_path = LOGS_DIR / f"task3_level{level_id}_{algorithm}.csv"
            q_table = train(
                level_id=level_id,
                algorithm=algorithm,
                seed=seed,
                render=False,
                csv_log_path=csv_path,
            )
            # Persist THESE tables so write_batch_eval_table() evaluates the
            # run we just logged, not a stale models/level/... file.
            save_qtable(q_table, qtable_path(level_id, algorithm))
            curve_csv_map[level_id][algorithm] = csv_path

    figure_path = _plot_key_chest_curves(curve_csv_map)
    eval_table_path = write_batch_eval_table(seed=seed)
    return {"figure": figure_path, "eval_table": eval_table_path, "curves": curve_csv_map}


if __name__ == "__main__":
    paths = run_comparison()
    print(f"figure:     {paths['figure']}")
    print(f"eval_table: {paths['eval_table']}")
    for level_id, algo_paths in paths["curves"].items():  # type: ignore[attr-defined]
        for algorithm, p in algo_paths.items():
            print(f"log level{level_id} {algorithm}: {p}")
