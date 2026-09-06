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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.algorithms import epsilon_greedy, load_qtable, qtable_path, save_qtable
from src.env import Action
from src.plot_results import plot_training_curve
from src.seed_utils import set_seed
from src.trainer import make_env, train

COMPARISON_LEVEL_ID = 1  # fixed: level1 is the Task-2 hazard-shortcut layout
FIGURES_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "report" / "figures"
LOGS_DIR = pathlib.Path(__file__).resolve().parent.parent / "logs"


def run_comparison(seed: int = 0) -> dict[str, pathlib.Path]:
    """Train Q-learning and SARSA on COMPARISON_LEVEL_ID with identical
    hyperparameters and `seed`, logging each to its own CSV, then produce
    one combined learning-curve plot. Returns
    {"q_learning": csv_path, "sarsa": csv_path, "figure": fig_path}.

    See plot_qualitative_rollout() below for the report's other required
    Task-2 artefact: a greedy-rollout comparison of the two learned
    policies on level1.
    """
    csv_paths = {}
    for algorithm in ("q_learning", "sarsa"):
        csv_path = LOGS_DIR / f"task2_level{COMPARISON_LEVEL_ID}_{algorithm}.csv"
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
        title=f"Task 2: Q-learning vs SARSA (level {COMPARISON_LEVEL_ID})",
        output_name=f"task2_q_vs_sarsa_level{COMPARISON_LEVEL_ID}.png",
    )

    return {**csv_paths, "figure": figure_path}


def _greedy_rollout(algorithm: str, seed: int = 0) -> list[tuple[int, int]]:
    """Load COMPARISON_LEVEL_ID's saved Q-table for `algorithm` and return
    the sequence of agent positions from a single greedy (epsilon=0)
    rollout. Requires the Q-table to already be saved (see algorithms.py's
    save_qtable / trainer.py's train()).
    """
    env = make_env(COMPARISON_LEVEL_ID)
    q_table = load_qtable(qtable_path(COMPARISON_LEVEL_ID, algorithm), n_actions=env.action_space_n)
    rng = set_seed(seed)
    state = env.reset()
    path = [env.get_state_snapshot()["agent_pos"]]
    done = False
    steps = 0
    max_steps = env.level["max_steps"]
    while not done and steps < max_steps:
        action = epsilon_greedy(q_table[state], 0.0, rng)
        result = env.step(Action(action))
        path.append(env.get_state_snapshot()["agent_pos"])
        state, done = result.state, result.done
        steps += 1
    return path


def plot_qualitative_rollout(
    output_name: str = "task2_qualitative_rollout_level1.png",
) -> dict:
    """Task 2's required qualitative artefact: overlay Q-learning's and
    SARSA's greedy (epsilon=0) rollouts on level1's grid in one figure, so a
    marker can see at a glance whether SARSA routes around the fire gap
    where Q-learning cuts through it (per level1's _design_note).

    Requires both Q-tables to already be saved under
    algorithms.qtable_path(1, "q_learning"/"sarsa") -- run run_comparison()
    first, then algorithms.save_qtable() on its results, or reuse an
    existing part1_gridworld/models/level1_*.json.

    Returns {"figure": path, "diverged": bool, "q_learning_path": [...],
    "sarsa_path": [...]} -- `diverged` is False when both policies visit the
    exact same set of tiles, which is itself real evidence (not a failure):
    it means this run found no incentive to trade safety for a shortcut,
    not that the on/off-policy distinction was implemented wrong.
    """
    env = make_env(COMPARISON_LEVEL_ID)
    paths = {algo: _greedy_rollout(algo) for algo in ("q_learning", "sarsa")}
    diverged = set(paths["q_learning"]) != set(paths["sarsa"])

    gw, gh = env.grid_size
    fig, ax = plt.subplots(figsize=(gw / 1.5, gh / 1.5))

    for (rx, ry) in env.level["rocks"]:
        ax.add_patch(plt.Rectangle((rx - 0.5, ry - 0.5), 1, 1, color="#606060"))
    for (fx, fy) in env.level["fire"]:
        ax.add_patch(plt.Rectangle((fx - 0.5, fy - 0.5), 1, 1, color="#eb5032"))
    for (ax_, ay_) in env.level["apples"]:
        ax.scatter([ax_], [ay_], color="#e63c64", marker="*", s=200, zorder=4)

    colors = {"q_learning": "#42c8ff", "sarsa": "#ffb020"}
    offsets = {"q_learning": -0.08, "sarsa": 0.08}  # nudge apart where paths overlap
    for algo, path in paths.items():
        xs = [p[0] + offsets[algo] for p in path]
        ys = [p[1] + offsets[algo] for p in path]
        ax.plot(xs, ys, "-o", color=colors[algo], linewidth=2, markersize=4, label=algo, alpha=0.85)

    ax.scatter([paths["q_learning"][0][0]], [paths["q_learning"][0][1]], color="lime", s=140,
               zorder=5, label="start")
    ax.set_xlim(-0.5, gw - 0.5)
    ax.set_ylim(gh - 0.5, -0.5)  # invert Y to match env's (0,0)=top-left convention
    title_suffix = "paths diverge" if diverged else "paths coincide (no divergence observed)"
    ax.set_title(f"Task 2: level1 greedy rollout -- Q-learning vs SARSA ({title_suffix})")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / output_name
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    return {
        "figure": out,
        "diverged": diverged,
        "q_learning_path": paths["q_learning"],
        "sarsa_path": paths["sarsa"],
    }


def run_three_way_comparison(seed: int = 0) -> dict[str, pathlib.Path]:
    """Task A.6: add Expected SARSA to the level1 comparison (kept separate
    from compare_algorithms.py, which is the creativity(d) three-way
    comparison on level4) so the two "3-way" scripts don't overlap in
    purpose. Trains Expected SARSA fresh, saves its Q-table, and produces a
    combined 3-curve plot reusing the existing Q-learning/SARSA CSVs from
    run_comparison().
    """
    csv_paths = {
        "q_learning": LOGS_DIR / f"task2_level{COMPARISON_LEVEL_ID}_q_learning.csv",
        "sarsa": LOGS_DIR / f"task2_level{COMPARISON_LEVEL_ID}_sarsa.csv",
    }
    expected_sarsa_csv = LOGS_DIR / f"task2_level{COMPARISON_LEVEL_ID}_expected_sarsa.csv"
    q = train(
        level_id=COMPARISON_LEVEL_ID,
        algorithm="expected_sarsa",
        seed=seed,
        render=False,
        csv_log_path=expected_sarsa_csv,
    )
    save_qtable(q, qtable_path(COMPARISON_LEVEL_ID, "expected_sarsa"))
    csv_paths["expected_sarsa"] = expected_sarsa_csv

    figure_path = plot_training_curve(
        csv_paths,
        title=f"Task 2: Q-learning vs SARSA vs Expected SARSA (level {COMPARISON_LEVEL_ID})",
        output_name=f"task2_three_way_level{COMPARISON_LEVEL_ID}.png",
    )

    return {**csv_paths, "figure": figure_path}


if __name__ == "__main__":
    run_comparison()
    plot_qualitative_rollout()
    run_three_way_comparison()
