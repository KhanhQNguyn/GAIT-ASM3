"""Plot the R_DEATH ablation (docs/KHANG.md C.3).

Closes the grader criticism "there is no ablation or justification that
simpler rewards failed" -- a reward magnitude stated without evidence
invites exactly that comment again.

METHODOLOGICAL NOTE, and the reason this is not a one-line ep_rew_mean
plot: the two runs OPTIMISE DIFFERENT REWARD FUNCTIONS. A run with
R_DEATH = -30 collects a mechanically higher rollout/ep_rew_mean than one
with R_DEATH = -100 without playing any better, because dying simply costs
it less. So the primary panels use metrics that do not depend on R_DEATH:

  rollout/ep_len_mean                   -- how long the agent survives
  reward_terms_episode/kill_enemy       -- proportional to enemies killed
  reward_terms_episode/kill_spawner     -- proportional to spawners killed
                                           (R_KILL_ENEMY / R_KILL_SPAWNER
                                           are NOT overridden)

rollout/ep_rew_mean is drawn last, explicitly labelled as not directly
comparable, so a reader can see it without being misled by it.

Usage (after both runs from Task C.3 have finished):
    cd part2_arena && python scripts/plot_death_penalty_ablation.py
"""

from __future__ import annotations

import argparse
import pathlib
import sys

# `python scripts/plot_death_penalty_ablation.py` only puts this file's own
# directory (scripts/) on sys.path, not part2_arena/ -- add it before
# importing arena.* (kept for consistency with the other scripts).
_PART2_ARENA_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PART2_ARENA_ROOT) not in sys.path:
    sys.path.insert(0, str(_PART2_ARENA_ROOT))

import matplotlib.pyplot as plt  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

LOGS_DIR = pathlib.Path(__file__).resolve().parent.parent / "logs"
FIGURES_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "report" / "figures"

# (tag, panel title, comparable-across-runs?)
PANELS = [
    ("rollout/ep_len_mean", "Survival: mean episode length", True),
    ("reward_terms_episode/kill_enemy", "Offense: enemy-kill reward per episode", True),
    ("reward_terms_episode/kill_spawner", "Objective: spawner-kill reward per episode", True),
    ("rollout/ep_rew_mean", "Mean episode return (NOT comparable -- see note)", False),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--style", type=int, choices=[1, 2], default=1)
    parser.add_argument("--algo", type=str, choices=["ppo", "dqn"], default="ppo")
    parser.add_argument("--config", type=str, default="tuned_v1")
    parser.add_argument("--curriculum", type=str, choices=["on", "off"], default="on")
    parser.add_argument(
        "--penalties",
        type=float,
        nargs="+",
        default=[-100.0, -30.0],
        help="the --death-penalty values that were trained, in legend order",
    )
    return parser.parse_args()


def find_ablation_log_dir(
    style: int, algo: str, config: str, curriculum: str, penalty: float
) -> pathlib.Path:
    """Locate the newest TensorBoard directory for one ablation run.

    Mirrors train.py::run_name exactly. The "ablation_" prefix is what
    keeps these directories from being picked up by
    plot_reward_decomposition.py::find_log_dir's prefix glob for the main
    run -- do not "simplify" it away.
    """
    curriculum_suffix = "_curriculum" if curriculum == "on" else ""
    prefix = f"ablation_style{style}_{algo}_{config}{curriculum_suffix}_death{int(penalty)}_"
    candidates = sorted(
        (path for path in LOGS_DIR.glob(f"{prefix}*") if path.is_dir()),
        key=lambda path: (
            int(path.name.rsplit("_", 1)[-1]) if path.name.rsplit("_", 1)[-1].isdigit() else -1
        ),
    )
    if not candidates:
        raise FileNotFoundError(
            f"No TensorBoard run under {LOGS_DIR} matching '{prefix}*' -- run:\n"
            f"  python scripts/train.py --style {style} --algo {algo} --config {config} "
            f"--curriculum {curriculum} --timesteps 100000 --seed 0 "
            f"--death-penalty {int(penalty)}"
        )
    return candidates[-1]


def read_scalar(log_dir: pathlib.Path, tag: str):
    """Read one scalar series; return (steps, values), or ([], []) if the
    tag was never written (e.g. a term that never fired in this run).
    """
    accumulator = EventAccumulator(str(log_dir))
    accumulator.Reload()
    if tag not in set(accumulator.Tags().get("scalars", [])):
        return [], []
    events = accumulator.Scalars(tag)
    return [event.step for event in events], [event.value for event in events]


def plot_ablation(series_by_penalty: dict, output_name: str) -> pathlib.Path:
    """Render one panel per PANELS entry, both runs overlaid on each."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))

    for axis, (tag, title, comparable) in zip(axes.flat, PANELS, strict=True):
        plotted = False
        for label, series in series_by_penalty.items():
            steps, values = series.get(tag, ([], []))
            if not steps:
                continue
            axis.plot(steps, values, label=label)
            plotted = True
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("Timesteps")
        if not comparable:
            axis.set_facecolor("#f6f2ea")
        if plotted:
            axis.legend(fontsize=8)
        else:
            axis.text(0.5, 0.5, f"no data for\n{tag}", ha="center", va="center")

    figure.suptitle(
        "R_DEATH ablation, control style 1 -- bottom-right panel: return is measured\n"
        "in different units per run and is NOT a like-for-like comparison",
        fontsize=11,
    )
    figure.tight_layout()

    output_path = FIGURES_DIR / output_name
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path


def main() -> None:
    args = parse_args()

    series_by_penalty: dict[str, dict] = {}
    for penalty in args.penalties:
        log_dir = find_ablation_log_dir(
            args.style, args.algo, args.config, args.curriculum, penalty
        )
        print(f"R_DEATH={int(penalty)}: reading {log_dir}")
        series_by_penalty[f"R_DEATH = {int(penalty)}"] = {
            tag: read_scalar(log_dir, tag) for tag, _title, _comparable in PANELS
        }

    output_path = plot_ablation(
        series_by_penalty, f"death_penalty_ablation_style{args.style}.png"
    )
    print(f"Saved ablation figure to: {output_path}")

    print("\nFinal-value summary (paste into docs/DECISIONS.md):")
    for label, series in series_by_penalty.items():
        print(f"  {label}")
        for tag, _title, comparable in PANELS:
            _steps, values = series.get(tag, ([], []))
            note = "" if comparable else "   (not comparable across runs)"
            final = f"{values[-1]:.2f}" if values else "n/a"
            print(f"    {tag:<42} final={final}{note}")


if __name__ == "__main__":
    main()
