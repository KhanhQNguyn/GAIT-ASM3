"""Creativity hook (b): reward decomposition dashboard. Reads the per-term
reward scalars that train.py logs to TensorBoard (see rewards.py's
RewardBreakdown and its per-field logging in train.py) and renders a
stacked-area chart showing each term's contribution to total reward over
training time.

This pairs with the architecture principle capping reward-shaping terms at
6-8 named constants -- decomposition only stays readable because the term
count is capped.

Usage:
    python scripts/plot_reward_decomposition.py --style 1 --algo ppo
"""

from __future__ import annotations

import argparse
import pathlib
import sys

# `python scripts/plot_reward_decomposition.py` only puts this file's own
# directory (scripts/) on sys.path, not part2_arena/ -- add it before
# importing arena.* (not needed directly here, but kept consistent with the
# rest of the Member D scripts in case a future revision needs it).
_PART2_ARENA_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PART2_ARENA_ROOT) not in sys.path:
    sys.path.insert(0, str(_PART2_ARENA_ROOT))

import matplotlib.pyplot as plt  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

LOGS_DIR = pathlib.Path(__file__).resolve().parent.parent / "logs"
FIGURES_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "report" / "figures"

REWARD_TERM_TAGS = [
    "reward_terms/kill_enemy",
    "reward_terms/kill_spawner",
    "reward_terms/phase_progress",
    "reward_terms/damage_taken",
    "reward_terms/death",
    "reward_terms/approach_nearest_enemy",
    "reward_terms/shoot_while_no_target",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--style", type=int, choices=[1, 2], default=1)
    parser.add_argument("--algo", type=str, choices=["ppo", "dqn"], default="ppo")
    parser.add_argument("--curriculum", type=str, choices=["on", "off"], default="off")
    return parser.parse_args()


def find_log_dir(style: int, algo: str, curriculum: str) -> pathlib.Path:
    """Locate the most recent TensorBoard run directory for (style, algo,
    curriculum), matching the tb_log_name train.py's main() passes to
    model.learn() (e.g. "style1_ppo", "style2_dqn_curriculum"), including
    SB3's auto-appended "_N" run suffix. Picks the highest-numbered (most
    recent) run if train.py was run more than once for this combination.
    """
    suffix = "_curriculum" if curriculum == "on" else ""
    prefix = f"style{style}_{algo}{suffix}_"
    candidates = sorted(
        (p for p in LOGS_DIR.glob(f"{prefix}*") if p.is_dir()),
        key=lambda p: int(p.name.rsplit("_", 1)[-1]) if p.name.rsplit("_", 1)[-1].isdigit() else -1,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No TensorBoard run directory found under {LOGS_DIR} matching '{prefix}*' "
            f"-- run scripts/train.py --style {style} --algo {algo} "
            f"--curriculum {curriculum} first."
        )
    return candidates[-1]


def read_all_term_scalars(log_dir: pathlib.Path) -> dict:
    """Read each tag in REWARD_TERM_TAGS from the TensorBoard event file(s)
    under log_dir into a {tag: (steps, values)} dict. Tags that were never
    logged (e.g. a term that never fired during this run) are omitted
    rather than raising.
    """
    accumulator = EventAccumulator(str(log_dir))
    accumulator.Reload()
    available = set(accumulator.Tags().get("scalars", []))

    result = {}
    for tag in REWARD_TERM_TAGS:
        if tag not in available:
            continue
        events = accumulator.Scalars(tag)
        result[tag] = ([event.step for event in events], [event.value for event in events])
    return result


def plot_stacked_area(
    term_series: dict, output_name: str = "reward_decomposition.png"
) -> pathlib.Path:
    """Render a stacked-area chart of each reward term's contribution over
    training steps, save to FIGURES_DIR / output_name.
    """
    if not term_series:
        raise ValueError(
            "term_series is empty -- no reward_terms/* tags found in the log "
            "directory (was RewardBreakdownCallback active during training?)"
        )

    # All terms are logged together every rollout by the same callback, so
    # their series should already be the same length; truncate to the
    # shortest just in case a term started logging a step later than others.
    min_len = min(len(steps) for steps, _values in term_series.values())
    steps = next(iter(term_series.values()))[0][:min_len]
    labels = [tag.split("/", 1)[-1] for tag in term_series]
    series = [values[:min_len] for _steps, values in term_series.values()]

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.stackplot(steps, *series, labels=labels)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Reward contribution (rolling mean per rollout)")
    ax.set_title("Reward term decomposition over training")
    ax.legend(loc="upper left", fontsize="small")

    output_path = FIGURES_DIR / output_name
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    args = parse_args()
    log_dir = find_log_dir(args.style, args.algo, args.curriculum)
    term_series = read_all_term_scalars(log_dir)
    output_name = f"reward_decomposition_style{args.style}_{args.algo}.png"
    output_path = plot_stacked_area(term_series, output_name)
    print(f"Saved reward decomposition plot to: {output_path}")
