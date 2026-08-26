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
    return parser.parse_args()


def read_all_term_scalars(log_dir: pathlib.Path) -> dict:
    """Read each tag in REWARD_TERM_TAGS from the TensorBoard event file(s)
    under log_dir into a {tag: (steps, values)} dict.

    TODO: implement (same TensorBoard-reading approach as
    compare_ppo_dqn.read_tensorboard_scalars -- consider factoring a shared
    helper once both are implemented).
    """
    raise NotImplementedError


def plot_stacked_area(
    term_series: dict, output_name: str = "reward_decomposition.png"
) -> pathlib.Path:
    """Render a stacked-area chart of each reward term's contribution over
    training steps, save to FIGURES_DIR / output_name.

    TODO: implement with matplotlib (plt.stackplot).
    """
    raise NotImplementedError


if __name__ == "__main__":
    args = parse_args()
    # TODO: locate the right log_dir for (style, algo), read_all_term_scalars,
    # plot_stacked_area.
    raise NotImplementedError
