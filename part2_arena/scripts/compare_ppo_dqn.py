"""Creativity hook (a): PPO vs. DQN ablation. Trains (or reads existing
TensorBoard logs for) both algorithms on the SAME control style,
observation space, and reward function, then plots convergence speed,
stability, and sample efficiency side by side.

The spec only requires ONE of PPO/DQN -- doing this comparison is the
bonus, and it's a genuine ablation (identical env/reward/hyperparameter
budget) rather than "two models for the sake of it."

Usage:
    python scripts/compare_ppo_dqn.py --style 1 --timesteps 300000
"""

from __future__ import annotations

import argparse
import pathlib

LOGS_DIR = pathlib.Path(__file__).resolve().parent.parent / "logs"
FIGURES_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "report" / "figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--style", type=int, choices=[1, 2], default=1)
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def train_both(style: int, timesteps: int, seed: int) -> dict[str, pathlib.Path]:
    """Run train.py's build_model/training path twice (algo="ppo" and
    algo="dqn") with identical env/timesteps/seed, returning each run's
    TensorBoard log directory for read_tensorboard_scalars().

    TODO: implement, likely by importing and reusing scripts/train.py's
    build_model() rather than reimplementing training here.
    """
    raise NotImplementedError


def read_tensorboard_scalars(log_dir: pathlib.Path, tag: str = "rollout/ep_rew_mean"):
    """Read a scalar time series back out of a TensorBoard event file for
    plotting (e.g. via tensorboard.backend.event_processing or the
    tbparse package).

    TODO: implement.
    """
    raise NotImplementedError


def plot_comparison(ppo_series, dqn_series, output_name: str = "ppo_vs_dqn.png") -> pathlib.Path:
    """Plot both algorithms' reward-over-timesteps curves on the same
    axes, save to FIGURES_DIR / output_name.

    TODO: implement with matplotlib.
    """
    raise NotImplementedError


if __name__ == "__main__":
    args = parse_args()
    # TODO: wire train_both -> read_tensorboard_scalars -> plot_comparison.
    raise NotImplementedError
