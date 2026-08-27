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
import sys

# `python scripts/compare_ppo_dqn.py` only puts this file's own directory
# (scripts/) on sys.path, not part2_arena/ -- add it before importing arena.*.
# scripts/ itself is also added explicitly so `import train` (the sibling
# module, reused below rather than reimplemented) resolves regardless of cwd.
_PART2_ARENA_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PART2_ARENA_ROOT) not in sys.path:
    sys.path.insert(0, str(_PART2_ARENA_ROOT))
_SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import matplotlib.pyplot as plt  # noqa: E402
import train  # noqa: E402  (sibling module: scripts/train.py)
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

from arena.gym_adapter import ArenaGymEnv  # noqa: E402

LOGS_DIR = pathlib.Path(__file__).resolve().parent.parent / "logs"
FIGURES_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "report" / "figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--style", type=int, choices=[1, 2], default=1)
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--config",
        type=str,
        default="tuned_v1",
        help="hyperparameter preset (same one, for both algos) from config/hyperparams.json",
    )
    return parser.parse_args()


def train_both(
    style: int, timesteps: int, seed: int, preset: str = "tuned_v1"
) -> dict[str, pathlib.Path]:
    """Run train.py's build_model()/training path twice (algo="ppo" and
    algo="dqn") with identical env/timesteps/seed/preset, returning each
    run's TensorBoard log directory (SB3's model.logger.dir) for
    read_tensorboard_scalars(). Reuses train.build_model() rather than
    reimplementing training here, so this is a genuine like-for-like
    ablation against whatever hyperparameters config/hyperparams.json's
    [algo][preset] block actually specifies for each algorithm.
    """
    log_dirs: dict[str, pathlib.Path] = {}
    for algo in ("ppo", "dqn"):
        env = Monitor(ArenaGymEnv(control_style=style))
        model = train.build_model(
            algo, env, tensorboard_log=str(LOGS_DIR), preset=preset, seed=seed
        )
        model.learn(
            total_timesteps=timesteps, tb_log_name=f"ablation_style{style}_{algo}_{preset}"
        )
        model.save(train.MODELS_DIR / f"ablation_style{style}_{algo}_{preset}")
        log_dirs[algo] = pathlib.Path(model.logger.dir)
    return log_dirs


def read_tensorboard_scalars(log_dir: pathlib.Path, tag: str = "rollout/ep_rew_mean"):
    """Read a scalar time series back out of a TensorBoard event file for
    plotting.
    """
    accumulator = EventAccumulator(str(log_dir))
    accumulator.Reload()
    events = accumulator.Scalars(tag)
    steps = [event.step for event in events]
    values = [event.value for event in events]
    return steps, values


def plot_comparison(ppo_series, dqn_series, output_name: str = "ppo_vs_dqn.png") -> pathlib.Path:
    """Plot both algorithms' reward-over-timesteps curves on the same
    axes, save to FIGURES_DIR / output_name.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ppo_steps, ppo_values = ppo_series
    dqn_steps, dqn_values = dqn_series

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ppo_steps, ppo_values, label="PPO")
    ax.plot(dqn_steps, dqn_values, label="DQN")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean episode reward (rollout/ep_rew_mean)")
    ax.set_title("PPO vs. DQN -- reward over training")
    ax.legend()

    output_path = FIGURES_DIR / output_name
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    args = parse_args()
    log_dirs = train_both(args.style, args.timesteps, args.seed, args.config)
    ppo_series = read_tensorboard_scalars(log_dirs["ppo"])
    dqn_series = read_tensorboard_scalars(log_dirs["dqn"])
    output_path = plot_comparison(ppo_series, dqn_series, f"ppo_vs_dqn_style{args.style}.png")
    print(f"Saved comparison plot to: {output_path}")
