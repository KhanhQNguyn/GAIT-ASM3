"""Train a Stable-Baselines3 agent on the arena, for one control style.

Supports both required deliverables (one model per control style) and the
creativity hooks:
  (a) --algo {ppo,dqn}: train either algorithm, enabling a PPO-vs-DQN
      ablation when run twice with the same --style.
  (c) --curriculum {on,off}: toggles PhaseManager's curriculum ramp,
      enabling a with/without curriculum learning-speed comparison.

Usage:
    python scripts/train.py --style 1 --algo ppo --timesteps 300000
    python scripts/train.py --style 2 --algo dqn --timesteps 300000 --curriculum on
"""

from __future__ import annotations

import argparse
import pathlib

MODELS_DIR = pathlib.Path(__file__).resolve().parent.parent / "models"
LOGS_DIR = pathlib.Path(__file__).resolve().parent.parent / "logs"
CONFIG_DIR = pathlib.Path(__file__).resolve().parent.parent / "config"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--style", type=int, choices=[1, 2], required=True)
    parser.add_argument("--algo", type=str, choices=["ppo", "dqn"], required=True)
    parser.add_argument("--curriculum", type=str, choices=["on", "off"], default="off")
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--config",
        type=str,
        default="tuned_v1",
        help="hyperparameter preset name in config/hyperparams.json (e.g. baseline, tuned_v1)",
    )
    return parser.parse_args()


def build_model(algo: str, env, tensorboard_log: str, preset: str = "tuned_v1"):
    """Construct a stable_baselines3.PPO or DQN model with a small MLP
    policy and MEANINGFULLY TUNED hyperparameters -- not left at library
    defaults, per the rubric. Document the tuned values (and why) in
    report/report_template.md section 4 once finalized.

    TODO: load config/hyperparams.json (CONFIG_DIR / "hyperparams.json"),
    pick the [algo][preset] block, pop "policy" and "net_arch" into
    policy_kwargs, and pass the rest straight through as PPO(...) / DQN(...)
    kwargs. Do NOT hardcode the numbers here -- keeping them in the JSON is
    what makes the section-4 hyperparameter sweep reproducible.
    """
    raise NotImplementedError


def model_save_path(style: int, algo: str, curriculum: str, preset: str = "tuned_v1") -> pathlib.Path:
    suffix = "_curriculum" if curriculum == "on" else ""
    # Preset is part of the filename so a hyperparameter sweep does not
    # overwrite its own earlier runs.
    return MODELS_DIR / f"style{style}_{algo}_{preset}{suffix}"


def main() -> None:
    """TODO:
      1. args = parse_args(); seed_utils.set_seed(args.seed) (import from the
         scripts/ dir) and pass seed=args.seed to the model constructor too.
      2. Build an arena.gym_adapter.ArenaGymEnv(control_style=args.style,
         curriculum_enabled=(args.curriculum == "on")), likely wrapped in
         a stable_baselines3.common.vec_env.DummyVecEnv/Monitor.
      3. model = build_model(args.algo, env, tensorboard_log=str(LOGS_DIR),
         preset=args.config).
      4. model.learn(total_timesteps=args.timesteps,
         callback=callbacks.RewardTermLoggingCallback(),
         tb_log_name=f"style{args.style}_{args.algo}_{args.config}").
      5. model.save(model_save_path(args.style, args.algo, args.curriculum,
         args.config)).
      6. Print a summary of where the model + TensorBoard logs landed.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
