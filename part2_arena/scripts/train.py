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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--style", type=int, choices=[1, 2], required=True)
    parser.add_argument("--algo", type=str, choices=["ppo", "dqn"], required=True)
    parser.add_argument("--curriculum", type=str, choices=["on", "off"], default="off")
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def build_model(algo: str, env, tensorboard_log: str):
    """Construct a stable_baselines3.PPO or DQN model with a small MLP
    policy and MEANINGFULLY TUNED hyperparameters -- not left at library
    defaults, per the rubric. Document the tuned values (and why) in
    report/report_template.md section 4 once finalized.

    TODO: implement, e.g.:
        from stable_baselines3 import PPO, DQN
        policy_kwargs = dict(net_arch=[...])
        if algo == "ppo":
            return PPO("MlpPolicy", env, policy_kwargs=policy_kwargs,
                       learning_rate=..., n_steps=..., batch_size=...,
                       gamma=..., tensorboard_log=tensorboard_log, verbose=1)
        else:
            return DQN("MlpPolicy", env, policy_kwargs=policy_kwargs,
                       learning_rate=..., buffer_size=..., batch_size=...,
                       gamma=..., exploration_fraction=...,
                       tensorboard_log=tensorboard_log, verbose=1)
    """
    raise NotImplementedError


def model_save_path(style: int, algo: str, curriculum: str) -> pathlib.Path:
    suffix = "_curriculum" if curriculum == "on" else ""
    return MODELS_DIR / f"style{style}_{algo}{suffix}"


def main() -> None:
    """TODO:
      1. args = parse_args(); seed everything.
      2. Build an arena.gym_adapter.ArenaGymEnv(control_style=args.style,
         curriculum_enabled=(args.curriculum == "on")), likely wrapped in
         a stable_baselines3.common.vec_env.DummyVecEnv/Monitor.
      3. model = build_model(args.algo, env, tensorboard_log=str(LOGS_DIR)).
      4. model.learn(total_timesteps=args.timesteps).
      5. model.save(model_save_path(args.style, args.algo, args.curriculum)).
      6. Print a summary of where the model + TensorBoard logs landed.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
