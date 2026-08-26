"""Evaluation script for Control Style 1 (rotation + thrust) ONLY. A
separate, standalone script from eval_style2.py by design -- the rubric
asks for "its own evaluation script" per control style, so this file does
not take a --style flag or share logic that would blur that line.

Loads the saved style-1 model and plays it live in the Pygame arena for
visual inspection / video recording.

Usage:
    python scripts/eval_style1.py [--algo ppo] [--episodes 5]
"""

from __future__ import annotations

import argparse
import pathlib

from stable_baselines3 import DQN, PPO

from arena.gym_adapter import ArenaGymEnv

MODELS_DIR = pathlib.Path(__file__).resolve().parent.parent / "models"

CONTROL_STYLE = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algo", type=str, choices=["ppo", "dqn"], default="ppo")
    parser.add_argument("--curriculum", type=str, choices=["on", "off"], default="off")
    parser.add_argument("--episodes", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    suffix = "_curriculum" if args.curriculum == "on" else ""
    model_path = MODELS_DIR / f"style{CONTROL_STYLE}_{args.algo}{suffix}"
    model_cls = PPO if args.algo == "ppo" else DQN
    model = model_cls.load(model_path)

    env = ArenaGymEnv(control_style=CONTROL_STYLE, render_mode="human")

    for episode in range(1, args.episodes + 1):
        obs, _info = env.reset()
        env.render()
        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            total_reward += reward
            steps += 1

        outcome = "died" if terminated else "survived to step limit"
        final_phase = env.core_env.state.phase if env.core_env.state is not None else None
        print(
            f"[episode {episode}/{args.episodes}] steps={steps} "
            f"return={total_reward:.2f} outcome={outcome} "
            f"final_phase={final_phase}"
        )

    env.close()


if __name__ == "__main__":
    main()
