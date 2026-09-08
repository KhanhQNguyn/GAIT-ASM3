"""Evaluation script for Control Style 2 (direct directional movement)
ONLY. A separate, standalone script from eval_style1.py by design -- see
eval_style1.py's module docstring for why these are kept distinct rather
than parametrized into one shared script.

Usage:
    python scripts/eval_style2.py [--algo ppo] [--episodes 5] [--sampling stochastic]
"""

from __future__ import annotations

import argparse
import pathlib
import sys

# `python scripts/eval_style2.py` only puts this file's own directory
# (scripts/) on sys.path, not part2_arena/ -- add it before importing arena.*.
_PART2_ARENA_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PART2_ARENA_ROOT) not in sys.path:
    sys.path.insert(0, str(_PART2_ARENA_ROOT))

import pygame  # noqa: E402
from stable_baselines3 import DQN, PPO  # noqa: E402

from arena.gym_adapter import ArenaGymEnv  # noqa: E402

MODELS_DIR = pathlib.Path(__file__).resolve().parent.parent / "models"

CONTROL_STYLE = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algo", type=str, choices=["ppo", "dqn"], default="ppo")
    parser.add_argument("--curriculum", type=str, choices=["on", "off"], default="on")
    parser.add_argument(
        "--config",
        type=str,
        default="tuned_v3",
        help="hyperparameter preset the model was trained with (part of its filename). "
        "tuned_v3 since the 2026-09-08c fix (was tuned_v4 / gamma 0.999 -- retired, see "
        "config/hyperparams.json _notes).",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument(
        "--sampling",
        type=str,
        choices=["stochastic", "deterministic"],
        default="stochastic",
        help="PPO is a stochastic policy: sampling from the trained policy's "
        "action distribution (stochastic) shows its full learned behaviour, "
        "including phase progression; argmax (deterministic) collapses to "
        "the policy's single most-likely action. DQN users may prefer "
        "deterministic.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="frame-rate cap for the render loop so playback is watchable / recordable",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    suffix = "_curriculum" if args.curriculum == "on" else ""
    model_path = MODELS_DIR / f"style{CONTROL_STYLE}_{args.algo}_{args.config}{suffix}"
    model_cls = PPO if args.algo == "ppo" else DQN
    model = model_cls.load(model_path)

    env = ArenaGymEnv(
        control_style=CONTROL_STYLE,
        curriculum_enabled=(args.curriculum == "on"),  # match training conditions
        render_mode="human",
    )
    clock = pygame.time.Clock()

    # Space pauses, '[' / ']' change playback speed, R restarts the current
    # episode, N skips to the next one (see ArenaRenderer.handle_events()).
    episode = 1
    while episode <= args.episodes:
        obs, _info = env.reset()
        if not env.render():
            break
        clock.tick(args.fps * env.speed_multiplier)
        total_reward, steps = 0.0, 0
        terminated = truncated = False

        while not (terminated or truncated):
            while env.is_paused:
                if not env.render():
                    terminated = truncated = True
                    break
                clock.tick(args.fps * env.speed_multiplier)
            if terminated or truncated:
                break
            if env.consume_restart_request():
                obs, _info = env.reset()
                total_reward, steps = 0.0, 0
                continue
            if env.consume_skip_request():
                break

            action, _state = model.predict(
                obs, deterministic=(args.sampling == "deterministic")
            )
            obs, reward, terminated, truncated, info = env.step(action)
            if not env.render():
                terminated = truncated = True
            clock.tick(args.fps * env.speed_multiplier)
            total_reward += reward
            steps += 1

        outcome = "died" if terminated else "survived to step limit"
        final_phase = env.core_env.state.phase if env.core_env.state is not None else None
        env.show_episode_end_banner(
            {"return": total_reward, "steps": steps, "phase": final_phase, "outcome": outcome}
        )
        for _ in range(int(args.fps * 2)):  # hold the banner ~2s before the next episode
            if not env.render():
                break
            clock.tick(args.fps)
        print(
            f"[episode {episode}/{args.episodes}] steps={steps} "
            f"return={total_reward:.2f} outcome={outcome} "
            f"final_phase={final_phase}"
        )
        episode += 1

    env.close()


if __name__ == "__main__":
    main()
