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

MODELS_DIR = pathlib.Path(__file__).resolve().parent.parent / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algo", type=str, choices=["ppo", "dqn"], default="ppo")
    parser.add_argument("--curriculum", type=str, choices=["on", "off"], default="off")
    parser.add_argument("--config", type=str, default="tuned_v1",
                        help="hyperparameter preset the model was trained with (part of its filename)")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--fps", type=int, default=60,
                        help="frame-rate cap for the render loop so playback is watchable / recordable")
    return parser.parse_args()


def main() -> None:
    """TODO:
      1. args = parse_args(). seed_utils.set_seed(0) for a reproducible demo.
      2. Build arena.gym_adapter.ArenaGymEnv(control_style=1,
         render_mode="human").
      3. Load the matching saved model (PPO.load / DQN.load) from
         MODELS_DIR / f"style1_{args.algo}_{args.config}"
         (+ "_curriculum" if args.curriculum == "on").
      4. Run args.episodes episodes with deterministic=True actions,
         calling env.render() each step so the Pygame window updates live.
         Pace the loop with a pygame.time.Clock().tick(args.fps) each step
         -- without it the window blurs past far too fast to film.
      5. Print per-episode return/outcome summaries.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
