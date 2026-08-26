"""Evaluation script for Control Style 2 (direct directional movement)
ONLY. A separate, standalone script from eval_style1.py by design -- see
eval_style1.py's module docstring for why these are kept distinct rather
than parametrized into one shared script.

Usage:
    python scripts/eval_style2.py [--algo ppo] [--episodes 5]
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
                        help="hyperparameter preset the model was trained with")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--fps", type=int, default=60,
                        help="frame-rate cap for the render loop (watchable playback)")
    return parser.parse_args()


def main() -> None:
    """TODO: mirror eval_style1.main() exactly, but with control_style=2
    and loading MODELS_DIR / f"style2_{args.algo}_{args.config}"
    (+ "_curriculum" if args.curriculum == "on"). Same
    pygame.time.Clock().tick(args.fps) pacing in the render loop.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
