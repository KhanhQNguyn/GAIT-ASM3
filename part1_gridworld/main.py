"""Entry point for Part I. Launches the interactive Pygame menu, then either
trains live (with rendering) or loads a saved policy to watch it act --
satisfying the "visually rendered, interactive, no console display" rule.

Usage:
    python main.py
"""

from __future__ import annotations

import sys

import pygame

from src.menu import run_menu
from src.trainer import evaluate_policy, train


def main() -> None:
    """TODO:
      1. pygame.init(), create a window.
      2. selection = run_menu(screen); exit if None (window closed).
      3. If selection.watch_only: load a saved QTable for
         (selection.level_id, selection.algorithm) and call evaluate_policy
         with render=True.
         Else: call train(level_id=..., algorithm=..., render=True).
      4. Clean up (renderer.close()) and exit.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
