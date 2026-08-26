"""Entry point for Part I. Launches the interactive Pygame menu, then either
trains live (with rendering) or loads a saved policy to watch it act --
satisfying the "visually rendered, interactive, no console display" rule.

Usage:
    python main.py            # from inside part1_gridworld/
    python part1_gridworld/main.py   # from the repo root
"""

from __future__ import annotations

import pathlib
import sys

# Import-path shim (plumbing, not assignment logic): main.py uses
# `from src.trainer import ...` but the modules inside src/ import each
# other bare (`from env import ...`, `from config.rewards_constants import
# ...`), so BOTH this directory and its src/ subdir must be importable
# regardless of the current working directory (see docs/AUDIT_main.md 8.3).
_HERE = pathlib.Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pygame  # noqa: E402  (must follow the sys.path shim above)

from src.menu import run_menu  # noqa: E402
from src.trainer import evaluate_policy, train  # noqa: E402


def main() -> None:
    """TODO:
      1. pygame.init(), create a window.
      2. selection = run_menu(screen); exit if None (window closed).
      3. If selection.watch_only: load a saved QTable via
         algorithms.load_qtable(algorithms.qtable_path(selection.level_id,
         selection.algorithm)) and call evaluate_policy with render=True.
         Else: q = train(level_id=..., algorithm=..., render=True), then
         algorithms.save_qtable(q, algorithms.qtable_path(...)) so the
         policy can be replayed later for the video demo.
      4. Clean up (renderer.close()) and exit.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
