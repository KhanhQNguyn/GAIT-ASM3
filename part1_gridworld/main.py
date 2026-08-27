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
from src.trainer import evaluate_policy, make_env, train  # noqa: E402
from src.algorithms import load_qtable, qtable_path, save_qtable  # noqa: E402
from src.render import GridWorldRenderer, TILE_SIZE_PX  # noqa: E402

# Window dimensions — the renderer adds a HUD strip on top; we need the base
# grid size from the level, which we get after menu selection.
_DEFAULT_GRID = (10, 10)


def main() -> None:
    """Full entry point:
      1. pygame.init(), create a window (pre-menu size, resized after selection).
      2. selection = run_menu(screen); exit if None (window closed).
      3. If selection.watch_only: load the saved QTable and call
         evaluate_policy(env, q_table, render=True).
         Else: q_table = train(..., render=True), then save the QTable so
         the policy can be replayed later for the video demo.
      4. Clean up (renderer.close()) and exit.
    """
    pygame.init()

    # Menu window — fixed size 700×600 gives plenty of room for the picker
    MENU_W, MENU_H = 700, 600
    screen = pygame.display.set_mode((MENU_W, MENU_H))
    pygame.display.set_caption("Gridworld RL — Menu")

    selection = run_menu(screen)
    if selection is None:
        # User closed the window — clean exit
        pygame.quit()
        return

    # Build the environment so we know the grid size for the renderer window
    env = make_env(selection.level_id)
    gw, gh = env.grid_size

    # Resize window to fit the gridworld + HUD
    from src.render import GridWorldRenderer, TILE_SIZE_PX
    HUD_H = 56
    win_w = gw * TILE_SIZE_PX
    win_h = gh * TILE_SIZE_PX + HUD_H
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption(
        f"Gridworld — Level {selection.level_id} | {selection.algorithm.replace('_', ' ').title()}"
    )

    renderer = GridWorldRenderer(grid_size=(gw, gh), caption=pygame.display.get_caption()[0])

    try:
        if selection.watch_only:
            # Load saved Q-table and run greedy rollouts
            qpath = qtable_path(selection.level_id, selection.algorithm)
            if not qpath.exists():
                _show_error(
                    screen,
                    f"No saved policy found for level {selection.level_id} / "
                    f"{selection.algorithm}.\n\nTrain first (uncheck Watch Only).",
                )
                renderer.close()
                return

            q_table = load_qtable(qpath, n_actions=env.action_space_n)
            # evaluate_policy runs greedy rollouts and renders each frame
            evaluate_policy(env, q_table, render=True)

        else:
            # Train from scratch with live rendering
            q_table = train(
                level_id=selection.level_id,
                algorithm=selection.algorithm,
                render=True,
                seed=0,
            )

            # Save the learned policy so watch-only mode can replay it later
            qpath = qtable_path(selection.level_id, selection.algorithm)
            save_qtable(q_table, qpath)
            print(f"Policy saved to {qpath}")

    finally:
        renderer.close()


def _show_error(screen: pygame.Surface, message: str) -> None:
    """Display an error message inside the pygame window until closed."""
    pygame.font.init()
    try:
        font = pygame.font.SysFont("Segoe UI", 18)
    except Exception:
        font = pygame.font.Font(None, 22)

    clock = pygame.time.Clock()
    lines = message.split("\n")
    running = True
    while running:
        screen.fill((20, 10, 10))
        y = 60
        for line in lines:
            surf = font.render(line.strip(), True, (240, 100, 100))
            screen.blit(surf, (20, y))
            y += 30
        hint = font.render("Press any key or close the window to exit.", True, (160, 160, 160))
        screen.blit(hint, (20, y + 20))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                running = False
        clock.tick(30)


if __name__ == "__main__":
    main()
