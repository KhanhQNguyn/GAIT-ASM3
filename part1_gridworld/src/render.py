"""Pygame rendering ONLY. This module receives a GridWorldEnv's state and
draws it -- it must never contain game logic, reward computation, or RL
decision-making. Keeping this separate from env.py is what lets env.py stay
pygame-free and unit-testable headlessly.
"""

from __future__ import annotations

import pygame

TILE_SIZE_PX = 48
COLORS = {
    "background": (30, 30, 30),
    "grid_line": (60, 60, 60),
    "agent": (66, 135, 245),
    "rock": (110, 110, 110),
    "fire": (220, 60, 40),
    "apple": (220, 40, 90),
    "key": (245, 210, 60),
    "chest": (180, 130, 40),
    "monster": (150, 40, 200),
}


class GridWorldRenderer:
    """Owns the pygame window/surface and draws one GridWorldEnv frame at a
    time. Does not own the environment or the training loop.
    """

    def __init__(self, grid_size: tuple[int, int], caption: str = "Gridworld"):
        self.grid_size = grid_size
        # TODO: pygame.init(), create the window sized to
        # grid_size * TILE_SIZE_PX, set caption.
        raise NotImplementedError

    def draw(self, env_state: dict) -> None:
        """Draw one frame from a plain-data snapshot of the environment's
        current state (agent position, remaining apples, rocks, fire, key,
        chest, monster positions, has_key flag, etc.) -- NOT the
        GridWorldEnv object itself, to keep this module decoupled from
        env.py's internals.

        TODO: clear screen, draw grid lines, draw each entity type with its
        COLORS entry, pygame.display.flip().
        """
        raise NotImplementedError

    def handle_events(self) -> bool:
        """Pump the pygame event queue; return False if the window was
        closed (so the caller can stop the loop), True otherwise.

        TODO: implement.
        """
        raise NotImplementedError

    def close(self) -> None:
        """Tear down the pygame window.

        TODO: pygame.quit().
        """
        raise NotImplementedError
