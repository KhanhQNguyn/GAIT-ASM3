"""Pygame rendering ONLY, for the arena. Mirrors part1_gridworld's render.py
separation: this module reads an ArenaState snapshot and draws it; it must
never mutate game state or compute rewards.
"""

from __future__ import annotations

import pygame

from arena.entities import ArenaState

COLORS = {
    "background": (12, 14, 20),
    "player": (66, 200, 245),
    "enemy": (230, 70, 70),
    "spawner": (200, 140, 40),
    "projectile_player": (240, 240, 120),
    "projectile_enemy": (240, 90, 90),
    "health_bar_bg": (60, 60, 60),
    "health_bar_fg": (70, 200, 90),
}


class ArenaRenderer:
    """Owns the pygame window and draws one ArenaState frame at a time."""

    def __init__(self, width: int, height: int, caption: str = "Arena"):
        self.width = width
        self.height = height
        # TODO: pygame.init(), create window sized (width, height), set caption.
        raise NotImplementedError

    def draw(self, state: ArenaState) -> None:
        """Draw the player, enemies, spawners, projectiles (with health
        bars where relevant) and a small HUD showing phase/health.

        TODO: implement.
        """
        raise NotImplementedError

    def handle_events(self) -> bool:
        """Pump the pygame event queue; return False if the window was
        closed.

        TODO: implement.
        """
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError
