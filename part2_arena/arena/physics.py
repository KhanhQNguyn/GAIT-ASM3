"""Minimal 2D physics/movement helpers for the arena. Deliberately simple
per the feasibility guide -- this is not meant to be a full physics engine,
just enough continuous movement and collision detection to make the arena
feel real-time rather than tile-based.
"""

from __future__ import annotations

import math


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def wrap_or_clamp_to_bounds(x: float, y: float, width: float, height: float) -> tuple[float, float]:
    """Keep an entity within the arena bounds.

    TODO: decide wrap-around vs. clamp-at-edge (clamp is simpler and
    probably sufficient) and implement.
    """
    raise NotImplementedError


def integrate_position(x: float, y: float, vx: float, vy: float, dt: float) -> tuple[float, float]:
    """Simple Euler integration step: x/y += v * dt.

    TODO: implement.
    """
    raise NotImplementedError


def distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x2 - x1, y2 - y1)


def relative_direction(from_x: float, from_y: float, to_x: float, to_y: float) -> float:
    """Angle in radians from (from_x, from_y) to (to_x, to_y), used by
    obs.py for the 'relative direction to nearest enemy/spawner' features.

    TODO: implement (math.atan2).
    """
    raise NotImplementedError


def circle_collision(x1: float, y1: float, r1: float, x2: float, y2: float, r2: float) -> bool:
    """Simple circle-circle collision test, used for projectile <-> ship
    and projectile <-> enemy/spawner collisions.

    TODO: implement.
    """
    raise NotImplementedError
