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
    """Keep an entity within the arena bounds by clamping at the edge
    (chosen over wrap-around: simpler, and it makes the arena feel like a
    bounded room rather than a torus, per the feasibility guide). The caller
    is responsible for zeroing velocity on the clamped axis if it wants the
    entity to stop rather than slide along the wall.
    """
    return clamp(x, 0.0, width), clamp(y, 0.0, height)


def integrate_position(x: float, y: float, vx: float, vy: float, dt: float) -> tuple[float, float]:
    """Simple explicit-Euler step: x += vx*dt, y += vy*dt. The arena sim is
    step-based; core_env passes dt=1.0 so velocities in config/arena.json
    are expressed directly in world-units-per-step.
    """
    return x + vx * dt, y + vy * dt


def distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x2 - x1, y2 - y1)


def relative_direction(from_x: float, from_y: float, to_x: float, to_y: float) -> float:
    """Angle in radians from (from_x, from_y) to (to_x, to_y), used by
    obs.py for the 'relative direction to nearest enemy/spawner' features
    and by core_env for enemy chase / projectile aiming. Returns 0.0 when
    the two points coincide (atan2(0, 0) is 0.0, which is fine here).
    """
    return math.atan2(to_y - from_y, to_x - from_x)


def circle_collision(x1: float, y1: float, r1: float, x2: float, y2: float, r2: float) -> bool:
    """Circle-circle overlap test (touching counts as colliding). Used for
    projectile<->enemy, projectile<->spawner, and enemy<->player contact
    checks in core_env.
    """
    return distance(x1, y1, x2, y2) <= r1 + r2
