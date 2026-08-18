"""Plain-data entity classes for the arena. No pygame, no gymnasium, no SB3
here -- this module only describes game state, not how it's drawn or
trained against.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Player:
    """The controllable ship."""

    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    orientation: float = 0.0  # radians; only meaningful for ControlStyle1
    health: float = 100.0
    max_health: float = 100.0

    # TODO: add whatever additional fields physics.py / rewards.py need
    # (e.g. cooldown timer for shooting).


@dataclass
class Enemy:
    """A single enemy unit that navigates toward the player."""

    x: float
    y: float
    health: float
    max_health: float
    speed: float

    # TODO: add navigation/AI state fields as needed (e.g. target, behavior).


@dataclass
class Spawner:
    """Periodically creates enemies. Destroying every currently-active
    spawner is what advances the phase system (see phases.py).
    """

    x: float
    y: float
    health: float
    max_health: float
    spawn_interval_steps: int
    steps_since_last_spawn: int = 0
    active: bool = True


@dataclass
class Projectile:
    """A single fired shot (from the player or, if the design calls for it,
    from enemies too).
    """

    x: float
    y: float
    vx: float
    vy: float
    owner: str  # "player" or "enemy"
    damage: float


@dataclass
class ArenaState:
    """Full mutable state of one arena episode. core_env.py owns and
    mutates an instance of this; render_pygame.py only reads it.
    """

    player: Player
    enemies: list[Enemy] = field(default_factory=list)
    spawners: list[Spawner] = field(default_factory=list)
    projectiles: list[Projectile] = field(default_factory=list)
    phase: int = 0
    step_count: int = 0
