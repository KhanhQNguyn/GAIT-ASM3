"""Builds the fixed-size numeric observation vector for the arena. No
pixels/screenshots per the spec -- every feature here is a scalar derived
from ArenaState. Feature order/meaning is documented below and must be
kept in sync with report section 2 (Observation Design).
"""

from __future__ import annotations

import numpy as np

from arena.entities import ArenaState

# Index -> (name, description). Keep this in sync with build_observation's
# actual output order -- tests/test_obs_shape.py checks the length matches.
OBSERVATION_SPEC: list[tuple[str, str]] = [
    ("player_x", "Player x position, normalized to arena width"),
    ("player_y", "Player y position, normalized to arena height"),
    ("player_vx", "Player x velocity, normalized to max speed"),
    ("player_vy", "Player y velocity, normalized to max speed"),
    ("player_orientation_sin", "sin(player orientation) -- only meaningful for ControlStyle1, 0 otherwise"),
    ("player_orientation_cos", "cos(player orientation) -- only meaningful for ControlStyle1, 1 otherwise"),
    ("player_health_frac", "Player health / max health, in [0, 1]"),
    ("nearest_enemy_distance", "Distance to nearest enemy, normalized by arena diagonal"),
    ("nearest_enemy_direction_sin", "sin(angle to nearest enemy) relative to player"),
    ("nearest_enemy_direction_cos", "cos(angle to nearest enemy) relative to player"),
    ("nearest_spawner_distance", "Distance to nearest active spawner, normalized"),
    ("nearest_spawner_direction_sin", "sin(angle to nearest active spawner)"),
    ("nearest_spawner_direction_cos", "cos(angle to nearest active spawner)"),
    ("current_phase_frac", "Current phase number / max expected phase, roughly normalized"),
    ("num_active_enemies_frac", "Count of active enemies / an assumed max, clipped to [0, 1]"),
]

OBS_DIM = len(OBSERVATION_SPEC)


def build_observation(state: ArenaState, arena_width: float, arena_height: float) -> np.ndarray:
    """Produce the fixed-size float32 observation vector described by
    OBSERVATION_SPEC, in that exact order.

    TODO: implement using arena.physics.distance / relative_direction
    helpers; when there are no enemies/spawners left, define a sane
    fallback (e.g. distance = 1.0 normalized max, direction = 0) rather
    than raising or returning NaN -- tests/test_obs_shape.py checks for
    NaNs.
    """
    raise NotImplementedError
