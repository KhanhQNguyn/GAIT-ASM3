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
# Convention: ALL features are normalized to [-1, 1]. Features that are
# naturally in [0, 1] (e.g. fractions, distances) are linearly rescaled to
# [-1, 1] via x_normalized = 2 * x_unit - 1. sin/cos features are already
# in [-1, 1] by definition. This matches the gym_adapter.py Box bounds.
OBSERVATION_SPEC: list[tuple[str, str]] = [
    ("player_x", "Player x position, normalized to [-1, 1] across arena width"),
    ("player_y", "Player y position, normalized to [-1, 1] across arena height"),
    ("player_vx", "Player x velocity, normalized to [-1, 1] relative to max speed"),
    ("player_vy", "Player y velocity, normalized to [-1, 1] relative to max speed"),
    ("player_orientation_sin", "sin(player orientation), in [-1, 1]; 0 if ControlStyle2"),
    ("player_orientation_cos", "cos(player orientation), in [-1, 1]; 1 if ControlStyle2"),
    ("player_health_frac", "Player health / max health, rescaled [0,1] -> [-1,1]"),
    ("nearest_enemy_distance", "Dist to nearest enemy / arena diagonal, rescaled [0,1] -> [-1,1]"),
    ("nearest_enemy_direction_sin", "sin(angle to nearest enemy) relative to player, in [-1, 1]"),
    ("nearest_enemy_direction_cos", "cos(angle to nearest enemy) relative to player, in [-1, 1]"),
    ("nearest_spawner_distance", "Nearest spawner dist / arena diagonal, rescaled [0,1]->[-1,1]"),
    ("nearest_spawner_direction_sin", "sin(angle to nearest active spawner), in [-1, 1]"),
    ("nearest_spawner_direction_cos", "cos(angle to nearest active spawner), in [-1, 1]"),
    ("current_phase_frac", "Current phase / max expected phase, rescaled [0,1] -> [-1,1]"),
    ("num_active_enemies_frac", "Active enemies / assumed max clipped [0,1], rescaled->[-1,1]"),
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
