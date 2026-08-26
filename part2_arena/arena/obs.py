"""Builds the fixed-size numeric observation vector for the arena. No
pixels/screenshots per the spec -- every feature here is a scalar derived
from ArenaState. Feature order/meaning is documented below and must be
kept in sync with report section 2 (Observation Design).
"""

from __future__ import annotations

import json
import math
import pathlib

import numpy as np

from arena.entities import ArenaState
from arena.physics import clamp, distance, relative_direction

_CONFIG_PATH = pathlib.Path(__file__).resolve().parent.parent / "config" / "arena.json"

# Index -> (name, description). Keep this in sync with build_observation's
# actual output order -- tests/test_obs_shape.py checks the length matches
# and that the spec-required feature names are all present.
#
# Design note: enemies do NOT shoot in this arena (they deal contact
# damage), so there is no "incoming projectile" feature -- the agent only
# needs to perceive enemies, spawners, its own kinematics, health, and the
# phase. Keeping the vector to the spec minimum (+ phase + enemy count) also
# keeps the observation honest: it describes the world, it does not hand the
# agent a strategy.
#
# Normalisation convention: EVERY feature is in [-1, 1].
#   - positions:      2 * clamp(coord / arena_size, 0, 1) - 1
#   - velocities:     clamp(v / max_speed, -1, 1)      (already signed)
#   - fractions:      2 * clamp(x_unit, 0, 1) - 1      (health, phase, count)
#   - distances:      2 * clamp(d / arena_diagonal, 0, 1) - 1   (near = -1, far = +1)
#   - sin/cos:        passed through, already in [-1, 1]
# This matches the spaces.Box(-1, 1) bounds in gym_adapter.py.
OBSERVATION_SPEC: list[tuple[str, str]] = [
    ("player_x", "Player x position, 2*(x/width)-1 -> [-1, 1]"),
    ("player_y", "Player y position, 2*(y/height)-1 -> [-1, 1]"),
    ("player_vx", "Player x velocity / max_speed, clipped to [-1, 1]"),
    ("player_vy", "Player y velocity / max_speed, clipped to [-1, 1]"),
    ("player_orientation_sin", "sin(heading); forced 0 for ControlStyle2 (no facing)"),
    ("player_orientation_cos", "cos(heading); forced 1 for ControlStyle2"),
    ("player_health_frac", "health/max_health, [0,1] -> [-1,1] (full=1, dead=-1)"),
    ("nearest_enemy_distance", "dist to nearest enemy / diagonal, [0,1] -> [-1,1]; +1 if none"),
    ("nearest_enemy_direction_sin", "sin(world angle player->nearest enemy); 0 if none"),
    ("nearest_enemy_direction_cos", "cos(world angle player->nearest enemy); 1 if none"),
    ("nearest_spawner_distance", "dist to nearest ACTIVE spawner / diagonal -> [-1,1]; +1 if none"),
    ("nearest_spawner_direction_sin", "sin(world angle player->nearest active spawner); 0 if none"),
    ("nearest_spawner_direction_cos", "cos(world angle player->nearest active spawner); 1 if none"),
    ("current_phase_frac", "phase / max_expected_phase, [0,1] -> [-1,1]"),
    ("num_active_enemies_frac", "live enemies / NUM_ACTIVE_ENEMIES_MAX, [0,1] -> [-1,1]"),
]

OBS_DIM = len(OBSERVATION_SPEC)


def _load_obs_params() -> tuple[float, float, float]:
    """(max_speed, max_expected_phase, num_active_enemies_max) from
    config/arena.json, with fallbacks if the file is missing/partial.
    """
    max_speed, max_phase, enemies_max = 6.0, 6.0, 12.0
    try:
        cfg = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
        max_speed = float(cfg.get("player", {}).get("max_speed", max_speed))
        max_phase = float(cfg.get("phase_curve", {}).get("max_expected_phase", max_phase))
        enemies_max = float(cfg.get("observation", {}).get("num_active_enemies_max", enemies_max))
    except (OSError, ValueError, TypeError):
        pass
    return max_speed, max_phase, enemies_max


_MAX_SPEED, _MAX_EXPECTED_PHASE, _NUM_ACTIVE_ENEMIES_MAX_CFG = _load_obs_params()

# Public constant (authoritative value lives in config/arena.json ->
# observation.num_active_enemies_max; this mirrors it as a fallback).
NUM_ACTIVE_ENEMIES_MAX = int(_NUM_ACTIVE_ENEMIES_MAX_CFG)


def _unit_to_pm1(x_unit: float) -> float:
    """Map a value expected in [0, 1] to [-1, 1], clamping out-of-range."""
    return 2.0 * clamp(x_unit, 0.0, 1.0) - 1.0


def build_observation(state: ArenaState, arena_width: float, arena_height: float) -> np.ndarray:
    """Produce the fixed-size float32 observation vector described by
    OBSERVATION_SPEC, in that exact order. Every element is in [-1, 1] and
    finite. The zero-enemies / zero-active-spawners cases use the documented
    fallbacks (distance feature = +1, direction sin/cos = 0/1) -- never NaN,
    never a division by zero.
    """
    p = state.player
    diag = math.hypot(arena_width, arena_height) or 1.0
    max_health = p.max_health if p.max_health > 0 else 1.0

    # --- player kinematics ---
    feats: list[float] = [
        _unit_to_pm1(p.x / arena_width if arena_width else 0.0),
        _unit_to_pm1(p.y / arena_height if arena_height else 0.0),
        clamp(p.vx / _MAX_SPEED, -1.0, 1.0),
        clamp(p.vy / _MAX_SPEED, -1.0, 1.0),
    ]
    if state.control_style == 2:
        feats += [0.0, 1.0]
    else:
        feats += [math.sin(p.orientation), math.cos(p.orientation)]
    feats.append(_unit_to_pm1(p.health / max_health))

    # --- nearest enemy ---
    if state.enemies:
        e = min(state.enemies, key=lambda en: distance(p.x, p.y, en.x, en.y))
        d = distance(p.x, p.y, e.x, e.y)
        ang = relative_direction(p.x, p.y, e.x, e.y)
        feats += [_unit_to_pm1(d / diag), math.sin(ang), math.cos(ang)]
    else:
        feats += [1.0, 0.0, 1.0]

    # --- nearest ACTIVE spawner ---
    active_spawners = [s for s in state.spawners if s.active]
    if active_spawners:
        s = min(active_spawners, key=lambda sp: distance(p.x, p.y, sp.x, sp.y))
        d = distance(p.x, p.y, s.x, s.y)
        ang = relative_direction(p.x, p.y, s.x, s.y)
        feats += [_unit_to_pm1(d / diag), math.sin(ang), math.cos(ang)]
    else:
        feats += [1.0, 0.0, 1.0]

    # --- phase + crowding ---
    phase_unit = state.phase / _MAX_EXPECTED_PHASE if _MAX_EXPECTED_PHASE else 0.0
    crowd_unit = len(state.enemies) / NUM_ACTIVE_ENEMIES_MAX if NUM_ACTIVE_ENEMIES_MAX else 0.0
    feats.append(_unit_to_pm1(phase_unit))
    feats.append(_unit_to_pm1(crowd_unit))

    arr = np.asarray(feats, dtype=np.float32)
    # Defensive: clip float rounding that could push sin/cos a hair past ±1.
    return np.clip(arr, -1.0, 1.0)
