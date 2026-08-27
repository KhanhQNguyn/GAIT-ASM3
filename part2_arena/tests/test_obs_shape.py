"""Correctness tests for arena/obs.py -- evidence for rubric row Part II-H's
"fixed-size vector with required features" requirement.
"""

import numpy as np
import pytest

from arena.entities import ArenaState, Enemy, Player, Spawner
from arena.obs import OBS_DIM, OBSERVATION_SPEC, build_observation

W, H = 960.0, 680.0


def _state(n_enemies=0, n_spawners=0, control_style=1, player=None):
    p = player or Player(x=W / 2, y=H / 2, health=100.0, max_health=100.0)
    enemies = [
        Enemy(x=100 + 40 * i, y=120, health=30, max_health=30, speed=1.5) for i in range(n_enemies)
    ]
    spawners = [
        Spawner(x=80 + 60 * i, y=60, health=120, max_health=120, spawn_interval_steps=100)
        for i in range(n_spawners)
    ]
    return ArenaState(player=p, enemies=enemies, spawners=spawners, control_style=control_style)


@pytest.mark.parametrize("n_enemies,n_spawners", [(0, 0), (0, 2), (3, 0), (5, 3)])
def test_observation_has_fixed_shape(n_enemies, n_spawners):
    """build_observation() must always return a 1D float32 array of length
    OBS_DIM, regardless of how many enemies/spawners exist (incl. zero).
    """
    obs = build_observation(_state(n_enemies, n_spawners), W, H)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32


@pytest.mark.parametrize("n_enemies,n_spawners,style", [(0, 0, 1), (0, 0, 2), (4, 2, 1), (4, 2, 2)])
def test_observation_has_no_nans_or_infs(n_enemies, n_spawners, style):
    """Finite and within [-1, 1] for any valid state, including the
    degenerate zero-enemies / zero-spawners case (no division by zero, no
    NaN from a nearest-distance over an empty list).
    """
    obs = build_observation(_state(n_enemies, n_spawners, control_style=style), W, H)
    assert np.all(np.isfinite(obs)), obs
    assert np.all(obs >= -1.0) and np.all(obs <= 1.0), (obs.min(), obs.max())


def test_observation_zero_entities_uses_documented_fallbacks():
    """No enemies / no active spawners -> distance features = +1 (nothing
    near), direction sin/cos = 0 / 1. Matches OBSERVATION_SPEC's text.
    """
    obs = build_observation(_state(0, 0), W, H)
    names = [n for n, _ in OBSERVATION_SPEC]
    idx = {n: i for i, n in enumerate(names)}
    assert obs[idx["nearest_enemy_distance"]] == pytest.approx(1.0)
    assert obs[idx["nearest_enemy_direction_sin"]] == pytest.approx(0.0)
    assert obs[idx["nearest_enemy_direction_cos"]] == pytest.approx(1.0)
    assert obs[idx["nearest_spawner_distance"]] == pytest.approx(1.0)


def test_observation_includes_all_required_features():
    """OBSERVATION_SPEC must include every feature the spec names as the
    minimum, and OBS_DIM must equal its length.
    """
    names = {n for n, _ in OBSERVATION_SPEC}
    required = {
        "player_x",
        "player_y",
        "player_vx",
        "player_vy",
        "player_orientation_sin",
        "player_orientation_cos",
        "player_health_frac",
        "nearest_enemy_distance",
        "nearest_enemy_direction_sin",
        "nearest_enemy_direction_cos",
        "nearest_spawner_distance",
        "nearest_spawner_direction_sin",
        "nearest_spawner_direction_cos",
        "current_phase_frac",
    }
    assert required <= names, required - names
    assert OBS_DIM == len(OBSERVATION_SPEC)
    # enemies do not shoot -> no incoming-projectile feature (see obs.py)
    assert not any("projectile" in n for n in names)


@pytest.mark.parametrize("style", [1, 2])
def test_observation_bounds_at_extreme_positions(style):
    """Stay within [-1, 1] at arena corners, sitting exactly on an enemy,
    and with velocity far above max_speed -- the clamps must hold.
    (Going-beyond item, MEMBER_C section 4.)
    """
    for px, py in [(0.0, 0.0), (W, H), (W, 0.0), (0.0, H)]:
        p = Player(x=px, y=py, vx=999.0, vy=-999.0, health=100.0, max_health=100.0)
        st = _state(3, 2, control_style=style, player=p)
        # drop an enemy exactly on the player
        st.enemies[0].x, st.enemies[0].y = px, py
        obs = build_observation(st, W, H)
        assert np.all(np.isfinite(obs))
        assert np.all(obs >= -1.0) and np.all(obs <= 1.0), (px, py, obs.min(), obs.max())
