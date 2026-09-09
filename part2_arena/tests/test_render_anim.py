"""Render-only animation tests (2026-09-08 asset pass).

Uses SDL's dummy video driver so ArenaRenderer can be constructed headless;
these tests pin the CONTRACTS the user asked for:
- ship sprite tiers at the exact health thresholds,
- the enemy turn animation always sweeps through the middle frame in the
  order l2 -> l1 -> m -> r1 -> r2 (never jumps over frames),
- the explosion animation plays its frames in order and ends after the
  last one.
"""

import os

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from arena.entities import ArenaState, Enemy, Player  # noqa: E402
from arena.render_pygame import _ENEMY_TURN_FRAMES, ArenaRenderer  # noqa: E402


@pytest.fixture(scope="module")
def renderer():
    return ArenaRenderer(960, 680)


def test_ship_sprite_tiers():
    """100-76 full, 75-51 slight, 50-26 damaged, <=25 very damaged."""
    tiers = ArenaRenderer._ship_sprite_name
    assert tiers(1.00) == "ship/ship_fullhealth"
    assert tiers(0.76) == "ship/ship_fullhealth"
    assert tiers(0.75) == "ship/ship_slightdamaged"
    assert tiers(0.51) == "ship/ship_slightdamaged"
    assert tiers(0.50) == "ship/ship_damaged"
    assert tiers(0.26) == "ship/ship_damaged"
    assert tiers(0.25) == "ship/ship_verydamaged"
    assert tiers(0.00) == "ship/ship_verydamaged"


def test_enemy_turn_sweep_order():
    """A direction flip must walk l2 -> l1 -> m -> r1 -> r2 one index per
    rendered frame (never skipping the middle frame), per the asset spec.
    """
    r = ArenaRenderer(960, 680)
    p = Player(x=480.0, y=340.0, health=100.0, max_health=100.0)
    en = Enemy(x=200.0, y=340.0, health=25.0, max_health=25.0, speed=2.0)

    # Enemy LEFT of the player -> travel direction points right -> r2 (4).
    # First sight snaps to the target (no mid-frame sweep on spawn).
    assert r._enemy_turn_frame(en, p) == 4

    # Enemy teleports to the RIGHT of the player -> must sweep back
    # r1 -> m -> l1 -> l2, one index per frame, in exactly that order.
    en.x = 800.0
    sweep = [r._enemy_turn_frame(en, p) for _ in range(8)]
    assert sweep == [3, 2, 1, 0, 0, 0, 0, 0]


def test_enemy_turn_frame_names_order():
    """The frame table itself must be ordered full-left .. full-right."""
    assert _ENEMY_TURN_FRAMES == (
        "enemy/enemy_l2",
        "enemy/enemy_l1",
        "enemy/enemy_m",
        "enemy/enemy_r1",
        "enemy/enemy_r2",
    )


def test_explosion_plays_frames_in_order(renderer):
    """Each explosion survives exactly _n frames and the index advances one
    per draw -- explosion_01 through explosion_11 play in order.
    """
    state = ArenaState(
        player=Player(x=480.0, y=340.0, health=100.0, max_health=100.0), control_style=1
    )
    r = renderer
    r._explosions.append([100.0, 100.0, 0])
    seen_indices = []
    for _ in range(12):
        if not r._explosions:
            break
        seen_indices.append(r._explosions[0][2])
        r.draw(state)
    assert seen_indices == list(range(11))  # 0..10 -> explosion_01..11
    assert r._explosions == []  # finished, nothing lingers
