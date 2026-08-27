"""Mechanic-correctness tests for src/env.py -- these are the regression
tests that defend "rewards and mechanics must not be altered" for Part I,
and cover Task 3's "correct episode termination and reward accounting."

All tests use synthetic level fixtures built in-memory (via tmp_path or
a helper that writes a JSON file) so they never depend on the live
levelN.json files — this keeps the tests fast and free of side-effects.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from src.env import Action, GridWorldEnv
from config.rewards_constants import REWARD_APPLE, REWARD_CHEST, REWARD_KEY


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_level(tmp_path: pathlib.Path, **overrides) -> pathlib.Path:
    """Write a minimal valid level JSON to tmp_path and return the path.

    The base level is a 5x5 open grid with the agent at (0,0), one apple
    at (4,4), no key/chest, no monsters, no rocks/fire. Individual tests
    override whatever they need via **overrides.
    """
    base = {
        "level_id": 99,
        "name": "test_level",
        "grid_size": [5, 5],
        "agent_start": [0, 0],
        "max_steps": 100,
        "rocks": [],
        "fire": [],
        "apples": [[4, 4]],
        "key": None,
        "chest": None,
        "monsters": [],
    }
    base.update(overrides)
    p = tmp_path / "test_level.json"
    p.write_text(json.dumps(base), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_rock_blocks_movement_no_crash(tmp_path):
    """Moving into a rock tile results in the agent staying in place (not
    an exception, not a reset, not a death).
    """
    # Rock immediately to the right of agent start
    lp = _make_level(tmp_path, rocks=[[1, 0]])
    env = GridWorldEnv(lp)
    state = env.reset()
    assert env._agent_pos == (0, 0)

    # Try to walk RIGHT into the rock
    result = env.step(Action.RIGHT)
    assert not result.done, "Rock contact must not end the episode"
    assert env._agent_pos == (0, 0), "Agent must stay in place when blocked by rock"
    assert result.reward == pytest.approx(0.0), "No reward for hitting a wall"


def test_fire_causes_immediate_death(tmp_path):
    """Stepping onto a fire tile ends the episode immediately (done=True)
    with no further monster movement resolved that turn.
    """
    # Fire immediately to the right of agent start; also add a monster far away
    # to confirm it doesn't move (if it did, it'd be at a different known pos)
    lp = _make_level(
        tmp_path,
        fire=[[1, 0]],
        apples=[[4, 4]],
        monsters=[{"start": [4, 0], "move_prob": 1.0}],  # move_prob=1.0 would always move
    )
    env = GridWorldEnv(lp)
    env.reset()

    monster_before = env._monsters[0].position

    result = env.step(Action.RIGHT)  # step into fire
    assert result.done, "Fire contact must end the episode"
    assert env._agent_pos == (1, 0), "Agent must be at fire tile"
    assert result.info.get("cause") == "fire"

    # Monster must NOT have moved (fire death skips monster resolution)
    assert env._monsters[0].position == monster_before, (
        "Monster must not move after agent's fire death"
    )


def test_apple_gives_reward_and_is_consumed(tmp_path):
    """Picking up an apple gives REWARD_APPLE exactly once; stepping onto
    the same tile again does not re-award it.
    """
    # Apple right next to agent start
    lp = _make_level(tmp_path, apples=[[1, 0]])
    env = GridWorldEnv(lp)
    env.reset()

    result1 = env.step(Action.RIGHT)  # collect apple at (1,0)
    assert result1.reward == pytest.approx(REWARD_APPLE), (
        f"Expected REWARD_APPLE={REWARD_APPLE}, got {result1.reward}"
    )

    # Walk away then come back
    env.step(Action.LEFT)  # back to (0,0)
    result2 = env.step(Action.RIGHT)  # step onto (1,0) again — apple gone
    assert result2.reward == pytest.approx(0.0), (
        "Stepping onto a consumed apple tile must give zero reward"
    )

    # Verify bitmask is zero (apple truly consumed)
    assert env._apples_bitmask == 0


def test_key_gives_no_reward(tmp_path):
    """Picking up the key gives REWARD_KEY (0), not some other value --
    this directly defends the spec's explicit 'keys give no reward' rule.
    """
    # Key right next to agent, chest far away so we don't accidentally open it
    lp = _make_level(tmp_path, apples=[], key=[1, 0], chest=[4, 4])
    env = GridWorldEnv(lp)
    env.reset()

    result = env.step(Action.RIGHT)  # pick up key at (1,0)
    assert result.reward == pytest.approx(REWARD_KEY), (
        f"Key pickup must give REWARD_KEY={REWARD_KEY}, got {result.reward}"
    )
    assert REWARD_KEY == 0.0, "REWARD_KEY must be exactly 0.0 per spec"
    assert env._has_key, "has_key must be True after picking up the key"


def test_chest_requires_key_to_open(tmp_path):
    """Reaching the chest tile WITHOUT the key gives no reward and does not
    end the level; reaching it WITH the key gives REWARD_CHEST and consumes
    the key.
    """
    # Chest at (2,0), key at (4,0), agent starts at (0,0).
    # No apples so episode ends only when chest opened.
    lp = _make_level(tmp_path, apples=[], key=[4, 0], chest=[2, 0])
    env = GridWorldEnv(lp)
    env.reset()

    # Step into chest WITHOUT key
    env.step(Action.RIGHT)   # (1,0)
    result_no_key = env.step(Action.RIGHT)  # (2,0) — chest, no key
    assert result_no_key.reward == pytest.approx(0.0), (
        "Chest without key must give no reward"
    )
    assert not result_no_key.done, "Chest without key must not end the episode"
    assert not env._chest_open

    # Now verify opening chest WITH key using a clean fresh level
    p2 = tmp_path / "chest_with_key.json"
    base2 = {
        "level_id": 98,
        "name": "chest_test2",
        "grid_size": [5, 5],
        "agent_start": [0, 0],
        "max_steps": 100,
        "rocks": [],
        "fire": [],
        "apples": [],
        "key": [1, 0],
        "chest": [3, 0],
        "monsters": [],
    }
    import json as _json
    p2.write_text(_json.dumps(base2), encoding="utf-8")
    env3 = GridWorldEnv(p2)
    env3.reset()
    env3.step(Action.RIGHT)  # (1,0): pick up key
    assert env3._has_key
    env3.step(Action.RIGHT)  # (2,0)
    result_chest = env3.step(Action.RIGHT)  # (3,0): open chest
    assert result_chest.reward == pytest.approx(REWARD_CHEST), (
        f"Chest with key must give REWARD_CHEST={REWARD_CHEST}"
    )
    assert env3._chest_open
    assert not env3._has_key, "Key must be consumed after opening chest"
    assert result_chest.done, "Episode must end when chest opened (only reward left)"


def test_episode_ends_when_all_rewards_collected(tmp_path):
    """On a level with multiple apples + key + chest, the episode is done
    only once every apple is collected AND the chest is opened -- not earlier.
    """
    # Compact 5x5 layout: apples at (1,0) and (2,0), key at (3,0), chest at (4,0)
    p = tmp_path / "multi.json"
    lvl = {
        "level_id": 97,
        "name": "multi_apple_key_chest",
        "grid_size": [5, 5],
        "agent_start": [0, 0],
        "max_steps": 50,
        "rocks": [],
        "fire": [],
        "apples": [[1, 0], [2, 0]],
        "key": [3, 0],
        "chest": [4, 0],
        "monsters": [],
    }
    p.write_text(json.dumps(lvl), encoding="utf-8")
    env = GridWorldEnv(p)
    env.reset()

    r1 = env.step(Action.RIGHT)  # (1,0) -> apple
    assert r1.reward == pytest.approx(REWARD_APPLE)
    assert not r1.done, "Episode must not end after collecting first apple only"

    r2 = env.step(Action.RIGHT)  # (2,0) -> apple
    assert r2.reward == pytest.approx(REWARD_APPLE)
    assert not r2.done, "Episode must not end after collecting second apple (chest still locked)"

    r3 = env.step(Action.RIGHT)  # (3,0) -> key
    assert r3.reward == pytest.approx(REWARD_KEY)
    assert not r3.done, "Episode must not end after picking up key"

    r4 = env.step(Action.RIGHT)  # (4,0) -> open chest
    assert r4.reward == pytest.approx(REWARD_CHEST)
    assert r4.done, "Episode must end after opening chest (all rewards collected)"


def test_episode_ends_on_death_even_with_rewards_remaining(tmp_path):
    """Dying (fire) ends the episode immediately even if apples are still
    uncollected.
    """
    # Fire at (1,0), apples far away
    lp = _make_level(tmp_path, fire=[[1, 0]], apples=[[4, 4]])
    env = GridWorldEnv(lp)
    env.reset()

    result = env.step(Action.RIGHT)  # step into fire
    assert result.done, "Death by fire must end the episode"
    assert env._apples_bitmask != 0, "Apples should still be uncollected"
