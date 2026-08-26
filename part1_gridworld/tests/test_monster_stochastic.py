"""Statistical test for the monsters' 40% per-action move probability
(Task 4), and for the "monster moving into the agent kills it" rule under
that stochastic process.
"""

from __future__ import annotations

import json
import random

import pytest

from src.env import Action, GridWorldEnv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_level(tmp_path, lvl: dict):
    """Write level dict as JSON to tmp_path and return the path."""
    p = tmp_path / f"test_{lvl['level_id']}.json"
    p.write_text(json.dumps(lvl), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_monster_move_probability_converges_to_40_percent(tmp_path):
    """Over many independent trials of GridWorldEnv._resolve_monster_moves
    (with a seeded RNG and a monster that always has at least one unblocked
    direction available), the empirical fraction of trials in which the
    monster moved should converge close to 0.4.

    Uses a generous tolerance (±0.03 over ≥2000 trials) to avoid
    flaky failures.
    """
    # Monster at (2,2) in open 5x5 grid — always has 4 unblocked directions
    lvl = {
        "level_id": 90,
        "name": "monster_prob_test",
        "grid_size": [5, 5],
        "agent_start": [0, 4],
        "max_steps": 1000,
        "rocks": [],
        "fire": [],
        "apples": [[4, 0]],  # far away so we don't win accidentally
        "key": None,
        "chest": None,
        "monsters": [{"start": [2, 2], "move_prob": 0.4}],
    }
    lp = _write_level(tmp_path, lvl)
    env = GridWorldEnv(lp)
    env.reset()

    # Use a seeded RNG for reproducibility
    env._rng = random.Random(42)

    NUM_TRIALS = 3000
    TOLERANCE = 0.03
    moved_count = 0

    for _ in range(NUM_TRIALS):
        env.reset()
        # Place monster at (2,2) manually to ensure it always has room
        env._monsters[0].position = (2, 2)
        before = env._monsters[0].position

        env._resolve_monster_moves()

        after = env._monsters[0].position
        if after != before:
            moved_count += 1

    empirical_rate = moved_count / NUM_TRIALS
    assert abs(empirical_rate - 0.4) <= TOLERANCE, (
        f"Empirical move rate {empirical_rate:.4f} is outside "
        f"[{0.4 - TOLERANCE}, {0.4 + TOLERANCE}]"
    )


def test_monster_moving_onto_agent_causes_death(tmp_path):
    """Construct a scenario where the agent's action does not end the
    episode, but the subsequent forced monster move lands exactly on the
    agent's tile -- the episode must end in death on that same step.

    Setup (6x6 grid):
      - Agent starts at (1, 2).
      - Monster starts at (3, 2), move_prob=1.0 (always moves).
      - Rocks at (3,1), (3,3), (4,2) block monster's UP, DOWN, and RIGHT.
        Monster's only unblocked direction: LEFT -> (2,2).
      - Agent moves RIGHT to (2,2).
        * Step 3: monster still at (3,2) != (2,2) -> no immediate death.
        * Step 6: monster moves LEFT to (2,2) (its only option).
        * Step 7: monster at (2,2) == agent at (2,2) -> DEATH.
    """
    lvl = {
        "level_id": 94,
        "name": "monster_onto_agent_test",
        "grid_size": [6, 6],
        "agent_start": [1, 2],
        "max_steps": 100,
        # Block monster's UP (3,1), DOWN (3,3), and RIGHT (4,2) — only LEFT (2,2) is free
        "rocks": [[3, 1], [3, 3], [4, 2]],
        "fire": [],
        "apples": [[5, 5]],  # far away so no accidental win
        "key": None,
        "chest": None,
        "monsters": [{"start": [3, 2], "move_prob": 1.0}],
    }
    p = tmp_path / "monster_onto_agent.json"
    p.write_text(json.dumps(lvl), encoding="utf-8")
    env = GridWorldEnv(p)
    env.reset()

    assert env._agent_pos == (1, 2)
    assert env._monsters[0].position == (3, 2)

    # Agent moves RIGHT to (2,2).
    # Step 3: monster still at (3,2) != (2,2) -> no agent-into-monster death.
    # Step 6: monster moves LEFT to (2,2) (its only free direction).
    # Step 7: (2,2) == agent_pos -> DEATH.
    result = env.step(Action.RIGHT)
    assert result.done, (
        "Monster moving onto agent's tile must cause death (done=True)"
    )
    assert result.info.get("cause") == "monster_into_agent", (
        f"Death cause must be 'monster_into_agent', got: {result.info.get('cause')}"
    )


def test_agent_moving_onto_monster_causes_death(tmp_path):
    """Construct a scenario where the agent's own action moves it onto a tile
    currently occupied by a monster -- the episode must end in death
    immediately, before any monster movement occurs that turn.

    The monster has move_prob=0.0 so it never moves, isolating step 3
    (agent-into-monster check) cleanly.
    """
    # Monster at (2,2), agent at (1,2). Agent moves RIGHT -> (2,2) == monster pos.
    lvl = {
        "level_id": 95,
        "name": "agent_into_monster_test",
        "grid_size": [5, 5],
        "agent_start": [1, 2],
        "max_steps": 100,
        "rocks": [],
        "fire": [],
        "apples": [[4, 4]],
        "key": None,
        "chest": None,
        "monsters": [{"start": [2, 2], "move_prob": 0.0}],  # never moves
    }
    lp = _write_level(tmp_path, lvl)
    env = GridWorldEnv(lp)
    env.reset()

    assert env._agent_pos == (1, 2)
    assert env._monsters[0].position == (2, 2)

    # Agent walks RIGHT into monster
    result = env.step(Action.RIGHT)
    assert result.done, (
        "Agent moving onto monster's tile must cause death (done=True)"
    )
    assert env._agent_pos == (2, 2), "Agent must be at monster's tile"
    assert result.info.get("cause") == "agent_into_monster", (
        f"Death cause must be 'agent_into_monster', got: {result.info.get('cause')}"
    )

    # Monster must NOT have moved (death occurs before monster resolution in step 3)
    assert env._monsters[0].position == (2, 2), (
        "Monster must not move when agent-into-monster death occurs in step 3"
    )
