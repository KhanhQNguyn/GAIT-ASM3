"""Statistical test for the monsters' 40% per-action move probability
(Task 4), and for the "monster moving into the agent kills it" rule under
that stochastic process.
"""

import pytest


def test_monster_move_probability_converges_to_40_percent():
    """Over many independent trials of GridWorldEnv._resolve_monster_moves
    (with a seeded RNG and a monster that always has at least one unblocked
    direction available), the empirical fraction of trials in which the
    monster moved should converge close to 0.4.

    TODO: implement once env.GridWorldEnv._resolve_monster_moves exists --
    use a generous tolerance (e.g. +/- 0.03 over >= 2000 trials) to avoid
    flaky failures.
    """
    pytest.skip("TODO: implement once GridWorldEnv._resolve_monster_moves is implemented")


def test_monster_moving_onto_agent_causes_death():
    """Construct a scenario where the agent's action does not end the
    episode, but the subsequent forced monster move lands exactly on the
    agent's tile -- the episode must end in death on that same step.

    TODO: implement (may require forcing move_prob=1.0 and a fixed monster
    position in a synthetic level fixture to make this deterministic).
    """
    pytest.skip("TODO: implement once GridWorldEnv is implemented")


def test_agent_moving_onto_monster_causes_death():
    """Construct a scenario where the agent's own action moves it onto a tile
    currently occupied by a monster -- the episode must end in death
    immediately, before any monster movement occurs that turn.

    This tests the second of the two monster-death conditions: the agent
    walking into a monster (complement to test_monster_moving_onto_agent_causes_death
    which tests the monster walking into the agent).

    TODO: implement (may require forcing a monster to a fixed position and
    directing the agent straight into it in a synthetic level fixture to make
    this deterministic; monster movement should ideally be disabled for this
    turn so the test isolates step 3 of the step() order of operations).
    """
    pytest.skip("TODO: implement once GridWorldEnv is implemented")

