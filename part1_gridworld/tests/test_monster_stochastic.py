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
