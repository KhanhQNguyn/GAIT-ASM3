"""Mechanic-correctness tests for src/env.py -- these are the regression
tests that defend "rewards and mechanics must not be altered" for Part I,
and cover Task 3's "correct episode termination and reward accounting."
"""

import pytest


def test_rock_blocks_movement_no_crash():
    """Moving into a rock tile results in the agent staying in place (not
    an exception, not a reset, not a death).

    TODO: implement using a small synthetic level fixture with one rock
    adjacent to the agent's start.
    """
    pytest.skip("TODO: implement once env.GridWorldEnv is implemented")


def test_fire_causes_immediate_death():
    """Stepping onto a fire tile ends the episode immediately (done=True)
    with no further monster movement resolved that turn.

    TODO: implement.
    """
    pytest.skip("TODO: implement once env.GridWorldEnv is implemented")


def test_apple_gives_reward_and_is_consumed():
    """Picking up an apple gives REWARD_APPLE exactly once; stepping onto
    the same tile again (impossible once consumed, but verify the tile no
    longer registers as an apple) does not re-award it.

    TODO: implement.
    """
    pytest.skip("TODO: implement once env.GridWorldEnv is implemented")


def test_key_gives_no_reward():
    """Picking up the key gives REWARD_KEY (0), not some other value --
    this directly defends the spec's explicit 'keys give no reward' rule
    against future refactors that might accidentally treat all pickups
    uniformly.

    TODO: implement.
    """
    pytest.skip("TODO: implement once env.GridWorldEnv is implemented")


def test_chest_requires_key_to_open():
    """Reaching the chest tile WITHOUT the key gives no reward and does not
    end the level; reaching it WITH the key gives REWARD_CHEST and consumes
    the key.

    TODO: implement.
    """
    pytest.skip("TODO: implement once env.GridWorldEnv is implemented")


def test_episode_ends_when_all_rewards_collected():
    """On a level with multiple apples + key + chest, the episode is done
    only once every apple is collected AND the chest is opened -- not
    earlier.

    TODO: implement using level2/level3-style fixture.
    """
    pytest.skip("TODO: implement once env.GridWorldEnv is implemented")


def test_episode_ends_on_death_even_with_rewards_remaining():
    """Dying (fire or monster contact) ends the episode immediately even if
    apples/chest are still uncollected.

    TODO: implement.
    """
    pytest.skip("TODO: implement once env.GridWorldEnv is implemented")
