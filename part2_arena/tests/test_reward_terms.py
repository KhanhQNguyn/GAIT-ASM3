"""Correctness tests for arena/rewards.py -- evidence for rubric row
Part II-J's reward structure requirement, and guards the "no inline reward
math outside rewards.py" architecture principle.
"""

import pytest


def test_kill_enemy_reward_fires_independently():
    """A step_events dict with only enemies_killed=1 set must produce a
    RewardBreakdown with kill_enemy == R_KILL_ENEMY and every other field
    == 0.

    TODO: implement once rewards.compute_reward is implemented.
    """
    pytest.skip("TODO: implement once rewards.compute_reward is implemented")


def test_kill_spawner_reward_exceeds_kill_enemy_reward():
    """Sanity-check the design intent documented in rewards_config.py:
    R_KILL_SPAWNER must be strictly greater than R_KILL_ENEMY, or the agent
    has no incentive to prioritize spawners over farming enemies.

    TODO: implement (this can just assert on the constants directly, but
    is placed here as an explicit contract test rather than left implicit).
    """
    pytest.skip("TODO: implement once rewards_config constants are finalized")


def test_death_penalty_dominates_a_full_episode_of_positive_reward():
    """R_DEATH must be large enough in magnitude that a somewhat successful
    episode's positive rewards (a handful of kills) followed by death
    nets negative overall -- otherwise the agent has no incentive to avoid
    a suicidal aggressive playstyle.

    TODO: implement once rewards_config constants are finalized -- this is
    a design-intent test, adjust the exact scenario numbers to match
    whatever the team tunes R_KILL_ENEMY / R_KILL_SPAWNER / R_DEATH to.
    """
    pytest.skip("TODO: implement once rewards_config constants are finalized")


def test_reward_breakdown_sums_to_total():
    """RewardBreakdown.total must equal the sum of its individual fields --
    guards against a term being added to the dataclass but forgotten in the
    total property.

    TODO: implement once rewards.RewardBreakdown is implemented.
    """
    pytest.skip("TODO: implement once rewards.RewardBreakdown is implemented")
