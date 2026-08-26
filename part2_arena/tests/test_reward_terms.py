"""Correctness tests for arena/rewards.py -- evidence for rubric row
Part II-J's reward structure requirement, and guards the "no inline reward
math outside rewards.py" architecture principle.
"""

from arena.rewards import compute_reward
from arena.rewards_config import R_DEATH, R_KILL_ENEMY, R_KILL_SPAWNER


def test_kill_enemy_reward_fires_independently():
    """A step_events dict with only enemies_killed=1 set must produce a
    RewardBreakdown with kill_enemy == R_KILL_ENEMY and every other field
    == 0.
    """
    breakdown = compute_reward({"enemies_killed": 1})

    assert breakdown.kill_enemy == R_KILL_ENEMY
    assert breakdown.kill_spawner == 0.0
    assert breakdown.phase_progress == 0.0
    assert breakdown.damage_taken == 0.0
    assert breakdown.death == 0.0
    assert breakdown.approach_nearest_enemy == 0.0
    assert breakdown.shoot_while_no_target == 0.0


def test_kill_spawner_reward_exceeds_kill_enemy_reward():
    """Sanity-check the design intent documented in rewards_config.py:
    R_KILL_SPAWNER must be strictly greater than R_KILL_ENEMY, or the agent
    has no incentive to prioritize spawners over farming enemies.
    """
    assert R_KILL_SPAWNER > R_KILL_ENEMY


def test_death_penalty_dominates_a_full_episode_of_positive_reward():
    """R_DEATH must be large enough in magnitude that a somewhat successful
    episode's positive rewards (a handful of kills) followed by death
    nets negative overall -- otherwise the agent has no incentive to avoid
    a suicidal aggressive playstyle.
    """
    breakdown = compute_reward(
        {
            "enemies_killed": 5,
            "spawners_killed": 1,
            "died": True,
        }
    )

    assert breakdown.death == R_DEATH
    assert breakdown.total < 0


def test_reward_breakdown_sums_to_total():
    """RewardBreakdown.total must equal the sum of its individual fields --
    guards against a term being added to the dataclass but forgotten in the
    total property.
    """
    breakdown = compute_reward(
        {
            "enemies_killed": 2,
            "spawners_killed": 1,
            "phase_advanced": True,
            "damage_taken": 10.0,
            "died": False,
            "distance_delta_to_nearest_enemy": -5.0,
            "shot_fired_with_no_target": True,
        }
    )

    expected_total = (
        breakdown.kill_enemy
        + breakdown.kill_spawner
        + breakdown.phase_progress
        + breakdown.damage_taken
        + breakdown.death
        + breakdown.approach_nearest_enemy
        + breakdown.shoot_while_no_target
    )
    assert breakdown.total == expected_total
