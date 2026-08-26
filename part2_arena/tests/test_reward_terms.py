"""Correctness tests for arena/rewards.py -- evidence for rubric row
Part II-J's reward structure requirement, and guards the "no inline reward
math outside rewards.py" architecture principle.
"""

import pytest

from arena.rewards import APPROACH_REWARD_EPISODE_CAP, compute_reward
from arena.rewards_config import (
    R_APPROACH_NEAREST_ENEMY,
    R_DEATH,
    R_KILL_ENEMY,
    R_KILL_SPAWNER,
    SHOT_NO_TARGET_RADIUS,
)


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


def test_approach_reward_gated_off_within_engage_range():
    """R_APPROACH_NEAREST_ENEMY must NOT pay out once the nearest enemy is
    within engage/weapon range (per rewards_config.py's REQUIRED
    implementation shape) -- at that point the agent should be shooting,
    not still collecting a "getting closer" shaping bonus.
    """
    breakdown = compute_reward(
        {
            "distance_delta_to_nearest_enemy": -5.0,
            "nearest_enemy_distance": SHOT_NO_TARGET_RADIUS - 1.0,
        }
    )
    assert breakdown.approach_nearest_enemy == 0.0


def test_approach_reward_pays_out_when_closing_distance_outside_engage_range():
    """Sanity check the normal, ungated path: closing distance while still
    outside engage range pays the expected raw amount.
    """
    breakdown = compute_reward(
        {
            "distance_delta_to_nearest_enemy": -5.0,
            "nearest_enemy_distance": SHOT_NO_TARGET_RADIUS + 1.0,
        }
    )
    assert breakdown.approach_nearest_enemy == 5.0 * R_APPROACH_NEAREST_ENEMY


def test_approach_reward_never_exceeds_per_episode_cap():
    """R_APPROACH_NEAREST_ENEMY's cumulative per-episode contribution must
    be clamped at APPROACH_REWARD_EPISODE_CAP even when the caller reports
    a large per-step closing distance, and even when most of the budget is
    already spent (cumulative_approach_reward close to the cap).
    """
    breakdown = compute_reward(
        {
            "distance_delta_to_nearest_enemy": -1000.0,
            "nearest_enemy_distance": SHOT_NO_TARGET_RADIUS + 1.0,
            "cumulative_approach_reward": APPROACH_REWARD_EPISODE_CAP - 0.02,
        }
    )
    assert breakdown.approach_nearest_enemy == pytest.approx(0.02)

    breakdown_already_capped = compute_reward(
        {
            "distance_delta_to_nearest_enemy": -1000.0,
            "nearest_enemy_distance": SHOT_NO_TARGET_RADIUS + 1.0,
            "cumulative_approach_reward": APPROACH_REWARD_EPISODE_CAP,
        }
    )
    assert breakdown_already_capped.approach_nearest_enemy == 0.0
