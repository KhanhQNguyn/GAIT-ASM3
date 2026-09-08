"""Correctness tests for arena/rewards.py -- evidence for rubric row
Part II-J's reward structure requirement, and guards the "no inline reward
math outside rewards.py" architecture principle.
"""

import pytest

import arena.rewards as rewards_module
from arena.core_env import ArenaCoreEnv
from arena.entities import Enemy
from arena.rewards import APPROACH_REWARD_EPISODE_CAP, compute_reward
from arena.rewards_config import (
    R_APPROACH_NEAREST_ENEMY,
    R_DAMAGE_DEALT_PER_HP,
    R_DAMAGE_TAKEN_PER_HP,
    R_DEATH,
    R_KILL_ENEMY,
    R_KILL_SPAWNER,
    R_SHOOT_TOWARD_ENEMY,
    R_TIME_STEP_PENALTY,
    R_WALL_PROXIMITY_PER_STEP,
    SHOT_NO_TARGET_RADIUS,
    WALL_PROXIMITY_MARGIN,
)

# Test-time stand-in for the approach reward's pre-rebalance value: the
# shipped constant is 0.0 (disabled, see the decision test below), but the
# gating/cap MACHINERY in compute_reward is still live and must stay
# correct for a one-line re-enable. Monkeypatching the rewards module's
# binding works because compute_reward reads the module global at call time.
_ACTIVE_APPROACH_FOR_MACHINERY_TESTS = 0.01


def test_kill_enemy_reward_fires_independently():
    """A step_events dict with only enemies_killed=1 set must produce a
    RewardBreakdown with kill_enemy == R_KILL_ENEMY and every other field
    == 0 (except the unconditional per-step time penalty).
    """
    breakdown = compute_reward({"enemies_killed": 1})

    assert breakdown.kill_enemy == R_KILL_ENEMY
    assert breakdown.kill_spawner == 0.0
    assert breakdown.phase_progress == 0.0
    assert breakdown.damage_taken == 0.0
    assert breakdown.damage_dealt == 0.0
    assert breakdown.death == 0.0
    assert breakdown.approach_nearest_enemy == 0.0
    assert breakdown.shoot_while_no_target == 0.0
    assert breakdown.time_penalty == R_TIME_STEP_PENALTY


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
            "damage_dealt": 25.0,
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
        + breakdown.damage_dealt
        + breakdown.death
        + breakdown.approach_nearest_enemy
        + breakdown.shoot_while_no_target
        + breakdown.shoot_toward_enemy
        + breakdown.wall_proximity
        + breakdown.time_penalty
    )
    assert breakdown.total == expected_total


def test_damage_dealt_reward_scales_with_hp_removed():
    """R_DAMAGE_DEALT_PER_HP (2026-08-28 rebalance) must pay out per HP of
    enemy damage dealt by player projectiles, BEFORE any kill: the dense
    signal that gives early training a gradient toward shooting at all.
    """
    breakdown = compute_reward({"damage_dealt": 25.0})
    assert breakdown.damage_dealt == pytest.approx(25.0 * R_DAMAGE_DEALT_PER_HP)

    # A full 30-HP enemy (two 25-damage hits, second one clamped by
    # min(damage, health) in core_env) earns 30 * R_DAMAGE_DEALT_PER_HP of
    # this term plus R_KILL_ENEMY for the kill itself.
    full_enemy = compute_reward({"damage_dealt": 30.0, "enemies_killed": 1})
    assert full_enemy.damage_dealt == pytest.approx(30.0 * R_DAMAGE_DEALT_PER_HP)
    assert full_enemy.kill_enemy == R_KILL_ENEMY


def test_time_penalty_applies_unconditionally_every_step():
    """R_TIME_STEP_PENALTY (2026-08-28 rebalance) must be applied on EVERY
    step with no event required -- this is what makes "hide in a corner and
    do nothing for 1200 steps" cost -12 instead of being free.
    """
    empty = compute_reward({})
    assert empty.time_penalty == R_TIME_STEP_PENALTY
    assert empty.total == R_TIME_STEP_PENALTY  # no other term fires

    busy = compute_reward({"enemies_killed": 1, "phase_advanced": True, "died": True})
    assert busy.time_penalty == R_TIME_STEP_PENALTY

    # Anti-starvation sanity: an idle full-length episode (-12) must still be
    # far better than death (-100), so the penalty shapes pacing, not risk.
    assert 1200 * R_TIME_STEP_PENALTY > R_DEATH


def test_wall_proximity_ramps_linearly_and_stops_outside_margin():
    """R_WALL_PROXIMITY_PER_STEP (2026-08-28 rebalance) must be 0 outside
    WALL_PROXIMITY_MARGIN, ramp linearly from 0 (at the margin) to the full
    penalty (at the wall), and treat a missing wall_distance key as "no
    penalty" (the safe default for old callers/tests).
    """
    # Far from any wall: no penalty.
    far = compute_reward({"wall_distance": WALL_PROXIMITY_MARGIN + 1.0})
    assert far.wall_proximity == 0.0

    # Exactly at the wall: full penalty.
    at_wall = compute_reward({"wall_distance": 0.0})
    assert at_wall.wall_proximity == pytest.approx(R_WALL_PROXIMITY_PER_STEP)

    # Halfway into the margin: exactly half the penalty (linear ramp).
    half = compute_reward({"wall_distance": WALL_PROXIMITY_MARGIN / 2.0})
    assert half.wall_proximity == pytest.approx(R_WALL_PROXIMITY_PER_STEP / 2.0)

    # Missing key defaults to no penalty.
    absent = compute_reward({})
    assert absent.wall_proximity == 0.0


def test_shoot_toward_enemy_reward_fires_on_aimed_shot():
    """R_SHOOT_TOWARD_ENEMY (2026-08-28 rebalance) must pay out exactly when
    step_events says the player fired at a nearby enemy, and not otherwise.
    """
    aimed = compute_reward({"shot_toward_enemy": True})
    assert aimed.shoot_toward_enemy == R_SHOOT_TOWARD_ENEMY

    not_aimed = compute_reward({"shot_toward_enemy": False})
    assert not_aimed.shoot_toward_enemy == 0.0

    absent = compute_reward({})
    assert absent.shoot_toward_enemy == 0.0


def test_style2_shoot_preserves_velocity_and_noop_brakes():
    """ENV-BACKED pin of the 2026-08-28 kiting change: in style 2, SHOOT
    PRESERVES the player's velocity (glide while firing -- previously SHOOT
    hard-stopped, which made open-field shooting a sitting duck and pushed
    the policy onto walls as firing positions). NO_OP remains the deliberate
    brake.
    """
    env = ArenaCoreEnv(control_style=2)
    env.reset(seed=0)
    p = env.state.player

    # Simulate a MOVE_RIGHT step: velocity set to max speed along +x.
    p.vx, p.vy = 6.0, 0.0
    x_before = p.x
    obs, reward, done, info = env.step(int(5))  # ControlStyle2.SHOOT
    assert (p.vx, p.vy) == (6.0, 0.0)  # velocity preserved while firing
    assert p.x > x_before  # and the player keeps gliding

    # NO_OP brakes: velocity zeroes, position no longer advances.
    obs, reward, done, info = env.step(int(0))  # ControlStyle2.NO_OP
    assert (p.vx, p.vy) == (0.0, 0.0)
    x_after_brake = p.x
    obs, reward, done, info = env.step(int(0))
    assert p.x == x_after_brake


def test_approach_reward_disabled_by_decision():
    """DECISION (2026-08-28 rebalance): R_APPROACH_NEAREST_ENEMY is DISABLED
    at 0.0 -- enemies already seek the player, so paying for closing
    distance rewarded passivity. This test pins the decision so a silent
    revert is caught; re-enabling requires updating rewards_config.py's
    docstring AND this test deliberately.
    """
    assert R_APPROACH_NEAREST_ENEMY == 0.0
    breakdown = compute_reward(
        {
            "distance_delta_to_nearest_enemy": -100.0,
            "nearest_enemy_distance": SHOT_NO_TARGET_RADIUS + 1.0,
        }
    )
    assert breakdown.approach_nearest_enemy == 0.0


def test_approach_reward_machinery_still_gates_within_engage_range(monkeypatch):
    """The gating/cap MACHINERY for the (disabled) approach term must stay
    correct so a future re-enable is a one-line constant change. Patch the
    constant back to its pre-rebalance value for machinery verification.
    """
    monkeypatch.setattr(
        rewards_module, "R_APPROACH_NEAREST_ENEMY", _ACTIVE_APPROACH_FOR_MACHINERY_TESTS
    )

    # Within engage range: gated off entirely.
    gated = compute_reward(
        {
            "distance_delta_to_nearest_enemy": -5.0,
            "nearest_enemy_distance": SHOT_NO_TARGET_RADIUS - 1.0,
        }
    )
    assert gated.approach_nearest_enemy == 0.0

    # Outside engage range: pays -delta * constant.
    paying = compute_reward(
        {
            "distance_delta_to_nearest_enemy": -5.0,
            "nearest_enemy_distance": SHOT_NO_TARGET_RADIUS + 1.0,
        }
    )
    assert paying.approach_nearest_enemy == pytest.approx(
        5.0 * _ACTIVE_APPROACH_FOR_MACHINERY_TESTS
    )


def test_approach_reward_machinery_never_exceeds_per_episode_cap(monkeypatch):
    """The cumulative per-episode clamp (APPROACH_REWARD_EPISODE_CAP) must
    still bind when the term is active (patched on) even with a huge
    per-step closing distance and a nearly-spent budget.
    """
    monkeypatch.setattr(
        rewards_module, "R_APPROACH_NEAREST_ENEMY", _ACTIVE_APPROACH_FOR_MACHINERY_TESTS
    )

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


def test_contact_enemy_persists_cooldowns_and_gives_no_kill_credit():
    """ENV-BACKED pin of the NON-KAMIKAZE design rule (2026-08-28): an enemy
    touching the player deals contact damage ONCE, enters a per-enemy damage
    cooldown, SURVIVES, and awards NO R_KILL_ENEMY (kills are projectile-
    only). While in cooldown, further contact deals no damage. Shooting is
    the only way to remove enemies -- body-blocking can never farm kills,
    and tanking repeated contact now kills the player.
    """
    env = ArenaCoreEnv(control_style=2)
    env.reset(seed=0)
    p = env.state.player
    contact_damage = float(env._ecfg["contact_damage"])
    max_health = float(env._pcfg["max_health"])
    enemy = Enemy(x=p.x, y=p.y, health=30.0, max_health=30.0, speed=0.0)
    env.state.enemies = [enemy]

    # Contact #1: damage dealt, cooldown started, enemy SURVIVES.
    obs, reward, done, info = env.step(int(0))  # ControlStyle2.NO_OP
    rb = info["reward_breakdown"]
    assert rb.kill_enemy == 0.0  # NO kill credit for contact
    assert rb.damage_dealt == 0.0  # the player's projectiles did nothing
    assert rb.damage_taken == pytest.approx(R_DAMAGE_TAKEN_PER_HP * contact_damage)
    assert len(env.state.enemies) == 1  # the enemy persists (non-kamikaze)
    assert enemy.damage_cooldown == int(env._ecfg["contact_damage_cooldown_steps"])
    assert p.health == pytest.approx(max_health - contact_damage)

    # While in cooldown, contact deals NO further damage (i-frames).
    health_after_first = p.health
    for _ in range(3):
        obs, reward, done, info = env.step(int(0))
        assert info["reward_breakdown"].damage_taken == 0.0
    assert p.health == pytest.approx(health_after_first)

    # Fast-forward the cooldown; the next contact damages again.
    for _ in range(46):
        obs, reward, done, info = env.step(int(0))
    assert p.health < health_after_first  # second contact landed after cooldown
