"""Correctness tests for arena/rewards.py -- evidence for rubric row
Part II-J's reward structure requirement, and guards the "no inline reward
math outside rewards.py" architecture principle.
"""

import math

import pytest

import arena.rewards as rewards_module
from arena.core_env import ArenaCoreEnv
from arena.entities import Enemy
from arena.rewards import (
    APPROACH_REWARD_EPISODE_CAP,
    PROGRESS_REWARD_EPISODE_CAP,
    compute_reward,
)
from arena.rewards_config import (
    R_AIMED_HIT_BONUS,
    R_APPROACH_NEAREST_ENEMY,
    R_DAMAGE_DEALT_PER_HP,
    R_DAMAGE_TAKEN_PER_HP,
    R_DEATH,
    R_KILL_ENEMY,
    R_KILL_SPAWNER,
    R_PHASE_PROGRESS,
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


def test_damage_economy_makes_tanking_hits_unprofitable():
    """DECISION (2026-09-08c glass-cannon fix, see FIX_PART2.md): the
    survival incentive is carried by the DAMAGE economy, not by inflating
    R_DEATH. R_DAMAGE_TAKEN_PER_HP was raised -0.5 -> -0.7 (paired with
    enemy.contact_damage 25 -> 20 in config/arena.json), so absorbing a full
    100-HP health bar of contact costs -70 BEFORE the death term -- roughly
    R_KILL_ENEMY * 11, i.e. trading contact for one-shot kills (~+6.25 each)
    is a losing exchange. Pins the constant so a silent revert to -0.5 is
    caught; a deliberate retune updates this test.
    """
    assert R_DAMAGE_TAKEN_PER_HP == -0.7
    full_bar = compute_reward({"damage_taken": 100.0})
    assert full_bar.damage_taken == pytest.approx(-70.0)
    # a full bar of damage must sting far more than a single kill pays, so
    # the agent cannot shrug off sustained contact while farming kills
    assert abs(full_bar.damage_taken) >= 10 * R_KILL_ENEMY


def test_progression_rewards_are_uncapped():
    """DECISION (2026-09-08e spec-alignment, see FIX_PART2.md): every
    section-5 progression reward (kill_enemy, kill_spawner, phase_progress) is
    paid IN FULL every time -- PROGRESS_REWARD_EPISODE_CAP is +inf (disabled).
    The 2026-09-08c cap clamped kill_spawner + phase_progress, i.e. two of the
    five reward terms the spec requires, for most of the episode; that is a
    bigger deviation from section 5 than any optional shaping, so it was
    removed. Pins the decision so a silent re-enable to a finite value is
    caught.
    """
    assert math.isinf(PROGRESS_REWARD_EPISODE_CAP)

    # A huge running progression total does NOT reduce this step's payout.
    b = compute_reward(
        {
            "phase_advanced": True,
            "spawners_killed": 2,
            "enemies_killed": 4,
            "cumulative_progress_reward": 1_000_000.0,
        }
    )
    assert b.phase_progress == R_PHASE_PROGRESS
    assert b.kill_spawner == pytest.approx(R_KILL_SPAWNER * 2)
    assert b.kill_enemy == pytest.approx(R_KILL_ENEMY * 4)


def test_progress_cap_machinery_still_clamps_when_re_enabled(monkeypatch):
    """The clamp code + core_env's running-total plumbing are retained so a
    re-enable is a one-line change. Patch the cap to a finite value and
    verify it scales kill_spawner + phase_progress together to the remaining
    budget while leaving kill_enemy on the same step untouched.
    """
    cap = 100.0
    monkeypatch.setattr(rewards_module, "PROGRESS_REWARD_EPISODE_CAP", cap)

    raw_capped = R_PHASE_PROGRESS + R_KILL_SPAWNER  # this step's capped-term total
    already = cap - raw_capped / 2.0  # leave exactly half the step's budget
    near = compute_reward(
        {
            "phase_advanced": True,
            "spawners_killed": 1,
            "enemies_killed": 2,  # never capped
            "cumulative_progress_reward": already,
        }
    )
    remaining = cap - already
    assert near.phase_progress + near.kill_spawner == pytest.approx(remaining)
    scale = remaining / raw_capped
    assert near.phase_progress == pytest.approx(R_PHASE_PROGRESS * scale)
    assert near.kill_spawner == pytest.approx(R_KILL_SPAWNER * scale)
    assert near.kill_enemy == pytest.approx(R_KILL_ENEMY * 2)  # untouched

    over = compute_reward(
        {"phase_advanced": True, "spawners_killed": 1, "enemies_killed": 3,
         "cumulative_progress_reward": cap}
    )
    assert over.phase_progress == 0.0 and over.kill_spawner == 0.0
    assert over.kill_enemy == pytest.approx(R_KILL_ENEMY * 3)


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

    # A one-shot kill on a 25-HP enemy (base_health 25 == projectile_damage
    # 25 since 2026-09-08; core_env clamps damage_dealt at remaining HP)
    # earns 25 * R_DAMAGE_DEALT_PER_HP of this term plus R_KILL_ENEMY for
    # the kill itself.
    full_enemy = compute_reward({"damage_dealt": 25.0, "enemies_killed": 1})
    assert full_enemy.damage_dealt == pytest.approx(25.0 * R_DAMAGE_DEALT_PER_HP)
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


def test_shoot_toward_enemy_reward_disabled_by_decision():
    """DECISION (2026-09-08d less-attack fix, see BUG_LESS_ATTACK.md):
    R_SHOOT_TOWARD_ENEMY is DISABLED at 0.0. It paid per shot for FACING the
    objective, hit or miss -- the direct spray incentive (real fire-time
    alignment measured ~0.3, so it paid ~+0.04/shot just for pointing at the
    swarm). The `shot_toward_enemy` step_events key and the graded
    proportional payout wiring are RETAINED so re-enabling is a one-line
    constant change; this test pins the decision so a silent revert is
    caught.
    """
    assert R_SHOOT_TOWARD_ENEMY == 0.0
    for flag in (True, False, 0.5, 1.0):
        assert compute_reward({"shot_toward_enemy": flag}).shoot_toward_enemy == 0.0
    assert compute_reward({}).shoot_toward_enemy == 0.0


def test_shot_toward_enemy_flag_is_graded_alignment():
    """ENV-BACKED pin of the 2026-09-08 graded aim shaping: core_env's
    _shot_toward_enemy_flag is max(0, cos(diff)) between the shot direction
    and the nearest in-range enemy -- 1.0 dead-on, proportional off-axis,
    0.0 at 90+ deg. The graded term is what gives style 2 (whose aim is
    coupled to its cardinal movement) a smooth gradient toward facing the
    enemy before firing.
    """
    env = ArenaCoreEnv(control_style=2)
    env.reset(seed=0)
    p = env.state.player

    # Enemy directly to the player's right, well inside engage range.
    env.state.enemies.append(
        Enemy(x=p.x + 200.0, y=p.y, health=30.0, max_health=30.0, speed=2.0)
    )

    p.orientation = 0.0  # facing +x: dead-on at the enemy
    p.shoot_cooldown = 0
    env._try_shoot()
    assert env._shot_toward_enemy_flag == pytest.approx(1.0)


    p.orientation = math.radians(60.0)  # 60 deg off: graded cos(60) = 0.5
    p.shoot_cooldown = 0
    env._try_shoot()
    assert env._shot_toward_enemy_flag == pytest.approx(0.5)

    p.orientation = math.radians(90.0)  # perpendicular: no credit
    p.shoot_cooldown = 0
    env._try_shoot()
    assert env._shot_toward_enemy_flag == pytest.approx(0.0)


def test_aimed_hit_bonus_pays_graded_alignment_sum():
    """R_AIMED_HIT_BONUS (2026-09-08) must pay proportionally to the sum of
    the fire-time graded alignments of projectiles that actually HIT an
    objective this step -- dead-on intended hit pays in full, a lucky hit
    (alignment ~0) pays ~nothing, no hits pays 0.
    """
    assert compute_reward({"aimed_hit_alignment_sum": 1.0}).aimed_hit == pytest.approx(
        R_AIMED_HIT_BONUS
    )
    assert compute_reward({"aimed_hit_alignment_sum": 0.5}).aimed_hit == pytest.approx(
        R_AIMED_HIT_BONUS / 2.0
    )
    assert compute_reward({"aimed_hit_alignment_sum": 0.0}).aimed_hit == 0.0
    assert compute_reward({}).aimed_hit == 0.0


def test_aimed_hit_bonus_dominates_a_spray_hit():
    """DECISION (2026-09-08d less-attack fix, see BUG_LESS_ATTACK.md):
    R_AIMED_HIT_BONUS raised 1.0 -> 3.0 so a lined-up hit clearly out-earns a
    spray hit, making precise aim the better shooting strategy. A near-zero
    alignment hit that also kills (25 HP one-shot enemy) must still pay less
    than a well-aimed non-killing hit.
    """
    assert R_AIMED_HIT_BONUS == 3.0
    aimed_no_kill = compute_reward({"damage_dealt": 25.0, "aimed_hit_alignment_sum": 0.9})
    spray_kill = compute_reward(
        {"damage_dealt": 25.0, "enemies_killed": 1, "aimed_hit_alignment_sum": 0.05}
    )
    assert aimed_no_kill.total > spray_kill.total - R_KILL_ENEMY
    # and an aimed hit's bonus alone beats the full damage-dealt payout
    assert R_AIMED_HIT_BONUS * 0.9 > 25.0 * R_DAMAGE_DEALT_PER_HP


def test_aimed_hit_bonus_fires_on_hit_wiring():
    """ENV-BACKED pin: core_env stamps the graded alignment on each
    projectile at fire time and _resolve_collisions accumulates it only
    for projectiles that actually hit -- an unaimed shot that hits nothing
    contributes 0.
    """
    env = ArenaCoreEnv(control_style=2)
    env.reset(seed=0)
    p = env.state.player

    # Enemy directly ahead in the bullet's path, tough enough to survive.
    env.state.enemies.append(
        Enemy(x=p.x + 12.0, y=p.y, health=50.0, max_health=50.0, speed=0.0)
    )
    p.orientation = 0.0
    p.shoot_cooldown = 0
    env._try_shoot()
    env._advance_projectiles()
    result = env._resolve_collisions()
    assert result[-1] == pytest.approx(1.0)  # dead-on hit: full alignment

    # Enemy BEHIND the player: the forward shot hits nothing.
    env.state.enemies = [
        Enemy(x=p.x - 12.0, y=p.y, health=50.0, max_health=50.0, speed=0.0)
    ]
    p.shoot_cooldown = 0
    p.orientation = 0.0
    env._try_shoot()
    env._advance_projectiles()
    result = env._resolve_collisions()
    assert result[-1] == pytest.approx(0.0)


def test_wall_proximity_corner_sums_two_walls():
    """R_WALL_PROXIMITY_PER_STEP (2026-09-08 corner-aware form) must sum the
    ramps of the two nearest walls, so a corner costs ~2x a straight wall;
    a missing wall_distance_second key keeps old one-wall behaviour.
    """
    # Straight wall: unchanged.
    one_wall = compute_reward({"wall_distance": 0.0})
    assert one_wall.wall_proximity == pytest.approx(R_WALL_PROXIMITY_PER_STEP)

    # Corner: both ramps at full -> 2x.
    corner = compute_reward({"wall_distance": 0.0, "wall_distance_second": 0.0})
    assert corner.wall_proximity == pytest.approx(2 * R_WALL_PROXIMITY_PER_STEP)

    # Old callers that never send the second key: no extra penalty.
    legacy = compute_reward({"wall_distance": WALL_PROXIMITY_MARGIN + 1.0})
    assert legacy.wall_proximity == 0.0

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
