"""The ONLY place a Part II reward value is computed. core_env.py must call
compute_reward() and use its returned total -- never add ad hoc reward
math inline in step(). This is what keeps rewards_config.py truthful and
what makes reward-term unit tests (tests/test_reward_terms.py) and the
creativity(b) reward-decomposition dashboard possible: every term is
individually named and returned, not just summed silently.
"""

from __future__ import annotations

from dataclasses import dataclass

from arena.rewards_config import (
    R_APPROACH_NEAREST_ENEMY,
    R_DAMAGE_TAKEN_PER_HP,
    R_DEATH,
    R_KILL_ENEMY,
    R_KILL_SPAWNER,
    R_PHASE_PROGRESS,
    R_SHOOT_WHILE_NO_TARGET,
    SHOT_NO_TARGET_RADIUS,
)

# Per-episode cap on R_APPROACH_NEAREST_ENEMY's cumulative contribution (see
# its docstring in rewards_config.py, and docs/AUDIT_main.md 5.8) -- kept
# equal to R_KILL_ENEMY so this shaping term can never out-earn a single
# real kill, however long the agent loiters near an enemy.
APPROACH_REWARD_EPISODE_CAP: float = R_KILL_ENEMY


@dataclass
class RewardBreakdown:
    """Per-term reward contribution for one step, plus the total. Returned
    alongside the scalar reward so train.py can log each term to
    TensorBoard separately (reward decomposition, creativity hook b).
    """

    kill_enemy: float = 0.0
    kill_spawner: float = 0.0
    phase_progress: float = 0.0
    damage_taken: float = 0.0
    death: float = 0.0
    approach_nearest_enemy: float = 0.0
    shoot_while_no_target: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.kill_enemy
            + self.kill_spawner
            + self.phase_progress
            + self.damage_taken
            + self.death
            + self.approach_nearest_enemy
            + self.shoot_while_no_target
        )


def compute_reward(step_events: dict) -> RewardBreakdown:
    """Translate this step's game events into a RewardBreakdown using ONLY
    the constants in rewards_config.py.

    FINALIZED step_events shape (Member D/C contract -- core_env.py's
    step() must produce a dict with exactly these keys; any key may be
    omitted by the caller and is treated as its listed default, so partial
    dicts -- e.g. in unit tests -- work without every key being spelled out):

        "enemies_killed"                    int,   default 0
        "spawners_killed"                   int,   default 0
        "phase_advanced"                    bool,  default False
        "damage_taken"                      float, default 0.0 (>= 0, HP lost this step)
        "died"                              bool,  default False
        "distance_delta_to_nearest_enemy"   float, default 0.0
            (negative = agent got closer to the nearest enemy this step,
            positive = got farther, 0.0 if there is no enemy to measure to)
        "nearest_enemy_distance"            float, default +inf
            (distance from the agent to the nearest enemy, in arena world
            units, AFTER this step's movement; +inf, or simply omitting
            this key, both mean "no enemy / outside engage range" for
            R_APPROACH_NEAREST_ENEMY's gating -- see below)
        "cumulative_approach_reward"        float, default 0.0
            (running total of approach_nearest_enemy already awarded THIS
            EPISODE, BEFORE this step -- the caller/core_env.py must track
            this across steps; used to enforce R_APPROACH_NEAREST_ENEMY's
            per-episode cap, see APPROACH_REWARD_EPISODE_CAP above)
        "shot_fired_with_no_target"         bool,  default False
            (True iff a shot was fired AND the nearest enemy was farther
            than SHOT_NO_TARGET_RADIUS at that moment -- NOT "zero enemies
            exist anywhere in the arena")

    R_APPROACH_NEAREST_ENEMY is gated + capped per its rewards_config.py
    docstring: it only pays out when the agent got strictly closer this
    step (distance_delta < 0) AND the nearest enemy is still outside
    engage range (nearest_enemy_distance > SHOT_NO_TARGET_RADIUS), and the
    payout is clamped so cumulative_approach_reward + this step's amount
    never exceeds APPROACH_REWARD_EPISODE_CAP.
    """
    enemies_killed = step_events.get("enemies_killed", 0)
    spawners_killed = step_events.get("spawners_killed", 0)
    phase_advanced = step_events.get("phase_advanced", False)
    damage_taken = step_events.get("damage_taken", 0.0)
    died = step_events.get("died", False)
    distance_delta = step_events.get("distance_delta_to_nearest_enemy", 0.0)
    nearest_enemy_distance = step_events.get("nearest_enemy_distance", float("inf"))
    cumulative_approach_reward = step_events.get("cumulative_approach_reward", 0.0)
    shot_fired_with_no_target = step_events.get("shot_fired_with_no_target", False)

    approach_nearest_enemy = 0.0
    if distance_delta < 0 and nearest_enemy_distance > SHOT_NO_TARGET_RADIUS:
        raw_approach_reward = -distance_delta * R_APPROACH_NEAREST_ENEMY
        remaining_budget = max(0.0, APPROACH_REWARD_EPISODE_CAP - cumulative_approach_reward)
        approach_nearest_enemy = min(raw_approach_reward, remaining_budget)

    return RewardBreakdown(
        kill_enemy=enemies_killed * R_KILL_ENEMY,
        kill_spawner=spawners_killed * R_KILL_SPAWNER,
        phase_progress=R_PHASE_PROGRESS if phase_advanced else 0.0,
        damage_taken=damage_taken * R_DAMAGE_TAKEN_PER_HP,
        death=R_DEATH if died else 0.0,
        approach_nearest_enemy=approach_nearest_enemy,
        shoot_while_no_target=R_SHOOT_WHILE_NO_TARGET if shot_fired_with_no_target else 0.0,
    )
