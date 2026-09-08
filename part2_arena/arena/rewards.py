"""The ONLY place a Part II reward value is computed. core_env.py must call
compute_reward() and use its returned total -- never add ad hoc reward
math inline in step(). This is what keeps rewards_config.py truthful and
what makes reward-term unit tests (tests/test_reward_terms.py) and the
creativity(b) reward-decomposition dashboard possible: every term is
individually named and returned, not just summed silently.

--- Ownership note --------------------------------------------------------
compute_reward()'s body was originally implemented by Member C to unblock
core_env.py (hard dependency); reconciled with Member D's already-merged
R_APPROACH_NEAREST_ENEMY gating/cap requirement (docs/AUDIT_main.md 5.8)
when feat/member-c-khang was merged -- see APPROACH_REWARD_EPISODE_CAP and
the two OPTIONAL step_events keys below. Member D owns: the constant
VALUES in rewards_config.py and tests/test_reward_terms.py.

Final reward-shaping decision: see rewards_config.py's per-constant
docstrings (dated team decision) -- do not reintroduce unresolved language
here. RATIFIED as of the 2026-08-28 rebalance: R_APPROACH_NEAREST_ENEMY
DISABLED at 0.0 (machinery retained + tested) and R_SHOOT_WHILE_NO_TARGET
disabled at 0.0, replaced by two ACTIVE shaping terms --
R_DAMAGE_DEALT_PER_HP (dense per-hit signal; the fix for the degenerate
corner-camping policies that never learned to shoot) and
R_TIME_STEP_PENALTY (makes passivity costly). Member C's original
recommendation (drop approach to 0.0) was ultimately adopted by this
rebalance; the earlier disagreement history is still recorded in
docs/DECISIONS.md.
R_DEATH is decided too -- KEPT at -100.0; see rewards_config.py::R_DEATH's
docstring for the ablation evidence and rationale.
-------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass

from arena.rewards_config import (
    R_APPROACH_NEAREST_ENEMY,
    R_DAMAGE_DEALT_PER_HP,
    R_DAMAGE_TAKEN_PER_HP,
    R_DEATH,
    R_KILL_ENEMY,
    R_KILL_SPAWNER,
    R_PHASE_PROGRESS,
    R_SHOOT_TOWARD_ENEMY,
    R_SHOOT_WHILE_NO_TARGET,
    R_TIME_STEP_PENALTY,
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
    damage_dealt: float = 0.0
    death: float = 0.0
    approach_nearest_enemy: float = 0.0
    shoot_while_no_target: float = 0.0
    shoot_toward_enemy: float = 0.0
    time_penalty: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.kill_enemy
            + self.kill_spawner
            + self.phase_progress
            + self.damage_taken
            + self.damage_dealt
            + self.death
            + self.approach_nearest_enemy
            + self.shoot_while_no_target
            + self.shoot_toward_enemy
            + self.time_penalty
        )


def compute_reward(
    step_events: dict, reward_overrides: dict[str, float] | None = None
) -> RewardBreakdown:
    """Translate this step's game events into a RewardBreakdown using ONLY
    the constants in rewards_config.py.

    LOCKED `step_events` CONTRACT (core_env.step() must produce exactly this
    shape; keep this block identical to the one in core_env.step()'s
    docstring -- a silent key-name drift produces wrong rewards with no
    error):

        step_events = {
            "enemies_killed":                  int,    # enemies destroyed this step
            "spawners_killed":                 int,    # spawners destroyed this step
            "phase_advanced":                  bool,   # phase incremented this step
            "damage_taken":                    float,  # player HP lost this step, >= 0
            "damage_dealt":                    float,  # enemy HP removed by player
                                                       #   projectiles this step, >= 0
            "died":                            bool,   # player HP reached 0 this step
            "distance_delta_to_nearest_enemy": float,  # signed; < 0 means the player
                                                       #   got closer to the nearest
                                                       #   enemy this step; 0.0 when
                                                       #   there is no enemy
            "shot_fired_with_no_target":       bool,   # player fired this step while
                                                       #   the nearest enemy was
                                                       #   farther than
                                                       #   SHOT_NO_TARGET_RADIUS (or
                                                       #   no enemy existed)
            "shot_toward_enemy":               bool,   # player fired this step at its
                                                       #   current objective (nearest
                                                       #   enemy within
                                                       #   SHOT_NO_TARGET_RADIUS, or --
                                                       #   when none is in range --
                                                       #   the nearest active spawner),
                                                       #   within the aim window
        }

    Every key is always present. Missing keys are treated as 0 / False so a
    partial dict in a unit test still works.

    OPTIONAL extension keys (Member D's R_APPROACH_NEAREST_ENEMY gating/cap,
    docs/AUDIT_main.md 5.8 -- both default to values that make the gate/cap
    a no-op, so core_env.py implementations that don't populate them still
    get a correct, just ungated/uncapped, approach_nearest_enemy):

        "nearest_enemy_distance"     float, default +inf
            (distance from the player to the nearest enemy, in arena world
            units, AFTER this step's movement; used to gate
            approach_nearest_enemy off once the nearest enemy is within
            engage range -- see SHOT_NO_TARGET_RADIUS)
        "cumulative_approach_reward" float, default 0.0
            (running total of approach_nearest_enemy already awarded THIS
            EPISODE, BEFORE this step -- the caller must track this across
            steps for the per-episode cap, APPROACH_REWARD_EPISODE_CAP, to
            actually bind)

    Term mapping:
        kill_enemy             = R_KILL_ENEMY            * enemies_killed
        kill_spawner           = R_KILL_SPAWNER          * spawners_killed
        phase_progress         = R_PHASE_PROGRESS        * phase_advanced
        damage_taken            = R_DAMAGE_TAKEN_PER_HP  * damage_taken  (already negative)
        damage_dealt            = R_DAMAGE_DEALT_PER_HP  * damage_dealt
        death                  = R_DEATH                 * died
        approach_nearest_enemy = R_APPROACH_NEAREST_ENEMY * max(-distance_delta, 0),
                                  gated to only pay out while
                                  nearest_enemy_distance > SHOT_NO_TARGET_RADIUS,
                                  and clamped so cumulative_approach_reward +
                                  this step's amount never exceeds
                                  APPROACH_REWARD_EPISODE_CAP
                                  (currently DISABLED: the constant is 0.0, so
                                  this term is always 0 -- machinery retained)
        shoot_while_no_target  = R_SHOOT_WHILE_NO_TARGET * shot_fired_with_no_target
        shoot_toward_enemy     = R_SHOOT_TOWARD_ENEMY     * shot_toward_enemy
        time_penalty           = R_TIME_STEP_PENALTY     (unconditional, every step)

    `reward_overrides` (optional): a {constant_name: value} dict that
    replaces a reward constant for this call only. Currently only
    "R_DEATH" is honoured -- it exists so scripts/train.py --death-penalty
    can run a real R_DEATH ablation (docs/KHANG.md C.3). This module still
    computes every reward number (Global Invariant #1); the override is a
    call-time substitution read inside this function, not reward math
    happening somewhere else. Unknown keys are ignored. Note that
    `from arena.rewards_config import R_DEATH` binds by value, so
    monkeypatching rewards_config at runtime does NOT work -- this
    parameter is the only correct mechanism.
    """
    ev = step_events or {}
    overrides = reward_overrides or {}
    r_death = float(overrides.get("R_DEATH", R_DEATH))
    delta = float(ev.get("distance_delta_to_nearest_enemy", 0.0))
    nearest_enemy_distance = float(ev.get("nearest_enemy_distance", float("inf")))
    cumulative_approach_reward = float(ev.get("cumulative_approach_reward", 0.0))

    approach_nearest_enemy = 0.0
    if delta < 0 and nearest_enemy_distance > SHOT_NO_TARGET_RADIUS:
        raw_approach_reward = -delta * R_APPROACH_NEAREST_ENEMY
        remaining_budget = max(0.0, APPROACH_REWARD_EPISODE_CAP - cumulative_approach_reward)
        approach_nearest_enemy = min(raw_approach_reward, remaining_budget)

    return RewardBreakdown(
        kill_enemy=R_KILL_ENEMY * int(ev.get("enemies_killed", 0)),
        kill_spawner=R_KILL_SPAWNER * int(ev.get("spawners_killed", 0)),
        phase_progress=R_PHASE_PROGRESS * (1.0 if ev.get("phase_advanced") else 0.0),
        damage_taken=R_DAMAGE_TAKEN_PER_HP * float(ev.get("damage_taken", 0.0)),
        damage_dealt=R_DAMAGE_DEALT_PER_HP * float(ev.get("damage_dealt", 0.0)),
        death=r_death * (1.0 if ev.get("died") else 0.0),
        approach_nearest_enemy=approach_nearest_enemy,
        shoot_while_no_target=(
            R_SHOOT_WHILE_NO_TARGET * (1.0 if ev.get("shot_fired_with_no_target") else 0.0)
        ),
        shoot_toward_enemy=(
            R_SHOOT_TOWARD_ENEMY * (1.0 if ev.get("shot_toward_enemy") else 0.0)
        ),
        time_penalty=R_TIME_STEP_PENALTY,
    )
