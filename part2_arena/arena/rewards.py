"""The ONLY place a Part II reward value is computed. core_env.py must call
compute_reward() and use its returned total -- never add ad hoc reward
math inline in step(). This is what keeps rewards_config.py truthful and
what makes reward-term unit tests (tests/test_reward_terms.py) and the
creativity(b) reward-decomposition dashboard possible: every term is
individually named and returned, not just summed silently.

--- Ownership note --------------------------------------------------------
compute_reward()'s body was implemented by Member C to unblock core_env.py
(hard dependency). Member D owns: the constant VALUES in rewards_config.py
and tests/test_reward_terms.py. Member C's recommendation, per
docs/message.txt (keep shaping minimal; don't reward-shape the strategy):
keep R_APPROACH_NEAREST_ENEMY = 0.0 and R_SHOOT_WHILE_NO_TARGET = 0.0 so
only the 5 spec-required terms are live, and reconsider R_DEATH = -100
(20x a kill -- large enough to dominate the signal; justify with evidence
in report section 3 or reduce it).
-------------------------------------------------------------------------
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
)

# NOTE: SHOT_NO_TARGET_RADIUS lives in rewards_config.py and is applied by
# core_env.step() when it decides step_events["shot_fired_with_no_target"];
# compute_reward() only prices the resulting boolean.


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

    LOCKED `step_events` CONTRACT (core_env.step() must produce exactly this
    shape; keep this block identical to the one in core_env.step()'s
    docstring -- a silent key-name drift produces wrong rewards with no
    error):

        step_events = {
            "enemies_killed":                  int,    # enemies destroyed this step
            "spawners_killed":                 int,    # spawners destroyed this step
            "phase_advanced":                  bool,   # phase incremented this step
            "damage_taken":                    float,  # player HP lost this step, >= 0
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
        }

    Every key is always present. Missing keys are treated as 0 / False so a
    partial dict in a unit test still works.

    Term mapping:
        kill_enemy            = R_KILL_ENEMY            * enemies_killed
        kill_spawner          = R_KILL_SPAWNER          * spawners_killed
        phase_progress        = R_PHASE_PROGRESS        * phase_advanced
        damage_taken          = R_DAMAGE_TAKEN_PER_HP   * damage_taken   (already negative constant)
        death                 = R_DEATH                 * died
        approach_nearest_enemy= R_APPROACH_NEAREST_ENEMY * max(-distance_delta, 0)
                                (rewards ONLY closing distance; 0 while the
                                 constant is 0, which is the recommendation)
        shoot_while_no_target = R_SHOOT_WHILE_NO_TARGET * shot_fired_with_no_target
    """
    ev = step_events or {}
    delta = float(ev.get("distance_delta_to_nearest_enemy", 0.0))
    return RewardBreakdown(
        kill_enemy=R_KILL_ENEMY * int(ev.get("enemies_killed", 0)),
        kill_spawner=R_KILL_SPAWNER * int(ev.get("spawners_killed", 0)),
        phase_progress=R_PHASE_PROGRESS * (1.0 if ev.get("phase_advanced") else 0.0),
        damage_taken=R_DAMAGE_TAKEN_PER_HP * float(ev.get("damage_taken", 0.0)),
        death=R_DEATH * (1.0 if ev.get("died") else 0.0),
        approach_nearest_enemy=R_APPROACH_NEAREST_ENEMY * max(-delta, 0.0),
        shoot_while_no_target=(
            R_SHOOT_WHILE_NO_TARGET * (1.0 if ev.get("shot_fired_with_no_target") else 0.0)
        ),
    )
