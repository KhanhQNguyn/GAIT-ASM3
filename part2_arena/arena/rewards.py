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
)

# NOTE: SHOT_NO_TARGET_RADIUS is deliberately NOT imported here -- it is
# consumed by core_env.py (which measures the live distance to the nearest
# enemy at the moment a shot is fired and sets step_events
# ["shot_fired_with_no_target"] accordingly), not by compute_reward, which
# only ever sees the already-computed boolean. See rewards_config.py's
# docstring for the constant's derivation.


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
        "shot_fired_with_no_target"         bool,  default False
            (True iff a shot was fired AND the nearest enemy was farther
            than SHOT_NO_TARGET_RADIUS at that moment -- NOT "zero enemies
            exist anywhere in the arena")
    """
    enemies_killed = step_events.get("enemies_killed", 0)
    spawners_killed = step_events.get("spawners_killed", 0)
    phase_advanced = step_events.get("phase_advanced", False)
    damage_taken = step_events.get("damage_taken", 0.0)
    died = step_events.get("died", False)
    distance_delta = step_events.get("distance_delta_to_nearest_enemy", 0.0)
    shot_fired_with_no_target = step_events.get("shot_fired_with_no_target", False)

    return RewardBreakdown(
        kill_enemy=enemies_killed * R_KILL_ENEMY,
        kill_spawner=spawners_killed * R_KILL_SPAWNER,
        phase_progress=R_PHASE_PROGRESS if phase_advanced else 0.0,
        damage_taken=damage_taken * R_DAMAGE_TAKEN_PER_HP,
        death=R_DEATH if died else 0.0,
        approach_nearest_enemy=-distance_delta * R_APPROACH_NEAREST_ENEMY,
        shoot_while_no_target=R_SHOOT_WHILE_NO_TARGET if shot_fired_with_no_target else 0.0,
    )
