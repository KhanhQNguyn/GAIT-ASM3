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

    `step_events` is expected to carry whatever core_env.py's step() logic
    determined happened this step. Example (illustrative, not executable):
    The dictionary should contain keys such as "enemies_killed" (int),
    "spawners_killed" (int), "phase_advanced" (bool), "damage_taken" (float),
    "died" (bool), "distance_delta_to_nearest_enemy" (float, negative means
    got closer), and "shot_fired_with_no_target" (bool).
    "shot_fired_with_no_target" means no enemy was within the detection
    radius at the moment the shot was fired -- NOT "zero enemies exist
    anywhere in the arena." Specifically: if the distance to the nearest
    enemy exceeds SHOT_NO_TARGET_RADIUS (defined in rewards_config.py;
    see its docstring for the tuning TODO and scale rationale), the shot
    counts as having no target regardless of how many enemies are alive
    elsewhere in the arena.
    The exact key set is up to the implementer -- document it here once
    finalized, since core_env.py must produce exactly this shape.

    TODO: implement, multiplying each event by its rewards_config constant.
    """
    raise NotImplementedError
