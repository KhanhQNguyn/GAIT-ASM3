"""Phase system: tracks currently-active spawners and advances the arena's
difficulty phase once all of them are destroyed. Also hosts the creativity
hook (c) curriculum-learning ramp.
"""

from __future__ import annotations

from dataclasses import dataclass

from arena.entities import Spawner


@dataclass
class PhaseConfig:
    """Difficulty parameters for one phase. difficulty_for_phase() below
    produces one of these per phase number.
    """

    enemy_speed: float
    enemy_spawn_interval_steps: int
    num_spawners: int


class PhaseManager:
    """Owns the current phase number and decides when to advance it.

    curriculum_enabled (creativity hook c): when True, early phases are
    deliberately easier than difficulty_for_phase() would normally produce
    (slower enemies / sparser spawns), ramping up to full difficulty over
    the first few phases, instead of full difficulty from phase 0. Used by
    scripts/train.py --curriculum {on,off} to produce a learning-speed
    comparison for the report.
    """

    def __init__(self, curriculum_enabled: bool = False):
        self.phase = 0
        self.curriculum_enabled = curriculum_enabled

    def difficulty_for_phase(self, phase: int) -> PhaseConfig:
        """Return the (possibly curriculum-adjusted) difficulty parameters
        for the given phase number.

        TODO: implement a base difficulty curve (e.g. enemy_speed and
        spawn frequency increasing with phase, num_spawners increasing
        every couple of phases), and when self.curriculum_enabled, scale
        early phases down toward an easier starting point that converges to
        the same curve by some target phase (document the ramp schedule
        here once decided, since the report needs to describe it).
        """
        raise NotImplementedError

    def all_spawners_destroyed(self, spawners: list[Spawner]) -> bool:
        return all(not s.active for s in spawners)

    def maybe_advance_phase(self, spawners: list[Spawner]) -> bool:
        """Check whether every currently-active spawner has been
        destroyed; if so, increment self.phase and return True (the caller
        is then responsible for spawning the next phase's spawners using
        difficulty_for_phase(self.phase) and awarding R_PHASE_PROGRESS via
        rewards.py). Returns False if the phase did not advance.

        TODO: implement.
        """
        raise NotImplementedError
