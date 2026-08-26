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

        DECIDED base curve (parameters from config/arena.json -> phase_curve):
            enemy_speed          = base_enemy_speed
                                   + phase * enemy_speed_gain_per_phase
            enemy_spawn_interval = max(min_spawn_interval_steps,
                                       base_spawn_interval_steps
                                       - phase * spawn_interval_decay_per_phase)
            num_spawners         = base_num_spawners
                                   + phase // extra_spawner_every_n_phases

        DECIDED curriculum ramp (config/arena.json -> curriculum), applied
        only when self.curriculum_enabled:
            Let R = curriculum.enabled_ramp_phases (default 3),
                s0 = curriculum.start_difficulty_fraction (default 0.5).
            For phase < R, scale the *difficulty-increasing* deltas by
                frac = s0 + (1 - s0) * (phase / R)
            i.e. enemy_speed and (base - interval) and the num_spawners
            increment are each multiplied by `frac`; for phase >= R the
            curve is identical to the base curve. So a curriculum run and a
            non-curriculum run converge to the same difficulty by phase R
            and differ only in how gently they get there -- which is exactly
            the comparison creativity hook (c) / report section 8 needs.

        TODO: implement the two formulas above (read the config once, cache
        it on the instance). Do not invent a different schedule -- if it
        changes, change it here AND in config/arena.json's _notes AND in the
        report together.
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
