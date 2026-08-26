"""Phase system: tracks currently-active spawners and advances the arena's
difficulty phase once all of them are destroyed. Also hosts the creativity
hook (c) curriculum-learning ramp.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

from arena.entities import Spawner

_CONFIG_PATH = pathlib.Path(__file__).resolve().parent.parent / "config" / "arena.json"

# Fallbacks used when config/arena.json is missing or partial. The
# authoritative values live in that file (phase_curve / curriculum blocks).
_DEFAULT_PHASE_CURVE = {
    "base_enemy_speed": 1.6,
    "enemy_speed_gain_per_phase": 0.35,
    "base_spawn_interval_steps": 120,
    "spawn_interval_decay_per_phase": 12,
    "min_spawn_interval_steps": 40,
    "base_num_spawners": 1,
    "extra_spawner_every_n_phases": 2,
    "max_expected_phase": 6,
}
_DEFAULT_CURRICULUM = {"enabled_ramp_phases": 3, "start_difficulty_fraction": 0.5}


def _load_arena_config() -> dict:
    try:
        return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


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
        cfg = _load_arena_config()
        self._phase_curve = {**_DEFAULT_PHASE_CURVE, **cfg.get("phase_curve", {})}
        self._curriculum = {**_DEFAULT_CURRICULUM, **cfg.get("curriculum", {})}

    def difficulty_for_phase(self, phase: int) -> PhaseConfig:
        """Return the (possibly curriculum-adjusted) difficulty parameters
        for the given phase number.

        Base curve (parameters from config/arena.json -> phase_curve):
            enemy_speed          = base_enemy_speed
                                   + phase * enemy_speed_gain_per_phase
            enemy_spawn_interval = max(min_spawn_interval_steps,
                                       base_spawn_interval_steps
                                       - phase * spawn_interval_decay_per_phase)
            num_spawners         = base_num_spawners
                                   + phase // extra_spawner_every_n_phases

        Curriculum ramp (config/arena.json -> curriculum), applied only when
        self.curriculum_enabled and phase < R = enabled_ramp_phases:
            frac = s0 + (1 - s0) * (phase / R),  s0 = start_difficulty_fraction
        and each difficulty-INCREASING delta (the enemy_speed gain, the
        spawn-interval reduction, and the extra-spawner increment) is scaled
        by `frac`. For phase >= R, frac = 1 and the curve is identical to the
        base curve -- so a curriculum run and a non-curriculum run converge
        to the same difficulty by phase R and differ only in how gently they
        get there (exactly what creativity hook (c) / report section 8 needs).
        """
        pc = self._phase_curve
        phase = max(0, int(phase))

        frac = 1.0
        if self.curriculum_enabled:
            ramp = int(self._curriculum["enabled_ramp_phases"])
            s0 = float(self._curriculum["start_difficulty_fraction"])
            if ramp > 0 and phase < ramp:
                frac = s0 + (1.0 - s0) * (phase / ramp)

        enemy_speed = pc["base_enemy_speed"] + frac * phase * pc["enemy_speed_gain_per_phase"]

        interval_drop = frac * phase * pc["spawn_interval_decay_per_phase"]
        interval_raw = pc["base_spawn_interval_steps"] - interval_drop
        enemy_spawn_interval_steps = max(
            int(pc["min_spawn_interval_steps"]), int(round(interval_raw))
        )

        every = int(pc["extra_spawner_every_n_phases"])
        extra = (phase // every) if every > 0 else 0
        num_spawners = max(1, int(pc["base_num_spawners"]) + int(frac * extra))

        return PhaseConfig(
            enemy_speed=float(enemy_speed),
            enemy_spawn_interval_steps=enemy_spawn_interval_steps,
            num_spawners=num_spawners,
        )

    def all_spawners_destroyed(self, spawners: list[Spawner]) -> bool:
        return all(not s.active for s in spawners)

    def maybe_advance_phase(self, spawners: list[Spawner]) -> bool:
        """Advance the phase iff there is at least one spawner and every one
        of them is destroyed. Returns True on advance (the caller then spawns
        the next phase's spawners via difficulty_for_phase(self.phase) and
        marks R_PHASE_PROGRESS), False otherwise.

        The empty-list guard matters: `all(... for _ in [])` is vacuously
        True, so without it an episode with no spawners would advance the
        phase every single step.
        """
        if not spawners:
            return False
        if self.all_spawners_destroyed(spawners):
            self.phase += 1
            return True
        return False
