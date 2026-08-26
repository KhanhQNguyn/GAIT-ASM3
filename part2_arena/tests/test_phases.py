"""Tests for arena/phases.py -- the phase system and curriculum ramp
(creativity hook c). Going-beyond coverage for MEMBER_C section 4.
"""

from arena.entities import Spawner
from arena.phases import PhaseManager


def _spawner(active=True):
    return Spawner(x=0, y=0, health=10 if active else 0, max_health=10,
                   spawn_interval_steps=100, active=active)


def test_phase_does_not_advance_with_partial_destruction():
    pm = PhaseManager()
    start = pm.phase
    assert pm.maybe_advance_phase([_spawner(True), _spawner(False)]) is False
    assert pm.phase == start


def test_phase_advances_only_when_all_spawners_destroyed():
    pm = PhaseManager()
    start = pm.phase
    assert pm.maybe_advance_phase([_spawner(False), _spawner(False)]) is True
    assert pm.phase == start + 1


def test_empty_spawner_list_never_advances():
    """all(... for _ in []) is vacuously True -- the guard must stop that."""
    pm = PhaseManager()
    for _ in range(5):
        assert pm.maybe_advance_phase([]) is False
    assert pm.phase == 0


def test_base_difficulty_curve_is_monotonic():
    pm = PhaseManager()
    d0, d3 = pm.difficulty_for_phase(0), pm.difficulty_for_phase(3)
    assert d3.enemy_speed > d0.enemy_speed
    assert d3.enemy_spawn_interval_steps <= d0.enemy_spawn_interval_steps
    assert d3.num_spawners >= d0.num_spawners
    assert d0.enemy_spawn_interval_steps >= pm._phase_curve["min_spawn_interval_steps"]


def test_curriculum_is_easier_early_and_converges_to_base_curve():
    base = PhaseManager(curriculum_enabled=False)
    cur = PhaseManager(curriculum_enabled=True)
    ramp = cur._curriculum["enabled_ramp_phases"]

    assert cur.difficulty_for_phase(1).enemy_speed < base.difficulty_for_phase(1).enemy_speed
    # past the ramp the two curves are identical
    assert cur.difficulty_for_phase(ramp + 2) == base.difficulty_for_phase(ramp + 2)
