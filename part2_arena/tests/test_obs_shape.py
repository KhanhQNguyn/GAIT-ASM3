"""Correctness tests for arena/obs.py -- evidence for rubric row Part II-H's
"fixed-size vector with required features" requirement.
"""

import pytest


def test_observation_has_fixed_shape():
    """build_observation() must always return a 1D float32 array of length
    obs.OBS_DIM, regardless of how many enemies/spawners currently exist
    (including zero of either).

    TODO: implement once obs.build_observation is implemented.
    """
    pytest.skip("TODO: implement once obs.build_observation is implemented")


def test_observation_has_no_nans_or_infs():
    """Especially check the zero-enemies / zero-spawners edge case, where a
    naive nearest-distance calculation could divide by zero or return NaN.
    Once implemented, this test should also assert that every value in the
    returned array falls within [-1, 1] (the decided normalization convention
    for all features; see obs.py's OBSERVATION_SPEC and the rescaling formula
    there). Both the NaN/Inf check and the bounds check must pass for any
    valid arena state, including degenerate cases with no enemies/spawners.

    TODO: implement.
    """
    pytest.skip("TODO: implement once obs.build_observation is implemented")


def test_observation_includes_all_required_features():
    """obs.OBSERVATION_SPEC must include, at minimum, player position,
    player velocity, distance+direction to nearest enemy, distance+
    direction to nearest spawner, player health, and current phase --
    cross-check the spec names against OBSERVATION_SPEC's entries. It also
    carries three nearest_incoming_projectile_* features (see obs.py's
    header comment); assert those are present too, and that
    obs.OBS_DIM == len(OBSERVATION_SPEC).

    TODO: implement (can be a static assertion over OBSERVATION_SPEC's
    names, doesn't require a live env).
    """
    pytest.skip("TODO: implement, cross-referencing obs.OBSERVATION_SPEC")
