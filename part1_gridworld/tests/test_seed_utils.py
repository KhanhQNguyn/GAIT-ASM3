"""Tests for src/seed_utils.py -- reproducibility seeding."""

import random

from src.seed_utils import set_seed


def test_set_seed_returns_dedicated_rng_and_seeds_globals():
    r1 = set_seed(123)
    assert isinstance(r1, random.Random)
    a = random.random()
    set_seed(123)
    b = random.random()
    assert a == b  # global state reproducibly re-seeded
    r2 = set_seed(123)
    assert r1.random() == r2.random()  # dedicated RNGs reproducible and independent
