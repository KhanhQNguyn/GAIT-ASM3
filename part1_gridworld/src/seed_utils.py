"""Reproducibility helper. Call set_seed() once at the start of any
training/eval script so runs are reproducible for the report and demo video
(same seed -> same monster movement rolls, same tie-breaking, same epsilon
exploration draws).
"""

from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int) -> random.Random:
    """Seed Python's random and NumPy's global RNG, and return a dedicated
    random.Random instance for callers (e.g. algorithms.epsilon_greedy,
    env._resolve_monster_moves) that prefer an explicit RNG object over the
    global one -- makes it easy to run multiple independent seeded trials
    side by side without them interfering.

    TODO: implement.
    """
    raise NotImplementedError
