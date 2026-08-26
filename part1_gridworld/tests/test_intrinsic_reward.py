"""Correctness tests for src/intrinsic.py -- Task 5's exact-formula and
per-episode-reset requirements.
"""

import math

import pytest
from intrinsic import IntrinsicRewardTracker


def test_intrinsic_reward_formula_matches_spec():
    tracker = IntrinsicRewardTracker(strength=2.0)
    # first visit: pre-visit count n=0 -> 2.0/sqrt(0+1) = 2.0
    assert tracker.visit_and_get_bonus("A") == pytest.approx(2.0)
    # second visit: n=1 -> 2.0/sqrt(2)
    assert tracker.visit_and_get_bonus("A") == pytest.approx(2.0 / math.sqrt(2))
    # fresh state's first visit: n=0 -> 2.0 again
    assert tracker.visit_and_get_bonus("B") == pytest.approx(2.0)


def test_visit_counter_resets_between_episodes():
    tracker = IntrinsicRewardTracker(strength=1.0)
    tracker.visit_and_get_bonus("A")
    tracker.visit_and_get_bonus("A")
    assert tracker.visit_count("A") == 2
    tracker.reset_episode()
    assert tracker.visit_count("A") == 0
    assert tracker.visit_and_get_bonus("A") == pytest.approx(1.0)  # n=0 -> 1/sqrt(1)


def test_intrinsic_reward_does_not_mutate_environment_reward():
    """Training with intrinsic reward enabled must not change the logged
    ENVIRONMENT-only return for a fixed sequence of actions/outcomes --
    only the value used for the Q-update should differ. Guards against a
    bug where intrinsic reward gets added directly into env.step()'s
    returned reward instead of being combined only at the update site.

    TODO: implement once trainer.train supports use_intrinsic_reward.
    """
    pytest.skip("TODO: implement once trainer.train is implemented")
