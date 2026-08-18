"""Correctness tests for src/intrinsic.py -- Task 5's exact-formula and
per-episode-reset requirements.
"""

import pytest


def test_intrinsic_reward_formula_matches_spec():
    """r_i must equal strength / sqrt(n(s) + 1) for the visit count AT the
    time of the call. Check against a hand-computed value for a few visit
    counts (n=0, 1, 5).

    TODO: implement once intrinsic.IntrinsicRewardTracker is implemented.
    """
    pytest.skip("TODO: implement once IntrinsicRewardTracker is implemented")


def test_visit_counter_resets_between_episodes():
    """After reset_episode(), a previously heavily-visited state's count
    must go back to 0 (i.e. its next bonus equals the n=0 value again) --
    this is the single most important behavior since it's the difference
    between 'per-episode' and 'whole-training-run' novelty, which the spec
    requires to be per-episode.

    TODO: implement.
    """
    pytest.skip("TODO: implement once IntrinsicRewardTracker is implemented")


def test_intrinsic_reward_does_not_mutate_environment_reward():
    """Training with intrinsic reward enabled must not change the logged
    ENVIRONMENT-only return for a fixed sequence of actions/outcomes --
    only the value used for the Q-update should differ. Guards against a
    bug where intrinsic reward gets added directly into env.step()'s
    returned reward instead of being combined only at the update site.

    TODO: implement once trainer.train supports use_intrinsic_reward.
    """
    pytest.skip("TODO: implement once trainer.train is implemented")
