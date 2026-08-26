"""Correctness tests for src/intrinsic.py -- Task 5's exact-formula and
per-episode-reset requirements.
"""

import math

import pytest

from src.intrinsic import IntrinsicRewardTracker


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


def test_intrinsic_reward_does_not_mutate_environment_reward(tmp_path):
    """Training with intrinsic reward enabled must not change the logged
    ENVIRONMENT-only return for a fixed sequence of actions/outcomes --
    only the value used for the Q-update should differ. Guards against a
    bug where intrinsic reward gets added directly into env.step()'s
    returned reward instead of being combined only at the update site.

    Uses the fact that training_config.json's epsilon_start is 1.0, so
    episode 0's action sequence is pure RNG (epsilon_greedy always takes
    the "explore" branch, never consulting -- or being biased by -- the
    Q-table). With an identical seed, episode 0's actions must therefore be
    bit-identical whether or not intrinsic reward is enabled, since
    IntrinsicRewardTracker never touches the shared RNG. So episode 0's
    logged (environment-only) return must match exactly between the two
    runs; a bonus leaking into the logged value would break that equality.
    """
    from src.trainer import train

    csv_without = tmp_path / "without_intrinsic.csv"
    csv_with = tmp_path / "with_intrinsic.csv"

    train(
        level_id=0, algorithm="q_learning", seed=0,
        use_intrinsic_reward=False, csv_log_path=csv_without,
    )
    train(
        level_id=0, algorithm="q_learning", seed=0,
        use_intrinsic_reward=True, csv_log_path=csv_with,
    )

    import csv as csv_module

    def first_episode_return(path):
        with open(path, newline="", encoding="utf-8") as f:
            row = next(csv_module.DictReader(f))
            return float(row["return"])

    assert first_episode_return(csv_without) == first_episode_return(csv_with)
