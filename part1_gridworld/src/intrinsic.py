"""Intrinsic reward (Task 5, Level 6) kept in its own module, separate from
env.py, specifically so it stays trivially unit-testable
(tests/test_intrinsic_reward.py) and so it's obvious at a glance that it
never touches or alters the environment's own reward values -- it only adds
a bonus on top, computed independently.

Formula (must match exactly):
    r_i = intrinsic_reward_strength / sqrt(n(s) + 1)
where n(s) is the number of times the CURRENT state has been visited so far
DURING THE CURRENT EPISODE ONLY. n(s) resets to empty at the start of every
episode -- it is not a running total across the whole training run.
"""

from __future__ import annotations

import math
from collections import defaultdict


class IntrinsicRewardTracker:
    """Tracks per-episode state visit counts and computes the intrinsic
    reward bonus for Task 5 / Level 6.

    Usage (see trainer.py):
        tracker = IntrinsicRewardTracker(strength=cfg.intrinsic_reward_strength)
        tracker.reset_episode()          # call at the start of EVERY episode
        for step in episode:
            r_i = tracker.visit_and_get_bonus(state)   # call BEFORE or AFTER
                                                          # env.step, but be
                                                          # consistent about
                                                          # which state (s or
                                                          # s') is counted --
                                                          # document the choice
            total_reward = env_reward + r_i
    """

    def __init__(self, strength: float):
        self.strength = strength
        self._visit_counts: dict = defaultdict(int)

    def reset_episode(self) -> None:
        """Clear the per-episode visit counter. Must be called at the start
        of every episode -- forgetting this silently turns the intrinsic
        reward into a whole-training-run novelty bonus instead of a
        per-episode one, which is a spec violation.

        TODO: implement.
        """
        raise NotImplementedError

    def visit_and_get_bonus(self, state) -> float:
        """Record a visit to `state` for the current episode, then return
        strength / sqrt(n(s) + 1) using the count AFTER this visit is
        recorded (or before -- pick one and be consistent; document the
        choice here once decided, since it changes the very first visit's
        bonus value).

        TODO: implement.
        """
        raise NotImplementedError

    def visit_count(self, state) -> int:
        """Read-only accessor for the current per-episode visit count of
        `state`, useful for tests and debugging.
        """
        return self._visit_counts[state]
