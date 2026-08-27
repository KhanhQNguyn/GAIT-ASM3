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
            r_i = tracker.visit_and_get_bonus(state)   # call with the state
                                                          # BEFORE env.step (i.e.
                                                          # current state s, not
                                                          # next state s'); the
                                                          # visit is recorded first,
                                                          # then the bonus uses the
                                                          # POST-visit count so the
                                                          # very first visit yields
                                                          # strength / sqrt(1).
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

        """
        self._visit_counts.clear()

    def visit_and_get_bonus(self, state) -> float:
        """Record a visit to `state` for the current episode, then return
        the intrinsic bonus using the count AFTER this visit is recorded.

        Implementation instruction: increment n(s) first, then return
            strength / sqrt(n(s))
        where n(s) is the post-increment count. The first visit yields
        strength / sqrt(1) = strength, the second yields strength / sqrt(2),
        and so on.

        Note on the module-level formula: the module docstring writes the
        same formula as  r_i = strength / sqrt(n(s) + 1)  using the
        PRE-visit count. These are two equivalent ways to write the same
        computation — post-increment n(s) equals (pre-visit n(s) + 1) on
        every visit, not just the first. There is no difference in the
        numeric result; choose whichever form is easier to read when
        implementing, as long as both are consistent with the instruction
        above.

        TODO: implement.
        """
        n = self._visit_counts[state]  # pre-visit count (0 on first visit)
        bonus = self.strength / math.sqrt(n + 1)
        self._visit_counts[state] = n + 1
        return bonus

    def visit_count(self, state) -> int:
        """Read-only accessor for the current per-episode visit count of
        `state`, useful for tests and debugging.
        """
        return self._visit_counts[state]
