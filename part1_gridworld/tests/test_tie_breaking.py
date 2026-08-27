"""Focused statistical test for epsilon-greedy's random tie-breaking rule
(Task 1 rubric requirement), separated from test_algorithms.py so it's easy
to point a marker at directly as evidence for that specific rubric line.
"""

import random

from src.algorithms import epsilon_greedy


def test_tied_best_actions_each_selected_with_nonzero_frequency():
    """Given a Q-value row with, say, 2 of 4 actions tied for the max and
    epsilon=0, run epsilon_greedy many times with different RNG states and
    assert BOTH tied actions appear in the results (not just one of them
    every time), and that the non-tied actions never appear.

    TODO: implement once algorithms.epsilon_greedy is implemented.
    """
    rng = random.Random(7)
    q_values = [3.0, 3.0, 0.0, 0.0]
    seen = set()
    for _ in range(500):
        seen.add(epsilon_greedy(q_values, epsilon=0.0, rng=rng))
    assert seen == {0, 1}
