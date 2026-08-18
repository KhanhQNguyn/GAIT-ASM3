"""Correctness tests for the tabular update rules in src/algorithms.py.
These are the tests referenced by rubric rows Part I-B and Part I-C -- they
are the concrete evidence that the update rules are implemented correctly,
not just "seems to learn something."
"""

import pytest


def test_q_learning_update_uses_max_over_next_actions():
    """Q-learning's target must use max_a' Q(s',a'), regardless of which
    action would actually be taken next. Construct a QTable where the
    greedy next action differs from an arbitrarily chosen 'actual' next
    action, run q_learning_update, and assert the update used the MAX
    value, not the arbitrary one.

    TODO: implement once algorithms.q_learning_update exists.
    """
    pytest.skip("TODO: implement after algorithms.q_learning_update is implemented")


def test_sarsa_update_uses_actual_next_action_not_max():
    """SARSA's target must use Q(s', a') for the SPECIFIC next_action
    passed in, even when a different action has a higher Q-value at s'.
    Construct a case where next_action is deliberately NOT the argmax and
    assert the update reflects next_action's value, not the max.

    TODO: implement once algorithms.sarsa_update exists.
    """
    pytest.skip("TODO: implement after algorithms.sarsa_update is implemented")


def test_expected_sarsa_update_uses_policy_expectation():
    """Expected SARSA's target must equal the epsilon-greedy expectation
    over ALL next-state actions, not a sampled action and not a bare max.
    Verify against a hand-computed expectation for a small fixed Q(s', *).

    TODO: implement once algorithms.expected_sarsa_update exists.
    """
    pytest.skip("TODO: implement after algorithms.expected_sarsa_update is implemented")


def test_linear_epsilon_decay_endpoints_and_linearity():
    """epsilon at episode 0 == epsilon_start, epsilon at the final episode
    == epsilon_end, and the decay is linear (constant step size) in
    between -- not exponential.

    TODO: implement once algorithms.linear_epsilon_decay is implemented.
    """
    pytest.skip("TODO: implement after algorithms.linear_epsilon_decay is implemented")


def test_epsilon_greedy_random_tie_breaking_distribution():
    """When multiple actions share the best Q-value, epsilon_greedy (with
    epsilon=0, to isolate the tie-break path) must select among them
    roughly uniformly over many trials, not always the lowest index.

    TODO: implement once algorithms.epsilon_greedy is implemented -- run
    many trials with a seeded RNG and a chi-square-style sanity check on
    the distribution of selected tied actions.
    """
    pytest.skip("TODO: implement after algorithms.epsilon_greedy is implemented")
