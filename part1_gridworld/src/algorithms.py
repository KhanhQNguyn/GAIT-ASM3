"""Tabular RL algorithms for Part I: Q-learning (Task 1), SARSA (Task 2),
and Expected SARSA (creativity hook d). Kept independent of env.py and
render.py so these functions are trivially unit-testable in isolation
(tests/test_algorithms.py) with a fake/mock Q-table.
"""

from __future__ import annotations

import random
from collections import defaultdict


class QTable:
    """Maps (state, action) -> float, defaulting unseen entries to 0.0.

    Access pattern (illustrative, not executable):
    The Q-table should be accessed by first retrieving the state, and then
    the action. For instance, accessing the state yields a list of Q-values
    for all actions, which can then be indexed by the action integer.
    The formerly present .values(state) method has been removed -- all
    callers in trainer.py, compare_algorithms.py, and tests must use
    this state-then-action access pattern instead.
    """

    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        self._table: dict = defaultdict(lambda: [0.0] * n_actions)

    def __getitem__(self, state):
        return self._table[state]


def linear_epsilon_decay(
    episode: int, total_episodes: int, epsilon_start: float, epsilon_end: float
) -> float:
    """Linearly interpolate epsilon from epsilon_start (episode 0) to
    epsilon_end (episode total_episodes - 1), per config-driven
    epsilonStart/epsilonEnd. Must be linear, not exponential -- the spec is
    explicit about this.

    TODO: implement the linear interpolation.
    """
    raise NotImplementedError


def epsilon_greedy(q_values: list[float], epsilon: float, rng: random.Random) -> int:
    """Epsilon-greedy action selection with RANDOM TIE-BREAKING among
    actions sharing the best Q-value (a plain argmax silently always picks
    the first-index tie, which the spec explicitly disallows).

    TODO:
      - with probability epsilon, return a uniformly random action index.
      - otherwise, find all indices tied for max(q_values) and pick one of
        them uniformly at random via `rng`.
    """
    raise NotImplementedError


def q_learning_update(
    q_table: QTable,
    state,
    action: int,
    reward: float,
    next_state,
    done: bool,
    alpha: float,
    gamma: float,
) -> None:
    """Off-policy Q-learning update:
        Q(s,a) <- Q(s,a) + alpha * (reward + gamma * max_a' Q(s',a') - Q(s,a))
    where the bootstrap target uses the MAX over next-state actions,
    regardless of which action the current policy would actually take next.
    If done, the bootstrap term is 0 (no next state to continue into).

    TODO: implement, mutating q_table in place.
    """
    raise NotImplementedError


def sarsa_update(
    q_table: QTable,
    state,
    action: int,
    reward: float,
    next_state,
    next_action: int,
    done: bool,
    alpha: float,
    gamma: float,
) -> None:
    """On-policy SARSA update:
        Q(s,a) <- Q(s,a) + alpha * (reward + gamma * Q(s',a') - Q(s,a))
    where a' is the action the current epsilon-greedy policy ACTUALLY
    selected for next_state (passed in as next_action), NOT the max. This
    is the key difference from q_learning_update -- do not accidentally
    reimplement Q-learning here.

    TODO: implement, mutating q_table in place.
    """
    raise NotImplementedError


def expected_sarsa_update(
    q_table: QTable,
    state,
    action: int,
    reward: float,
    next_state,
    done: bool,
    alpha: float,
    gamma: float,
    epsilon: float,
) -> None:
    """Creativity hook (d): Expected SARSA update:
        Q(s,a) <- Q(s,a) + alpha * (reward + gamma * E_{a'~pi}[Q(s',a')] - Q(s,a))
    where the expectation is taken over the CURRENT epsilon-greedy policy's
    action distribution at next_state (not the sampled next action like
    SARSA, and not the max like Q-learning). With epsilon-greedy:
      E[Q(s',a')] = (epsilon / n_actions) * sum_a' Q(s',a')
                    + (1 - epsilon) * max_a' Q(s',a')
    (accounting correctly for the greedy action also receiving its share of
    the epsilon/n_actions exploration mass -- don't double count or omit it).

    TODO: implement, mutating q_table in place.
    """
    raise NotImplementedError
