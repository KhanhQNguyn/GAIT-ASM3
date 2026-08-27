"""Tabular RL algorithms for Part I: Q-learning (Task 1), SARSA (Task 2),
and Expected SARSA (creativity hook d). Kept independent of env.py and
render.py so these functions are trivially unit-testable in isolation
(tests/test_algorithms.py) with a fake/mock Q-table.
"""

from __future__ import annotations

import json
import pathlib
import random
from collections import defaultdict

# Trained Q-tables are persisted here so a learned policy can be replayed for
# the video demo without retraining live (see docs/AUDIT_main.md 5.3).
MODELS_DIR = pathlib.Path(__file__).resolve().parent.parent / "models"


class QTable:
    """Maps (state, action) -> float, defaulting unseen entries to 0.0.

    Access pattern (illustrative, not executable):
    The Q-table should be accessed by first retrieving the state, and then
    the action -- i.e. `q_table[state][action]`. Accessing `q_table[state]`
    yields a list of Q-values for all actions, which is then indexed by the
    action integer. `q_table[state][action]` is the ONLY supported access
    pattern. The formerly present .values(state) method has been removed --
    all callers in trainer.py, compare_algorithms.py, and tests must use
    this state-then-action access pattern instead.
    """

    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        self._table: dict = defaultdict(lambda: [0.0] * n_actions)

    def __getitem__(self, state):
        return self._table[state]


def qtable_path(level_id: int, algorithm: str) -> pathlib.Path:
    """Canonical on-disk location for a trained Q-table:
    MODELS_DIR / f"level{level_id}_{algorithm}.json". One fixed convention so
    trainer.train() (writer), main.py's watch-only path (reader), and any
    eval/comparison script all agree without passing paths around.
    """
    return MODELS_DIR / f"level{level_id}_{algorithm}.json"


def _state_to_jsonable(state) -> list:
    """Encode a GridWorldEnv state tuple
    (agent_x, agent_y, apples_bitmask, has_key, chest_open, monsters_tuple)
    into a JSON-safe list. monsters_tuple (a tuple of (x, y) pairs) becomes
    a list of 2-element lists; everything else is already JSON-safe.
    """
    ax, ay, bitmask, has_key, chest_open, monsters = state
    return [ax, ay, bitmask, has_key, chest_open, [list(m) for m in monsters]]


def _state_from_jsonable(state_repr: list) -> tuple:
    """Inverse of _state_to_jsonable -- rebuilds the exact tuple shape
    GridWorldEnv produces (see its class docstring) so the loaded table
    indexes identically to a live environment's states.
    """
    ax, ay, bitmask, has_key, chest_open, monsters = state_repr
    return (ax, ay, bitmask, bool(has_key), bool(chest_open), tuple(tuple(m) for m in monsters))


def save_qtable(q_table: "QTable", path: str | pathlib.Path) -> None:
    """Serialise a trained QTable to `path` as JSON.

    Format: {"n_actions": int, "entries": [[state_repr, [q0, q1, ...]], ...]}
    where state_repr is state's JSON-safe encoding. Only entries that were
    actually visited (i.e. present in the underlying defaultdict) are
    written -- unvisited states still default to [0.0]*n_actions on load.
    """
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = [
        [_state_to_jsonable(state), list(values)] for state, values in q_table._table.items()
    ]
    data = {"n_actions": q_table.n_actions, "entries": entries}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_qtable(path: str | pathlib.Path, n_actions: int) -> "QTable":
    """Inverse of save_qtable: rebuild a QTable from the JSON at `path`.

    Raises ValueError if the file's n_actions disagrees with the argument
    (e.g. env's action space changed since the table was trained).
    """
    path = pathlib.Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    file_n_actions = data["n_actions"]
    if file_n_actions != n_actions:
        raise ValueError(
            f"{path}: saved Q-table has n_actions={file_n_actions}, "
            f"but n_actions={n_actions} was requested"
        )

    q_table = QTable(n_actions)
    for state_repr, values in data["entries"]:
        q_table._table[_state_from_jsonable(state_repr)] = list(values)
    return q_table


def linear_epsilon_decay(
    episode: int, total_episodes: int, epsilon_start: float, epsilon_end: float
) -> float:
    """Linearly interpolate epsilon from epsilon_start (episode 0) to
    epsilon_end (episode total_episodes - 1), per config-driven
    epsilonStart/epsilonEnd. Must be linear, not exponential -- the spec is
    explicit about this.

    TODO: implement the linear interpolation.
    """
    if total_episodes <= 1:
        return epsilon_start
    frac = episode / (total_episodes - 1)
    return epsilon_start + (epsilon_end - epsilon_start) * frac


def epsilon_greedy(q_values: list[float], epsilon: float, rng: random.Random) -> int:
    """Epsilon-greedy action selection with RANDOM TIE-BREAKING among
    actions sharing the best Q-value (a plain argmax silently always picks
    the first-index tie, which the spec explicitly disallows).

    TODO:
      - with probability epsilon, return a uniformly random action index.
      - otherwise, find all indices tied for max(q_values) and pick one of
            them uniformly at random via `rng`.
    """
    n_actions = len(q_values)
    if rng.random() < epsilon:
        return rng.randrange(n_actions)
    max_q = max(q_values)
    tied = [i for i, q in enumerate(q_values) if q == max_q]
    return rng.choice(tied)


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
    q = q_table[state]
    if done:
        target = reward
    else:
        target = reward + gamma * max(q_table[next_state])
    q[action] += alpha * (target - q[action])


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
    q = q_table[state]
    if done:
        target = reward
    else:
        target = reward + gamma * q_table[next_state][next_action]
    q[action] += alpha * (target - q[action])


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
    q = q_table[state]
    if done:
        target = reward
    else:
        n_actions = len(q_table[next_state])
        q_next = q_table[next_state]
        expectation = (epsilon / n_actions) * sum(q_next) + (1 - epsilon) * max(q_next)
        target = reward + gamma * expectation
    q[action] += alpha * (target - q[action])
