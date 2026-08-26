"""Orchestrates one training run: wires together GridWorldEnv (env.py),
the tabular algorithms (algorithms.py), optional intrinsic reward
(intrinsic.py), optional live rendering (render.py), and episode logging
(logger.py). Nothing else in the codebase should contain a training loop --
keep this the single place that happens.
"""

from __future__ import annotations

import json
import pathlib

# pyrefly: ignore [missing-import]
from algorithms import (
    QTable,
    epsilon_greedy,
    expected_sarsa_update,
    linear_epsilon_decay,
    q_learning_update,
    sarsa_update,
)
# pyrefly: ignore [missing-import]
from env import GridWorldEnv
from intrinsic import IntrinsicRewardTracker
from logger import EpisodeLogger
from seed_utils import set_seed

CONFIG_DIR = pathlib.Path(__file__).resolve().parent.parent / "config"


def _validate_config(cfg: dict) -> None:
    alpha = cfg.get("alpha")
    gamma = cfg.get("gamma")
    eps_start = cfg.get("epsilon_start")
    eps_end = cfg.get("epsilon_end")
    episodes = cfg.get("episodes")
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"training_config 'alpha' must be in (0, 1], got {alpha!r}")
    if not (0.0 < gamma <= 1.0):
        raise ValueError(f"training_config 'gamma' must be in (0, 1], got {gamma!r}")
    if not (0.0 <= eps_end <= eps_start <= 1.0):
        raise ValueError(
            "training_config needs 0 <= epsilon_end <= epsilon_start <= 1, "
            f"got start={eps_start!r} end={eps_end!r}"
        )
    if not (isinstance(episodes, int) and episodes > 0):
        raise ValueError(f"training_config 'episodes' must be an int > 0, got {episodes!r}")


def load_training_config(level_id: int) -> dict:
    """Load config/training_config.json, merge the 'default' block with any
    level_overrides for this level_id, and validate the result.

    TODO: implement (json.load + dict merge). Once the loading code is
    written, add the following validation checks (raise ValueError naming
    the offending key and value so misconfigurations fail loudly):
      - alpha must be in (0, 1]  (learning rate; 0 is a no-op, > 1 diverges)
      - gamma must be in (0, 1]  (discount factor; 0 ignores future rewards)
      - epsilon_end must be <= epsilon_start, with both in [0, 1]
      - episodes must be > 0
    """
    with open(CONFIG_DIR / "training_config.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    cfg = dict(data["default"])
    override = data.get("level_overrides", {}).get(str(level_id))
    if override:
        cfg.update(override)
    _validate_config(cfg)
    return cfg


def make_env(level_id: int) -> GridWorldEnv:
    """Single place that turns a level_id into a GridWorldEnv, so train(),
    evaluate_policy()'s callers (main.py), and the comparison scripts all
    construct the environment identically (path convention:
    CONFIG_DIR / f"level{level_id}.json").

    TODO: implement (return GridWorldEnv(CONFIG_DIR / f"level{level_id}.json")).
    """
    raise NotImplementedError


def train(
    level_id: int,
    algorithm: str,
    seed: int = 0,
    render: bool = False,
    use_intrinsic_reward: bool = False,
    csv_log_path: str | pathlib.Path | None = None,
) -> QTable:
    """Run one full training session and return the learned QTable.

    Build the environment via make_env(level_id) (do not construct
    GridWorldEnv directly here). Callers that want to replay the policy
    later persist the returned QTable themselves via
    algorithms.save_qtable(q, algorithms.qtable_path(level_id, algorithm)).

    algorithm must be one of "q_learning", "sarsa", "expected_sarsa".
    use_intrinsic_reward should only meaningfully be used with level 6 (see
    intrinsic.py) but is accepted generically so compare_algorithms.py /
    plot_results.py can run controlled on/off comparisons on any level.

    Loop shape explanation (illustrative, not executable):
    First, the environment and intrinsic tracker (if enabled) should be
    reset at the start of each episode. Then, calculate epsilon using the
    linear decay function. Next, select the initial action using the
    epsilon-greedy strategy. Finally, step the environment. If the chosen
    algorithm is SARSA or Expected-SARSA, ensure the next action is
    selected BEFORE computing the update, as SARSA requires the next action
    up front; Expected-SARSA does not require the selected action but
    shares the flow structure. Add the intrinsic bonus to the reward used
    for the update ONLY -- never mutate the environment's own reward.
    Call the appropriate *_update function. Log the episode's total
    (environment-only, for comparability) reward to the EpisodeLogger,
    and return the final QTable once all episodes complete. Optionally
    render if requested.

    TODO: implement, dispatching to q_learning_update / sarsa_update /
    expected_sarsa_update based on `algorithm`.
    """
    raise NotImplementedError


def evaluate_policy(env: GridWorldEnv, q_table: QTable, render: bool = True) -> dict:
    """Run one greedy (epsilon=0) episode with a trained QTable and return a
    summary dict (steps, total_return, died). Used both for the video demo
    ("learned policy, not random" evidence) and for verifying convergence.

    TODO: implement.
    """
    raise NotImplementedError
