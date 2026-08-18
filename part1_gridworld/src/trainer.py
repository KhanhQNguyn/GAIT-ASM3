"""Orchestrates one training run: wires together GridWorldEnv (env.py),
the tabular algorithms (algorithms.py), optional intrinsic reward
(intrinsic.py), optional live rendering (render.py), and episode logging
(logger.py). Nothing else in the codebase should contain a training loop --
keep this the single place that happens.
"""

from __future__ import annotations

import pathlib

from algorithms import (
    QTable,
    epsilon_greedy,
    expected_sarsa_update,
    linear_epsilon_decay,
    q_learning_update,
    sarsa_update,
)
from env import GridWorldEnv
from intrinsic import IntrinsicRewardTracker
from logger import EpisodeLogger
from seed_utils import set_seed

CONFIG_DIR = pathlib.Path(__file__).resolve().parent.parent / "config"


def load_training_config(level_id: int) -> dict:
    """Load config/training_config.json, merge the 'default' block with any
    level_overrides for this level_id.

    TODO: implement (json.load + dict merge).
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

    algorithm must be one of "q_learning", "sarsa", "expected_sarsa".
    use_intrinsic_reward should only meaningfully be used with level 6 (see
    intrinsic.py) but is accepted generically so compare_algorithms.py /
    plot_results.py can run controlled on/off comparisons on any level.

    Loop shape (per episode):
      1. env.reset(), intrinsic_tracker.reset_episode() if enabled.
      2. epsilon = linear_epsilon_decay(episode, total_episodes, ...).
      3. Choose the first action via epsilon_greedy.
      4. Step the environment; if using SARSA/Expected-SARSA, choose the
         next action BEFORE computing the update (SARSA needs a' up front;
         Expected SARSA needs epsilon, not a sampled a').
      5. Add the intrinsic bonus to the reward used for the update ONLY --
         never mutate the environment's own reward.
      6. Call the appropriate *_update function.
      7. Log the episode's total (environment-only, for comparability)
         return via EpisodeLogger.
      8. Optionally render.

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
