"""Stable-Baselines3 training callbacks for the arena.

Currently hosts one callback: RewardTermLoggingCallback, which is what makes
creativity hook (b) -- the reward-decomposition dashboard
(scripts/plot_reward_decomposition.py) -- possible. SB3 does NOT log the
contents of the per-step `info` dict to TensorBoard automatically, so
without this callback the `reward_terms/*` scalars that
plot_reward_decomposition.py reads back simply do not exist (see
docs/AUDIT_main.md 5.7).

Wire it in from scripts/train.py:
    from callbacks import RewardTermLoggingCallback
    model.learn(total_timesteps=..., callback=RewardTermLoggingCallback())
"""

from __future__ import annotations

# pyrefly: ignore [missing-import]
from stable_baselines3.common.callbacks import BaseCallback

# TensorBoard scalar tags this callback must emit, one per field of
# arena.rewards.RewardBreakdown. Keep this list in sync with
# scripts/plot_reward_decomposition.py::REWARD_TERM_TAGS and with
# RewardBreakdown's fields -- a mismatch silently drops a term from the
# decomposition chart.
REWARD_TERM_TAGS = [
    "reward_terms/kill_enemy",
    "reward_terms/kill_spawner",
    "reward_terms/phase_progress",
    "reward_terms/damage_taken",
    "reward_terms/death",
    "reward_terms/approach_nearest_enemy",
    "reward_terms/shoot_while_no_target",
]


class RewardTermLoggingCallback(BaseCallback):
    """Reads `info["reward_breakdown"]` (an arena.rewards.RewardBreakdown,
    put there by ArenaCoreEnv.step()) on every environment step and writes
    each term to TensorBoard under its `reward_terms/<name>` tag, plus a
    running per-episode sum per term.

    Design notes for the implementer (do NOT implement in this pass):
      - `self.locals["infos"]` holds one info dict per vectorised env on
        each `_on_step` call; accumulate each RewardBreakdown field into a
        per-env running total, and on episode end (`self.locals["dones"]`)
        log the episode totals via `self.logger.record(tag, value)` then
        reset that env's accumulators.
      - Also log the instantaneous per-step term values (mean across envs)
        so the stacked-area chart has fine-grained data, not just one point
        per episode.
      - `self.logger.record` writes to whatever TensorBoard run
        `model.learn(tb_log_name=...)` opened; no extra file handling here.
      - Return True always (returning False aborts training).
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        # TODO: initialise per-env, per-term running accumulators once the
        # number of envs is known (self.training_env.num_envs in _init_callback).

    def _on_step(self) -> bool:
        """Called by SB3 after every `env.step()` batch.

        TODO: implement per the design notes in the class docstring.
        """
        raise NotImplementedError
