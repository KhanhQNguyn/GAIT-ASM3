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

from collections import defaultdict

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
REWARD_TERM_FIELDS = [tag.split("/", 1)[-1] for tag in REWARD_TERM_TAGS]


class RewardTermLoggingCallback(BaseCallback):
    """Reads `info["reward_breakdown"]` (an arena.rewards.RewardBreakdown,
    put there by ArenaCoreEnv.step()) on every environment step and writes
    each term to TensorBoard under its `reward_terms/<name>` tag, plus a
    running per-episode sum per term under `reward_terms_episode/<name>`.

    - `self.locals["infos"]` holds one info dict per vectorised env on each
      `_on_step` call; each RewardBreakdown field is accumulated into a
      per-env running total, and on episode end (`self.locals["dones"]`)
      the episode totals are logged then that env's accumulators reset.
    - The instantaneous per-step term values are also logged (as a running
      mean across the current logging interval, via `record_mean`, SB3's
      own convention for `rollout/ep_rew_mean`) so the stacked-area chart
      has fine-grained data, not just one point per episode.
    - Writes to whatever TensorBoard run `model.learn(tb_log_name=...)`
      opened; no extra file handling here.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._episode_totals: list[dict[str, float]] = []

    def _init_callback(self) -> None:
        num_envs = self.training_env.num_envs
        self._episode_totals = [defaultdict(float) for _ in range(num_envs)]

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", ())
        dones = self.locals.get("dones", ())
        for env_idx, info in enumerate(infos):
            breakdown = info.get("reward_breakdown")
            if breakdown is None:
                continue
            for field_name in REWARD_TERM_FIELDS:
                value = getattr(breakdown, field_name, 0.0)
                self.logger.record_mean(f"reward_terms/{field_name}", value)
                self._episode_totals[env_idx][field_name] += value

            if env_idx < len(dones) and dones[env_idx]:
                for field_name in REWARD_TERM_FIELDS:
                    total = self._episode_totals[env_idx][field_name]
                    self.logger.record_mean(f"reward_terms_episode/{field_name}", total)
                self._episode_totals[env_idx] = defaultdict(float)
        return True
