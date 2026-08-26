"""Thin gymnasium.Env adapter around ArenaCoreEnv, used ONLY for
Stable-Baselines3 training/evaluation. core_env.py stays the literal,
spec-compliant 4-tuple implementation; this module exists purely because
SB3 requires Gymnasium's 5-tuple step contract (terminated/truncated split
instead of a single done flag), which is a training-library requirement,
not part of the assignment's own API wording.

Do not put any game logic here -- this file's only job is protocol
translation. Example (illustrative, not executable):
    core_env_step_returns_obs_reward_done_info
    gym_adapter_step_returns_obs_reward_terminated_truncated_info
with terminated = the player died, and truncated = the step/time limit was
hit, derived from info["died"] / info["truncated"] that core_env.step()
already provides (see core_env.py's step() docstring).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from arena.actions import action_enum_for_style
from arena.core_env import ArenaCoreEnv
from arena.obs import OBS_DIM


class ArenaGymEnv(gym.Env):
    """gymnasium.Env wrapper for one control style, suitable for
    stable_baselines3.PPO / DQN.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        control_style: int,
        curriculum_enabled: bool = False,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.core_env = ArenaCoreEnv(
            control_style=control_style, curriculum_enabled=curriculum_enabled
        )
        self.render_mode = render_mode
        self._renderer = None  # created lazily in render() if render_mode == "human"

        action_enum = action_enum_for_style(control_style)
        self.action_space = spaces.Discrete(len(action_enum))
        # Observation bounds are [-1, 1] for ALL features -- this is the
        # decided, final normalization convention (not a placeholder).
        # See arena/obs.py's OBSERVATION_SPEC for per-feature descriptions
        # and the rescaling formula. Every feature is either naturally in
        # [-1, 1] (sin/cos) or linearly rescaled from [0, 1] to [-1, 1]
        # via x_normalized = 2 * x_unit - 1.
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Gymnasium reset contract: return (observation, info)."""
        super().reset(seed=seed)  # seeds self.np_random per the gymnasium convention
        obs = self.core_env.reset(seed=seed)
        return obs, {}

    def step(self, action: int):
        """Gymnasium step contract: return
        (observation, reward, terminated, truncated, info).

        core_env returns the legacy (obs, reward, done, info); we split
        `done` using the flags core_env already put in `info`:
            terminated = the player died
            truncated  = the step limit was hit without dying
        `and not terminated` guarantees the two are never both True.
        """
        obs, reward, _done, info = self.core_env.step(int(action))
        terminated = bool(info["died"])
        truncated = bool(info["truncated"]) and not terminated
        return obs, float(reward), terminated, truncated, info

    def render(self):
        """Lazily create an ArenaRenderer on first call (only when
        render_mode == "human"), then delegate to core_env.render().
        """
        if self.render_mode != "human":
            return None
        if self._renderer is None:
            from arena.render_pygame import ArenaRenderer

            self._renderer = ArenaRenderer(
                int(self.core_env.arena_width),
                int(self.core_env.arena_height),
                caption=f"Arena - control style {self.core_env.control_style}",
            )
        self.core_env.render(self._renderer)
        return None

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
