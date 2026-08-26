"""Contract tests for the two-layer Gym API design (see core_env.py and
gym_adapter.py's module docstrings for why there are two layers). These
tests are the concrete evidence for rubric row Part II-H.
"""

import numpy as np
import pytest

from arena.core_env import ArenaCoreEnv
from arena.obs import OBS_DIM


@pytest.mark.parametrize("style", [1, 2])
def test_core_env_reset_returns_bare_observation(style):
    """ArenaCoreEnv.reset() must return the observation alone (not a tuple),
    matching the spec's literal wording: reset() -> observation.
    """
    env = ArenaCoreEnv(control_style=style)
    obs = env.reset()

    assert isinstance(obs, np.ndarray), f"reset() returned {type(obs)}, not a bare ndarray"
    assert not isinstance(obs, tuple)
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    assert np.all(np.isfinite(obs))
    assert np.all(obs >= -1.0) and np.all(obs <= 1.0)


@pytest.mark.parametrize("style", [1, 2])
def test_core_env_step_returns_literal_4_tuple(style):
    """ArenaCoreEnv.step(action) must return exactly
    (observation, reward, done, info) -- a 4-tuple, with `done` a single
    bool. `info` must carry reward_breakdown / died / truncated.
    """
    env = ArenaCoreEnv(control_style=style)
    env.reset()

    result = env.step(0)
    assert isinstance(result, tuple) and len(result) == 4

    obs, reward, done, info = result
    assert isinstance(obs, np.ndarray) and obs.shape == (OBS_DIM,) and obs.dtype == np.float32
    assert np.all(np.isfinite(obs))
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    assert {"reward_breakdown", "died", "truncated"} <= set(info)
    assert isinstance(info["died"], bool) and isinstance(info["truncated"], bool)
    assert done is False  # one step from a fresh reset cannot end the episode


def test_gym_adapter_step_returns_gymnasium_5_tuple():
    """ArenaGymEnv.step(action) must return exactly
    (observation, reward, terminated, truncated, info), with terminated and
    truncated as separate bools that are never both True on the same step.
    """
    from arena.gym_adapter import ArenaGymEnv

    env = ArenaGymEnv(control_style=1)
    env.reset(seed=0)
    for _ in range(50):
        result = env.step(env.action_space.sample())
        assert isinstance(result, tuple) and len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert env.observation_space.contains(obs)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool) and isinstance(truncated, bool)
        assert not (terminated and truncated)
        if terminated or truncated:
            env.reset()


def test_gym_adapter_terminated_on_death_truncated_on_max_steps():
    """Force a death -> terminated=True, truncated=False. Force hitting the
    step limit without dying -> terminated=False, truncated=True.
    """
    from arena.entities import Enemy
    from arena.gym_adapter import ArenaGymEnv

    # death: drop player HP and park an enemy on top of it
    env = ArenaGymEnv(control_style=1)
    env.reset()
    p = env.core_env.state.player
    p.health = 1.0
    env.core_env.state.enemies.append(Enemy(x=p.x, y=p.y, health=30, max_health=30, speed=1.0))
    _, _, terminated, truncated, _ = env.step(0)  # NO_OP
    assert terminated is True and truncated is False

    # truncation: shrink the step limit, never take damage
    env = ArenaGymEnv(control_style=2)
    env.reset()
    env.core_env.max_steps = 4
    terminated = truncated = False
    for _ in range(4):
        _, _, terminated, truncated, _ = env.step(0)
    assert terminated is False and truncated is True


@pytest.mark.parametrize("style", [1, 2])
def test_gymnasium_env_checker_passes(style):
    """gymnasium's env_checker against ArenaGymEnv for both control styles --
    catches spaces/shape/dtype mismatches that would otherwise only surface
    as a cryptic Stable-Baselines3 error mid-training.
    """
    from gymnasium.utils.env_checker import check_env

    from arena.gym_adapter import ArenaGymEnv

    check_env(ArenaGymEnv(control_style=style), skip_render_check=True)
