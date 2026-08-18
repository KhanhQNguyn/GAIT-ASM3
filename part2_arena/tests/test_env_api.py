"""Contract tests for the two-layer Gym API design (see core_env.py and
gym_adapter.py's module docstrings for why there are two layers). These
tests are the concrete evidence for rubric row Part II-H.
"""

import pytest


def test_core_env_reset_returns_bare_observation():
    """ArenaCoreEnv.reset() must return the observation alone (not a tuple),
    matching the spec's literal wording: reset() -> observation.

    TODO: implement once core_env.ArenaCoreEnv.reset is implemented.
    """
    pytest.skip("TODO: implement once ArenaCoreEnv.reset is implemented")


def test_core_env_step_returns_literal_4_tuple():
    """ArenaCoreEnv.step(action) must return exactly
    (observation, reward, done, info) -- a 4-tuple, with `done` a single
    bool, matching the spec's literal wording.

    TODO: implement once ArenaCoreEnv.step is implemented.
    """
    pytest.skip("TODO: implement once ArenaCoreEnv.step is implemented")


def test_gym_adapter_step_returns_gymnasium_5_tuple():
    """ArenaGymEnv.step(action) must return exactly
    (observation, reward, terminated, truncated, info), with terminated and
    truncated as separate bools that are never both True on the same step.

    TODO: implement once ArenaGymEnv.step is implemented.
    """
    pytest.skip("TODO: implement once ArenaGymEnv.step is implemented")


def test_gym_adapter_terminated_on_death_truncated_on_max_steps():
    """Force a death -> terminated=True, truncated=False. Force hitting the
    step limit without dying -> terminated=False, truncated=True.

    TODO: implement (may need to drive the env directly rather than through
    a full episode, once core_env exposes enough state for this to be
    deterministic).
    """
    pytest.skip("TODO: implement once ArenaCoreEnv/ArenaGymEnv are implemented")


def test_gymnasium_env_checker_passes():
    """Run gymnasium.utils.env_checker.check_env against ArenaGymEnv for
    both control styles -- catches spaces/shape/dtype mismatches that would
    otherwise only surface as a cryptic Stable-Baselines3 error mid-training.

    TODO: implement once ArenaGymEnv is implemented.
    """
    pytest.skip("TODO: implement once ArenaGymEnv is implemented")
