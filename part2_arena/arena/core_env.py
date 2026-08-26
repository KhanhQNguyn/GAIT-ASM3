"""The core arena environment, implementing the Gym-style API EXACTLY as
worded in the assignment spec:

    reset() -> observation
    step(action) -> (observation, reward, done, info)
    render() -> displays the scene

This is a deliberate, literal 4-tuple `step()` -- the legacy Gym contract,
matching the spec's own words. This class has ZERO dependency on gymnasium
or stable-baselines3, and is what satisfies the rubric's "Gym-style API"
row on its own terms, independent of any training-library requirement.

Stable-Baselines3 needs Gymnasium's 5-tuple contract (terminated/truncated
split) instead of a single `done` flag. Rather than compromise this class's
literal spec-compliance to satisfy that, see gym_adapter.py -- a separate,
thin wrapper that adapts this class for SB3 training/eval only. Do not
merge the two; each one's job is to satisfy a different requirement
cleanly.
"""

from __future__ import annotations

import pathlib

from arena.actions import action_enum_for_style
from arena.entities import ArenaState, Player
from arena.obs import build_observation
from arena.phases import PhaseManager
from arena.rewards import compute_reward

CONFIG_DIR = pathlib.Path(__file__).resolve().parent.parent / "config"

# Fallback constants. The authoritative values live in config/arena.json
# (CONFIG_DIR / "arena.json"); __init__ should load that file and fall back
# to these only if it is missing. DEFAULT_MAX_STEPS was cut from 3000 to
# 1200 so a ~300k-timestep training run sees a usable number of episode
# terminations (see docs/AUDIT_main.md 5.4).
ARENA_WIDTH = 960
ARENA_HEIGHT = 680
DEFAULT_MAX_STEPS = 1200


class ArenaCoreEnv:
    """Headless-capable arena environment for one control style.

    control_style: 1 (rotation+thrust) or 2 (direct directional) -- see
    arena/actions.py. Determines the action enum used to interpret
    `action` in step().
    """

    def __init__(self, control_style: int, curriculum_enabled: bool = False):
        self.control_style = control_style
        self.action_enum = action_enum_for_style(control_style)
        self.phase_manager = PhaseManager(curriculum_enabled=curriculum_enabled)
        self.state: ArenaState | None = None
        # TODO: load config/arena.json (CONFIG_DIR / "arena.json") and set
        # self.max_steps, self.arena_width/height, player/enemy/spawner
        # params, and the observation.num_active_enemies_max from it; fall
        # back to the module constants above if the file is absent. Also
        # seed an RNG here (scripts/seed_utils.set_seed handles the global
        # seeding; this is the env-local generator).

    def reset(self):
        """Reset to a fresh episode: new Player at a default position,
        spawners for phase 0 (via self.phase_manager.difficulty_for_phase),
        no enemies/projectiles yet, step_count=0. Return the observation
        vector via arena.obs.build_observation.

        TODO: implement.
        """
        raise NotImplementedError

    def step(self, action: int):
        """Apply one control action, advance physics/AI/collisions one
        tick, resolve the phase system, compute the reward via
        arena.rewards.compute_reward, and return the literal 4-tuple
        (observation, reward, done, info).

        `info` should include at minimum the RewardBreakdown (see
        rewards.py) under info["reward_breakdown"], so scripts/train.py can
        log per-term TensorBoard scalars for the creativity(b) reward
        decomposition dashboard, and info["died"] / info["truncated"] so
        gym_adapter.py can split `done` into terminated/truncated without
        re-deriving it.

        Order of operations (documented here since core_env.py is the
        single place this logic should exist):
          1. Interpret `action` via self.action_enum; apply to the player
             (movement/rotation/shoot) via arena.physics helpers.
          2. Advance enemy AI (move toward player) and spawner timers
             (spawn new enemies on interval).
          3. Advance projectiles; resolve collisions (projectile-enemy,
             projectile-spawner, projectile-player, enemy-player contact).
          4. Apply damage/deaths; collect step_events for compute_reward.
          5. self.phase_manager.maybe_advance_phase(...) if all active
             spawners are destroyed; spawn the next phase's spawners.
          6. done = player died OR step_count >= max_steps.
          7. Build observation, compute reward, return the 4-tuple.

        TODO: implement.
        """
        raise NotImplementedError

    def render(self, renderer=None) -> None:
        """Display the current scene. Evaluation scripts pass in an
        arena.render_pygame.ArenaRenderer instance; training does not call
        this at all (render only during evaluation per the feasibility
        guide).

        TODO: implement (delegate to renderer.draw(self.state)).
        """
        raise NotImplementedError
