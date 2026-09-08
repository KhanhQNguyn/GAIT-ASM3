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

Enemies do NOT fire projectiles in this arena -- they deal contact damage
and PERSIST after touching the player (per-enemy damage cooldown, see
_resolve_collisions). Only the player shoots; shooting is the only way to
remove enemies, which is what makes combat the survival strategy. That
keeps the observation vector at the spec minimum (no "incoming projectile"
feature) and the mechanics simple. Projectile-vs-enemy and
projectile-vs-spawner collisions still satisfy the rubric's "projectile
collisions" requirement.
"""

from __future__ import annotations

import json
import math
import pathlib
import random

from arena.actions import ControlStyle1, ControlStyle2, action_enum_for_style
from arena.entities import ArenaState, Enemy, Player, Projectile, Spawner
from arena.obs import build_observation
from arena.phases import PhaseManager
from arena.physics import (
    circle_collision,
    distance,
    integrate_position,
    relative_direction,
    wrap_or_clamp_to_bounds,
)
from arena.rewards import compute_reward
from arena.rewards_config import SHOT_NO_TARGET_RADIUS

CONFIG_DIR = pathlib.Path(__file__).resolve().parent.parent / "config"
_CONFIG_PATH = CONFIG_DIR / "arena.json"

# Fallback constants. The authoritative values live in config/arena.json
# (CONFIG_DIR / "arena.json"); __init__ loads that file and falls back to
# these only if it is missing/partial. DEFAULT_MAX_STEPS was cut from 3000
# to 1200 so a ~300k-timestep training run sees a usable number of episode
# terminations (see docs/AUDIT_main.md 5.4). ARENA_WIDTH / ARENA_HEIGHT are
# also imported by gym_adapter.py -- keep them defined here.
ARENA_WIDTH = 960
ARENA_HEIGHT = 680
DEFAULT_MAX_STEPS = 1200

_DEFAULTS = {
    "arena": {"width": ARENA_WIDTH, "height": ARENA_HEIGHT, "max_steps": DEFAULT_MAX_STEPS},
    "player": {
        "max_health": 100.0,
        "max_speed": 6.0,
        "thrust_accel": 0.5,
        "friction": 0.97,
        "rotate_speed_rad": 0.20,
        "radius": 14.0,
        "shoot_cooldown_steps": 8,
        "projectile_speed": 12.0,
        "projectile_damage": 25.0,
    },
    "enemy": {
        "radius": 12.0,
        "contact_damage": 20.0,
        "contact_damage_cooldown_steps": 45,
        "base_health": 25.0,
        "max_concurrent_enemies": 18,
    },
    "spawner": {"radius": 18.0, "base_health": 120.0},
}

# 2026-09-08: doubled from 4.0 to 8.0 (visual radius kept in sync in
# render_pygame.py). Style 2's aim is coupled to its movement (projectiles
# fire along the velocity direction, which is cardinal-only), so the hit
# cone is what makes jousting -- charging an enemy head-on along an axis
# while firing -- learnable at all: a 4px bullet gives only a ~6 deg cone
# at 300px range, too tight for PPO to discover. 8px roughly doubles it.
# Style 1 rotates freely and loses nothing.
_PROJECTILE_RADIUS = 8.0
_SPAWNER_MARGIN = 90.0


def _load_config() -> dict:
    """arena.json merged over _DEFAULTS (block by block), so a partial file
    still yields every key.
    """
    merged = {k: dict(v) for k, v in _DEFAULTS.items()}
    try:
        raw = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raw = {}
    for block, vals in merged.items():
        vals.update(raw.get(block, {}) or {})
    return merged


class ArenaCoreEnv:
    """Headless-capable arena environment for one control style.

    control_style: 1 (rotation+thrust) or 2 (direct directional) -- see
    arena/actions.py. Determines the action enum used to interpret
    `action` in step().

    The base environment is DETERMINISTIC given the action sequence:
    spawner timers are interval-based, enemy motion is straight-line toward
    the player, and spawner placement is fixed per phase. `reset()` accepts
    an ignored `seed` for gym-API symmetry.
    """

    def __init__(
        self,
        control_style: int,
        curriculum_enabled: bool = False,
        reward_overrides: dict[str, float] | None = None,
    ):
        self.control_style = int(control_style)
        self.action_enum = action_enum_for_style(self.control_style)
        self.phase_manager = PhaseManager(curriculum_enabled=curriculum_enabled)

        cfg = _load_config()
        self._arena_cfg = cfg["arena"]
        self._pcfg = cfg["player"]
        self._ecfg = cfg["enemy"]
        self._scfg = cfg["spawner"]
        self.arena_width = float(self._arena_cfg["width"])
        self.arena_height = float(self._arena_cfg["height"])
        self.max_steps = int(self._arena_cfg["max_steps"])

        self.state: ArenaState | None = None
        self._current_enemy_speed = 1.6
        self._shot_no_target_flag = False
        self._shot_toward_enemy_flag = 0.0
        # transient per-step data exposed for rendering / debugging
        self._last_obs = None
        self._last_step_events: dict = self._empty_step_events()
        self._render_events: list = []
        # running per-episode total of RewardBreakdown.approach_nearest_enemy,
        # fed back into step_events so compute_reward() can enforce its
        # per-episode cap (see arena/rewards.py APPROACH_REWARD_EPISODE_CAP).
        self._cumulative_approach_reward: float = 0.0
        # running per-episode total of (kill_spawner + phase_progress), fed
        # back into step_events for PROGRESS_REWARD_EPISODE_CAP. That cap is
        # DISABLED (+inf) as of 2026-09-08e -- it clamped spec-required
        # reward terms (see rewards.py) -- so this is inert plumbing kept
        # (like _cumulative_approach_reward) for a one-line re-enable.
        self._cumulative_progress_reward: float = 0.0
        # total player shots actually fired this episode (cooldown was clear).
        # scripts/eval_aim_stats.py diffs this instead of len(projectiles),
        # which silently drops shots that collide the same step they fire
        # (2026-09-08d, BUG_LESS_ATTACK.md fix D).
        self._shots_fired: int = 0
        # Optional {constant_name: value} replacements handed to
        # rewards.compute_reward on every step -- training-time only, used
        # by scripts/train.py --death-penalty for the R_DEATH ablation
        # (docs/KHANG.md C.3). None for every normal run.
        self._reward_overrides = reward_overrides

    # ------------------------------------------------------------------- API
    def reset(self, *, seed: int | None = None):
        """Reset to a fresh episode and return the observation ALONE (not a
        tuple -- matches the spec's literal `reset() -> observation`).

        Layout: the player starts at the arena centre (symmetric, room to
        manoeuvre in every direction, neither control style advantaged);
        phase-0 spawners are placed at RANDOM points inside the spawn margins
        (rejection-sampled to keep a minimum distance from the player's
        centre start), using the RNG seeded by `seed` -- pass the same seed
        to get the identical layout back (Gymnasium determinism contract,
        exercised by the env_checker tests); leave it None for fresh layouts
        every episode, which is what training wants for variety.
        """
        self._rng = random.Random(seed)
        self.phase_manager.phase = 0
        player = Player(
            x=self.arena_width / 2.0,
            y=self.arena_height / 2.0,
            health=float(self._pcfg["max_health"]),
            max_health=float(self._pcfg["max_health"]),
            orientation=-math.pi / 2.0,  # facing "up"
            shoot_cooldown=0,
        )
        self.state = ArenaState(player=player, control_style=self.control_style)
        self._spawn_phase_spawners(0)
        self.state.phase = 0
        self.state.step_count = 0
        self._shot_no_target_flag = False
        self._shot_toward_enemy_flag = 0.0
        self._render_events = []
        self._last_step_events = self._empty_step_events()
        self._cumulative_approach_reward = 0.0
        self._cumulative_progress_reward = 0.0
        self._shots_fired = 0

        obs = build_observation(self.state, self.arena_width, self.arena_height)
        self._last_obs = obs
        return obs

    def step(self, action: int):
        """Apply one control action, advance one tick, resolve the phase
        system, compute the reward, and return the literal 4-tuple
        (observation, reward, done, info).

        info always carries:
            info["reward_breakdown"] -> arena.rewards.RewardBreakdown
            info["died"]             -> bool  (player HP hit 0 this step)
            info["truncated"]        -> bool  (step limit hit, not dead)

        Order of operations:
          1. Interpret `action` via self.action_enum; apply to the player
             (Style 1: rotate / inertial thrust / shoot; Style 2: direct
             velocity / shoot). SHOOT spawns one player Projectile if the
             cooldown is clear.
          2. Advance enemy AI (straight-line toward the player) and spawner
             timers (spawn an Enemy on interval, up to the concurrent cap).
          3. Advance projectiles; resolve collisions via circle_collision:
             projectile-enemy, projectile-spawner, enemy-player contact.
             (No enemy projectiles exist -- see the module docstring.)
          4. Apply damage/deaths; build step_events (see LOCKED CONTRACT).
          5. maybe_advance_phase(): if every active spawner is destroyed,
             increment the phase, spawn the next phase's spawners, and set
             step_events["phase_advanced"] = True.
          6. done = player died OR step_count >= max_steps.
          7. Build observation, compute reward, return the 4-tuple.

        LOCKED step_events CONTRACT (must stay byte-identical to the block in
        arena/rewards.py::compute_reward -- a silent key-name drift produces
        wrong rewards with no error):

            step_events = {
                "enemies_killed":                  int,    # enemies destroyed this step
                "spawners_killed":                 int,    # spawners destroyed this step
                "phase_advanced":                  bool,   # phase incremented this step
                "damage_taken":                    float,  # player HP lost this step, >= 0
                "died":                            bool,   # player HP reached 0 this step
                "distance_delta_to_nearest_enemy": float,  # signed; < 0 = got closer;
                                                           #   0.0 when there is no enemy
                                                           #   before AND after the step
                "shot_fired_with_no_target":       bool,   # player fired while the nearest
                                                           #   enemy was farther than
                                                           #   SHOT_NO_TARGET_RADIUS (or none)
            }

        Plus OPTIONAL keys feeding compute_reward's per-episode caps (see
        its docstring -- defaults there make each cap a no-op if omitted,
        but this class populates them for real): "nearest_enemy_distance"
        (float, post-move distance to the nearest enemy, or +inf if none)
        and "cumulative_approach_reward" (float, this episode's running
        total of approach_nearest_enemy BEFORE this step) for
        R_APPROACH_NEAREST_ENEMY; "cumulative_progress_reward" (float, this
        episode's running total of kill_enemy + kill_spawner +
        phase_progress BEFORE this step) for PROGRESS_REWARD_EPISODE_CAP.
        """
        if self.state is None:
            raise RuntimeError("ArenaCoreEnv.step() called before reset()")

        st = self.state
        p = st.player
        self._render_events = []
        ev = self._empty_step_events()

        prev_nearest = self._nearest_enemy_distance()

        # 1. player action
        self._apply_player_action(int(action))
        if p.shoot_cooldown > 0:
            p.shoot_cooldown -= 1

        # 2. enemies + spawners
        self._advance_enemies()
        self._advance_spawners()

        # 3-4. projectiles, collisions, damage
        self._advance_projectiles()
        enemies_killed, spawners_killed, dmg_taken, dmg_dealt, aimed_hit_sum = (
            self._resolve_collisions()
        )
        ev["enemies_killed"] = enemies_killed
        ev["spawners_killed"] = spawners_killed
        ev["damage_taken"] = dmg_taken
        ev["damage_dealt"] = dmg_dealt
        ev["aimed_hit_alignment_sum"] = aimed_hit_sum

        new_nearest = self._nearest_enemy_distance()
        if prev_nearest is not None and new_nearest is not None:
            ev["distance_delta_to_nearest_enemy"] = new_nearest - prev_nearest
        else:
            ev["distance_delta_to_nearest_enemy"] = 0.0
        ev["nearest_enemy_distance"] = new_nearest if new_nearest is not None else float("inf")
        ev["cumulative_approach_reward"] = self._cumulative_approach_reward
        ev["cumulative_progress_reward"] = self._cumulative_progress_reward

        ev["shot_fired_with_no_target"] = self._shot_no_target_flag
        ev["shot_toward_enemy"] = self._shot_toward_enemy_flag

        # Distances to the nearest and SECOND-nearest wall, for the
        # wall-proximity shaping term (see rewards_config.py
        # R_WALL_PROXIMITY_PER_STEP): summing the two ramps makes a corner
        # cost ~2x a straight wall automatically -- the "stuck in corner"
        # deterrent without a separate harsh penalty.
        wall_ds = sorted(
            (
                float(p.x),
                float(p.y),
                self.arena_width - float(p.x),
                self.arena_height - float(p.y),
            )
        )
        ev["wall_distance"] = wall_ds[0]
        ev["wall_distance_second"] = wall_ds[1]

        # 5. phase system
        if self.phase_manager.maybe_advance_phase(st.spawners):
            ev["phase_advanced"] = True
            self._spawn_phase_spawners(self.phase_manager.phase)
        st.phase = self.phase_manager.phase

        # 6. termination
        st.step_count += 1
        died = p.health <= 0.0
        if died:
            p.health = 0.0
        truncated = (st.step_count >= self.max_steps) and not died
        done = died or truncated
        ev["died"] = bool(died)

        # 7. observation + reward
        obs = build_observation(st, self.arena_width, self.arena_height)
        rb = compute_reward(ev, self._reward_overrides)
        self._cumulative_approach_reward += rb.approach_nearest_enemy
        # kill_enemy is intentionally excluded -- it is uncapped (2026-09-08d,
        # BUG_LESS_ATTACK.md); only phase/spawner progression is capped.
        self._cumulative_progress_reward += rb.kill_spawner + rb.phase_progress
        self._last_obs = obs
        self._last_step_events = ev
        info = {
            "reward_breakdown": rb,
            "died": bool(died),
            "truncated": bool(truncated),
        }
        return obs, float(rb.total), bool(done), info

    def render(self, renderer=None) -> None:
        """Draw the current scene through an
        arena.render_pygame.ArenaRenderer (passed by the eval scripts).
        Training never calls this. Passes transient hit/kill effects plus
        the last obs / step_events so the renderer's debug overlay can show
        them; a bare `renderer.draw(state)` renderer also works because the
        extras arg is optional.
        """
        if renderer is None:
            return
        extra = {
            "events": list(self._render_events),
            "obs": self._last_obs,
            "step_events": self._last_step_events,
        }
        renderer.draw(self.state, extra)

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _empty_step_events() -> dict:
        return {
            "enemies_killed": 0,
            "spawners_killed": 0,
            "phase_advanced": False,
            "damage_taken": 0.0,
            "damage_dealt": 0.0,
            "died": False,
            "distance_delta_to_nearest_enemy": 0.0,
            "nearest_enemy_distance": float("inf"),
            "cumulative_approach_reward": 0.0,
            "cumulative_progress_reward": 0.0,
            "shot_fired_with_no_target": False,
            "shot_toward_enemy": 0.0,
            "wall_distance": float("inf"),
            "wall_distance_second": float("inf"),
            "aimed_hit_alignment_sum": 0.0,
        }

    def _nearest_enemy_distance(self) -> float | None:
        st = self.state
        if not st.enemies:
            return None
        p = st.player
        return min(distance(p.x, p.y, e.x, e.y) for e in st.enemies)

    def _nearest_enemy(self):
        """The nearest living enemy entity, or None if there are none."""
        st = self.state
        if not st.enemies:
            return None
        p = st.player
        return min(st.enemies, key=lambda e: distance(p.x, p.y, e.x, e.y))

    def _spawn_phase_spawners(self, phase: int) -> None:
        cfg = self.phase_manager.difficulty_for_phase(phase)
        self._current_enemy_speed = cfg.enemy_speed
        n = max(1, cfg.num_spawners)
        lo_x, hi_x = _SPAWNER_MARGIN, self.arena_width - _SPAWNER_MARGIN
        lo_y, hi_y = _SPAWNER_MARGIN, self.arena_height - _SPAWNER_MARGIN
        # Rejection-sample spawner positions so none spawns on top of the
        # player (who starts at the arena centre). RNG comes from reset(seed)
        # so layouts are reproducible under a fixed seed and varied otherwise.
        rng = getattr(self, "_rng", None) or random.Random()
        px, py = self.arena_width / 2.0, self.arena_height / 2.0
        min_player_dist = 150.0
        spawners = []
        for _ in range(n):
            for _attempt in range(20):
                sx = rng.uniform(lo_x, hi_x)
                sy = rng.uniform(lo_y, hi_y)
                if distance(sx, sy, px, py) >= min_player_dist:
                    break
            spawners.append(
                Spawner(
                    x=sx,
                    y=sy,
                    health=float(self._scfg["base_health"]),
                    max_health=float(self._scfg["base_health"]),
                    spawn_interval_steps=int(cfg.enemy_spawn_interval_steps),
                    steps_since_last_spawn=0,
                    active=True,
                )
            )
        self.state.spawners = spawners

    def _apply_player_action(self, action: int) -> None:
        st = self.state
        p = st.player
        pc = self._pcfg
        max_speed = float(pc["max_speed"])
        self._shot_no_target_flag = False
        self._shot_toward_enemy_flag = 0.0

        try:
            act = self.action_enum(action)
        except ValueError:
            act = None  # out-of-range action -> treat as NO_OP (defensive)

        if self.control_style == 1:
            if act is ControlStyle1.ROTATE_LEFT:
                p.orientation -= float(pc["rotate_speed_rad"])
            elif act is ControlStyle1.ROTATE_RIGHT:
                p.orientation += float(pc["rotate_speed_rad"])
            elif act is ControlStyle1.THRUST_FORWARD:
                p.vx += math.cos(p.orientation) * float(pc["thrust_accel"])
                p.vy += math.sin(p.orientation) * float(pc["thrust_accel"])
            elif act is ControlStyle1.SHOOT:
                self._try_shoot()
            # NO_OP / None: coast

            # inertial: friction every step, then clamp speed
            fr = float(pc["friction"])
            p.vx *= fr
            p.vy *= fr
            spd = math.hypot(p.vx, p.vy)
            if spd > max_speed:
                p.vx *= max_speed / spd
                p.vy *= max_speed / spd

        else:  # control_style == 2, direct directional
            if act is ControlStyle2.MOVE_UP:
                p.vx, p.vy = 0.0, -max_speed
            elif act is ControlStyle2.MOVE_DOWN:
                p.vx, p.vy = 0.0, max_speed
            elif act is ControlStyle2.MOVE_LEFT:
                p.vx, p.vy = -max_speed, 0.0
            elif act is ControlStyle2.MOVE_RIGHT:
                p.vx, p.vy = max_speed, 0.0
            elif act is ControlStyle2.SHOOT:
                self._try_shoot()
                # 2026-08-28 design change: SHOOT no longer hard-stops the
                # player -- velocity is PRESERVED, so the player glides in
                # its last direction while firing (kiting). Previously SHOOT
                # zeroed velocity, which made open-field shooting a sitting
                # duck and taught the policy to use walls as firing positions
                # (measured: 84-92% of style-2 eval steps spent hugging
                # walls). NO_OP is still the deliberate brake. Matches
                # style 1, where SHOOT also lets the ship coast.
            else:  # NO_OP / None -> snappy stop
                p.vx, p.vy = 0.0, 0.0

            # keep `orientation` aimed along the current velocity so SHOOT
            # fires where the player is heading (also gives glide-shooting a
            # coherent direction after the 2026-08-28 SHOOT-preserves-velocity
            # change); leave it untouched on a hard stop (NO_OP).
            if (p.vx, p.vy) != (0.0, 0.0):
                p.orientation = math.atan2(p.vy, p.vx)

        # integrate + clamp to the arena. Zero ONLY the axis that hit a wall
        # (wall-tangent slide: a glancing approach keeps sliding along the
        # wall, only a head-on or into-a-corner approach dead-stops). For
        # style 1 the remaining escape cost is reorientation -- pc
        # ["rotate_speed_rad"] was raised 0.14 -> 0.2 on 2026-09-08c
        # (FIX_PART2.md symptom 4: style-1 policies nosed into a wall while
        # being swarmed and took ~22 rotate steps to turn away; ~16 now).
        nx, ny = integrate_position(p.x, p.y, p.vx, p.vy, 1.0)
        cx, cy = wrap_or_clamp_to_bounds(nx, ny, self.arena_width, self.arena_height)
        if cx != nx:
            p.vx = 0.0
        if cy != ny:
            p.vy = 0.0
        p.x, p.y = cx, cy

    def _try_shoot(self) -> None:
        st = self.state
        p = st.player
        pc = self._pcfg
        if p.shoot_cooldown > 0:
            return
        self._shots_fired += 1
        spd = float(pc["projectile_speed"])

        # Fire-time aim alignment, stamped on the projectile so
        # R_AIMED_HIT_BONUS can pay for INTENDED hits only (an alignment of
        # ~0 means the hit, if any, was luck). `_shot_toward_enemy_flag` is
        # also set from it for the (now disabled, 2026-09-08d)
        # R_SHOOT_TOWARD_ENEMY term and for eval instrumentation. Objective
        # is the nearest enemy within SHOT_NO_TARGET_RADIUS, else the
        # nearest active spawner -- see _graded_aim_alignment.
        alignment = self._graded_aim_alignment()

        # The same graded value is stamped onto the projectile so
        # R_AIMED_HIT_BONUS can pay for INTENDED hits only (an alignment of
        # ~0 means the hit, if any, was luck).
        st.projectiles.append(
            Projectile(
                x=p.x,
                y=p.y,
                vx=math.cos(p.orientation) * spd,
                vy=math.sin(p.orientation) * spd,
                owner="player",
                damage=float(pc["projectile_damage"]),
                aim_alignment=alignment,
            )
        )
        p.shoot_cooldown = int(pc["shoot_cooldown_steps"])
        nd = self._nearest_enemy_distance()
        self._shot_no_target_flag = (nd is None) or (nd > SHOT_NO_TARGET_RADIUS)
        self._shot_toward_enemy_flag = alignment

    def _graded_aim_alignment(self) -> float:
        """Graded alignment of the player's CURRENT facing with its current
        objective (nearest in-range enemy, else nearest active spawner), in
        [0, 1]. See _try_shoot's comment for the design rationale.
        """
        st = self.state
        p = st.player
        ne = self._nearest_enemy()
        if ne is not None:
            d = distance(p.x, p.y, ne.x, ne.y)
            if d <= SHOT_NO_TARGET_RADIUS:
                target_x, target_y = ne.x, ne.y
                aimed_at_objective = True
            else:
                aimed_at_objective = False
        else:
            aimed_at_objective = False
        if not aimed_at_objective and st.spawners:
            active = [s for s in st.spawners if s.active]
            if active:
                sp = min(active, key=lambda s: distance(p.x, p.y, s.x, s.y))
                target_x, target_y = sp.x, sp.y
                aimed_at_objective = True
        if not aimed_at_objective:
            return 0.0
        ang_to_target = relative_direction(p.x, p.y, target_x, target_y)
        diff = abs((ang_to_target - p.orientation + math.pi) % (2 * math.pi) - math.pi)
        return max(0.0, math.cos(diff))

    def _advance_enemies(self) -> None:
        st = self.state
        p = st.player
        for e in st.enemies:
            ang = relative_direction(e.x, e.y, p.x, p.y)
            e.x += math.cos(ang) * e.speed
            e.y += math.sin(ang) * e.speed
            e.x, e.y = wrap_or_clamp_to_bounds(e.x, e.y, self.arena_width, self.arena_height)

    def _advance_spawners(self) -> None:
        st = self.state
        cap = int(self._ecfg["max_concurrent_enemies"])
        base_hp = float(self._ecfg["base_health"])
        for s in st.spawners:
            if not s.active:
                continue
            s.steps_since_last_spawn += 1
            if s.steps_since_last_spawn >= s.spawn_interval_steps and len(st.enemies) < cap:
                st.enemies.append(
                    Enemy(
                        x=s.x,
                        y=s.y,
                        health=base_hp,
                        max_health=base_hp,
                        speed=self._current_enemy_speed,
                    )
                )
                s.steps_since_last_spawn = 0

    def _advance_projectiles(self) -> None:
        st = self.state
        w, h = self.arena_width, self.arena_height
        for pr in st.projectiles:
            pr.x, pr.y = integrate_position(pr.x, pr.y, pr.vx, pr.vy, 1.0)
        st.projectiles = [
            pr for pr in st.projectiles if -20.0 <= pr.x <= w + 20.0 and -20.0 <= pr.y <= h + 20.0
        ]

    def _resolve_collisions(self) -> tuple[int, int, float, float, float]:
        st = self.state
        p = st.player
        p_r = float(self._pcfg["radius"])
        e_r = float(self._ecfg["radius"])
        s_r = float(self._scfg["radius"])

        enemies_killed = 0
        spawners_killed = 0
        dmg_taken = 0.0
        dmg_dealt = 0.0
        # Sum of the graded aim alignment of every player projectile that
        # hits an objective this step (see R_AIMED_HIT_BONUS in
        # rewards_config.py) -- pays for intended hits, not lucky ones.
        aimed_hit_sum = 0.0
        # player projectiles vs enemies / spawners (one hit consumes the shot)
        surviving = []
        for pr in st.projectiles:
            hit = False
            for e in st.enemies:
                if e.health > 0 and circle_collision(pr.x, pr.y, _PROJECTILE_RADIUS, e.x, e.y, e_r):
                    dealt = min(pr.damage, e.health)  # no overkill credit
                    e.health -= pr.damage
                    dmg_dealt += dealt
                    aimed_hit_sum += pr.aim_alignment
                    hit = True
                    if e.health <= 0:
                        enemies_killed += 1
                        self._render_events.append(("kill", e.x, e.y))
                    break
            if not hit:
                for s in st.spawners:
                    if s.active and circle_collision(pr.x, pr.y, _PROJECTILE_RADIUS, s.x, s.y, s_r):
                        s.health -= pr.damage
                        hit = True
                        aimed_hit_sum += pr.aim_alignment
                        if s.health <= 0:
                            s.active = False
                            spawners_killed += 1
                            self._render_events.append(("kill", s.x, s.y))
                        break
            if not hit:
                surviving.append(pr)
        st.projectiles = surviving
        st.enemies = [e for e in st.enemies if e.health > 0]

        # enemy-player contact (NON-KAMIKAZE, 2026-08-28 design change): an
        # enemy deals contact damage ONCE, enters a per-enemy damage cooldown,
        # SURVIVES, and keeps chasing. With the old kamikaze rule the enemy
        # died on touching the player, so camping let enemies suicide into you
        # and shooting was never needed for survival -- every trained policy
        # converged to corner-camping with zero kills. Persisting enemies make
        # shooting the only way to reduce incoming threat. Enemies still only
        # die from player projectiles, so enemies_killed remains projectile-
        # kills-only (no kill credit for contact -- see rewards_config.py).
        contact = float(self._ecfg["contact_damage"])
        cooldown = int(self._ecfg["contact_damage_cooldown_steps"])
        for e in st.enemies:
            if e.damage_cooldown > 0:
                e.damage_cooldown -= 1
                continue
            if circle_collision(p.x, p.y, p_r, e.x, e.y, e_r):
                p.health -= contact
                dmg_taken += contact
                e.damage_cooldown = cooldown
                self._render_events.append(("player_hit",))
        if p.health < 0.0:
            p.health = 0.0

        return enemies_killed, spawners_killed, dmg_taken, dmg_dealt, aimed_hit_sum
