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
and are destroyed on touching the player. Only the player shoots. That
keeps the observation vector at the spec minimum (no "incoming projectile"
feature) and the mechanics simple. Projectile-vs-enemy and
projectile-vs-spawner collisions still satisfy the rubric's "projectile
collisions" requirement.
"""

from __future__ import annotations

import json
import math
import pathlib

from arena.actions import ControlStyle1, ControlStyle2, action_enum_for_style
from arena.entities import ArenaState, Enemy, Player, Projectile, Spawner
from arena.obs import build_observation
from arena.phases import PhaseManager
from arena.physics import (
    circle_collision,
    clamp,
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
        "rotate_speed_rad": 0.14,
        "radius": 14.0,
        "shoot_cooldown_steps": 8,
        "projectile_speed": 12.0,
        "projectile_damage": 25.0,
    },
    "enemy": {
        "radius": 12.0,
        "contact_damage": 12.0,
        "base_health": 30.0,
        "max_concurrent_enemies": 18,
    },
    "spawner": {"radius": 18.0, "base_health": 120.0},
}

_PROJECTILE_RADIUS = 4.0
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

    def __init__(self, control_style: int, curriculum_enabled: bool = False):
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
        # transient per-step data exposed for rendering / debugging
        self._last_obs = None
        self._last_step_events: dict = self._empty_step_events()
        self._render_events: list = []

    # ------------------------------------------------------------------- API
    def reset(self, *, seed: int | None = None):
        """Reset to a fresh episode and return the observation ALONE (not a
        tuple -- matches the spec's literal `reset() -> observation`).

        Layout: the player starts at the arena centre (symmetric, room to
        manoeuvre in every direction, neither control style advantaged);
        phase-0 spawners are placed at evenly-spaced points on an inset
        ellipse around the centre; no enemies or projectiles yet.
        """
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
        self._render_events = []
        self._last_step_events = self._empty_step_events()

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
        enemies_killed, spawners_killed, dmg_taken = self._resolve_collisions()
        ev["enemies_killed"] = enemies_killed
        ev["spawners_killed"] = spawners_killed
        ev["damage_taken"] = dmg_taken

        new_nearest = self._nearest_enemy_distance()
        if prev_nearest is not None and new_nearest is not None:
            ev["distance_delta_to_nearest_enemy"] = new_nearest - prev_nearest
        else:
            ev["distance_delta_to_nearest_enemy"] = 0.0

        ev["shot_fired_with_no_target"] = self._shot_no_target_flag

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
        rb = compute_reward(ev)
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
            "died": False,
            "distance_delta_to_nearest_enemy": 0.0,
            "shot_fired_with_no_target": False,
        }

    def _nearest_enemy_distance(self) -> float | None:
        st = self.state
        if not st.enemies:
            return None
        p = st.player
        return min(distance(p.x, p.y, e.x, e.y) for e in st.enemies)

    def _spawn_phase_spawners(self, phase: int) -> None:
        cfg = self.phase_manager.difficulty_for_phase(phase)
        self._current_enemy_speed = cfg.enemy_speed
        n = max(1, cfg.num_spawners)
        cx, cy = self.arena_width / 2.0, self.arena_height / 2.0
        rx = max(1.0, cx - _SPAWNER_MARGIN)
        ry = max(1.0, cy - _SPAWNER_MARGIN)
        spawners = []
        for k in range(n):
            ang = 2.0 * math.pi * k / n - math.pi / 2.0
            lo_x, hi_x = _SPAWNER_MARGIN, self.arena_width - _SPAWNER_MARGIN
            lo_y, hi_y = _SPAWNER_MARGIN, self.arena_height - _SPAWNER_MARGIN
            sx = clamp(cx + math.cos(ang) * rx, lo_x, hi_x)
            sy = clamp(cy + math.sin(ang) * ry, lo_y, hi_y)
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
                # keep moving while firing? no -- style 2 stops unless a move
                # action is held. SHOOT does not move.
                p.vx, p.vy = 0.0, 0.0
            else:  # NO_OP / None -> snappy stop
                p.vx, p.vy = 0.0, 0.0

            # keep `orientation` aimed along the last non-zero move so SHOOT
            # has a direction; leave it untouched on NO_OP / SHOOT.
            if (p.vx, p.vy) != (0.0, 0.0):
                p.orientation = math.atan2(p.vy, p.vx)

        # integrate + clamp to the arena
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
        spd = float(pc["projectile_speed"])
        st.projectiles.append(
            Projectile(
                x=p.x,
                y=p.y,
                vx=math.cos(p.orientation) * spd,
                vy=math.sin(p.orientation) * spd,
                owner="player",
                damage=float(pc["projectile_damage"]),
            )
        )
        p.shoot_cooldown = int(pc["shoot_cooldown_steps"])
        nd = self._nearest_enemy_distance()
        self._shot_no_target_flag = (nd is None) or (nd > SHOT_NO_TARGET_RADIUS)

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

    def _resolve_collisions(self) -> tuple[int, int, float]:
        st = self.state
        p = st.player
        p_r = float(self._pcfg["radius"])
        e_r = float(self._ecfg["radius"])
        s_r = float(self._scfg["radius"])

        enemies_killed = 0
        spawners_killed = 0
        dmg_taken = 0.0

        # player projectiles vs enemies / spawners (one hit consumes the shot)
        surviving = []
        for pr in st.projectiles:
            hit = False
            for e in st.enemies:
                if e.health > 0 and circle_collision(pr.x, pr.y, _PROJECTILE_RADIUS, e.x, e.y, e_r):
                    e.health -= pr.damage
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
                        if s.health <= 0:
                            s.active = False
                            spawners_killed += 1
                            self._render_events.append(("kill", s.x, s.y))
                        break
            if not hit:
                surviving.append(pr)
        st.projectiles = surviving
        st.enemies = [e for e in st.enemies if e.health > 0]

        # enemy-player contact: enemy is destroyed and deals contact damage once
        contact = float(self._ecfg["contact_damage"])
        still_alive = []
        for e in st.enemies:
            if circle_collision(p.x, p.y, p_r, e.x, e.y, e_r):
                p.health -= contact
                dmg_taken += contact
                self._render_events.append(("player_hit",))
            else:
                still_alive.append(e)
        st.enemies = still_alive
        if p.health < 0.0:
            p.health = 0.0

        return enemies_killed, spawners_killed, dmg_taken
