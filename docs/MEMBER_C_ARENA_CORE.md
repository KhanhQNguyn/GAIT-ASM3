# MEMBER C — Arena Core Environment

> **How to use this file:** This is your complete, self-contained context for the project. You
> don't need `PLAN.md` or `SPRINT.md` open alongside this. Work top to bottom. Every task is a
> checkbox: mark `- [x]` only when genuinely done (tests passing, output actually generated, not
> just code written). Append a dated line to the **Progress Log** at the bottom every time you
> finish a task group.

---

## 0. Project context (read once)

This is a 2-part, 40-point RL assignment: Part I is classical tabular RL in a Pygame gridworld;
Part II is deep RL (Stable-Baselines3) controlling a ship in a real-time Pygame arena. The
codebase is a fully-specified skeleton — every function has the right signature and a docstring
telling you exactly what to implement; every body currently `raise NotImplementedError`.

You are **Member C — Arena Core Environment**. You build the entire Part II simulation: the
entities, physics, observation vector, phase system, and the Gym-style API itself. **Your
`core_env.py` is the single most import-heavy file in the whole repository** — it pulls together
everyone else's Part II work, and everything downstream (training, evaluation, both control
schemes) is blocked until it exists and works.

---

## 1. Your ownership

| File | What it is |
|---|---|
| `part2_arena/arena/entities.py` | Plain-data classes: `Player`, `Enemy`, `Spawner`, `Projectile`, `ArenaState` |
| `part2_arena/arena/actions.py` | The two control schemes' action enums |
| `part2_arena/arena/physics.py` | Minimal 2D movement/collision math |
| `part2_arena/arena/obs.py` | The fixed-size observation vector |
| `part2_arena/arena/phases.py` | Difficulty phase system + curriculum ramp |
| `part2_arena/arena/core_env.py` | **The integration gate** — the literal 4-tuple Gym API |
| `part2_arena/arena/gym_adapter.py` | Gymnasium 5-tuple wrapper for SB3 |
| `part2_arena/arena/render_pygame.py` | Pygame rendering for the arena |
| `part2_arena/tests/test_env_api.py`, `test_obs_shape.py` | Your correctness tests |

**Rubric rows you deliver:** Part II-G (4.5 pts, arena environment requirements — the single
biggest line item in the whole rubric) and Part II-H (2.5 pts, Gym-style API + observation
design). That's 7 of 40 points sitting directly in your files.

---

## 2. Dependency contract — critical, read this before starting

- **You depend on:** nobody, for `entities.py`, `actions.py`, `physics.py`. Start immediately.
- **You depend on Member D's `rewards.py`** for `core_env.py` specifically — `core_env.step()`
  calls `rewards.compute_reward(step_events)`. **You and Member D must agree on the exact
  `step_events` dict shape before either of you writes against it** — this is the single riskiest
  handoff in the entire project (a silent key-name mismatch produces no error, just silently
  wrong rewards during training). Do this agreement in a real conversation, not by guessing —
  then write the agreed shape down in `rewards.py`'s docstring immediately.
- **Everyone in Part II depends on you.** `gym_adapter.py` (yours, but sequenced after
  `core_env.py`), `scripts/train.py` and both eval scripts (Member D's) cannot start until your
  `core_env.py` passes its tests. **The moment it does, tell Member D immediately.**
- `render_pygame.py` is the one exception — it only needs `entities.ArenaState`'s shape to be
  stable, so you can (and should) do it early, in parallel with `obs.py`/`phases.py`, not after
  `core_env.py`.

**Sequencing for you specifically:** `entities.py` + `actions.py` + `physics.py` (any order,
parallel-safe) → `render_pygame.py` can start here too → `obs.py` + `phases.py` (both need
`entities.py`) → **agree on `step_events` shape with Member D** → `core_env.py` → `gym_adapter.py`.

---

## 3. Task list

### 3.1 `arena/entities.py` — start here

- [x] Review `Player`, `Enemy`, `Spawner`, `Projectile`, `ArenaState` as already stubbed. Add any
      additional fields `physics.py` or `rewards.py` will need — the file already flags this,
      e.g. a shoot cooldown timer on `Player` so the "shoot" action can't fire every single
      frame. Decide this **now**, before other files depend on the dataclass shape, and post the
      decision so Member D can react if it affects `step_events`.

### 3.2 `arena/actions.py`

- [x] `action_enum_for_style(style: int)` — return `ControlStyle1` for `style == 1`,
      `ControlStyle2` for `style == 2`, `raise ValueError` for anything else so a typo'd
      `--style` flag fails loudly during training instead of silently using the wrong action set.

### 3.3 `arena/physics.py`

- [x] `wrap_or_clamp_to_bounds(x, y, width, height)` — clamp-at-edge (simpler than wrap-around,
      and sufficient per the assignment's feasibility guide).
- [x] `integrate_position(x, y, vx, vy, dt)` — simple Euler step: `x += vx*dt`, `y += vy*dt`.
- [x] `relative_direction(from_x, from_y, to_x, to_y)` — `math.atan2(to_y - from_y, to_x - from_x)`,
      used by `obs.py` for the "direction to nearest enemy/spawner" features.
- [x] `circle_collision(x1, y1, r1, x2, y2, r2)` — `distance(x1,y1,x2,y2) <= r1 + r2`. Used for
      all projectile/entity collision checks in `core_env.py`.

### 3.4 `arena/render_pygame.py` — can start in parallel with 3.5/3.6, only needs `entities.py`

- [x] `ArenaRenderer.__init__(width, height, caption)` — `pygame.init()`, create a window sized
      `(width, height)`, set the caption.
- [x] `.draw(state: ArenaState)` — draw the player, every enemy, every spawner, every projectile
      (with health bars on player/enemies/spawners where relevant), plus a small HUD showing
      phase number and player health. Uses the `COLORS` dict already defined in the file.
- [x] `.handle_events()` — poll pygame events (at minimum, detect window-close) so eval scripts
      can exit cleanly.

**Smoke test now** (before `core_env.py` exists) — hand-build a minimal `ArenaState` with one
`Player` and one `Enemy` and confirm `.draw()` runs without exceptions.

### 3.5 `arena/obs.py`

- [x] `build_observation(state: ArenaState, arena_width, arena_height)` — implement every feature
      in `OBSERVATION_SPEC`, **in that exact order** (the list is already fully specified in the
      file: player x/y, player vx/vy, orientation sin/cos, health fraction, nearest-enemy
      distance + direction sin/cos, nearest-spawner distance + direction sin/cos, current-phase
      fraction, active-enemy-count fraction). Use `physics.distance` / `physics.relative_direction`.
      **Convention: every feature is normalized to `[-1, 1]`** — features naturally in `[0,1]`
      get rescaled via `x_normalized = 2*x_unit - 1`; sin/cos features are already in range.
- [x] Handle the **zero-enemies / zero-spawners edge case explicitly** — return a sane fallback
      (distance = 1.0 normalized-max, direction = 0), never NaN, never a division by zero. This
      is directly tested, not a hypothetical.

**Acceptance:**
```bash
cd part2_arena && pytest tests/test_obs_shape.py -v
```
- [x] `test_observation_has_fixed_shape`
- [x] `test_observation_has_no_nans_or_infs` — also checks every value is within `[-1, 1]`.
- [x] `test_observation_includes_all_required_features`

### 3.6 `arena/phases.py`

- [x] `PhaseManager.difficulty_for_phase(phase)` — build a base difficulty curve: `enemy_speed`
      and spawn frequency increase with `phase`, `num_spawners` increases every couple of phases.
      When `self.curriculum_enabled` (creativity hook c), deliberately make early phases *easier*
      than the base curve would produce (slower enemies, sparser spawns), ramping up to the full
      curve by some target phase you choose. **Write the exact ramp schedule as a comment in this
      method** — the report needs to describe it precisely, and "I'll remember" is not
      retrievable later.
- [x] `maybe_advance_phase(spawners)` — `if self.all_spawners_destroyed(spawners): self.phase += 1; return True`, else `return False`. (`all_spawners_destroyed` is already implemented.)

### 3.7 `arena/core_env.py` — the big one, agree on `step_events` with Member D first

- [x] `ArenaCoreEnv.__init__(control_style, curriculum_enabled=False)` — store
      `self.action_enum = action_enum_for_style(control_style)`,
      `self.phase_manager = PhaseManager(curriculum_enabled=curriculum_enabled)`, plus fixed
      setup (`max_steps` — use `DEFAULT_MAX_STEPS` already defined, an RNG instance).
- [x] `reset()` — fresh `Player` at a default spawn position (e.g. arena center or a fixed
      corner — your call, document it), spawners for phase 0 via
      `self.phase_manager.difficulty_for_phase(0)`, no enemies/projectiles yet, `step_count=0`.
      Return **the observation alone**, not a tuple — matches the spec's literal wording
      `reset() -> observation`.
- [x] `step(action: int)` — follow this order exactly (already documented in the file, repeated
      here for completeness):
  1. Interpret `action` via `self.action_enum`; apply to the player via `physics` helpers
     (movement/rotation for Style 1, direct movement for Style 2; shoot spawns a `Projectile`
     respecting the cooldown you added in 3.1).
  2. Advance enemy AI (move each enemy toward the player at its `speed`) and spawner timers
     (spawn a new `Enemy` when `steps_since_last_spawn >= spawn_interval_steps`, then reset the
     counter).
  3. Advance all projectiles (`integrate_position`), resolve collisions via
     `physics.circle_collision`: projectile-enemy, projectile-spawner, projectile-player,
     enemy-player direct contact.
  4. Apply damage/deaths from step 3's collisions; **build the `step_events` dict** in exactly
     the shape you agreed with Member D (something like: `enemies_killed`, `spawners_killed`,
     `phase_advanced`, `damage_taken`, `died`, `distance_delta_to_nearest_enemy`,
     `shot_fired_with_no_target`).
  5. `self.phase_manager.maybe_advance_phase(spawners)` — if it returns `True`, spawn the next
     phase's spawners via `difficulty_for_phase(self.phase_manager.phase)` and mark
     `step_events["phase_advanced"] = True`.
  6. `done = player.health <= 0 OR self.step_count >= self.max_steps`.
  7. Build the observation via `obs.build_observation`, compute reward via
     `rewards.compute_reward(step_events)` (Member D's function), return the **literal 4-tuple**
     `(observation, reward, done, info)` with `info["reward_breakdown"] = reward_breakdown`,
     `info["died"] = <bool>`, `info["truncated"] = <bool, step-limit case only>`.
- [x] `render(renderer=None)` — delegate to `renderer.draw(self.state)`.

**Acceptance:**
```bash
pytest tests/test_env_api.py::test_core_env_reset_returns_bare_observation \
       tests/test_env_api.py::test_core_env_step_returns_literal_4_tuple -v
```
Then smoke-test headless for **both** control styles:
```bash
python -c "
from arena.core_env import ArenaCoreEnv
for style in (1, 2):
    env = ArenaCoreEnv(control_style=style)
    obs = env.reset()
    for _ in range(50):
        obs, reward, done, info = env.step(0)
        if done: obs = env.reset()
    print(f'style {style} OK')
"
```
Both must run cleanly with no exceptions. **The instant this works, tell Member D — training is
now unblocked.**

### 3.8 `arena/gym_adapter.py` — sequenced after `core_env.py`

- [x] `ArenaGymEnv(gym.Env).__init__(control_style, curriculum_enabled=False, render_mode=None)`
      — build `observation_space = spaces.Box(low=-1, high=1, shape=(OBS_DIM,), dtype=np.float32)`
      and `action_space = spaces.Discrete(len(action_enum_for_style(control_style)))`; wrap an
      internal `ArenaCoreEnv`.
- [x] `reset(seed=None, options=None)` — gymnasium's 2-return contract: `(observation, info)`.
- [x] `step(action)` — call `core_env.step(action)`, split `done` using
      `terminated = info["died"]`, `truncated = info["truncated"] and not terminated`. Return the
      literal 5-tuple `(obs, reward, terminated, truncated, info)`. **Never let both be `True` on
      the same step.**
- [x] `render()` — lazily construct an `ArenaRenderer` only when `render_mode == "human"`,
      delegate to `core_env.render(self._renderer)`.

**Acceptance:**
```bash
pytest tests/test_env_api.py -v   # all 4 tests in this file, 100%
```
- [x] `test_gym_adapter_step_returns_gymnasium_5_tuple`
- [x] `test_gym_adapter_terminated_on_death_truncated_on_max_steps`

### 3.9 Sanity pass on gameplay feel

- [x] Full pass on `core_env.py` after the plumbing works: confirm phase progression actually
      triggers when spawners die, enemies visibly navigate toward the player, and both control
      styles feel distinct (rotation+thrust should feel like inertial ship control; direct
      movement should feel snappy/tile-shooter-like) — this is a real gameplay-quality item, not
      just a passing test, since the video demo needs to show it convincingly.

---

## 4. Going beyond the minimum — how to actually stand out

You own the biggest single rubric row (II-G, 4.5 pts) — polish here is highly visible:

- [ ] **Simple enemy AI variety** instead of pure "always move straight at the player" — e.g. a
      couple of enemy archetypes with different `speed`/behavior (one that strafes, one that
      rushes), still trivially simple per the feasibility guide but noticeably more "game-like"
      in the video.
- [x] **Visual feedback on hits/deaths** in `render_pygame.py` — a brief flash or particle burst
      on enemy/spawner destruction, a screen-edge tint when the player takes damage. Cheap in
      Pygame (a few circles with a short lifespan), disproportionately improves how "finished"
      the demo looks.
- [x] **A debug overlay toggle** in `render_pygame.py` — show the raw observation vector values
      and current `step_events` live on screen during evaluation. Extremely useful for your own
      debugging, and doubles as compelling report/video evidence that the observation design
      actually reflects what's happening.
- [x] **Extra tests beyond the required set** — e.g. a test that `build_observation` stays within
      `[-1, 1]` even at extreme/edge positions (corner of the arena, exactly on top of an enemy),
      and a test that `maybe_advance_phase` doesn't advance when only *some* spawners are
      destroyed.
- [x] **Document the curriculum ramp schedule precisely** (see 3.6) with an actual formula, not
      just "starts easier" — e.g. "`enemy_speed(phase) = base_speed * min(1, 0.5 + phase/5)` for
      the first 5 phases" — a marker can verify a documented formula; they can't verify vibes.
- [x] **Type hints and docstrings** on every method you touch, consistent with the rest of the
      repo's style.

---

## 5. Report & video responsibilities

**Report — you write:**
- Section 1 (Description of both environments), Part II half — the arena's mechanics, entities,
  and phase system.
- Section 2 (Observation design) — the full feature list from `OBSERVATION_SPEC` with the
  justification for each (why position/velocity/orientation/distance/direction/health/phase were
  chosen as the minimum sufficient state for the agent to act on).

**Video — you help capture (alongside Member D):**
- The real-time Pygame arena running visually, enemies spawning and moving, projectiles and
  collisions functioning, at least one phase progression occurring. You're responsible for the
  environment looking correct and legible on camera — Member D drives the trained-agent gameplay
  itself once models exist.

---

## 6. Definition of done for Member C

- [x] `pytest part2_arena/tests/test_env_api.py tests/test_obs_shape.py -v` — 100% pass, zero
      skips.
- [x] Headless smoke test passes for both control styles (Section 3.7).
- [x] `render_pygame.py` draws a live frame correctly for a hand-built and a real `ArenaState`.
- [ ] Phase progression, enemy spawning, and both control schemes visibly work when watched.
- [x] At least one "going beyond the minimum" item from Section 4 is actually implemented.
- [x] Report Sections 1 (Part II half) and 2 are drafted.
- [x] `docs/CONTRIBUTIONS.md` has a current entry for your work.

---

## 7. Progress Log

*(Append a line every time a task group above is completed. Keep entries short and factual.)*

| Date | What was completed | Tests status |
|---|---|---|
| 2026-08-26 | 3.1 entities (Player.shoot_cooldown, ArenaState.control_style); 3.2 actions; 3.3 physics (all 4). | behaviour checks pass |
| 2026-08-26 | 3.5 obs (15 features — enemies don't shoot, projectile features NOT added per A1; every value in [-1,1], zero-entity fallbacks); 3.6 phases (base curve + curriculum ramp, empty-list guard). | `test_obs_shape.py` 12/12, `test_phases.py` 5/5 |
| 2026-08-26 | 3.4 render_pygame (draw/HUD/health bars/kill-flash/damage-tint/debug overlay, D-key toggle). Smoke test passes under SDL dummy driver (hand-built + real ArenaState, both styles). | smoke OK |
| 2026-08-26 | step_events contract locked (7 keys, byte-identical in core_env + rewards). `rewards.compute_reward` body implemented to unblock (Member D owns constants). | `compute_reward` checks pass |
| 2026-08-26 | 3.7 core_env (__init__ loads arena.json, reset→bare obs, 7-step order, deterministic base env, literal 4-tuple, info keys). CHECKPOINT: contract tests + headless smoke both styles. | `test_env_api.py::test_core_env_*` PASS; 300-step smoke both styles PASS |
| 2026-08-26 | 3.8 gym_adapter (reset→(obs,{}), 4→5-tuple split, lazy renderer, close). | `test_env_api.py` 8/8 incl. gymnasium `check_env` both styles |
| 2026-08-26 | 3.9 sanity: phase progression triggers on spawner death, enemies chase, shoot respects cooldown, forced death→terminated, max_steps→truncated. Going-beyond: hit/kill FX + debug overlay + `test_phases.py` extras. Docs: A1 sync (RUBRIC_MAP item 7, AUDIT §5.5, report §2), report §1.2/§2 draft, CONTRIBUTIONS entry. | full `part2_arena/tests` 25 passed / 4 skipped; `ruff` clean |
