# RL Assignment Report

**STRICT constraint: maximum 10 pages total, including images. No appendix —
anything beyond page 10 will not be considered by markers.** Budget roughly
one page per section below; trim aggressively rather than exceed the limit.

Generated tables/plots referenced below should live in `report/figures/` and
be produced by `scripts/generate_report_tables.py` and the various
`plot_*.py` scripts in each part, so the report never drifts from the code.

---

## 1. Environment Descriptions (~1-1.5 pages)

### 1.1 Part I — Gridworld
TODO: grid size, levels, mechanics, rules (rocks, fire, apples, keys,
chests, monsters), how Pygame rendering/interaction works.

### 1.2 Part II — Arena  *(draft — Member C)*

**Arena.** A continuous 960×680 world (`config/arena.json`). All entities are
circles/points moving in real number coordinates and integrated with an
explicit Euler step (`arena/physics.py`), so movement is smooth rather than
tile-locked — this is what makes it an "action arena" and not a grid. One
`env.step()` advances the simulation by one tick (`dt = 1.0`); the demo
renders at 60 FPS.

**Player.** A ship with position, velocity, a facing angle, health
(100 max), and a shoot cooldown (8 steps). Two control schemes, one trained
model each (`arena/actions.py`):
- *Style 1 — rotation + thrust:* `ROTATE_LEFT/RIGHT` turn the ship;
  `THRUST_FORWARD` adds acceleration along the facing; velocity carries
  between steps with light friction (0.97) and a speed cap — inertial feel.
- *Style 2 — direct directional:* `MOVE_UP/DOWN/LEFT/RIGHT` set the velocity
  vector directly, `NO_OP`/`SHOOT` stop it — snappy, twin-stick feel. The
  facing tracks the last move direction so `SHOOT` still has an aim.

**Enemies & spawners.** Spawners sit at fixed points on an inset ellipse
around the arena centre; each emits an enemy every `spawn_interval_steps`
(up to a concurrency cap of 18). Enemies move in a straight line toward the
player at their phase's speed and are destroyed on contact, dealing one
burst of contact damage. **Enemies do not fire projectiles** — keeping the
threat model to "don't get touched" keeps the observation at the spec
minimum and the mechanics easy to reason about.

**Projectiles.** Only the player shoots. A projectile travels along the
ship's facing and is consumed by the first enemy or active spawner it
overlaps (`physics.circle_collision`); enemy HP 30 (≈2 hits), spawner HP
120 (≈5 hits).

**Phase system.** When every active spawner in the current phase is
destroyed, the phase advances: `phases.PhaseManager.difficulty_for_phase`
produces the next `PhaseConfig` (faster enemies, shorter spawn interval,
another spawner every 2 phases) and the arena spawns that phase's spawners.
Enemies already on the field persist across the transition. The curriculum
option (creativity hook c) scales the *difficulty-increasing* deltas by
`frac = s0 + (1 − s0)·(phase/R)` for the first `R = 3` phases (`s0 = 0.5`),
so a curriculum run and a normal run converge to the identical curve by
phase 3 and differ only in how gently they ramp.

**Episode end.** Player health reaches 0 (**terminated**), or `max_steps`
(1200) is reached first (**truncated**). `ArenaCoreEnv` exposes the literal
`reset() -> obs`, `step(action) -> (obs, reward, done, info)` 4-tuple;
`ArenaGymEnv` is a thin wrapper that only splits `done` into
`terminated`/`truncated` for Stable-Baselines3.

## 2. Observation Design (~1 page)  *(draft — Member C)*

Fixed-size **15-float** vector, every element normalised to `[-1, 1]`
(`arena/obs.py::OBSERVATION_SPEC`; `spaces.Box(-1, 1, (15,), float32)`). No
pixels. Positions use `2·(x/size) − 1`; velocities `v/max_speed` (already
signed); fractions `2·f − 1`; distances `2·(d/diagonal) − 1` (near = −1,
far = +1); angles are given as `sin`/`cos` so the vector has no wrap-around
discontinuity.

| # | Feature | Why it is included |
|---|---|---|
| 0–1 | `player_x`, `player_y` | Absolute position — the agent must know where the walls are to avoid being cornered. |
| 2–3 | `player_vx`, `player_vy` | Current velocity — required for control under inertia (Style 1); lets the agent anticipate its own motion. |
| 4–5 | `player_orientation_sin/cos` | Facing — determines where a shot goes and which way `THRUST` pushes (Style 1). Forced to `0/1` for Style 2, where facing is not a controlled quantity. |
| 6 | `player_health_frac` | Remaining health — the agent needs it to trade aggression for caution as it gets low. |
| 7–9 | `nearest_enemy_distance`, `nearest_enemy_direction_sin/cos` | The immediate threat: how close and which way. Spec-required. Fallback `(+1, 0, 1)` when no enemy exists. |
| 10–12 | `nearest_spawner_distance`, `nearest_spawner_direction_sin/cos` | The objective: spawners must be destroyed to progress. Spec-required. Fallback `(+1, 0, 1)` when none are active. |
| 13 | `current_phase_frac` | Difficulty context — the same enemy layout is more dangerous at a later phase (faster enemies), so the agent's policy should be phase-aware. Spec-required. |
| 14 | `num_active_enemies_frac` | Crowding — one raw scalar for "how swarmed am I", which informs fight-vs-retreat without telling the agent what to do about it. |

This is the spec's minimum feature set plus phase and enemy-count. We
deliberately did **not** add features that would hand the agent a strategy
(e.g. "vector to the safest open space", "aim-corrected angle to target"):
the agent should learn evasion and aiming from the geometry it is given.
Directions are world-frame, not player-frame, for the same reason — raw
information, minimal interpretation baked in.

## 3. Reward Design (~1 page)

TODO: table of every reward term (Part I: apple/key/chest/death; Part II:
`rewards_config.py` constants) with justification for each, especially any
optional shaping terms. Pull directly from
`report/figures/reward_tables.md` (generated).

Two decisions to state and justify here:
  - Part I `REWARD_DEATH`: currently 0.0. If the SARSA-vs-Q-learning
    conservatism comparison (section 5 / Task 2) shows no meaningful
    difference on level1, a small negative value is the fix -- decide,
    change the constant, re-run `scripts/generate_report_tables.py`, and
    justify it here. (See docs/AUDIT_main.md 5.1.)
  - Part II shaping terms: `R_APPROACH_NEAREST_ENEMY` and
    `R_SHOOT_WHILE_NO_TARGET`. Member C's recommendation (per
    `docs/message.txt`: keep shaping minimal, don't reward-shape the
    strategy) is to keep BOTH at 0.0 and rely only on the 5 spec-required
    terms. If a shaping term is enabled, state why and show it did not
    change the learned behaviour by more than the noise band.
  - Part II `R_DEATH = -100` is ~20x `R_KILL_ENEMY`. Either justify that it
    must dominate a whole episode's positive reward (with an example
    calculation or a short ablation) or reduce it.

## 4. Hyperparameter Exploration (~1-1.5 pages)

TODO: what was tuned (learning rate, gamma, epsilon schedule, PPO/DQN
hyperparameters), what values were tried, and evidence (table or plot) of
the effect. Reference `part1_gridworld/config/training_config.json` and the
`train.py` runs logged to `part2_arena/logs/`.

## 5. Control Scheme Comparison (~1 page)

TODO: Style 1 (rotation+thrust) vs. Style 2 (directional) — training curves,
qualitative behavior differences, screenshots. If the PPO-vs-DQN ablation
was run, summarize it here or in a dedicated creativity subsection.

## 6. Training Evidence (~1-1.5 pages)

TODO: Part I training curves (Q-learning, SARSA, Expected SARSA, monster
levels, intrinsic reward on/off comparison — see
`part1_gridworld/src/plot_results.py` and `compare_algorithms.py`). Part II
TensorBoard screenshots / exported curves.

## 7. Originality Justification (~0.5 page)

TODO: what in this submission goes beyond the minimum spec, and why it's
your own work (reference any reused tutorial/assignment code per the reuse
policy).

## 8. Creativity Extensions (~0.5-1 page)

TODO: summarize whichever of the four extensions were completed —
PPO vs. DQN ablation, reward decomposition dashboard, curriculum learning,
Expected SARSA — with evidence for each.

## 9. Team

- Student numbers + contribution summary: see `CONTRIBUTIONS.md` (copy the
  final table in here).
- Video link: TODO.
