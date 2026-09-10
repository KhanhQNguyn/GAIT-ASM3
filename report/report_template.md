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

### 3.1 Part II reward — final structure

The five spec-required terms (`assessment_requirements_summary.md` §5), with the
magnitudes actually shipped (full table: `report/figures/reward_tables.md`):

| term | value | note |
|---|---|---|
| `R_KILL_ENEMY` | +5 | positive reward for destroying an enemy |
| `R_KILL_SPAWNER` | +20 | *larger* than an enemy kill — spawners gate phase progression |
| `R_PHASE_PROGRESS` | +30 | reaching a new phase (all active spawners destroyed); a top-tier event reward alongside `R_KILL_SPAWNER` |
| `R_DAMAGE_TAKEN_PER_HP` | −0.7 | per HP of contact damage |
| `R_DEATH` | −150 | one-off, on death — the "strong" §5 term (30× an enemy kill; R-REWARD-4 justified by the measured death-rate response, §3.2) |

**All three positive/progression terms are uncapped** — the agent is rewarded for
*every* enemy kill, *every* spawner kill and *every* phase advance, for the whole
episode. The intended learned behaviour is therefore: **seek and destroy spawners
to progress phases, killing or dodging enemies as needed, surviving as long as
possible against the escalating difficulty curve.** "Camp and farm enemies" is
suppressed by `R_KILL_SPAWNER` > `R_KILL_ENEMY` + `R_PHASE_PROGRESS`; "flee only"
earns nothing and never progresses a phase.

Four justified shaping terms (each targets a measured failure of an earlier
training iteration; §3.2): `R_DAMAGE_DEALT_PER_HP` +0.05/HP (dense combat
gradient), `R_TIME_STEP_PENALTY` −0.01/step (passivity has a cost),
`R_WALL_PROXIMITY_PER_STEP` −0.05/step ramped (anti wall-camp),
`R_AIMED_HIT_BONUS` +3.0 × fire-time alignment on a hit (rewards *aimed* fire, not
spray). Two further terms — `R_APPROACH_NEAREST_ENEMY`, `R_SHOOT_WHILE_NO_TARGET` —
and one structural mechanism — `PROGRESS_REWARD_EPISODE_CAP` — are retained
**inert** as documented considered-and-rejected designs (§3.3).

### 3.2 How the shaping terms were arrived at (iteration log — condense for the report)

- **2026-08-28:** `tuned_v1/v2` collapsed to corner-camping (0 kills). Added
  non-kamikaze enemies + `R_TIME_STEP_PENALTY` + `R_DAMAGE_DEALT_PER_HP`.
- **2026-09-08:** added `R_AIMED_HIT_BONUS` / `R_SHOOT_TOWARD_ENEMY` for aim;
  raised `enemy.contact_damage` to 25.
- **2026-09-08c "glass-cannon fix":** 1M `tuned_v3/v4` models learned "rush to a
  high phase, bank the reward, die ~step 700" (73 % / 47 % death). Damage cost
  raised (`R_DAMAGE_TAKEN_PER_HP` −0.5→−0.7, `contact_damage` 25→20), world kept
  hostile (`enemy_speed_gain_per_phase` 0.55→0.5, `max_concurrent` 18,
  `num_active_enemies_max` 12→18), `rotate_speed_rad` 0.14→0.2 (style-1 wall
  stall). Also added a per-episode cap on progression reward — **later reverted,
  see §3.3.**
- **2026-09-08d "less-attack fix":** the capped models dodged and sprayed instead
  of fighting; exact per-shot instrumentation (§3.4) showed real aim alignment
  ~0.3 / hit-rate ~0.1 and kills → 0 once the cap bound. `R_SHOOT_TOWARD_ENEMY`
  0.12→**0.0** (it paid per shot for *facing* the objective, hit or miss — the
  spray incentive; active shaping 5→4); `R_AIMED_HIT_BONUS` 1.0→**3.0** (an aimed
  hit now clearly out-earns a spray hit).
- **2026-09-08e "spec-alignment":** `PROGRESS_REWARD_EPISODE_CAP` disabled
  entirely — see §3.3. The cap-free 1M models cleared ~4 phases/episode (the
  spec-optimal spawner-focus, 5–9 spawner kills/ep, aim 0.73–0.90) but died
  ~50–90 % of the time — uncapped `R_PHASE_PROGRESS` = +50/phase made 4 phases
  (+200) worth 2× `R_DEATH`, so dying at phase 4 was reward-optimal.
- **2026-09-08f "magnitude tune":** `R_PHASE_PROGRESS` 50→**30** and `R_DEATH`
  −100→**−150** — both still §5-legal (a positive phase reward; a "strong"
  negative death reward), *not* a cap: every phase still pays in full. 4 phases
  = +120 < |`R_DEATH`|, so "survive and keep clearing" beats "rush to phase 4
  and die". R-REWARD-4 (30× `R_KILL_ENEMY`) is justified by the measured
  death-rate response, with `plot_death_penalty_ablation.py` available if a
  −100 vs −150 plot is wanted.
- **2026-09-09 "phase-tail softening" (env, not reward):** the 09-08f agent
  hunts spawners hard and clears ~4 phases (spec-optimal), but style 2 still
  died ~75–100 % — always at phase 4–5, where the difficulty curve (enemy
  speed 4.4–5.9 vs player 6.0, 30-step spawn floor × 3–4 spawners) is
  near-unsurvivable for *any* policy, not a policy defect. `phase_curve`:
  `min_spawn_interval_steps` 30→**45**, `enemy_speed_gain_per_phase`
  0.5→**0.4** (phase-6 enemy 4.8 < player 6.0). Paired with **target
  leading** in `core_env._graded_aim_alignment` so `R_AIMED_HIT_BONUS`
  rewards shots that connect on a kiting shooter's crossing target. No
  reward constant changed. Result (1M retrain, 25 stochastic eval eps):
  style 2 death rate 100 % → **52 %** (12/25 survive to the step cap),
  mean phase 4.4, 7.2 spawner kills/ep; style 1 death rate 56 % (11/25
  survive), mean phase 4.0, 6.2 spawner kills/ep — a balanced arcade
  profile (roughly half the episodes clear the timer at phase 4–5, half
  die at phase 4–6 after ~25 kills and ~7 spawner kills).

### 3.3 The progression cap — considered and rejected (§5 fidelity)

The 2026-09-08c cap clamped cumulative `phase_progress + kill_spawner` per episode.
Measured on the capped models the agent was destroying ~4 spawners per episode but
being *paid* for ~1.4, and phase advances past ~phase 2 paid nothing. `phase_progress`
and `kill_spawner` are **two of the five reward terms §5 requires** ("positive
reward for progressing to the next phase", "larger positive reward for destroying
spawners"); clamping a required term for most of the episode is a larger deviation
from §5 than any optional shaping and a Part II-J ("reward structure") risk. It was
removed (`PROGRESS_REWARD_EPISODE_CAP = +inf`; the clamp code and `core_env`
running-total plumbing are retained, monkeypatch-tested, for a one-line re-enable).
The glass-cannon is instead shaped only through §5's own negative terms (`R_DEATH`,
`R_DAMAGE_TAKEN_PER_HP`) plus the hostile world; an agent that aggressively clears
phases and eventually dies to the difficulty ramp is the spec-intended arcade
behaviour and a stronger phase-system demo than one camping a low phase.

### 3.4 `eval_aim_stats.py` metric bug (fixed, no retrain)

`eval_aim_stats.py` counted a shot only when `len(state.projectiles)` grew between
steps, which drops shots that collide the same step they fire — disproportionately
spray shots — so `mean_align` / `frac30` / hit-rate read ~3× too optimistic (0.87
vs the true ~0.3 for style 1). `ArenaCoreEnv` now exposes a real `_shots_fired`
counter (incremented in `_try_shoot`) and the script diffs that. Any aim number in
the report must come from the fixed script.

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
