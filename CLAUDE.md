# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

University RL assignment (RMIT, 40% assessment), two independent parts. **Both parts are now
fully implemented** — `part2_arena/` (arena core, training/eval scripts, config) and
`part1_gridworld/` (env, algorithms, trainer, intrinsic reward, logging, plotting, comparison
scripts) all have real bodies, and `pytest part1_gridworld/tests part2_arena/tests` passes 100%
with zero skips. Docstrings on already-implemented functions occasionally still say "TODO:
implement" as a leftover from the original skeleton pass — trust the code and the tests over a
stale docstring comment when the two disagree. The root `README.md` was rewritten and is now
broadly accurate, but it references `docs/PART2_TUNING_GUIDE.md`, which does not exist — the
tuning notes live in `docs/PLAN_FIX_PART2.md` and the per-team `docs/*.md` logs instead.

Two things worth knowing before extending this further:
- `part1_gridworld/logs/` (per-run CSV episode logs) and `part1_gridworld/models/` (saved
  Q-tables, JSON) are real generated evidence, not just scaffolding — `trainer.train()` writes to
  both by default. Re-running `compare_algorithms` / `compare_q_vs_sarsa` /
  `compare_monster_levels` / `compare_intrinsic_reward` (`part1_gridworld/src/`) regenerates the
  report-figure PNGs in `report/figures/`.
- Part II's `models/`/`logs/` hold **real 1M-timestep PPO runs**, not smoke tests. As of the
  2026-09-08c "glass-cannon fix" (`FIX_PART2.md`) the shipped checkpoints are
  **`style1_ppo_tuned_v3_curriculum`** and **`style2_ppo_tuned_v3_curriculum`** (`.zip` +
  `_best/`) — both `--config tuned_v3` (γ 0.995). `tuned_v4` (γ 0.999) is retired: it
  amplified a "rush to a high phase, then die" policy. Pre-fix checkpoints are frozen under
  `models/pre_glasscannonfix_2026-09-08c/` (and the earlier `models/pre_aimfix_backup_2026-09-08/`)
  for the report's before/after; older `tuned_v1`/`tuned_v2`, DQN, and `--death-penalty`
  ablations also remain in `models/`. Retraining overwrites the same-named file.

`docs/RUBRIC_MAP.md` maps each module/function to the exact rubric row and point value it
satisfies. Its **"Pre-implementation fixes applied"** section lists spec-fidelity decisions baked
into the codebase that must not be silently reverted — notably: agent moving onto a monster tile =
death (kept in sync across `config/schema.md` and `env.py`), the `QTable`
`q_table[state][action]`-only interface, the intrinsic-reward visit-order convention, the
`[-1, 1]` observation normalization, and `SHOT_NO_TARGET_RADIUS` in `rewards_config.py`.

`docs/AUDIT_main.md` is a full evidence-based audit of this branch's scaffold against the spec and
rubric (its "open design decisions" were §5, most now resolved — see below) — still worth reading
for the reasoning behind decisions baked into the current code.

`docs/RULES.md` is **mandatory pre-flight reading before editing any reward logic, tabular update
rule, the intrinsic tracker, the observation vector, or any eval/comparison script.** Every rule
(`R-ALG-*`, `R-REWARD-*`, `R-OBS-*`, `R-EVAL-*`, …) maps to a real correctness bug or a graded
"weak evidence" criticism from a previous submission of this same assignment. Its §7 is a
pre-submission grep-based self-audit; cite rule IDs in commit messages when a change complies with
one.

`GAIT-ASM3/` is an untracked nested clone of this same repo — ignore it; work only in the
top-level tree.

### Design decisions reviewed against lesson.md (docs/lesson.md)

`docs/lesson.md` is feedback from a *previous* team's assignment (same style of RL project) —
treated as guidelines, not new requirements. The two decisions most likely to be second-guessed:

- `REWARD_DEATH = 0.0` (`config/rewards_constants.py`) is **intentionally kept at 0.0, not a gap**.
  The spec's Part I reward list has no death term and explicitly says "rewards and mechanics must
  not be altered" — adding a penalty here would be exactly the unjustified spec deviation
  lesson.md warns against. Task 2's required SARSA-vs-Q-learning contrast on `level1` still
  emerges without one: death forfeits every uncollected apple (an emergent, structural cost), and
  SARSA's on-policy target already incorporates the exploring policy's real chance of stepping
  into the fire gap where Q-learning's off-policy max discounts that risk. See the constant's own
  comment for the full reasoning — do not change this value without a stronger justification than
  "it would make the comparison plot look cleaner."
- Part II's `R_APPROACH_NEAREST_ENEMY` (`arena/rewards_config.py`, kept at `0.01`, capped/gated) is
  a genuine, logged disagreement between team members: Member C's contribution log recommended
  removing it entirely for simplicity (closer to lesson.md's "avoid unnecessary shaping" advice);
  the shipped code keeps it, capped per-episode and gated outside engage range, with a written
  justification, and three tests assert its behavior. Both positions are defensible and spec-legal
  ("optional shaping rewards must be justified" — satisfied either way); this is a team judgment
  call, not a bug — leave it as-is unless the team decides otherwise.

Everything else flagged as an open design decision in earlier passes over this codebase (arena
config file, `max_steps` vs timestep budget, the projectile/observation inconsistency, the
TensorBoard reward-decomposition callback, `report/figures/` + root `.gitignore`, and — as of this
pass — the Part I `trainer.py`/`save_qtable`/`load_qtable`/comparison-script stubs) is resolved in
the current code.

## Commands

Run from the repo root unless noted. There is no build step.

```
pip install -r requirements.txt          # both parts; or use each part's own requirements.txt

ruff check .                             # lint (pyproject.toml: src = both packages, line-length 100, rules E/F/I, py311)
ruff check --fix .

pytest part1_gridworld/tests part2_arena/tests   # full suite — 100% pass, 0 skips
pytest part1_gridworld/tests/test_algorithms.py::test_linear_epsilon_decay_endpoints_and_linearity   # single test
pytest -k monster                        # filter by name
```

Part I (classical RL gridworld):
```
cd part1_gridworld && python main.py      # interactive Pygame menu -> train live or watch a saved policy
```

Part II (deep RL arena) — run from `part2_arena/`. `train.py` requires `--style` and `--algo`;
defaults are `--curriculum on --config tuned_v3 --timesteps 300000 --seed 0`:
```
python scripts/train.py --style 1 --algo ppo --config tuned_v3 --curriculum on --timesteps 1000000   # shipped style-1 run
python scripts/train.py --style 2 --algo ppo --config tuned_v3 --curriculum on --timesteps 1000000   # shipped style-2 run
python scripts/train.py --style 1 --algo ppo --death-penalty -30 [--wall-penalty 0]  # ablation overrides
python scripts/eval_style1.py [--algo ppo] [--episodes 5] [--sampling stochastic] [--checkpoint best]
python scripts/eval_style2.py
python scripts/eval_aim_stats.py --style 2 --config tuned_v3 --curriculum on   # headless aim/wall-hug metrics
python scripts/compare_styles.py               # REQUIRED rubric item: two-control-scheme comparison
python scripts/compare_ppo_dqn.py              # creativity hook (a)
python scripts/plot_reward_decomposition.py    # creativity hook (b), reads per-term TensorBoard scalars
python scripts/quantify_curriculum.py          # creativity hook (c) — curriculum on vs off, a real number
python scripts/plot_death_penalty_ablation.py  # R_DEATH magnitude ablation for the report
python scripts/print_env_io_example.py         # literal reset()/step() I/O dump for the report (rubric II-H)
tensorboard --logdir logs
```
`--death-penalty` runs are written with an `ablation_` prefix; `--wall-penalty` runs get a
`_wall<N>` suffix and are treated as main runs. Ablation flags override the corresponding
`rewards_config.py` constant for that run only.

Keep the report's reward tables in sync after changing any reward constant:
```
python scripts/generate_report_tables.py      # -> report/figures/reward_tables.md
```

### Import path quirks

- **Part II** uses clean package imports (`from arena.core_env import ...`) and expects to run
  with `part2_arena/` as CWD.
- **Part I** is inconsistent: `main.py` uses `from src.trainer import ...` (needs
  `part1_gridworld/` on the path) but modules inside `src/` import each other bare
  (`from algorithms import ...`, `from env import ...` — marked with `# pyrefly: ignore
  [missing-import]` in `trainer.py`, so the pattern is intentional). Bare imports need
  `part1_gridworld/src/` on the path; running `main.py` needs both.
- `pyproject.toml`'s `[tool.pytest.ini_options]` already puts `part1_gridworld`, `part2_arena`,
  and `part2_arena/scripts` on `pythonpath` (no `conftest.py` needed) so both parts' implicit
  namespace packages — and `part2_arena/scripts`' bare `from callbacks import ...` — collect
  correctly from the repo root.

## Architecture

### Part I — `part1_gridworld/`

Strict separation so the core is headless and unit-testable:

- **`src/env.py`** — `GridWorldEnv`, pure logic, **no pygame import**. Loads `config/levelN.json`
  (schema + invariants in `config/schema.md`; `_load_level` must raise `ValueError` naming the
  file and problem, not bare-assert). `step()` returns a `StepResult` and follows a fixed
  **8-step** order: (1) move agent (rocks/edges block) → (2) fire death → (3) agent moved onto a
  monster's tile = death → (4) pickups → (5) win check → (6) each monster moves with prob 0.4 in
  a random unblocked direction → (7) monster moved onto agent = death → (8) step counter /
  truncate. Both occupancy-collision directions (3 and 7) are equally valid death paths.
  `_resolve_monster_moves()` is a separate method (seeded-RNG testable); a fully boxed-in monster
  silently skips its move.
- **`src/algorithms.py`** — tabular updates only, independent of `env.py`/`render.py`. `QTable`
  is a thin `defaultdict` wrapper with **one** access pattern: `q_table[state]` returns the
  action-value list, then index by action (`.values(state)` was removed). `q_learning_update`
  (off-policy, `max`), `sarsa_update` (on-policy, actual `next_action`), `expected_sarsa_update`
  (creativity hook d, epsilon-greedy expectation), `epsilon_greedy` (random tie-breaking), and
  `linear_epsilon_decay` are all implemented and tested. `save_qtable`/`load_qtable` (keyed by
  `qtable_path(level_id, algorithm)`) serialize to JSON (`{"n_actions", "entries": [[state_repr,
  q_values], ...]}`, non-default entries only) — `main.py`'s watch-only path and every comparison
  script depend on this round-tripping exactly.
- **`src/trainer.py`** — the **only** training loop in the codebase. Wires env + algorithms +
  optional `intrinsic.py` + optional `render.py` + `logger.py`. `train(level_id, algorithm, ...)`
  dispatches on `algorithm` ∈ `{"q_learning","sarsa","expected_sarsa"}`; for SARSA/Expected-SARSA
  the next action is chosen before the update, every step. Intrinsic bonus is added to the update
  reward **only** — the episode return logged via `EpisodeLogger` is always env-only, defaulting
  to `part1_gridworld/logs/level{N}_{algorithm}[_intrinsic].csv` when `csv_log_path` isn't given.
  `train()` does **not** save the Q-table itself — callers (`main.py`) do that via
  `algorithms.save_qtable`. `load_training_config` merges `default` + `level_overrides` and also
  copies the JSON's top-level `intrinsic_reward_strength` into the returned dict (a real
  integration gap found and fixed while wiring `trainer.py` to `intrinsic.py` — the two were
  merged from different branches and didn't originally agree on this key).
- **`src/intrinsic.py`** — `IntrinsicRewardTracker`, Task 5 / Level 6. Formula
  `strength / sqrt(n(s) + 1)` (pre-visit count) ≡ `strength / sqrt(n(s))` (post-increment) —
  same number. Convention on this branch: call `visit_and_get_bonus(state)` with the **current
  state `s` BEFORE `env.step`**, incrementing the count first (first visit → `strength/sqrt(1)`).
  `reset_episode()` every episode — skipping it silently makes it a whole-run novelty bonus
  (spec violation).
- **`config/rewards_constants.py`** — single source of truth for Part I reward values
  (`REWARD_APPLE=1`, `REWARD_KEY=0`, `REWARD_CHEST=2`, `REWARD_DEATH=0`, `REWARD_STEP=0`).
  Everything computing a reward imports these; do not hardcode. Adding/changing a term requires
  updating `config/schema.md`, `tests/test_env_rules.py`, and the report.
- **`config/training_config.json`** — `default` block + `level_overrides` merged by `level_id`.
  `render.py` / `menu.py` are the Pygame layer; they read env state, never mutate it. `menu.py`
  existing (in-window level/algorithm selection) is itself part of the "interactive, visually
  rendered" rubric requirement.
- **`src/sprites.py`** — optional PNG sprite loader (`load_sprite(name, size)`), reading from
  `assets/sprites/<name>.png`. Every `render.py` draw call keeps its original `pygame.draw.*` shape
  as the fallback when a sprite is missing/invalid (`load_sprite` never raises) — dropping a
  correctly-named PNG into an empty `assets/sprites/` swaps that one shape with no code change.
  Deliberately duplicated (not shared) with Part II's `arena/sprites.py` so the two parts stay
  independently runnable/gradable.

Levels 0–6 map to tasks: 0=Q-learning, 1=SARSA, 2–3=key/chest, 4–5=monsters, 6=intrinsic reward
(see `config/schema.md` table). `level1` and `level6` carry `_design_note` fields explaining the
controlled comparison each enables.

### Part II — `part2_arena/`

Real-time Pygame arena controlled by an SB3 agent. Fully implemented. Deliberate two-layer env
design — **do not merge these**:

- **`arena/core_env.py`** — `ArenaCoreEnv`, the literal spec API: `reset() -> obs`,
  `step(action) -> (obs, reward, done, info)` (legacy 4-tuple), `render()`. **Zero dependency on
  gymnasium / SB3.** This is what satisfies the "Gym-style API" rubric row on its own terms.
  `info` carries `reward_breakdown`, `died`, and `truncated`. Fixed step order documented in the
  `step()` docstring. Module constants `ARENA_WIDTH=960`/`ARENA_HEIGHT=680`/`DEFAULT_MAX_STEPS=3000`
  are fallbacks only — the authoritative values load from `config/arena.json` (see below).
- **`arena/gym_adapter.py`** — `ArenaGymEnv(gym.Env)`, thin protocol translation **only** (no
  game logic). Exists purely because SB3 needs Gymnasium's 5-tuple (`terminated`/`truncated`
  split); derives those from `info["died"]` / `info["truncated"]`. `train.py` and both eval
  scripts use this layer. `observation_space` is `Box(-1, 1, (OBS_DIM,), float32)`. `render()`
  pumps the renderer's event queue and returns a bool (`False` = window closed), matching Part I's
  `GridWorldRenderer.handle_events()` convention; `is_paused`/`speed_multiplier`/
  `consume_restart_request()`/`consume_skip_request()`/`show_episode_end_banner()` are eval-only
  passthroughs to the lazily-created `ArenaRenderer` — they live here, never on `ArenaCoreEnv`, so
  the spec-compliant core stays pure.
- **`arena/sprites.py`** — Part II's copy of Part I's `src/sprites.py` sprite loader (same
  contract: `load_sprite(name, size)`, `assets/sprites/<name>.png`, never raises, callers keep
  their `pygame.draw.*` fallback). Kept as a separate duplicate file, not a shared import.
- **`arena/entities.py`** — plain dataclasses (`Player`, `Enemy`, `Spawner`, `Projectile`,
  `ArenaState`). No pygame/gym/SB3. `core_env.py` owns and mutates `ArenaState`;
  `render_pygame.py` only reads it. Enemies deal contact damage only (no enemy fire) — `Projectile`
  is player-shot-only, which is why `obs.py` has no incoming-projectile feature (see below).
- **`arena/obs.py`** — `build_observation()` produces a fixed-size float32 vector in the exact
  order of `OBSERVATION_SPEC` (15 features). **Every** feature normalized to `[-1, 1]` via
  `x_norm = 2*x_unit - 1` (sin/cos already in range); style-2 orientation features are `0`/`1`.
  No pixels. Keep `OBSERVATION_SPEC` in sync with report section 2 and `test_obs_shape.py`.
- **`arena/actions.py`** — `ControlStyle1` (rotation+thrust, 5 actions) and `ControlStyle2`
  (direct directional, 6 actions) `IntEnum`s. Single source of action ordering shared by
  training and eval — a mismatch silently produces nonsense play. `action_enum_for_style(style)`
  raises on anything but 1/2.
- **`arena/rewards.py`** + **`arena/rewards_config.py`** — `rewards.py` is the **only** place a
  Part II reward is computed; `core_env.step()` calls `compute_reward(step_events)` and uses the
  returned total, never inline math. `RewardBreakdown` names every term separately, and
  `scripts/callbacks.py::RewardTermLoggingCallback` (an SB3 `BaseCallback` wired into
  `train.py`) logs each to TensorBoard as `reward_terms/<name>` for
  `scripts/plot_reward_decomposition.py` (creativity hook b) to read back via `tbparse`.
  `rewards_config.py` holds all constants and **is the source of truth** — 5 spec-required +
  **4 ACTIVE** shaping terms (`R_DAMAGE_DEALT_PER_HP`, `R_TIME_STEP_PENALTY`,
  `R_WALL_PROXIMITY_PER_STEP`, `R_AIMED_HIT_BONUS = 3.0`), plus `R_SHOOT_TOWARD_ENEMY`
  (retired to `0.0` on 2026-09-08d — paid for facing, not hitting → spray incentive),
  `R_APPROACH_NEAREST_ENEMY` / `R_SHOOT_WHILE_NO_TARGET` inert at `0.0`. Each active term
  carries a one-line justification + anti-farming note (`docs/RULES.md` R-REWARD-3; still a
  Part II-J point — see the module docstring). **All three spec-required positive terms
  (`R_KILL_ENEMY` / `R_KILL_SPAWNER` / `R_PHASE_PROGRESS`) are uncapped** — the intended
  policy is "destroy spawners → progress phases → survive", not farm-enemies or flee.
  `rewards.py` also holds `APPROACH_REWARD_EPISODE_CAP` (live but inert, term is 0.0) and
  **`PROGRESS_REWARD_EPISODE_CAP` (disabled, `+inf`)** — the latter clamped `kill_spawner +
  phase_progress` (2026-09-08c) but was removed 2026-09-08e because those are spec-required
  §5 terms; its clamp code + `core_env._cumulative_progress_reward` plumbing are retained,
  monkeypatch-tested, for a one-line re-enable (`FIX_PART2.md` §3.3). `core_env` also
  exposes `_shots_fired` (real per-episode shot counter for `eval_aim_stats.py`, which
  otherwise under-counts spray shots).
- **`arena/phases.py`** — `PhaseManager`: advances the difficulty phase when all active spawners
  are destroyed (awarding `R_PHASE_PROGRESS`). `curriculum_enabled` (creativity hook c) makes
  early phases easier, ramping to the normal curve over `curriculum.enabled_ramp_phases`; toggled
  by `train.py --curriculum`. Both the base difficulty curve and the ramp schedule are driven by
  `config/arena.json`'s `phase_curve`/`curriculum` blocks.
- **`config/arena.json`** — single source of truth for arena/player/enemy/spawner/phase-curve/
  curriculum/observation-normalization constants (mirrors Part I's `training_config.json`
  pattern). `max_steps` is `1200` here (reduced from the `3000` module fallback specifically so a
  ~300k-timestep run sees enough full episodes to learn — see `docs/AUDIT_main.md` 5.4).
- **`config/hyperparams.json`** — named PPO presets (`baseline`, `tuned_v1`..`tuned_v4`) and DQN
  presets (`baseline`, `tuned_v1`) that `scripts/train.py --config <name>` loads via
  `build_model()`. **Both control styles ship on `tuned_v3` (γ 0.995)**; `tuned_v4` (= `tuned_v3`
  with γ 0.999) is retired but kept for reproducing the pre-2026-09-08c style-2 model. Add new
  presets rather than editing existing ones so past runs stay reproducible.
- **`scripts/train.py`** — one model per `--style`; `--algo {ppo,dqn}` (hook a),
  `--curriculum {on,off}` (hook c, **defaults `on`**), `--config` (hyperparameter preset,
  **defaults `tuned_v3`**), plus `--death-penalty` / `--wall-penalty` ablation overrides. Seeds
  via SB3's `set_random_seed(args.seed)` directly (Part II has no dedicated seed-utils module,
  unlike Part I). Models → `models/`, TensorBoard logs → `logs/`.
- **`scripts/eval_style1.py` / `eval_style2.py`** — deliberately standalone (no shared
  `--style` flag) because the rubric asks for a separate eval script per control style. Load the
  saved model, play live with `render_mode="human"`. `--sampling` **defaults to `stochastic`**
  (shows the full learned behaviour incl. phase progression); `deterministic` argmax can collapse
  to a degenerate wall/spray mode, so use stochastic for the demo video. `--checkpoint best` uses
  the EvalCallback's best-by-reward save. Their render loop
  checks `env.render()`'s returned bool (window-closed signal), respects `env.is_paused`
  (Space), reads `env.speed_multiplier` (`[`/`]`) for `clock.tick(fps * mult)`, and honors
  `env.consume_restart_request()`/`consume_skip_request()` (R/N) — all plumbed through
  `ArenaGymEnv` from `arena/render_pygame.py::ArenaRenderer.handle_events()`.
- **`arena/render_pygame.py`** — loads its HUD font from the bundled
  `assets/fonts/ShareTechMono-Regular.ttf` (OFL-licensed; see `assets/README.md`) instead of
  `pygame.font.SysFont`, so the HUD/debug overlay renders identically across operating systems.
  `TimedBanner` generalizes the existing kill-flash/damage-tint timed-effect pattern into one
  reusable on-screen message, used for the phase-transition banner (triggered in
  `_ingest_effects()` when `state.phase` increases) and the episode-end banner
  (`show_episode_end_banner()`, called by the eval scripts before the next `reset()`).

### Cross-part

`scripts/generate_report_tables.py` (repo root) imports **both** `rewards_constants` modules
directly and regenerates `report/figures/reward_tables.md` — never hand-copy reward values into
the report. `report/report_template.md` is the report skeleton (strict 10-page limit, no
appendix); `docs/` holds the authoritative spec (`assessment_requirements_summary.md`), rubric
map, submission checklist, video script, and `CONTRIBUTIONS.md` (student numbers still TODO).

### Reproducibility

`part1_gridworld/src/seed_utils.py::set_seed(seed)` seeds `random` + NumPy and returns a
dedicated `random.Random` for callers that want an explicit RNG (`epsilon_greedy`,
`_resolve_monster_moves`). Call it once at the start of any training/eval script so report
figures and the demo video are reproducible. Part II has no equivalent module — `scripts/train.py`
takes `--seed` and calls SB3's `set_random_seed(args.seed)` directly at startup instead.
