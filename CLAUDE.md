# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

University RL assignment (RMIT, 40% assessment), two independent parts. **Both parts are now
fully implemented** — `part2_arena/` (arena core, training/eval scripts, config) and
`part1_gridworld/` (env, algorithms, trainer, intrinsic reward, logging, plotting, comparison
scripts) all have real bodies, and `pytest part1_gridworld/tests part2_arena/tests` passes 100%
with zero skips. Docstrings on already-implemented functions occasionally still say "TODO:
implement" as a leftover from the original skeleton pass — trust the code and the tests over a
stale docstring comment when the two disagree.

Two things worth knowing before extending this further:
- `part1_gridworld/logs/` (per-run CSV episode logs) and `part1_gridworld/models/` (saved
  Q-tables, JSON) are real generated evidence, not just scaffolding — `trainer.train()` writes to
  both by default. Re-running `compare_algorithms.run_comparison()` /
  `compare_q_vs_sarsa.run_comparison()` regenerates the report-figure PNGs in `report/figures/`.
- Part II's `models/`/`logs/` currently only hold a 3000-timestep PPO smoke run per control style
  (`--style 1`/`--style 2`, seed 0) — enough to confirm `train.py → models/ → eval_style1.py` /
  `eval_style2.py` genuinely connects (both eval scripts run and produce sane, non-crashing
  rollouts; per-term TensorBoard logging under `reward_terms/*` is confirmed populating). The real
  100k–600k-timestep training runs the report/video need are still outstanding — don't mistake
  these smoke-test artifacts for finished evidence, and re-run `train.py` at full `--timesteps`
  before final submission (this will overwrite `models/style{1,2}_ppo_tuned_v1*`).

`docs/RUBRIC_MAP.md` maps each module/function to the exact rubric row and point value it
satisfies. Its **"Pre-implementation fixes applied"** section lists spec-fidelity decisions baked
into the codebase that must not be silently reverted — notably: agent moving onto a monster tile =
death (kept in sync across `config/schema.md` and `env.py`), the `QTable`
`q_table[state][action]`-only interface, the intrinsic-reward visit-order convention, the
`[-1, 1]` observation normalization, and `SHOT_NO_TARGET_RADIUS` in `rewards_config.py`.

`docs/AUDIT_main.md` is a full evidence-based audit of this branch's scaffold against the spec and
rubric (its "open design decisions" were §5, most now resolved — see below) — still worth reading
for the reasoning behind decisions baked into the current code.

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

Part II (deep RL arena) — run from `part2_arena/`:
```
python scripts/train.py --style 1 --algo ppo --timesteps 300000 [--curriculum on] [--config tuned_v1]
python scripts/eval_style1.py [--algo ppo] [--episodes 5]
python scripts/eval_style2.py
python scripts/compare_ppo_dqn.py              # creativity hook (a)
python scripts/plot_reward_decomposition.py    # creativity hook (b), reads per-term TensorBoard scalars
tensorboard --logdir logs
```

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
  scripts use this layer. `observation_space` is `Box(-1, 1, (OBS_DIM,), float32)`.
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
  `rewards_config.py` holds all constants (5 spec-required + 2 justified shaping terms, each with
  a one-line justification docstring): `R_APPROACH_NEAREST_ENEMY` (kept, per-episode capped at
  `R_KILL_ENEMY`, gated to outside engage range) and `R_SHOOT_WHILE_NO_TARGET` (kept disabled at
  `0.0` — revisit only if training runs show shot-spam hurting performance). `SHOT_NO_TARGET_RADIUS
  = 350.0` defines "shot fired with no target" for that term.
- **`arena/phases.py`** — `PhaseManager`: advances the difficulty phase when all active spawners
  are destroyed (awarding `R_PHASE_PROGRESS`). `curriculum_enabled` (creativity hook c) makes
  early phases easier, ramping to the normal curve over `curriculum.enabled_ramp_phases`; toggled
  by `train.py --curriculum`. Both the base difficulty curve and the ramp schedule are driven by
  `config/arena.json`'s `phase_curve`/`curriculum` blocks.
- **`config/arena.json`** — single source of truth for arena/player/enemy/spawner/phase-curve/
  curriculum/observation-normalization constants (mirrors Part I's `training_config.json`
  pattern). `max_steps` is `1200` here (reduced from the `3000` module fallback specifically so a
  ~300k-timestep run sees enough full episodes to learn — see `docs/AUDIT_main.md` 5.4).
- **`config/hyperparams.json`** — named PPO/DQN presets (`baseline`, `tuned_v1`, ...) that
  `scripts/train.py --config <name>` loads via `build_model()`; add new presets rather than
  editing existing ones so past runs stay reproducible.
- **`scripts/train.py`** — one model per `--style`; `--algo {ppo,dqn}` (hook a),
  `--curriculum {on,off}` (hook c), `--config` (hyperparameter preset, hook toward meaningfully
  tuned hyperparameters). Seeds via SB3's `set_random_seed(args.seed)` directly (Part II has no
  dedicated seed-utils module, unlike Part I). Models → `models/`, TensorBoard logs → `logs/`.
- **`scripts/eval_style1.py` / `eval_style2.py`** — deliberately standalone (no shared
  `--style` flag) because the rubric asks for a separate eval script per control style. Load the
  saved model, play live with `render_mode="human"`, `deterministic=True`.

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
