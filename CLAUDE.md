# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

University RL assignment (RMIT, 40% assessment), two independent parts. **Everything is a
skeleton**: nearly every function body is `raise NotImplementedError` with a detailed docstring
spelling out the required behaviour, and every test is `pytest.skip("TODO: ...")`. The docstrings
and `docs/` are the spec — implement to them; do not invent alternative designs.

`docs/RUBRIC_MAP.md` maps each module/function to the exact rubric row and point value it
satisfies. Its **"Pre-implementation fixes applied"** section (9 items) lists spec-fidelity
decisions already baked into the skeleton that must not be silently reverted — notably: agent
moving onto a monster tile = death (kept in sync across `config/schema.md` and `env.py`), the
`QTable` `q_table[state][action]`-only interface, the intrinsic-reward visit-order convention,
the `[-1, 1]` observation normalization, and `SHOT_NO_TARGET_RADIUS` in `rewards_config.py`.

`docs/AUDIT_main.md` is a full evidence-based audit of this branch's scaffold against the spec and
rubric — read it before implementing; the "open design decisions" below are its §5.

`GAIT-ASM3/` is an untracked nested clone of this same repo — ignore it; work only in the
top-level tree.

### Open design decisions that gate rubric marks

Deferred inside docstrings on this branch; each has a wrong-by-default failure mode:

- `REWARD_DEATH = 0.0` (`config/rewards_constants.py`) gives neither algorithm a reason to avoid
  hazards — undermines the required SARSA-vs-Q-learning conservatism comparison (`level1`) and
  monster avoidance. Likely needs a negative value, documented in report §3.
- Q-table **state encoding is undecided** in `env.py`'s class docstring, and tabular feasibility
  for levels 4–6 (stochastic monsters, large state space) is unverified. Pick an encoding and
  sanity-check reachable-state counts before generating curves; shrink levels if needed.
- **No Q-table save/load exists**, but `main.py` / `menu.py` `watch_only` need it for the video's
  "learned policy, not random" evidence.
- `part2_arena/arena/core_env.py` references `CONFIG_DIR = .../config` but **`part2_arena/config/`
  does not exist**. There is no arena/hyperparameter config file (Part I has `training_config.json`).
- Arena `DEFAULT_MAX_STEPS = 3000` (`core_env.py`) vs a ~300k-timestep budget ≈ only ~100
  episodes — too few to learn. Decide `max_steps` and the timestep budget together.
- `entities.py` has an enemy `Projectile` path but `obs.py`'s `OBSERVATION_SPEC` has **no
  projectile feature** — the agent cannot perceive enemy fire. Resolve the inconsistency.
- `plot_reward_decomposition.py` (creativity hook b) reads TensorBoard `reward_terms/*` scalars
  that nothing logs yet — needs an SB3 `BaseCallback` wired into `train.py`, plus `tbparse` in
  requirements.
- `report/figures/` does not exist yet (every `plot_*.py` writes there); there is no `.gitignore`.

## Commands

Run from the repo root unless noted. There is no build step.

```
pip install -r requirements.txt          # both parts; or use each part's own requirements.txt

ruff check .                             # lint (pyproject.toml: src = both packages, line-length 100, rules E/F/I, py311)
ruff check --fix .

pytest part1_gridworld/tests part2_arena/tests   # full suite (all currently skipped)
pytest part1_gridworld/tests/test_algorithms.py::test_linear_epsilon_decay_endpoints_and_linearity   # single test
pytest -k monster                        # filter by name
```

Part I (classical RL gridworld):
```
cd part1_gridworld && python main.py      # interactive Pygame menu -> train live or watch a saved policy
```

Part II (deep RL arena) — run from `part2_arena/`:
```
python scripts/train.py --style 1 --algo ppo --timesteps 300000 [--curriculum on]
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
  `part1_gridworld/src/` on the path; running `main.py` needs both. There is no `conftest.py`
  and `pyproject.toml` has no pytest config — expect to add `pythonpath` handling when wiring
  the tests up.

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
  (creativity hook d, epsilon-greedy expectation). `epsilon_greedy` must do **random
  tie-breaking**; `linear_epsilon_decay` must be linear, not exponential.
- **`src/trainer.py`** — the **only** training loop in the codebase. Wires env + algorithms +
  optional `intrinsic.py` + optional `render.py` + `logger.py`. `train(level_id, algorithm, ...)`
  dispatches on `algorithm` ∈ `{"q_learning","sarsa","expected_sarsa"}`. For SARSA/Expected-SARSA
  the next action is chosen before the update. Intrinsic bonus is added to the update reward
  **only** — never mutate env reward; the episode return logged is env-only for comparability.
  `load_training_config` merges `default` + `level_overrides` and must validate
  alpha/gamma ∈ (0,1], `epsilon_end ≤ epsilon_start` ∈ [0,1], `episodes > 0`.
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

Real-time Pygame arena controlled by an SB3 agent. Deliberate two-layer env design — **do not
merge these**:

- **`arena/core_env.py`** — `ArenaCoreEnv`, the literal spec API: `reset() -> obs`,
  `step(action) -> (obs, reward, done, info)` (legacy 4-tuple), `render()`. **Zero dependency on
  gymnasium / SB3.** This is what satisfies the "Gym-style API" rubric row on its own terms.
  `info` must carry `reward_breakdown`, `died`, and `truncated`. Fixed step order documented in
  the `step()` docstring. Module constants: `ARENA_WIDTH=960`, `ARENA_HEIGHT=680`,
  `DEFAULT_MAX_STEPS=3000`.
- **`arena/gym_adapter.py`** — `ArenaGymEnv(gym.Env)`, thin protocol translation **only** (no
  game logic). Exists purely because SB3 needs Gymnasium's 5-tuple (`terminated`/`truncated`
  split); derives those from `info["died"]` / `info["truncated"]`. `train.py` and both eval
  scripts use this layer. `observation_space` is `Box(-1, 1, (OBS_DIM,), float32)`.
- **`arena/entities.py`** — plain dataclasses (`Player`, `Enemy`, `Spawner`, `Projectile`,
  `ArenaState`). No pygame/gym/SB3. `core_env.py` owns and mutates `ArenaState`;
  `render_pygame.py` only reads it.
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
  returned total, never inline math. `RewardBreakdown` names every term separately (for the
  creativity-b TensorBoard decomposition). `rewards_config.py` holds all constants (≤ 8 terms:
  5 spec-required + ≤ 2 justified shaping terms, each with a one-line justification docstring),
  plus `SHOT_NO_TARGET_RADIUS=150.0` (tuning TODO) that defines "shot fired with no target" for
  the `R_SHOOT_WHILE_NO_TARGET` term.
- **`arena/phases.py`** — `PhaseManager`: advances the difficulty phase when all active spawners
  are destroyed (awarding `R_PHASE_PROGRESS`). `curriculum_enabled` (creativity hook c) makes
  early phases easier, ramping to the normal curve; toggled by `train.py --curriculum`. The base
  difficulty curve and the ramp schedule are both undefined TODOs.
- **`scripts/train.py`** — one model per `--style`; `--algo {ppo,dqn}` (hook a),
  `--curriculum {on,off}` (hook c). `build_model()` must use **meaningfully tuned**
  hyperparameters (rubric requirement), documented in the report. Models → `models/`,
  TensorBoard logs → `logs/`.
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
figures and the demo video are reproducible. Part II has no equivalent helper yet — `train.py`
takes `--seed` but nothing wires SB3 `set_random_seed` / `env.reset(seed=)` / action-space
seeding.
