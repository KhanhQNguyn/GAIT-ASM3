# Rubric Map

Every module/function below is mapped to the exact rubric row it satisfies
(point values from the assignment's 40-point breakdown table). Use this to
self-grade before submitting — if a row has no evidence yet, it's not done.

| Rubric Row | Points | Satisfied By | Status |
|---|---|---|---|
| Part I-A — Gridworld implementation & rules (visual/animated/interactive; mechanics match spec) | 2 | `part1_gridworld/src/env.py`, `render.py`, `menu.py` | TODO |
| Part I-B — Task 1: Q-learning (epsilon-greedy, correct update, linear decay, tie-breaking, shortest-path evidence) | 2.5 | `part1_gridworld/src/algorithms.py` (`epsilon_greedy`, `q_learning_update`), `config/level0.json`, `config/training_config.json`, `tests/test_algorithms.py`, `tests/test_tie_breaking.py` | TODO |
| Part I-C — Task 2: SARSA (on-policy update, same exploration schedule, comparison vs Q-learning) | 3 | `part1_gridworld/src/algorithms.py` (`sarsa_update`), `config/level1.json`, `src/compare_q_vs_sarsa.py` (required Q-vs-SARSA plot) | TODO |
| Part I-D — Task 3: Levels 2-3 (multiple apples, key, chest; correct termination/reward accounting) | 3 | `config/level2.json`, `config/level3.json`, `src/env.py`, `tests/test_env_rules.py` | TODO |
| Part I (Task 4, folded into I-D/general) — Monster levels 4-5 (stochastic movement, avoidance learning, training curves) | (see note below) | `config/level4.json`, `config/level5.json`, `src/env.py` (monster movement), `tests/test_monster_stochastic.py`, `src/plot_results.py` | TODO |
| Part I-F — Task 5: Intrinsic reward (correct formula, unchanged env rewards, per-episode visit counter, curve comparison + explanation) | 3 | `src/intrinsic.py`, `config/level6.json`, `tests/test_intrinsic_reward.py`, `src/plot_results.py` | TODO |
| Part II-G — Arena environment requirements (real-time, animated; core gameplay; health/phase systems; episode-end conditions) | 4.5 | `part2_arena/arena/core_env.py`, `entities.py`, `physics.py`, `phases.py`, `render_pygame.py`, `config/arena.json` (env + phase-curve params) | TODO |
| Part II-H — Gym-style API + observation design (reset/step/render; fixed-size vector with required features) | 2.5 | `arena/core_env.py` (literal 4-tuple API), `arena/gym_adapter.py` (SB3 5-tuple), `arena/obs.py`, `tests/test_env_api.py`, `tests/test_obs_shape.py` | TODO |
| Part II-I — Two control schemes + models (both control styles; separate trained/saved models; separate evaluation scripts) | 4 | `arena/actions.py`, `scripts/train.py --style {1,2}`, `scripts/eval_style1.py`, `scripts/eval_style2.py`, `models/` | TODO |
| Part II-J — Reward design & deep RL training quality (reward structure; SB3 DQN/PPO + TensorBoard; meaningful hyperparameter tuning) | 3 | `arena/rewards_config.py`, `arena/rewards.py`, `scripts/train.py`, `config/hyperparams.json` (preset sweep), `scripts/callbacks.py` (`RewardTermLoggingCallback`), `tests/test_reward_terms.py`, `logs/` (TensorBoard) | TODO |
| Report — page limit, environment/observation/reward descriptions, hyperparameter exploration, control-set comparison, originality justification | 2.5 | `report/report_template.md`, `report/figures/`, `scripts/generate_report_tables.py` | TODO |
| Video Demo — <=10 min, all members present, gridworld + arena shown, learned-policy evidence, both control schemes shown | 5 | `VIDEO_SCRIPT.md` | TODO |
| Creativity — going beyond expected requirements | 5 | (a) `part2_arena/scripts/compare_ppo_dqn.py`; (b) `part2_arena/scripts/plot_reward_decomposition.py`; (c) `part2_arena/arena/phases.py` curriculum hook + `train.py --curriculum`; (d) `part1_gridworld/src/algorithms.py::expected_sarsa_update` + `src/compare_algorithms.py` | TODO |

**Note on Task 4 (Monster Levels):** the spec's rubric table doesn't give
monster levels a separately labeled row — its evidence (monster behavior,
stochastic transition handling, training curves) is treated here as folded
into Part I-D and the general Part I-A implementation quality. Confirm with
the instructor if this is graded separately.

## How to use this file

Update the `Status` column as each piece lands (`TODO` -> `IN PROGRESS` ->
`DONE`, with a short note or commit reference). Before submission, every row
must be `DONE` with a pointer to the evidence (script output, plot, test
pass) referenced in the report.

---

## Pre-implementation fixes applied

The following documentation/spec-fidelity fixes were applied to the skeleton
**before any algorithm or environment logic was written**, so they are in
place from the first implementation commit. None of these changes implement
`NotImplementedError` function bodies.

1. **Monster-tile-entry spec gap** (`config/schema.md`, `src/env.py`,
   `tests/test_monster_stochastic.py`) — added explicit invariant that the
   agent moving onto a monster's tile is immediate death (independent of
   monster movement); removed the self-contradictory "should not happen"
   comment from `step()`'s docstring; added
   `test_agent_moving_onto_monster_causes_death` stub.

2. **`_load_level` TODO made concrete** (`src/env.py`, `src/trainer.py`) —
   replaced vague "json.load, validate" TODO with a full itemised checklist
   of required keys, bounds checks, and overlap checks, with ValueError
   requirements; `load_training_config` likewise documents alpha/gamma/
   epsilon/episodes validation requirements.

3. **QTable interface collapsed to `__getitem__`** (`src/algorithms.py`) —
   removed the duplicate `values(state)` method; class docstring now states
   `q_table[state][action]` as the only access pattern; all callers must use
   `q_table[state]` instead of `.values(`.

4. **Intrinsic-reward visit-order decided** (`src/intrinsic.py`) — documented
   that `visit_and_get_bonus` increments the count first then returns
   `strength / sqrt(n(s))` (post-visit count); usage example updated to
   reflect the call timing (current state `s`, before `env.step`).

5. **Observation-space bounds finalised** (`arena/obs.py`,
   `arena/gym_adapter.py`, `tests/test_obs_shape.py`) — chose `[-1, 1]` as
   the universal normalization convention; updated every `OBSERVATION_SPEC`
   description to state the target range; marked the `Box` bounds as final
   (not a placeholder); updated the test TODO to assert value range.

6. **Boxed-in-monster case documented** (`src/env.py`) — added one sentence
   to `_resolve_monster_moves`: a monster with zero unblocked directions
   skips its move silently (no crash on empty choices list).

7. **Lint config added** (`pyproject.toml` at repo root) — ruff configured
   for Python 3.11+, line length 100, `E/F/I` rule sets covering both
   `part1_gridworld` and `part2_arena`.

8. **This section** (`docs/RUBRIC_MAP.md`) — added for teammate visibility.

9. **Round-2 follow-ups** (`src/intrinsic.py`, `arena/rewards_config.py`,
   `arena/rewards.py`) — (a) rewrote `visit_and_get_bonus` docstring to
   state plainly that post-increment `n(s)` and `(pre-visit n(s) + 1)` are
   algebraically identical on every visit, not just the first (removes a
   misleading implication that implementers need to choose between two
   diverging conventions); (b) added `SHOT_NO_TARGET_RADIUS: float = 150.0`
   to `rewards_config.py` with a tuning TODO, and updated `rewards.py`'s
   import block and docstring to reference it (closes the "constant must be
   added" loose thread flagged in round 1).

**Keep the paired copies in sync.** The monster-tile-entry death rule (fix
1) is now stated in three places that must agree: `config/schema.md`'s
"Invariants" list, `src/env.py`'s `GridWorldEnv` class docstring, and
`src/env.py`'s `step()` order-of-operations docstring (steps 3 and 7). Do
not edit one of them independently — a change to the rule must be applied to
all three at once, or the contradiction this fix removed comes straight
back. The same applies to the `[-1, 1]` observation convention (fix 5),
which is duplicated across `arena/obs.py`, `arena/gym_adapter.py`, and
`tests/test_obs_shape.py`.

---

## Round-3 scaffold fixes (from docs/AUDIT_main.md)

Structural gaps and deferred design decisions closed at scaffold level --
no `NotImplementedError` bodies implemented, no reward-constant *values*
changed, no level tile-layouts changed. Cross-reference: `docs/AUDIT_main.md`.

1. **Q-table state encoding DECIDED** (`src/env.py` `GridWorldEnv` docstring)
   — `(agent_x, agent_y, apples_bitmask, has_key, chest_open, monsters)`,
   with feasibility note (raise per-level `episodes`, don't drop `monsters`).
   (AUDIT 5.2)
2. **Q-table save/load** (`src/algorithms.py`) — added `qtable_path()`,
   `save_qtable()`, `load_qtable()` stubs + `MODELS_DIR`; created
   `part1_gridworld/models/`. `main.py` and `menu.py` docstrings now name
   the path convention; `trainer.train()` notes callers persist the result.
   (AUDIT 5.3)
3. **`python main.py` import shim** (`part1_gridworld/main.py`) — adds this
   dir and `src/` to `sys.path` so the mixed `from src.x` / bare `from x`
   imports resolve regardless of CWD. (AUDIT 8.3)
4. **pytest can collect** (`pyproject.toml`) — added
   `[tool.pytest.ini_options]` with `pythonpath` (4 roots) + `testpaths`,
   and a minimal `[project]` table. (AUDIT 8.2)
5. **Part II config directory** — added `part2_arena/config/arena.json`
   (arena size, `max_steps`, player/enemy/spawner params, phase curve,
   curriculum ramp, `num_active_enemies_max`) and
   `part2_arena/config/hyperparams.json` (PPO/DQN `baseline` + `tuned_v1`
   presets). `core_env.py` / `phases.py` / `obs.py` / `train.py` docstrings
   now point at them. (AUDIT 5.6, 7.3, 7.4)
6. **Arena episode length** (`arena/core_env.py`) — `DEFAULT_MAX_STEPS`
   3000 -> 1200 (fallback; authoritative value in `arena.json`), with a
   comment on the timestep-budget reasoning. (AUDIT 5.4)
7. **Projectile / observation consistency** (`arena/obs.py`, `core_env.py`)
   — resolved AUDIT 5.5 the *simple* way during Member C implementation:
   enemies do NOT fire projectiles (they deal contact damage and die on
   touch), so there is no "incoming projectile" observation feature and
   OBSERVATION_SPEC stays at the spec-minimum **15**. Only the player
   shoots; projectile-vs-enemy / projectile-vs-spawner collisions still
   satisfy the rubric's "projectile collisions" requirement. This also
   keeps the observation from encoding a strategy (docs/message.txt).
   (The scaffold pass had briefly added 3 projectile features -> 18; that
   was reverted.)
8. **Reward-term TensorBoard logging** — added
   `part2_arena/scripts/callbacks.py::RewardTermLoggingCallback` stub;
   `train.py` docstring wires it into `model.learn(callback=...)`; added
   `tbparse` to both requirements files. This is what gives creativity
   hook (b) its data source. (AUDIT 5.7)
9. **Part II seeding** — added `part2_arena/scripts/seed_utils.py::set_seed`
   stub (SB3 + env + spaces); `train.py` / `eval_style*.py` docstrings call
   it. (AUDIT 7.2)
10. **Required Task-2 comparison script** —
    `part1_gridworld/src/compare_q_vs_sarsa.py` stub (Q-learning vs SARSA
    on level1). `compare_algorithms.py` stays the creativity(d) 3-algo
    version. (AUDIT 3, 6.7)
11. **Level solvability tests** — `part1_gridworld/tests/test_level_configs.py`
    stubs (validation + BFS reachability, per level). `_load_level`
    docstring notes reachability lives there, not in the load path.
    (AUDIT 6.2, 6.3)
12. **Death-rate plot** (`src/plot_results.py`) — added `plot_death_rate()`
    stub (rolling mean of the `died` column) for Task 4 avoidance evidence.
    (AUDIT 6.8)
13. **`make_env(level_id)` helper** (`src/trainer.py`) — one place that
    builds a `GridWorldEnv` from a level id, so `train()`, `evaluate_policy`
    callers, and the comparison scripts construct it identically. (AUDIT 6.5)
14. **Monster-vs-monster occupancy DECIDED** (`config/schema.md`,
    `src/env.py::_resolve_monster_moves`) — monsters don't block each other
    and may share a tile; only rocks/edges block. (AUDIT 6.4)
15. **Curriculum ramp DECIDED** (`arena/phases.py::difficulty_for_phase`
    docstring) — base curve formulas + a `frac = s0 + (1-s0)*(phase/R)`
    ramp that converges to the base curve by phase `R`. Params in
    `arena.json`. (AUDIT 7.3)
16. **`SHOT_NO_TARGET_RADIUS` rationale corrected** (`arena/rewards_config.py`)
    — comment now reasons from the real 960×680 arena (diag ≈ 1173), value
    unchanged. `R_APPROACH_NEAREST_ENEMY` docstring now specifies the
    mandatory gating (delta-only, outside-range, per-episode cap); value
    unchanged. (AUDIT 7.1, 5.8)
17. **`.gitignore` + report/figures/** — added `.gitignore` (keeps
    `models/`, `logs/`, generated Markdown tables tracked; ignores
    byte-code, caches, generated plot images, CSV logs); created
    `report/figures/.gitkeep` and `part1_gridworld/logs/.gitkeep`.
    (AUDIT 8.5, 9.1)
18. **eval frame pacing** (`scripts/eval_style1.py`, `eval_style2.py`) —
    added `--fps` (and `--config`) args; `main()` docstrings call for a
    `pygame.time.Clock().tick(args.fps)` in the render loop. (AUDIT 5.9)
19. **`--config` preset selection** (`scripts/train.py`) — `--config`
    arg; `build_model` loads `[algo][preset]` from `hyperparams.json`;
    preset name is baked into the model filename and TB run name so a sweep
    is self-documenting. (AUDIT 5.6)
20. **Report template notes** (`report/report_template.md`) — section 2
    flags the projectile features; section 3 flags the `REWARD_DEATH` and
    `R_APPROACH_NEAREST_ENEMY` decisions still owed.

Still open (need real implementation or a value/layout decision, out of
scope for a scaffold pass): every `NotImplementedError` body; the
`REWARD_DEATH` value question (5.1); pinning dependency versions (8.4);
filling `docs/CONTRIBUTIONS.md` student numbers (9.2); running `ruff check .`
and `pytest` once in a real venv (8.7).

