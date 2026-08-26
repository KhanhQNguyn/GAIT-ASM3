# Rubric Map

Every module/function below is mapped to the exact rubric row it satisfies
(point values from the assignment's 40-point breakdown table). Use this to
self-grade before submitting — if a row has no evidence yet, it's not done.

| Rubric Row | Points | Satisfied By | Status |
|---|---|---|---|
| Part I-A — Gridworld implementation & rules (visual/animated/interactive; mechanics match spec) | 2 | `part1_gridworld/src/env.py`, `render.py`, `menu.py` | TODO |
| Part I-B — Task 1: Q-learning (epsilon-greedy, correct update, linear decay, tie-breaking, shortest-path evidence) | 2.5 | `part1_gridworld/src/algorithms.py` (`epsilon_greedy`, `q_learning_update`), `config/level0.json`, `config/training_config.json`, `tests/test_algorithms.py`, `tests/test_tie_breaking.py` | TODO |
| Part I-C — Task 2: SARSA (on-policy update, same exploration schedule, comparison vs Q-learning) | 3 | `part1_gridworld/src/algorithms.py` (`sarsa_update`), `config/level1.json`, `src/compare_algorithms.py` | TODO |
| Part I-D — Task 3: Levels 2-3 (multiple apples, key, chest; correct termination/reward accounting) | 3 | `config/level2.json`, `config/level3.json`, `src/env.py`, `tests/test_env_rules.py` | TODO |
| Part I (Task 4, folded into I-D/general) — Monster levels 4-5 (stochastic movement, avoidance learning, training curves) | (see note below) | `config/level4.json`, `config/level5.json`, `src/env.py` (monster movement), `tests/test_monster_stochastic.py`, `src/plot_results.py` | TODO |
| Part I-F — Task 5: Intrinsic reward (correct formula, unchanged env rewards, per-episode visit counter, curve comparison + explanation) | 3 | `src/intrinsic.py`, `config/level6.json`, `tests/test_intrinsic_reward.py`, `src/plot_results.py` | TODO |
| Part II-G — Arena environment requirements (real-time, animated; core gameplay; health/phase systems; episode-end conditions) | 4.5 | `part2_arena/arena/core_env.py`, `entities.py`, `physics.py`, `phases.py`, `render_pygame.py` | TODO |
| Part II-H — Gym-style API + observation design (reset/step/render; fixed-size vector with required features) | 2.5 | `arena/core_env.py` (literal 4-tuple API), `arena/gym_adapter.py` (SB3 5-tuple), `arena/obs.py`, `tests/test_env_api.py`, `tests/test_obs_shape.py` | TODO |
| Part II-I — Two control schemes + models (both control styles; separate trained/saved models; separate evaluation scripts) | 4 | `arena/actions.py`, `scripts/train.py --style {1,2}`, `scripts/eval_style1.py`, `scripts/eval_style2.py`, `models/` | TODO |
| Part II-J — Reward design & deep RL training quality (reward structure; SB3 DQN/PPO + TensorBoard; meaningful hyperparameter tuning) | 3 | `arena/rewards_config.py`, `arena/rewards.py`, `scripts/train.py`, `tests/test_reward_terms.py`, `logs/` (TensorBoard) | TODO |
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

