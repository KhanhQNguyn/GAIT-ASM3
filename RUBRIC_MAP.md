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
