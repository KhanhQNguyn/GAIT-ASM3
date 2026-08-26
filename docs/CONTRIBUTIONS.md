# Contributions

Fill this in as work happens, not just at the end — it gets copy-pasted into
the report (required section: "student numbers and a contribution summary for
all team members").

| Student Number | Name | Contribution Summary |
|---|---|---|
| TODO | TODO | TODO |
| TODO | TODO | TODO |

## Detailed log (optional, useful for resolving disputes / writing the summary above)

- YYYY-MM-DD — who — what (e.g. "implemented Q-learning update rule + tests").
- 2026-08-26 — Member D — implemented the Part II reward system (`rewards_config.py`
  decisions + `rewards.py::compute_reward`, all 4 tests in `test_reward_terms.py` passing),
  `scripts/train.py` (tuned PPO/DQN `build_model()`, `main()`, `RewardBreakdownCallback`,
  `EvalCallback` best-checkpoint saving), `eval_style1.py`/`eval_style2.py`, and the two
  creativity scripts (`compare_ppo_dqn.py`, `plot_reward_decomposition.py`). All code verified
  against a `gymnasium.make("CartPole-v1")` stand-in env and synthetic/real TensorBoard scalar
  data. Real training runs / saved models / logs are still blocked on Member C's
  `arena/actions.py`, `core_env.py`, `gym_adapter.py`, `obs.py`, `phases.py`, `physics.py`,
  `entities.py` (all still `NotImplementedError`). Work landed across 5 branches:
  `feat/arena-rewards`, `feat/arena-train-script`, `feat/arena-eval-scripts`,
  `feat/arena-creativity-ablation`, `feat/arena-docs-update`.
