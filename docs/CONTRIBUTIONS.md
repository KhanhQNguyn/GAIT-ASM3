# Contributions

Fill this in as work happens, not just at the end — it gets copy-pasted into
the report (required section: "student numbers and a contribution summary for
all team members").

| Student Number | Name | Contribution Summary |
|---|---|---|
| TODO | TODO | TODO |
| TODO | TODO | TODO |
| TODO | Member C | Part II arena core: entities, actions, physics, observation vector, phase/curriculum system, `ArenaCoreEnv` (literal 4-tuple Gym API), `ArenaGymEnv` (SB3 wrapper), Pygame renderer with hit/kill FX + debug overlay. `rewards.compute_reward` body (to unblock; Member D owns the constants). Tests: `test_env_api.py`, `test_obs_shape.py`, `test_phases.py` (25 passing). Report sections 1.2 and 2 draft. |
| TODO | TODO | TODO |

## Detailed log (optional, useful for resolving disputes / writing the summary above)

- YYYY-MM-DD — who — what (e.g. "implemented Q-learning update rule + tests").
- 2026-08-26 — Member C — Implemented all 8 Part II arena modules + gym adapter + renderer.
  Decisions: enemies deal contact damage and do NOT shoot (OBSERVATION_SPEC stays at 15,
  no projectile feature — reverts the scaffold pass's 18); `arena.json` is the single
  Part II param file; `step_events` is a locked 7-key dict duplicated verbatim in
  `core_env.step()` and `rewards.compute_reward()`. Recommended to Member D:
  `R_APPROACH_NEAREST_ENEMY = 0` and `R_SHOOT_WHILE_NO_TARGET = 0` (keep only the 5
  spec reward terms); reconsider/justify `R_DEATH = -100`. Verified in a venv:
  `pytest part2_arena/tests` = 25 passed / 4 skipped (skips are Member D's
  `test_reward_terms.py`); `ruff check part2_arena/arena part2_arena/tests` clean;
  headless smoke + gymnasium `check_env` pass for both control styles; renderer smoke
  passes under the SDL dummy driver.
