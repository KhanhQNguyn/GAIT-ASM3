# Audit — branch `main`

Evidence-based review against `docs/assessment_requirements_summary.md` (40 marks) and
`docs/RUBRIC_MAP.md`. Every file under `part1_gridworld/` and `part2_arena/` was read, plus all
configs, tests, docs, and `report/`. Commands were attempted (see Appendix) — the active Python
interpreter has **none of the project dependencies installed**, so `ruff`, `pytest`, and import
smoke tests could not run. All findings below are from static reading.

Reviewed at: `git rev-parse HEAD` = `24dccdd` ("change"), working tree clean except untracked
`CLAUDE.md`.

---

## Update — scaffold pass applied

A scaffold-only follow-up pass (no `NotImplementedError` bodies, no reward-constant *value*
changes, no level-layout changes) addressed most of the structural gaps and deferred design
decisions below. See `docs/RUBRIC_MAP.md` "Round-3 scaffold fixes" for the itemised list.

**Resolved at scaffold level:** §5.2 (state encoding decided), §5.3 (save/load stubs +
`models/`), §5.4 (`DEFAULT_MAX_STEPS` 3000→1200 + config), §5.5 (3 projectile obs features
added), §5.6 (`part2_arena/config/arena.json` + `hyperparams.json` + `--config`), §5.7
(`RewardTermLoggingCallback` stub + `tbparse`), §5.9 (`--fps` + pacing note), §6.2/§6.3
(`test_level_configs.py` stubs), §6.4 (monster occupancy decided), §6.5 (`make_env`), §6.7
(`compare_q_vs_sarsa.py`), §6.8 (`plot_death_rate` stub), §7.1 (`SHOT_NO_TARGET_RADIUS`
rationale), §7.2 (`scripts/seed_utils.py` stub), §7.3 (curriculum ramp decided), §7.4
(`NUM_ACTIVE_ENEMIES_MAX`), §8.2 (pytest `pythonpath`), §8.3 (`main.py` shim), §8.5
(`.gitignore`), §9.1 (`report/figures/.gitkeep`).

**Partially addressed (documented, not resolved):** §5.1 `REWARD_DEATH` — constraint forbade
changing the value; the decision is now flagged in `report_template.md` §3 and
`rewards_constants.py`. §5.8 `R_APPROACH_NEAREST_ENEMY` — value unchanged; the mandatory gating
design is now written into its docstring.

**Still fully open (need implementation or a value decision):** every `NotImplementedError`
body (§3, §4 tables, §6.1); §8.1 tests still skip; §8.4 dependency pinning; §8.7 run
`ruff`/`pytest` in a real venv; §9.2 `CONTRIBUTIONS.md` student numbers; §9.3 report page budget.

---

## 1. Status snapshot

| Dimension | State |
|---|---|
| Design / architecture | ~65% — structure complete; several load-bearing decisions still `TODO` in docstrings |
| Implementation | **~2%** — only trivial members have bodies: `QTable` (`algorithms.py:13`), `RewardBreakdown` (`rewards.py:25`), `PhaseConfig` + `all_spawners_destroyed` (`phases.py:13,52`), `clamp`/`distance` (`physics.py:12,33`), `Action`/`ControlStyle*` enums, `MenuSelection` (`menu.py:17`), `GridWorldEnv.action_space_n` (`env.py:180`), `IntrinsicRewardTracker.visit_count` (`intrinsic.py:77`), all `parse_args` / `model_save_path`. Every other function body is `raise NotImplementedError`. |
| Tests | **31 test functions, 0 executable** — every one is `pytest.skip("TODO...")`. (part1: 19, part2: 12.) |
| Trained models / logs / figures | none — `part2_arena/models/` and `part2_arena/logs/` hold only `.gitkeep`; `report/figures/` **does not exist**; `part1_gridworld/` has no models dir at all |
| Report / contributions | `report/report_template.md` is a section skeleton; `docs/CONTRIBUTIONS.md` is entirely `TODO` incl. student numbers |

"How is it going": the **scaffold** is well-structured and the docstrings faithfully encode spec
detail. But it is still only a scaffold, and ~7 design decisions are deferred inside docstrings
in a state where the obvious implementation choice produces weak rubric evidence (Section 5).

---

## 2. What's solid

- **Headless-core / render-only separation**, both parts: `env.py` has no `pygame` import
  (`env.py:1-7`); `render.py` / `render_pygame.py` are draw-only (`render.py:1-5`,
  `render_pygame.py:1-4`). Same split in Part II (`core_env.py` vs `render_pygame.py`).
- **Single reward computation site** per part: `env.py` imports the constants
  (`env.py:16-22`); `arena/rewards.py` is "the ONLY place a Part II reward value is computed"
  (`rewards.py:1-7`), constants isolated in `rewards_config.py`. `generate_report_tables.py`
  reads both to keep the report in sync.
- **Two-layer Gym API** with written rationale: `core_env.py:1-19` (literal 4-tuple) vs
  `gym_adapter.py:1-15` (SB3 5-tuple). Correct and deliberate.
- **`RUBRIC_MAP.md` traceability** — every rubric row mapped to files; the "Pre-implementation
  fixes applied" section (9 items) is all verifiably in place on this branch, no regressions
  (checked each against the cited files).
- **Deliberate level→task mapping** (`config/schema.md:52-62`); `level1.json` and `level6.json`
  carry `_design_note` fields explaining the controlled comparison each enables.
- Four creativity hooks each scoped to a runnable artefact (`compare_ppo_dqn.py`,
  `plot_reward_decomposition.py`, `phases.py` curriculum, `expected_sarsa_update`).

---

## 3. Requirement coverage (spec clause → verdict → evidence / gap)

Legend: **SCAFFOLD** = file/API present, body not implemented (expected at this stage);
**DESIGNED** = decision recorded in a docstring/config; **MISSING** = no file/dir/mechanism;
**RISK** = present but a plausible implementation yields weak evidence (see §5).

### Part I

| Clause (spec §) | Verdict | Evidence / gap |
|---|---|---|
| Visually rendered in Pygame, interactive, animated; no console display | SCAFFOLD | `main.py:19` `raise NotImplementedError`; `menu.py:28`, `render.py:30` (constructor itself raises) all unimplemented |
| Multiple levels, different layouts | DONE (data) | `config/level0..6.json` all present, distinct layouts; bounds look in-range (not machine-verified — see §6) |
| Rewards/mechanics must not be altered | DESIGNED | `rewards_constants.py:16-36`; invariants `schema.md:32-50`; `env.py:71-88` |
| Movement up/down/left/right | DONE (data) | `Action` enum `env.py:27-33`; `ACTION_DELTAS` `env.py:36-41` |
| Rocks block; fire/monster = instant death; apple +1; key 0; chest +2; end on all-collected or death; monsters 40% move | DESIGNED, SCAFFOLD | fully specified `schema.md:32-50`, `env.py:136-159` step-order docstring; **no body** |
| **Task 1** Q-learning: ε-greedy, off-policy max update, linear ε decay, random tie-break, shortest-path evidence | SCAFFOLD + RISK | `algorithms.py:46,59,33` all `NotImplementedError`; state encoding undecided (`env.py:65`) → convergence/feasibility unverified (§5.2) |
| **Task 2** SARSA on-policy, same schedule, comparison vs Q-learning showing conservatism | SCAFFOLD + RISK | `sarsa_update` `algorithms.py:80`; **no script runs the required L1 Q-vs-SARSA comparison** — `compare_algorithms.py:18` is hard-wired to `COMPARISON_LEVEL_ID = 4` and is the 3-algo creativity plot. And `REWARD_DEATH = 0.0` removes the incentive the comparison depends on (§5.1) |
| **Task 3** Levels 2–3 (multi-apple, key, chest), correct termination/accounting | SCAFFOLD | `env.py` step/reset unimplemented; `test_env_rules.py` 7 tests all skipped |
| **Task 4** Monsters: stochastic 40% move, dies both collision directions, avoidance learning, training curves L4/L5 | SCAFFOLD + RISK | `_resolve_monster_moves` `env.py:161`; monster-vs-monster occupancy unspecified (§6.4); `plot_results.py:28` unimplemented; no death-rate metric |
| **Task 5** Intrinsic reward: exact formula, env reward unchanged, per-episode counter, on/off curve + explanation | SCAFFOLD (well-specified) | `intrinsic.py:44,54` unimplemented but formula + call convention + reset semantics fully pinned (`intrinsic.py:7-11,28-36,54-71`); `level6.json` isolates the variable |

### Part II

| Clause (spec §) | Verdict | Evidence / gap |
|---|---|---|
| Player ship move + shoot; enemy spawners; enemies navigate to player; health systems; projectile collisions; phase system; real-time (not tile); episode ends on death or max-steps | SCAFFOLD | `core_env.py:53-92` reset/step `NotImplementedError`; step-order + phase logic specified `core_env.py:63-92`, `phases.py:39-64`; `physics.py` helpers 4/6 unimplemented |
| Gym API: `reset()→obs`, `step()→(obs,reward,done,info)`, `render()` | SCAFFOLD (correct shape) | `core_env.py:53,63,94`; literal 4-tuple documented `core_env.py:1-19` |
| Observation: fixed-size numeric vector incl. pos, vel, orientation, nearest-enemy dist+dir, nearest-spawner dist+dir, health, phase; no pixels | DESIGNED + RISK | `OBSERVATION_SPEC` 15 features `obs.py:19-35` covers every named minimum; **but** enemy projectiles exist (`entities.py:65`) with no observation feature (§5.5) |
| Two control schemes, each its own trained model + own eval script | SCAFFOLD | `ControlStyle1/2` `actions.py:12-30`; `train.py --style {1,2}`; `eval_style1.py` / `eval_style2.py` present, both `main()` unimplemented |
| Reward: +enemy, ++spawner, +phase, −damage, −−death, shaping justified | DESIGNED | `rewards_config.py:15-52`; `R_KILL_SPAWNER(20) > R_KILL_ENEMY(5)`, `R_DEATH −100`; 2 shaping terms flagged "TODO justify or remove" (`rewards_config.py:39-52`) |
| SB3 DQN or PPO; MLP ≥1 hidden layer; TensorBoard logging; meaningful hyperparameter tuning; save to `models/`; eval script plays visually | SCAFFOLD + RISK | `train.py:34-53` `build_model` unimplemented (tuning is a docstring TODO); no hyperparameter config file, no sweep structure (§5.6); episode length vs timestep budget (§5.4) |

### Report / Video / Submission

| Clause | Verdict | Evidence / gap |
|---|---|---|
| Report ≤10 pages incl. images, no appendix | SCAFFOLD | `report/report_template.md:1-9` states the constraint; 9 sections budgeted ~1 page each — tight once figures land (§9) |
| Both env descriptions / observation design / reward design / hyperparameter exploration / control comparison / training evidence / originality | SCAFFOLD | all are `TODO` headers in the template |
| Student numbers + contribution summary | **MISSING** | `docs/CONTRIBUTIONS.md` table is `TODO / TODO / TODO` |
| Video link | MISSING (expected) | `report_template.md:71`, `VIDEO_SCRIPT.md` unfilled |
| Zip: gridworld + arena code + training scripts + **saved models** + **TensorBoard logs** | RISK | models/logs are `.gitkeep` only — nothing to submit yet; no `.gitignore` to keep the zip clean |
| Video must show learned-policy evidence (not random) for Part I | RISK | depends on watch-a-saved-policy flow that does not exist (§5.3) |

---

## 4. Rubric-row readiness

| Row | Pts | Ready? | Biggest single risk | Artefact a marker must see |
|---|---:|---|---|---|
| I-A Gridworld impl & rules | 2 | scaffold, 0% logic | no QTable save/load for the "learned policy" demo (§5.3) | Pygame window, animated, interactive menu; mechanics matching `schema.md` |
| I-B Q-learning | 2.5 | scaffold, 0% logic | state-space size vs 2000 episodes unverified (§5.2); tests skipped | `test_algorithms.py` green + a greedy run tracing a near-optimal apple route on `level0` |
| I-C SARSA + comparison | 3 | scaffold + **RISK** | `REWARD_DEATH=0` kills the conservatism signal (§5.1); no L1 comparison script (§3) | side-by-side L1 curves + a screenshot/trace where SARSA avoids the fire gap and Q-learning does not |
| I-D Levels 2–3 | 3 | scaffold, 0% logic | reachability of key/chest never validated (§6.3); accounting untested | `test_env_rules.py` green; a greedy episode collecting all apples + opening the chest |
| I Task 4 monsters (folded) | — | scaffold + RISK | monster-vs-monster occupancy unspecified (§6.4); no avoidance metric | `test_monster_stochastic.py` green; L4/L5 training curves; death-rate falling over episodes |
| I-F Intrinsic reward | 3 | scaffold, well-specified | cleanest row — mostly a matter of implementing to the docstring | `test_intrinsic_reward.py` green; L6 with/without curves + written explanation |
| II-G Arena environment | 4.5 | scaffold, 0% logic | projectile/observation inconsistency (§5.5); `max_steps` vs learnability (§5.4) | real-time arena video with spawns, collisions, ≥1 phase transition |
| II-H Gym API + observation | 2.5 | scaffold (shape correct) | `test_env_api.py` / `test_obs_shape.py` skipped; `gymnasium` env-checker never run | both test files green incl. `check_env`; observation-feature table in report |
| II-I Two control schemes + models | 4 | scaffold | too few episodes to learn in the timestep budget (§5.4); no Part II seeding (§7) | two saved models (`models/style1_*`, `models/style2_*`) + two eval videos of learned play |
| II-J Reward design + training quality | 3 | scaffold + RISK | no hyperparameter-exploration structure (§5.6); shaping term un-tuned (§5.7) | TensorBoard curves; a hyperparameter table with ≥2 configs compared |
| Report | 2.5 | template only | `report/figures/` missing; 10-page budget vs figure count (§9) | the PDF, all sections filled, figures generated by the scripts |
| Video Demo | 5 | script only | Part I learned-policy evidence depends on §5.3; eval frame pacing (§5.8) | ≤10 min, all members, both parts, both control schemes, learned behaviour |
| Creativity | 5 | 4 hooks scaffolded | hook (b) has no data source — nothing logs `reward_terms/*` (§5.7); `tbparse` missing (§7) | one artefact per hook (ablation plot, decomposition chart, curriculum on/off curves, Expected-SARSA curve) |

---

## 5. Critical design holes, ranked by points at risk

### 5.1 `REWARD_DEATH = 0.0` undermines the SARSA-vs-Q-learning comparison — I-C (3), Task 4 (folded)

`config/rewards_constants.py:31` sets `REWARD_DEATH: float = 0.0`, defended as "strictly literal
to the spec". `level1.json`'s `_design_note` expects SARSA to learn a more conservative route
around the 2-tile fire gap than Q-learning. With **no death penalty**, dying only costs the
remaining apples' opportunity value — the on-policy/off-policy difference the rubric wants
demonstrated will be marginal. Same weakening applies to "agent should learn to avoid monsters".

**Fix:** set a negative `REWARD_DEATH` (e.g. `-1.0`), regenerate `report/figures/reward_tables.md`,
and justify it in report §3. The constant's own docstring (`rewards_constants.py:26-31`) already
anticipates this as an allowed choice.

### 5.2 Q-table state encoding undecided + tabular feasibility unverified — I-B (2.5), I-D (3), Task 4

`env.py:65-69` leaves the state tuple as a `TODO` with a candidate including
`tuple(monster_positions)`. For `level4`/`level6`: agent (100) × apple subsets (2³) ×
2 monster positions (~100²) ≈ millions of states, explored stochastically, against only
`episodes: 2000`–`3000` (`training_config.json`). Nobody has estimated reachable-state counts or
confirmed tabular methods converge to the "shortest-path" / "avoidance" policies the rubric asks
for. Leaving monsters out of the state instead breaks the Markov assumption Task 4 relies on.

**Fix:** commit to an encoding in the docstring; do a reachable-state estimate per level; if L4–L6
are infeasible, shrink the grid or reduce apples/monsters (`schema.md:5-6` allows small grids).

### 5.3 No Q-table serialisation — Video (5), I-A (2)

`main.py:22-27` and `menu.py:20-25` (`watch_only`) both assume "load a saved QTable for
(level_id, algorithm)". **No save/load function exists** in `algorithms.py` or anywhere, no path
convention, and there is no `part1_gridworld/models/` directory. Training thousands of episodes
live on camera is not viable, so the "learned policy, not random" video evidence needs this path.

**Fix:** add `save_qtable` / `load_qtable` to `algorithms.py`, have `train()` write to
`part1_gridworld/models/level{N}_{algo}.json`, create the dir, wire `main.py`.

### 5.4 Arena `DEFAULT_MAX_STEPS = 3000` vs ~300k-timestep budget → ~100 episodes — II-I (4), II-J (3), II-G (4.5)

`core_env.py:35` sets `DEFAULT_MAX_STEPS = 3000`; `train.py:29` defaults `--timesteps 300_000`.
That is on the order of 100 episode terminations across a whole run — far too few for PPO/DQN to
learn survival + phase progression. The feasibility guide's 100k–600k budget assumes much shorter
episodes.

**Fix:** decide `max_steps` (~500–1000) and the timestep budget together, and/or use
`SubprocVecEnv` with N parallel envs; record the decision in the (missing) Part II config.

### 5.5 Enemy projectiles modelled but not observable — II-G (4.5), II-H (2.5)

`entities.py:65` — `Projectile.owner: str  # "player" or "enemy"`; `core_env.py:82-84` lists a
`projectile-player` collision case. But `OBSERVATION_SPEC` (`obs.py:19-35`) has **no projectile
feature**, so the agent cannot perceive incoming fire and cannot learn to dodge it. The design is
internally inconsistent.

**Fix:** either enemies do not shoot (remove the enemy-projectile path and the projectile-player
case), or add `nearest_incoming_projectile_distance` + `_direction_sin/_cos` to `OBSERVATION_SPEC`
and report §2.

### 5.6 No Part II config + no hyperparameter-sweep structure — Report §4, II-J (3)

`core_env.py:31` and `gym_adapter.py` reference `CONFIG_DIR = .../config`, but
**`part2_arena/config/` does not exist** (verified). Arena size, `max_steps`, the phase difficulty
curve (`phases.py:39` `NotImplementedError`), and tuning values are scattered as module constants
or docstring TODOs. `train.py:34-53` hardcodes "tuned" hyperparameters inline with no mechanism to
try several and record the effect — which is exactly what rubric §4 grades.

**Fix:** add `part2_arena/config/arena.json` (env + phase params) and
`config/hyperparams.json` (per-algo presets); `train.py --config <name>` stamps the preset into
the model filename and TB run name. Mirror Part I's `training_config.json`.

### 5.7 Reward-term TensorBoard scalars are never logged → creativity hook (b) has no data

`plot_reward_decomposition.py:23-31` reads tags `reward_terms/kill_enemy` etc. SB3 does not log
`info`-dict contents automatically, and `train.py`'s docstring (`train.py:61-72`) never mentions a
callback. Without a `BaseCallback` that reads `info["reward_breakdown"]` each step and writes the
scalars, the decomposition chart is empty.

**Fix:** add a `RewardTermLoggingCallback`, pass it to `model.learn(callback=...)`, add `tbparse`
to `requirements.txt` (also needed by `compare_ppo_dqn.py:42`).

### 5.8 `R_APPROACH_NEAREST_ENEMY = 0.01`/step is not small — II-J (3)

`rewards_config.py:39` — at `max_steps` 3000 that is +30/episode, vs `R_KILL_ENEMY = 5`; even at
`max_steps` 1000 it is +10. The docstring itself warns it "must be small relative to
R_KILL_ENEMY". High reward-hacking risk (loiter near an enemy, never shoot).

**Fix:** gate it (reward only the approach *delta* while outside weapon range) or drop it an order
of magnitude; document the final call in report §3.

### 5.9 Eval loops have no frame pacing — Video (5)

`eval_style1.py:29-40` / `eval_style2.py` call `env.render()` per step but nothing paces the loop
(`pygame.time.Clock().tick(...)`), so recorded playback will be unwatchably fast.

**Fix:** add a clock to the eval loop.

---

## 6. Part I gaps (beyond §5)

1. **`_load_level` implements none of its documented validation.** `env.py:96-123` specifies a
   5-step checklist (required keys, in-bounds coords, no tile overlaps, `ValueError` naming the
   problem); `env.py:124-125` is just `json.load`. Until implemented, a malformed level fails as a
   late `KeyError`/`IndexError` mid-training, exactly what the docstring says to avoid.
2. **No reachability/solvability check anywhere.** Even the documented `_load_level` checklist only
   covers bounds + overlaps, not "every apple / key / chest reachable from `agent_start` given the
   rocks". A typo could make a level silently unlearnable. Add a BFS check + one test per level.
3. **`level3.json` not obviously solvable.** Agent `[9,0]`, key `[7,3]`, chest `[1,8]`, with rock
   walls at `x=2` rows 0–4 and `x=6` rows 5–8, fire `[5,2]`. Looks passable via rows 9 / the `y>4`
   gap but has not been verified.
4. **Monster-vs-monster occupancy is unspecified.** `schema.md:49-50` and
   `_resolve_monster_moves` (`env.py:161-178`) only say rocks/edges block monsters. `level5.json`
   starts two monsters at `[3,4]` and `[3,5]` — the only 2 free tiles in a full-height `x=3` wall.
   Whether they can stack, or block each other, changes their mobility and the Task 4 stochastic
   demonstration. Decide and document.
5. **`evaluate_policy(env, q_table, ...)` vs `train()` env ownership.** `trainer.py:81` takes an
   env; `trainer.py:45` `train()` builds its own internally; there is no `make_env(level_id)`
   helper, so `main.py` and `compare_algorithms.py` will each construct envs differently.
6. **`level0` "shortest-path" wording.** 4 apples (`level0.json`) is a routing/ordering problem,
   not a single shortest path — fine for the mark, but the report/video should say "efficient
   collection route", not claim optimality.
7. **No script for the required Task 2 comparison** (repeated from §3 for completeness):
   `compare_algorithms.py` is the creativity 3-algo plot on `level4`; nothing runs Q-learning vs
   SARSA on `level1`.
8. **No monster-avoidance metric.** `logger.py:20` already records `died` per episode, so the data
   exists, but `plot_results.py` has no death-rate-over-episodes function — the cleanest Task 4
   evidence.

## 7. Part II gaps (beyond §5)

1. **`SHOT_NO_TARGET_RADIUS` rationale is internally stale.** `rewards_config.py:59-63` reasons
   from "a 400×400 arena (diagonal ≈ 566)" and picks `150.0` as ~0.3× that. But `core_env.py:33-34`
   sets the arena to `960×680` (diagonal ≈ 1173) — so `150` is ~13% of the real diagonal, not the
   intended ~30%. Re-tune against the real dimensions when implementing.
2. **No Part II seeding path.** `seed_utils.py` lives under `part1_gridworld/` only.
   `train.py:30` exposes `--seed` but nothing wires SB3 `set_random_seed`, `env.reset(seed=)`,
   or `action_space.seed()`. Reproducibility of the submitted logs/models is weaker than Part I.
3. **Curriculum ramp undefined.** `phases.py:39-50` `difficulty_for_phase` has neither a base
   difficulty curve nor a ramp schedule; creativity hook (c) and report §8 both need it described.
4. **`num_active_enemies_frac` "assumed max" undefined** (`obs.py:34`) — needs a constant, and it
   must match what the phase curve can actually produce or the feature saturates.
5. **`compare_ppo_dqn.py` `__main__` raises `NotImplementedError`** (`compare_ppo_dqn.py:61-64`) in
   addition to its helper stubs — the whole script is a stub, fine now, noted for completeness.

## 8. Testing / tooling / reproducibility gaps

1. **All 31 tests skip.** `RUBRIC_MAP.md` cites `test_algorithms.py`, `test_tie_breaking.py`,
   `test_env_rules.py`, `test_env_api.py`, `test_obs_shape.py`, `test_reward_terms.py` as the
   *evidence* for rows I-B, I-C, I-D, II-H, II-J. None back anything yet.
2. **Tests likely will not collect as written.** No `__init__.py` anywhere (implicit namespace
   packages), no `conftest.py`, and `pyproject.toml` has no `[tool.pytest.ini_options]` /
   `pythonpath`. `trainer.py:13-25` does bare `from algorithms import` / `from env import`
   (needs `part1_gridworld/src` on path); `env.py:16` does `from config.rewards_constants import`
   (needs `part1_gridworld` on path). Add pytest config with
   `pythonpath = ["part1_gridworld/src", "part1_gridworld", "part2_arena"]` or add `conftest.py`.
3. **`python main.py` is broken out of the box.** From `part1_gridworld/`, `main.py:15-16` does
   `from src.trainer import ...`; `trainer.py` then does `from env import ...`, which is not
   importable unless `part1_gridworld/src` is *also* on `sys.path`. Needs both paths, or a
   `sys.path` shim in `main.py`.
4. **Dependencies unpinned.** All three `requirements.txt` use `>=` only; the root file's own
   comment (`requirements.txt:2-3`) says to pin "once you start implementing". The submission
   bundles TB logs + saved models that a `gymnasium`/`sb3`/`torch` minor bump can make
   unloadable. `tbparse` (needed by two creativity scripts) is not listed at all.
5. **No `.gitignore`.** `__pycache__/` is already in the tree; without a `.gitignore` the
   submission zip will carry byte-code and stray artefacts. Keep `models/` and `logs/` tracked.
6. **`pyproject.toml` has no `[project]` metadata** and no test-runner config — only `[tool.ruff]`.
7. **Dev environment not set up here.** The interpreter used for this audit has no `numpy`,
   `pygame`, `gymnasium`, `stable_baselines3`, `torch`, `pytest`, or `ruff` (Appendix) — so lint
   and tests were not run. Whoever implements needs a working venv from `requirements.txt` first.

## 9. Report & process gaps

1. **`report/figures/` does not exist** — every `plot_*.py` (`plot_results.py:16`,
   `compare_algorithms.py:19`, `compare_ppo_dqn.py:20`, `plot_reward_decomposition.py:21`) and
   `generate_report_tables.py` writes there. Create it with a `.gitkeep`.
2. **`docs/CONTRIBUTIONS.md` is all `TODO`**, including student numbers — a scored report item and
   a submission requirement. Fill names + student numbers now; keep the dated log running.
3. **10-page limit vs figure count.** Sections 4, 5, 6, 8 each need plots (hyperparameter sweeps,
   control-scheme curves, five Part I training curves, PPO/DQN ablation, reward decomposition).
   At ~1 page/section that is tight — plan a figure budget (shared axes, one multi-panel figure
   per section).
4. Ensure `plot_results.py` output filenames match the figure names the report template's `TODO`
   lines will reference, so figures drop in without edits.

---

## 10. Prioritised action list

### Before writing any implementation bodies

1. Set `REWARD_DEATH` negative; regenerate reward tables; note it in report §3. (§5.1)
2. Commit to the Q-table state encoding in `env.py`'s docstring; estimate reachable states per
   level; shrink levels 4–6 if infeasible. (§5.2)
3. Decide arena `max_steps` **and** the training-timestep budget together. (§5.4)
4. Resolve the enemy-projectile question — perceivable (add obs features) or removed. (§5.5)
5. Create `part2_arena/config/` (arena params + hyperparameter presets); fix
   `SHOT_NO_TARGET_RADIUS` against real arena size; define `num_active_enemies` max and the
   curriculum ramp schedule. (§5.6, §7.1, §7.3, §7.4)
6. Add pytest config (`pythonpath`) or `conftest.py` so the suite collects; add a `sys.path` shim
   so `python main.py` runs. (§8.2, §8.3)
7. Create `report/figures/.gitkeep`; add a `.gitignore`; fill `docs/CONTRIBUTIONS.md`. (§8.5, §9)
8. Stand up a venv from `requirements.txt` and confirm `ruff` + `pytest` run. (§8.7)

### During implementation

9. `save_qtable` / `load_qtable` + `part1_gridworld/models/` + `main.py` watch-only path. (§5.3)
10. Implement `_load_level` validation + a BFS reachability check + one solvability test per
    level. (§6.1, §6.2)
11. `RewardTermLoggingCallback` wired into `train.py`; add `tbparse` to requirements. (§5.7)
12. A Task-2 script running Q-learning vs SARSA on `level1` into a shared plot. (§3, §6.7)
13. Death-rate-over-episodes plot in `plot_results.py` for Task 4. (§6.8)
14. Shared Part II seeding helper wired into `train.py`. (§7.2)
15. Decide monster-vs-monster occupancy; document in `schema.md` + `_resolve_monster_moves`. (§6.4)
16. `Clock().tick()` in both eval loops. (§5.9)
17. Gate or shrink `R_APPROACH_NEAREST_ENEMY`; document. (§5.8)
18. Implement every test body alongside its module; get the suite green.
19. Pin dependency versions once the first successful training run exists.
20. Re-run `generate_report_tables.py` after any reward-constant change.

---

## Appendix — command outputs

```
$ git branch --show-current
main

$ git rev-parse --short HEAD
24dccdd

$ git status --short
?? CLAUDE.md

$ ruff check .
ruff: command not found
python -m ruff check .  ->  No module named ruff        # ruff NOT installed in this interpreter

$ python -m pytest -q --collect-only part1_gridworld/tests part2_arena/tests
No module named pytest                                   # pytest NOT installed

$ python -c "import numpy, pygame"
ModuleNotFoundError: No module named 'numpy'             # project deps NOT installed

$ find . -name "__init__.py" -not -path "./GAIT-ASM3/*"
(none)                                                   # implicit namespace packages only

$ ls part2_arena/config
ls: cannot access 'part2_arena/config': No such file or directory

$ find report
report
report/report_template.md                               # report/figures/ does not exist

$ ls part2_arena/models part2_arena/logs
models/.gitkeep   logs/.gitkeep                          # no trained models / logs

$ cat .gitignore
No such file or directory

$ grep -n "pytest\|pythonpath" pyproject.toml
(no matches)                                             # [tool.ruff] only
```

Static test inventory (all `pytest.skip`):
`test_algorithms.py` 5, `test_env_rules.py` 7, `test_tie_breaking.py` 1,
`test_monster_stochastic.py` 3, `test_intrinsic_reward.py` 3,
`test_env_api.py` 5, `test_obs_shape.py` 3, `test_reward_terms.py` 4  — total 31.
