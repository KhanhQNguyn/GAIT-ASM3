# Scaffold Prompt v2 — RL Assignment Codebase

Paste the fenced prompt below into Claude Code (or another coding agent) to scaffold
the project skeleton. This is a revision of `scaffold_prompt.md`, corrected against
`assessment_requirements_summary.md` (the authoritative spec/rubric) and expanded to
cover gaps the v1 scaffold left out: the report, video demo, submission, and team
contribution artifacts, plus a resolved Gym-API conflict and four committed creativity
hooks. Skeleton only — no logic implementation yet.

## What changed vs. v1 (and why)

1. **Report/video/submission are now first-class scaffold artifacts**, not an
   afterthought. The rubric gives them 2.5 + 5 + (implicit, via the zip) points and v1
   had zero scaffolding for them — a strict 10-page/no-appendix report and a ≤10-minute
   video with per-member speaking requirements are easy to get wrong under deadline
   pressure without a template that already has the sections and shot list laid out.
2. **Gym API conflict resolved explicitly.** The spec's own wording for Part II
   (`step(action) -> (observation, reward, done, info)`) is the legacy Gym 4-tuple, but
   Stable-Baselines3 requires Gymnasium's 5-tuple (`terminated`/`truncated` split).
   Per your direction to treat the requirements doc as the strict source of truth: the
   core environment class implements the **literal 4-tuple API from the spec**
   (this is what satisfies the rubric's "Gym-style API" row), and a separate thin
   adapter module wraps it in Gymnasium's 5-tuple API purely so Stable-Baselines3 can
   train against it. Neither file lies about what it is.
3. **Two fully separate evaluation scripts** (`eval_style1.py`, `eval_style2.py`)
   instead of one `--style` flag, matching the rubric's literal "its own evaluation
   script" wording with no ambiguity for the grader.
4. **All four creativity angles get real hooks**: PPO vs. DQN ablation, reward
   decomposition dashboard, curriculum learning for phases, and Expected SARSA in
   Part I. Each is independently runnable — you are not forced to execute the full
   cross-product of style × algo × curriculum to get value from any one of them.
5. **Team artifacts added**: `CONTRIBUTIONS.md` (student numbers + contribution
   summary, required in the PDF) and `VIDEO_SCRIPT.md` (a timestamped shot list mapped
   to the rubric's "Must Show" checklist, with a slot for who presents each segment).
6. **Level configs enumerated explicitly** (`level0.json` … `level6.json`, 7 total)
   instead of a generic `config/` folder, since the spec ties each level number to a
   specific task and rubric row.
7. **`generate_report_tables.py` promoted to project root** and reads reward
   constants from *both* parts, so there is exactly one script that keeps the report's
   reward tables in sync with the code for the whole project, not just Part II.
8. **Reward immutability made testable in Part I too**: a `rewards_constants.py`
   single-source-of-truth file (mirroring Part II's `rewards_config.py`) plus a test
   asserting apples=+1, keys=0, chests=+2 never drift — directly defends the "helper
   functions allowed but rewards/mechanics must not be altered" rule.

---

## PROMPT

```
You are a senior software engineer specializing in game AI and reinforcement
learning, working with a small student team on a graded assignment. Scaffold a
complete Python codebase skeleton (folders, empty/stub files with docstrings,
function signatures, and clear TODOs — do NOT implement full logic yet) for a
two-part RL assignment.

PART I: Classical Gridworld (Pygame) with Q-learning, SARSA, and Expected SARSA
PART II: Real-time Arena (Pygame) with Deep RL via Stable-Baselines3 (PPO + DQN)

This is a team submission. Treat the report, video, and submission packaging as
first-class deliverables, not afterthoughts — they carry real rubric points.

=== FUNCTIONAL REQUIREMENTS — PART I (must match exactly, nothing missing) ===

General:
- Rendered in Pygame, animated, interactive. Console/text display is NOT permitted.
- 7 levels total, each mapped to a specific task:
  - level0.json — Task 1 (Q-learning): apples only, positioned on the right side.
  - level1.json — Task 2 (SARSA): apples only, DIFFERENT layout from level0, with
    enough hazard exposure (rocks/fire) that a Q-learning vs. SARSA behavioral
    comparison near hazards is meaningful. (The spec requires SARSA to look "more
    conservative around hazards" — design this level so that claim is testable.)
  - level2.json, level3.json — Task 3: multiple apples, a key, a chest each, two
    DIFFERENT layouts.
  - level4.json, level5.json — Task 4: monster levels, two DIFFERENT layouts.
  - level6.json — Task 5: intrinsic reward level (reuse a level4/5-style layout).
- Mechanics (fixed, must not be altered by helper functions):
  - 4-directional movement (up/down/left/right).
  - Rocks block movement (moving into one = no movement, not an error/crash).
  - Fire or monsters cause immediate death on contact.
  - Apples: +1 reward. Keys: 0 reward (unlock chests only). Chests: +2 reward,
    only openable if the agent holds a key.
  - Episode ends when ALL collectible rewards are obtained OR the agent dies.
  - After each agent action, each monster has a 40% chance to move, choosing
    randomly among its currently-allowed directions.
- Task 1 (Q-learning): epsilon-greedy action selection; off-policy update using
  max over next-state actions; LINEAR epsilon decay from epsilonStart to
  epsilonEnd, config-driven; random tie-breaking among equal-best Q-values;
  must demonstrably learn a shortest-path policy to the apples.
- Task 2 (SARSA): on-policy update using Q(s', a') where a' is the action
  ACTUALLY chosen by the current policy (not max); same exploration schedule as
  Q-learning; produce a short, evidenced comparison showing behavioral
  divergence from Q-learning.
- Task 3: both algorithms extended to levels 2-3 with correct termination and
  reward accounting when multiple apples + a key + a chest are present.
- Task 4: both algorithms must handle the monsters' stochastic transitions
  correctly and learn to avoid them while still completing objectives; capture
  training curves for levels 4 and 5.
- Task 5 (Level 6, intrinsic reward): r_i = intrinsicRewardStrength / sqrt(n(s)+1)
  where n(s) is the visit count for the CURRENT state DURING THE CURRENT EPISODE
  ONLY (reset every episode). Environment reward is unchanged; the update target
  uses env_reward + r_i. Produce training curves comparing with vs. without
  intrinsic reward, plus a short explanation of the observed effect.
- Creativity hook (Expected SARSA): implement a third tabular update
  (expected value over the policy's action distribution, not max and not the
  sampled next action) and compare all three algorithms' learning curves on the
  same hazard-containing level (reuse level4 or level5).
- Training hyperparameters (episodes, alpha, gamma, epsilonStart, epsilonEnd,
  intrinsicRewardStrength, max_steps_per_episode) come from a config file, not
  hardcoded.

=== FUNCTIONAL REQUIREMENTS — PART II (must match exactly, nothing missing) ===

Environment:
- Real-time, continuously animated Pygame arena — must NOT feel like a tile grid.
- Controllable player ship (movement + shooting), enemy spawners that
  periodically create enemies, enemies that navigate toward the player, player
  health, enemy health, projectile collisions, a phase system where destroying
  all currently-active spawners advances difficulty.
- All elements visually rendered.
- Episode ends on player death OR a max time/step count.

Gym-style API — IMPORTANT, read carefully:
- The core environment class (`arena/core_env.py`) must implement the API
  EXACTLY as the assignment spec describes it, literally:
    reset() -> observation
    step(action) -> (observation, reward, done, info)
    render() -> displays the scene
  This class must have ZERO dependency on gymnasium/gym and ZERO dependency on
  stable-baselines3. It is the thing that satisfies the rubric's "Gym-style API"
  requirement on its own literal terms.
- Because Stable-Baselines3 requires Gymnasium's 5-tuple step contract
  (terminated/truncated split, not a single done flag), add a SEPARATE adapter
  module `arena/gym_adapter.py` that wraps `core_env` in a real
  `gymnasium.Env` subclass for training/eval purposes only. It must derive
  `terminated` (player died) and `truncated` (max steps hit) from the core
  env's single `done` + its termination reason, without changing core_env's
  own literal signature. Document in both files' docstrings why this split
  exists and which one satisfies which requirement.

Observation (fixed-size numeric vector, 10-30 floats, NO pixels/screenshots):
- Player position, player velocity, player orientation (if relevant to the
  control style), distance + relative direction to nearest enemy, distance +
  relative direction to nearest spawner, player health, current phase.
- Document each vector index's meaning and justification in `arena/obs.py`.

Two control schemes, each with its own trained model AND its own evaluation
script (do not share one parametrized eval script between them):
- Style 1 — Rotation + Thrust: No-op, Thrust forward, Rotate left,
  Rotate right, Shoot.
- Style 2 — Direct Directional: No-op, Up, Down, Left, Right, Shoot.

Reward (single source of truth in `arena/rewards_config.py`, computed only in
`arena/rewards.py` — no inline reward math anywhere else):
- Positive for destroying an enemy; larger positive for destroying a spawner;
  positive for advancing to a new phase; negative when taking damage; strong
  negative on death.
- Cap total shaping terms at 6-8, each a named constant (e.g. R_KILL_ENEMY,
  R_KILL_SPAWNER, R_PHASE_PROGRESS, R_DAMAGE_TAKEN, R_DEATH, plus up to 2
  optional shaping terms). Every term needs a one-line docstring justification
  and must be logged to TensorBoard as a separate scalar (reward decomposition)
  — this also feeds the creativity dashboard below.

Deep RL training:
- Stable-Baselines3, PPO and/or DQN (both are wired up — see creativity hooks),
  MLP with at least one hidden layer, TensorBoard logging, meaningfully tuned
  hyperparameters (not left at library defaults), models saved under `models/`.
- Render only during evaluation, not during long training runs.

=== ARCHITECTURE PRINCIPLES (mandatory) ===

1. Strict separation of concerns: environment/game logic never imports pygame
   and knows nothing about the training loop. Rendering is its own module that
   receives state and draws it. The training loop wires env + rendering +
   logging together.
2. Single source of truth for ALL reward/mechanic constants, in BOTH parts:
   - Part I: `part1_gridworld/config/rewards_constants.py`
     (REWARD_APPLE=+1, REWARD_KEY=0, REWARD_CHEST=+2, plus death handling).
   - Part II: `part2_arena/arena/rewards_config.py`.
   - `scripts/generate_report_tables.py` at the PROJECT ROOT reads BOTH files
     and auto-generates Markdown/LaTeX tables into `report/figures/` — the
     report must never drift out of sync with the code, for either part.
3. No dead code: if a rewards/algorithms module exists, it must actually be
   called from env.py/trainer.py — never written then bypassed by inline logic.
4. Reproducibility: a small `seed_utils.py` (Part I) setting Python/NumPy seeds
   so training runs are reproducible for the report and demo.
5. Config-driven, not hardcoded: level layouts as JSON with a documented
   schema (`part1_gridworld/config/schema.md`); Part I training hyperparameters
   in a single `training_config.json`.
6. Unit tests (pytest):
   - Part I: correct Q-learning update, correct SARSA update, correct Expected
     SARSA update, correct intrinsic reward formula (including per-episode
     reset of n(s)), the "keys give no reward" rule, rocks block movement,
     fire/monster instant death, episode termination (death OR all rewards),
     random tie-breaking distribution, and monster 40% movement probability
     (statistical test over many trials).
   - Part II: `core_env` literally satisfies the spec's 4-tuple contract;
     `gym_adapter` literally satisfies Gymnasium's 5-tuple contract; each
     reward term fires independently and correctly; observation vector has a
     fixed shape, no NaNs, and documented bounds.
7. `requirements.txt` at the project root pinning pygame, numpy, gymnasium,
   stable-baselines3, torch, tensorboard, matplotlib, pytest.
8. `RUBRIC_MAP.md`: every module/function mapped to the exact rubric row it
   satisfies, WITH that row's point value (pull the point values from the
   40-point breakdown table in the assignment spec) so the team can self-grade
   before submitting.

=== CREATIVITY HOOKS (scaffold all four; each independently runnable) ===

a) PPO vs. DQN ablation (`part2_arena/scripts/train.py` supports
   `--algo {ppo,dqn}`; `part2_arena/scripts/compare_ppo_dqn.py` trains/reads
   TensorBoard logs for both on the SAME control style/observation/reward and
   plots convergence speed, stability, and sample efficiency side by side).
b) Reward decomposition dashboard (`part2_arena/scripts/
   plot_reward_decomposition.py` reads the per-term TensorBoard scalars from
   principle 6/req 5-above and renders a stacked-area chart of each reward
   term's contribution over training time).
c) Curriculum learning for phases (`part2_arena/arena/phases.py` includes a
   curriculum hook that ramps enemy speed/spawn rate as phases clear;
   `train.py --curriculum {on,off}` trains both variants for a learning-speed
   comparison).
d) Expected SARSA in Part I (`part1_gridworld/src/algorithms.py` adds
   `expected_sarsa_update`; `part1_gridworld/src/compare_algorithms.py` runs
   Q-learning, SARSA, and Expected SARSA on the same hazard level and plots
   all three learning curves together).

=== TEAM / REPORT / VIDEO / SUBMISSION SCAFFOLDING (new — do not skip) ===

- `report/report_template.md`: section skeleton with a rough page budget per
  section, pre-labeled with the 8 required sections from the spec (env
  description x2, observation design, reward design, hyperparameter
  exploration, control-set comparison, training evidence, originality
  justification) plus placeholders for student numbers, contribution summary,
  and the video link. Add an explicit "STRICT 10-page limit, no appendix" note
  at the top.
- `report/figures/`: empty folder where generated tables/plots/screenshots land.
- `CONTRIBUTIONS.md`: table of student number -> name -> contribution summary,
  to be filled in as work happens (not just at the end) and copy-pasted into
  the report.
- `VIDEO_SCRIPT.md`: a timestamped shot list totaling <= 10 minutes, structured
  as a checklist mirroring the spec's "Must Show" lists for Part I and Part II
  (gridworld window + learned-policy evidence + monster behavior; arena +
  trained agent + spawning/collisions + a phase progression + BOTH control
  schemes), with a column for which team member presents each segment.
- `SUBMISSION_CHECKLIST.md`: the 2 required submission items (zip link + report
  PDF), the zip's required contents (all gridworld code, arena code, training
  scripts, saved models, TensorBoard logs), the Canvas upload reminder, and the
  late-penalty note (10%/day).

=== PROJECT STRUCTURE ===

project/
  scripts/
    generate_report_tables.py   # root: reads BOTH parts' reward constants -> report/figures tables

  part1_gridworld/
    config/
      level0.json ... level6.json   # 7 levels, one per task/rubric row (see above)
      training_config.json          # episodes, alpha, gamma, epsilonStart/End, intrinsicRewardStrength
      rewards_constants.py          # single source of truth: REWARD_APPLE, REWARD_KEY, REWARD_CHEST
      schema.md                     # documents the level JSON schema
    src/
      env.py                # pure logic, no pygame dependency
      algorithms.py          # QTable, epsilon_greedy, q_learning_update, sarsa_update, expected_sarsa_update
      intrinsic.py            # intrinsic reward logic, kept separate for testability
      render.py                # pygame rendering only
      menu.py                   # level/algorithm select menu (also satisfies "interactive" requirement)
      trainer.py                 # orchestrates env + algorithms + render + logger
      logger.py                    # CSV/matplotlib episode-return logging (training curves)
      plot_results.py               # renders training curve comparisons (levels 4/5/6, intrinsic on/off)
      compare_algorithms.py          # creativity(d): Q-learning vs SARSA vs Expected SARSA on one hazard level
      seed_utils.py                   # reproducibility
    tests/
      test_algorithms.py
      test_env_rules.py       # rocks/fire/apples/keys/chests/termination
      test_intrinsic_reward.py
      test_tie_breaking.py
      test_monster_stochastic.py
    main.py
    requirements.txt

  part2_arena/
    arena/
      core_env.py         # LITERAL spec API: reset()->obs, step(action)->(obs,reward,done,info), render()
      gym_adapter.py        # gymnasium.Env wrapper (5-tuple) around core_env, for SB3 only
      entities.py
      physics.py
      obs.py                  # fixed-size observation vector; documents each feature
      rewards.py                # the only place reward is computed; calls rewards_config
      rewards_config.py           # single source of truth for all reward constants
      phases.py                     # phase system + creativity(c) curriculum hook
      actions.py                      # ControlStyle1 and ControlStyle2 action mappings
      render_pygame.py
    scripts/
      train.py                  # --style {1,2} --algo {ppo,dqn} --curriculum {on,off} --timesteps N
      eval_style1.py               # loads the style-1 model, plays it live
      eval_style2.py                 # loads the style-2 model, plays it live
      compare_ppo_dqn.py               # creativity(a)
      plot_reward_decomposition.py       # creativity(b)
    tests/
      test_env_api.py          # core_env's literal 4-tuple contract AND gym_adapter's 5-tuple contract
      test_reward_terms.py
      test_obs_shape.py
    models/
    logs/
    requirements.txt

  report/
    report_template.md
    figures/

  CONTRIBUTIONS.md
  VIDEO_SCRIPT.md
  SUBMISSION_CHECKLIST.md
  RUBRIC_MAP.md
  README.md
  requirements.txt

After scaffolding, print the list of files created and, for each one, a
one-line explanation of which part of the assignment (and which rubric row)
it addresses.
```
