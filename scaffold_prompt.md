# Scaffold Prompt for the RL Assignment Codebase (GAIT_A3-style)

Paste the prompt below into Claude Code (or whatever coding agent you use) to scaffold the project skeleton. This prompt already bakes in lessons learned from reviewing a strong sample submission (architecture worth keeping, mistakes worth avoiding), plus a suggested differentiation angle for the Creativity marks.

---

## PROMPT

```
You are a senior software engineer specializing in game AI and reinforcement
learning. Scaffold a complete Python codebase for an RL assignment with two
parts:

PART I: Classical Gridworld (Pygame) with Q-learning and SARSA
PART II: Real-time Arena (Pygame) with Deep RL via Stable-Baselines3

=== FUNCTIONAL REQUIREMENTS (must follow the spec exactly, nothing missing) ===

Part I - Gridworld:
- Must be rendered in Pygame, animated, and interactive (no console/text
  display allowed).
- Mechanics: 4-directional movement; rocks block movement; fire/monsters
  cause instant death; apples give +1; keys give NO reward (they only
  unlock chests); chests give +2 when opened with a key; episode ends
  when all collectible rewards are obtained OR the agent dies.
- After each agent action, monsters (if present) have a 40% chance to move.
- Task 1: Q-learning on Level 0 (apples only) — epsilon-greedy policy,
  correct off-policy update rule, LINEAR epsilon decay (start->end via
  config), random tie-breaking when multiple actions share the best
  Q-value, must learn a shortest-path policy.
- Task 2: SARSA on Level 1 — on-policy update (uses Q(s', a') where a' is
  the action ACTUALLY CHOSEN by the policy, not the max), same exploration
  schedule as Q-learning, must show evidence that SARSA behaves more
  conservatively around hazards than Q-learning.
- Task 3: Extend both algorithms to Levels 2-3 (multiple apples, a key,
  a chest).
- Task 4: Monster Levels 4-5 — both algorithms must handle stochastic
  transitions correctly, agent must learn to avoid monsters while still
  completing objectives.
- Task 5 (Level 6): Intrinsic reward EXACTLY per the formula
  r_i = intrinsicRewardStrength / sqrt(n(s) + 1), where n(s) = number of
  visits to that state DURING THE CURRENT EPISODE (reset every episode).
  Environment reward must stay unchanged. Total reward used for the
  update = env reward + r_i.

Part II - Arena:
- Real-time Pygame, continuous animation (must NOT feel like a tile grid).
- Player ship with movement + shooting; enemy spawners that periodically
  create enemies; enemies that navigate toward the player; player health;
  enemy health; projectile collisions; a phase system (destroying all
  active spawners increases difficulty).
- Gym-style API: reset() -> obs; step(action) -> (obs, reward, done, info);
  render() for evaluation.
- Observation: a FIXED-SIZE numeric vector (10-30 dims), no pixels, must
  include at minimum: position, velocity, orientation (if relevant),
  distance + direction to the nearest enemy, distance + direction to the
  nearest spawner, health, current phase.
- Two distinct control schemes, EACH with its own trained model + its own
  evaluation script:
  - Style 1: No-op, Thrust, Rotate left, Rotate right, Shoot
  - Style 2: No-op, Up, Down, Left, Right, Shoot
- Reward: positive for killing enemies, larger positive for destroying
  spawners, positive for reaching a new phase, negative when taking
  damage, strongly negative on death. ANY additional shaping reward must
  have a clear, documented justification (docstring/comment).
- Train with Stable-Baselines3 (PPO or DQN), MLP with at least one hidden
  layer, log to TensorBoard, save models under models/, provide an
  evaluation script that visually plays the trained agent in the arena.

=== ARCHITECTURE PRINCIPLES (mandatory) ===

1. Strict separation of concerns: environment logic (gridworld/arena) must
   NEVER import pygame or know anything about the training loop. Rendering
   is its own module that receives state and draws it. The training loop
   is its own module that wires env + rendering together.
2. A single source of truth for ALL reward constants: define them in one
   config file (constants.py or rewards.yaml), and:
   - Code reads from this file to compute rewards.
   - A generate_report_tables.py script reads the SAME file to auto-generate
     Markdown/LaTeX tables for the report.
   => Goal: the README/report must never drift out of sync with the actual
      code (this was a real problem I found reviewing a sample submission).
3. Cap the number of reward-shaping terms at 6-8, each clearly named
   (e.g. R_KILL_ENEMY, R_APPROACH_SPAWNER...). Do NOT scatter dozens of
   inline magic numbers inside step(). Each term must:
   - Have a one-line docstring explaining why it exists.
   - Be logged separately to TensorBoard (reward decomposition) for analysis.
4. No dead code — if you write a rewards.py module, it must actually be
   called from env.py, not written and then bypassed by inline reward
   logic elsewhere.
5. Write small unit tests (pytest) for: the correct Q-learning update rule,
   the correct SARSA update rule, the correct intrinsic reward formula,
   and the "keys give no reward" rule. These serve both as self-checks and
   as evidence of engineering rigor for the Creativity marks.
6. A RUBRIC_MAP.md file that maps every module/function to the specific
   rubric criterion it satisfies, so I can self-grade before submitting.

=== DIFFERENTIATION ANGLE FOR CREATIVITY MARKS (build the scaffolding for this) ===

Pick one or combine a few of these (avoid just doing "PPO + an unexplainably
complex reward function," which is the typical unremarkable approach):

a) Real algorithm comparison in Part II: train BOTH PPO and DQN on the same
   control style, same observation space, same reward function — then
   compare convergence speed, stability, and sample efficiency. This is a
   genuine ablation, not just "two models for the sake of it." The spec
   only requires one of the two algorithms, so doing both is a clear bonus
   and makes for a strong report section.

b) A "reward budget" + reward decomposition dashboard: strictly cap the
   number of reward terms, but in exchange produce a stacked-area chart
   showing each term's contribution over training time. This is both
   strong analytical evidence and keeps the report short and readable
   (the opposite of the "overloaded, under-explained shaping" problem
   found in the sample submission).

c) Curriculum learning for phase difficulty: instead of spawning full
   difficulty from the start, ramp up enemy speed/spawn rate as phases are
   cleared, and compare learning speed with vs. without curriculum. This
   is an advanced RL concept beyond the minimum requirement and makes for
   an impressive video demo segment.

d) Add a third tabular algorithm in Part I beyond Q-learning/SARSA (e.g.
   Expected SARSA) as a bonus, comparing all three learning curves on the
   same hazard level — small in scope but shows a deep understanding of
   on-policy vs. off-policy vs. expected updates.

Scaffold the SKELETON code (folders, empty files with docstrings describing
their responsibility, function signatures, clear TODOs) following the
structure below. Do NOT implement full logic yet — I will fill this in
myself:

project/
  part1_gridworld/
    config/            # per-level JSON config, clear schema
    src/
      env.py            # pure logic, no pygame dependency
      algorithms.py      # QTable, epsilon_greedy, q_learning_update, sarsa_update
      intrinsic.py        # intrinsic reward logic kept separate for testability
      render.py          # pygame rendering only
      menu.py
      trainer.py          # orchestrates: env + algorithms + render + logger
      tb_logger.py
      plot_results.py
    tests/
      test_algorithms.py
      test_env_rules.py
      test_intrinsic_reward.py
    main.py

  part2_arena/
    arena/
      env.py             # Gym-style API only
      entities.py
      physics.py
      obs.py
      rewards.py          # THE single place reward is computed, must actually be called
      rewards_config.py   # single source of truth for all reward constants
      render_pygame.py
    scripts/
      train.py             # supports both PPO and DQN via an --algo flag
      eval.py
      generate_report_tables.py    # reads rewards_config.py, generates report tables
    tests/
      test_env_api.py
      test_reward_terms.py
    models/
    logs/

  RUBRIC_MAP.md
  README.md

After scaffolding, print the list of files created and, for each one, a
one-line explanation of which part of the assignment it addresses.
```
