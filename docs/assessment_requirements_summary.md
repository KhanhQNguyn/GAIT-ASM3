# Assessment Requirements Summary — Reinforcement Learning Project

**Total Points: 40**

## Overview

This is a two-part reinforcement learning project focused on building autonomous systems, with applications to robotics, self-driving cars, drone navigation, warehouse automation, adaptive game AI, traffic control, and simulation-based training.

- **Part I** — Classical value-based RL (Q-learning, SARSA) in a visual Pygame gridworld.
- **Part II** — Deep RL (Stable Baselines3) in a real-time Pygame arena.

By the end, you must demonstrate: designed simulation environments, defined state/reward models, trained agents, analysis of learning behavior via logs, and visual demonstrations of intelligent behavior.

---

## Part I — Classical Reinforcement Learning (20 pts)

### General Requirements
- Gridworld must be implemented **and visually rendered in Pygame** — must support interaction and animation.
- **Console/text-based displays are NOT permitted.**
- Multiple levels with different layouts are required.
- Helper functions are allowed, but **rewards and mechanics must not be altered**.

### Gridworld Rules
- Agent movement: up, down, left, right.
- Rocks block movement (attempting to move into one results in no movement).
- Fire or monsters cause **immediate death** on contact.
- Apples give **+1 reward**.
- Keys give **no reward** but allow opening chests.
- Opening a chest gives **+2 reward**.
- Episode ends when all collectible rewards are obtained **or** the agent dies.
- After each agent action, monsters (if present) have a **40% chance to move**.

### Task 1: Basic Q-Learning (Level 0)
- Level 0 contains only apples, positioned on the right side of the map.
- Requirements:
  - Epsilon-greedy policy for action selection.
  - Correct Q-learning update rule (off-policy, using max over next-state actions).
  - Linear epsilon decay from `epsilonStart` to `epsilonEnd` (config-driven).
  - Random tie-breaking when multiple actions share the best Q-value.
  - Must demonstrate a learned shortest-path policy to the apples.
- Training parameters (episodes, alpha, gamma, epsilon ranges) will be supplied via a config file.

### Task 2: Basic SARSA (Level 1)
- Implement SARSA using **on-policy** updates (uses the actually chosen next action, not the max).
- Use the **same exploration schedule** as Q-learning.
- Must demonstrate that the SARSA policy **differs from Q-learning** — typically more conservative around hazards — with a short comparison.

### Task 3: Extend Q-Learning & SARSA to Levels 2–3
- These levels introduce:
  - Multiple apples
  - A key
  - A chest
- Both algorithms must run correctly with correct episode termination and reward accounting.

### Task 4: Monster Levels (Levels 4–5)
- Implement:
  - Monsters that move after each agent action.
  - Probabilistic movement (e.g., 40% chance to move).
  - Simple movement pattern (e.g., random choice among allowed directions).
  - Player dies if it enters a monster's tile or a monster moves into the player.
- RL requirements:
  - Q-learning and SARSA must handle **stochastic transitions**.
  - Agent should learn to avoid monsters while still completing objectives.
- Evidence to include:
  - Working monster movement in the gridworld.
  - Training curves showing learning behavior on Levels 4 and 5.

### Task 5: Intrinsic Reward (Level 6)
- Intrinsic reward formula:
  - `r_i = intrinsicRewardStrength / sqrt(n(s) + 1)`
  - where `n(s)` = number of visits to the current state during the episode.
  - Total reward = environment reward + intrinsic reward.
- Requirements:
  - Environment rewards must remain unchanged.
  - Maintain a **per-episode** visit counter for each state.
  - Agent must incorporate intrinsic reward into Q-learning/SARSA updates.
- Evidence to include in the report:
  - Training curves comparing learning with vs. without intrinsic reward.
  - Short explanation of the observed improvement.

---

## Part II — Deep Reinforcement Learning in a Pygame Arena (20 pts)

Design and build a real-time, visually animated Pygame arena, and train deep RL agents in it using **Stable Baselines3**. The agent learns from a continuous observation vector using a neural-network-based algorithm.

### 1. Environment Requirements
The arena must include:
- A controllable player ship with movement and shooting.
- Enemy spawners that periodically create enemies.
- Enemies that navigate toward the player.
- Player health and enemy health systems.
- Projectile collisions.
- A **phase system**: destroying all active spawners progresses the simulation to the next difficulty level.
- Must feel like a simplified action arena (continuous/semi-continuous movement), **not** a tile-based grid.
- All elements must be visually rendered.
- Episode ends when:
  - The player dies, **or**
  - A maximum time/step count is reached.

### 2. Gym-Style API
Must expose:
- `reset()` → returns initial observation.
- `step(action)` → returns `(observation, reward, done, info)`.
- `render()` → displays the scene for evaluation.

### 3. Observation Design
Fixed-size numeric feature vector including at minimum:
- Player position
- Player velocity
- Player orientation (if relevant)
- Distance and relative direction to nearest enemy
- Distance and relative direction to nearest spawner
- Player health
- Current phase
- **No pixels/screenshots** — use numeric feature vectors, fixed size.

### 4. Action Sets
Must implement **two distinct control schemes**, each with its own trained agent:

**Control Style 1 — Rotation + Thrust**
- No action
- Thrust forward
- Rotate left
- Rotate right
- Shoot

**Control Style 2 — Direct Directional Movement**
- No action
- Move up
- Move down
- Move left
- Move right
- Shoot

- Each control style needs its own trained model **and** its own evaluation script.

### 5. Reward Function
Must include (at minimum):
- Positive reward for destroying enemies.
- Larger positive reward for destroying spawners.
- Positive reward for progressing to the next phase.
- Negative reward when taking damage.
- Strong negative reward on death.
- Any optional/additional shaping rewards must be **justified**.

### 6. Deep RL Training
- Use **Stable Baselines3** with **DQN** or **PPO**.
- Neural network must have at least one hidden layer (MLP).
- Train while logging results to **TensorBoard**.
- Hyperparameters must be tuned meaningfully (not left at defaults only).
- Save trained models in a folder named `models`.
- Provide an evaluation script that can visually play the agent in the arena.

---

## Report Requirements (10 pts)

- **Strict maximum length: 10 pages**, including images, **no appendix** (anything beyond page 10 will not be considered).
- Must include:
  1. Description of both environments.
  2. Observation design (what each feature means and why it was chosen).
  3. Reward design (explained and justified, including any shaping).
  4. Hyperparameter exploration (with evidence, e.g., tables/plots).
  5. Comparison of the two control sets, with evidence (curves/screenshots/logs).
  6. Evidence of training with logs and screenshots.
  7. Originality justification.
  8. Student numbers and a contribution summary for all team members.
  9. A link to the video recording demonstration.

---

## Submission Requirements (STRICT)

You must submit **exactly 2 items**:

1. **A link to a zip file** containing:
   - All gridworld code
   - Arena simulation code
   - RL training scripts
   - Saved models
   - TensorBoard logs
2. **Your report PDF**, including:
   - All student numbers and a contribution summary
   - A link to the video recording demonstration

> Note: the actual zip file must also be uploaded to Canvas per the "Submission Instructions" section.

### Late Submission Policy
- 10% penalty per calendar day late.
- Notify the teaching team if submitting late so the correct version is downloaded.
- Special consideration information is available via RMIT's official page.

---

## Video Demonstration Requirements

- **Maximum length: 10 minutes.**
- **All team members must appear** and present at least one part.
- Upload to YouTube (unlisted) or university OneDrive; link it in the PDF report.

### Must Show — Part I (Gridworld)
- The gridworld visible in a Pygame window.
- The agent running with Q-learning or SARSA on one level.
- Monsters/items behaving correctly (if applicable to the level shown).
- Evidence the agent is following its **learned policy**, not acting randomly.

### Must Show — Part II (Arena)
- The real-time Pygame arena running visually.
- The trained deep RL agent controlling the player.
- Enemies spawning and moving.
- Projectiles and collisions functioning.
- At least one phase progression occurring.
- Learned behavior demonstrated for **both** control schemes (two short clips acceptable).

### Acceptability Requirements
- Environment and agent behavior must match the submitted code.
- Models shown must be the same ones included in the repository.
- Must show actual gameplay — not static screens.

---

## Reuse Policy
You may reuse code or materials from previous assignments or tutorials.

---

## Appendix: Technical Feasibility Guide (Not Mandatory, but Recommended)

### Gridworld
- Grid size: ~10×10 or 12×12.
- Use simple Pygame shapes.
- Q-learning/SARSA should run quickly.

### Arena Simulation
- Window size: ~800×600 or 960×680.
- Clear shapes for players, enemies, bullets.
- Keep enemy count/spawner frequency manageable.
- Physics should remain simple.
- Render only during evaluation, not during long training runs.

### Observation Vector
- ~10–30 float features.
- Fixed size.
- No pixels/images.

### RL Training
- Use Stable Baselines3.
- Small MLP networks.
- Train headless for speed.
- Typical total training: 100,000–600,000 timesteps.
- Use TensorBoard for monitoring.

---

## Learning Objectives Assessed
- **CLO1**: Apply various AI techniques and tools in the context of games.
- **CLO3**: Work effectively in a team environment to develop a complex software system.

---

## Grading Rubric Breakdown (Total: 40 pts)

| Section | Criteria | Points |
|---|---|---|
| Part I-A | Gridworld implementation & rules (visual/animated/interactive; mechanics match spec) | 2 |
| Part I-B | Task 1: Q-learning (epsilon-greedy, correct update rule, linear decay, tie-breaking, shortest-path evidence) | 2.5 |
| Part I-C | Task 2: SARSA (on-policy update, same exploration schedule, comparison vs Q-learning) | 3 |
| Part I-D | Task 3: Levels 2–3 (multiple apples, key, chest; correct termination/reward accounting) | 3 |
| Part I-F | Task 5: Intrinsic reward (correct formula, unchanged env rewards, per-episode visit counter, training curve comparison + explanation) | 3 |
| Part II-G | Arena environment requirements (real-time, animated; core gameplay; health/phase systems; episode-end conditions) | 4.5 |
| Part II-H | Gym-style API + observation design (reset/step/render; fixed-size vector with required features) | 2.5 |
| Part II-I | Two control schemes + models (both control styles; separate trained/saved models; separate evaluation scripts) | 4 |
| Part II-J | Reward design & deep RL training quality (reward structure; SB3 with DQN/PPO + TensorBoard logging; meaningful hyperparameter tuning) | 3 |
| Report | Page limit, environment/observation/reward descriptions, hyperparameter exploration, control-set comparison, originality justification | 2.5 |
| Video Demo | ≤10 min, all members present, gridworld + arena shown, learned policy evidence, both control schemes shown | 5 |
| Creativity | Going beyond expected requirements | 5 |

**Note:** Part I Task 4 (Monster Levels) is described in detail in the task list but does not appear as a separately labeled rubric row in the provided rubric — its evidence (monster behavior, stochastic transition handling, training curves) likely factors into the Part I-D/general implementation criteria; confirm with your instructor if unclear.

---

## Academic Integrity
- Standard RMIT academic integrity expectations apply.
- Properly acknowledge and reference all external words, data, diagrams, models, frameworks, and ideas (including internet sources).
- Plagiarism includes failure to document sources, copying material from the internet/databases, and collusion between students.
- This is treated as serious misconduct.

## Note on Spec Changes
This assignment specification may be corrected after release (to fix mistakes or add information). Any corrected version will be uploaded to Canvas with an announcement — check for updates.
