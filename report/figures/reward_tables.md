<!-- GENERATED FILE -- do not edit by hand.
     Regenerate with: python scripts/generate_report_tables.py -->

# Reward Tables

## Part I - Gridworld Reward Constants

| Constant | Value | Justification |
| --- | --- | --- |
| `REWARD_APPLE` | `1.0` | Collecting an apple. |
| `REWARD_KEY` | `0.0` | Picking up the key. The spec is explicit that this must be zero -- the key's only value is unlocking the chest, it is not a reward source itself. |
| `REWARD_CHEST` | `2.0` | Opening the chest (requires holding the key). |
| `REWARD_DEATH` | `0.0` | Reward on death (fire or monster contact). REVIEWED AND INTENTIONALLY KEPT AT 0.0 -- do not "fix" this into a negative value. The spec's reward list (apples/key/chest) has no death term, death is only specified as ending the episode, and the spec is explicit that "rewards and mechanics must not be altered." Per lesson.md (docs/lesson.md) -- feedback from a previous team's assignment -- do not change a spec-defined reward without a justification stronger than "it would make a comparison look cleaner." |
| `REWARD_STEP` | `0.0` | No per-step penalty is specified by the spec; kept at 0.0 and exposed here (rather than hardcoded in env.py) purely so it stays visible as a deliberate "not used" choice, discoverable by generate_report_tables.py. |

## Part II - Arena Reward Constants

| Constant | Value | Justification |
| --- | --- | --- |
| `R_KILL_ENEMY` | `5.0` | Destroying a single enemy. Baseline positive signal for offense. |
| `R_KILL_SPAWNER` | `20.0` | Destroying a spawner. Larger than R_KILL_ENEMY because spawners gate phase progression -- this must dominate enemy-killing in the agent's incentive structure or it will farm enemies forever instead of advancing. |
| `R_PHASE_PROGRESS` | `50.0` | Reaching a new phase (all active spawners destroyed). The single strongest positive signal, since phase progression is the closest thing this environment has to a 'win' condition within an episode. |
| `R_DAMAGE_TAKEN_PER_HP` | `-0.5` | Negative reward per HP of damage taken, scaled by HP rather than a flat per-hit penalty so a single graze and a heavy hit are distinguished. |
| `R_DEATH` | `-100.0` | DECISION: KEPT at -100.0. Strong negative reward on death, on top of R_DAMAGE_TAKEN_PER_HP for the killing blow -- must dominate any single episode's positive rewards so the agent reliably prioritizes survival. |
| `R_APPROACH_NEAREST_ENEMY` | `0.01` | DECISION (Member D): KEPT, at 0.01. Small per-step shaping reward for reducing distance to the nearest enemy, intended to speed up early training before the agent has discovered that engaging enemies is valuable at all. |
| `R_SHOOT_WHILE_NO_TARGET` | `0.0` | DECISION (Member D): KEPT DISABLED, at 0.0. The spec does not give the player a limited ammo/cooldown resource that a "wasted shot" would deplete, so spam-shooting has no direct mechanical cost worth discouraging via reward. Enabling a penalty here risks discouraging legitimate exploratory fire early in training, before the agent has learned to aim -- i.e. it would likely slow convergence more than it saves. Revisit only if real training runs show the agent spamming SHOOT in a way that visibly hurts performance (e.g. via the reward decomposition dashboard, scripts/plot_reward_decomposition.py). |
| `SHOT_NO_TARGET_RADIUS` | `350.0` | Distance threshold (in arena world units) beyond which the nearest enemy is not considered a valid target for R_SHOOT_WHILE_NO_TARGET purposes. A shot fired when the nearest enemy is farther than this value sets "shot_fired_with_no_target" in step_events (see rewards.py). |
