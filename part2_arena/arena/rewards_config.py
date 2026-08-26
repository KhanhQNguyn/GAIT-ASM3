"""Single source of truth for ALL Part II reward constants. rewards.py is
the ONLY module allowed to read these to compute a reward -- no other file
should hardcode a reward number. scripts/generate_report_tables.py (project
root) imports this module directly to keep the report's reward table in
sync with the code.

Capped at <= 8 terms per the architecture principle: 5 required by the spec
+ up to 2 optional shaping terms, each justified below. If you add a new
term, you must also add its one-line justification here (rubric checks for
this) and log it separately to TensorBoard in rewards.py.
"""

# --- Required by the spec ---

R_KILL_ENEMY: float = 5.0
"""Destroying a single enemy. Baseline positive signal for offense."""

R_KILL_SPAWNER: float = 20.0
"""Destroying a spawner. Larger than R_KILL_ENEMY because spawners gate
phase progression -- this must dominate enemy-killing in the agent's
incentive structure or it will farm enemies forever instead of advancing."""

R_PHASE_PROGRESS: float = 50.0
"""Reaching a new phase (all active spawners destroyed). The single
strongest positive signal, since phase progression is the closest thing
this environment has to a 'win' condition within an episode."""

R_DAMAGE_TAKEN_PER_HP: float = -0.5
"""Negative reward per HP of damage taken, scaled by HP rather than a flat
per-hit penalty so a single graze and a heavy hit are distinguished."""

R_DEATH: float = -100.0
"""Strong negative reward on death, on top of R_DAMAGE_TAKEN_PER_HP for the
killing blow -- must dominate any single episode's positive rewards so the
agent reliably prioritizes survival."""

# --- Optional shaping terms (<= 2, must be justified) ---

R_APPROACH_NEAREST_ENEMY: float = 0.01
"""DECISION (Member D): KEPT, at 0.01. Small per-step shaping reward for
reducing distance to the nearest enemy, intended to speed up early training
before the agent has discovered that engaging enemies is valuable at all.
Kept deliberately small -- a 500:1 ratio against R_KILL_ENEMY (5.0) -- so it
nudges the agent toward enemies without ever being large enough to make
loitering near an enemy (collecting shaping reward) more attractive than
actually shooting it. Document this decision in report/report_template.md
section 3."""

R_SHOOT_WHILE_NO_TARGET: float = 0.0
"""DECISION (Member D): KEPT DISABLED, at 0.0. The spec does not give the
player a limited ammo/cooldown resource that a "wasted shot" would deplete,
so spam-shooting has no direct mechanical cost worth discouraging via
reward. Enabling a penalty here risks discouraging legitimate exploratory
fire early in training, before the agent has learned to aim -- i.e. it
would likely slow convergence more than it saves. Revisit only if real
training runs show the agent spamming SHOOT in a way that visibly hurts
performance (e.g. via the reward decomposition dashboard,
scripts/plot_reward_decomposition.py)."""

SHOT_NO_TARGET_RADIUS: float = 350.0
"""Distance threshold (in arena world units) beyond which the nearest enemy
is not considered a valid target for R_SHOOT_WHILE_NO_TARGET purposes.
A shot fired when the nearest enemy is farther than this value sets
"shot_fired_with_no_target" in step_events (see rewards.py).

Derived (not a placeholder) from the real arena dimensions in core_env.py:
ARENA_WIDTH=960, ARENA_HEIGHT=680 -> diagonal = sqrt(960**2 + 680**2)
~= 1176.4 -> 0.3 * diagonal ~= 352.9, rounded to 350.0. Currently inert
since R_SHOOT_WHILE_NO_TARGET is disabled (see decision above); kept
accurate so step_events["shot_fired_with_no_target"] stays meaningful for
TensorBoard inspection and so the constant is correct if this term is
enabled later."""
