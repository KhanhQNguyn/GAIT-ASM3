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

REQUIRED implementation shape (not optional -- an ungated flat 0.01/step
over a 1200-step episode is +12, which rivals R_KILL_ENEMY=5 and lets the
agent farm this term by loitering, see docs/AUDIT_main.md 5.8). All three
are implemented in rewards.py::compute_reward:
  - reward only the per-step DECREASE in distance to the nearest enemy
    (distance_delta < 0), scaled by this constant -- not mere proximity;
  - apply it ONLY while the nearest enemy is outside weapon/engage range
    (reuses SHOT_NO_TARGET_RADIUS below as the engage-range threshold --
    the same distance already used to decide "close enough to be a valid
    shooting target" is exactly "close enough that the agent should be
    shooting, not still being paid to approach");
  - cap the cumulative per-episode contribution of this term at
    R_KILL_ENEMY, via the optional step_events["cumulative_approach_reward"]
    (the running per-episode total BEFORE this step, tracked by the
    caller) -- see compute_reward's docstring for the exact clamp.

Document the final decision (kept/removed/tuned) in
report/report_template.md section 3."""

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
~= 1176.4 -> 0.3 * diagonal ~= 352.9, rounded to 350.0 -- roughly 30% of
the diagonal, deliberately tight so a shot only counts as "on target" when
an enemy is fairly close. Also now doing double duty as
R_APPROACH_NEAREST_ENEMY's engage-range gate (see above): both uses share
the same underlying question ("is the nearest enemy close enough that the
agent should be shooting, not just approaching or spraying"), so one
tuned distance serves both rather than drifting into two similar
constants. R_SHOOT_WHILE_NO_TARGET itself is currently disabled (see its
decision above), so this value's shot-penalty role is inert for now, but
its distance/gating role for R_APPROACH_NEAREST_ENEMY is active. TODO:
tune against training behaviour (and against the real weapon/projectile
range in config/arena.json) and record the final value in report
section 3/4."""
