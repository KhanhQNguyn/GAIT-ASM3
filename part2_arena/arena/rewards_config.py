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
"""Small per-step shaping reward for reducing distance to the nearest enemy,
to speed up early training before the agent discovers that engaging enemies
is valuable at all. Document the final decision (kept/removed/tuned) in
report/report_template.md section 3.

REQUIRED implementation shape (not optional -- an ungated flat 0.01/step
over a 1200-step episode is +12, which rivals R_KILL_ENEMY=5 and lets the
agent farm this term by loitering, see docs/AUDIT_main.md 5.8):
  - reward only the per-step DECREASE in distance to the nearest enemy
    (distance_delta < 0), scaled by this constant -- not mere proximity;
  - apply it ONLY while the nearest enemy is outside weapon/engage range
    (so it stops paying out once the agent should be shooting, not chasing);
  - cap the cumulative per-episode contribution of this term (e.g. to
    <= R_KILL_ENEMY) so it can never dominate the real objective."""

R_SHOOT_WHILE_NO_TARGET: float = 0.0
"""TODO justify or remove: placeholder for a potential small penalty on
shooting with no enemy in range, to discourage spamming the shoot action.
Defaults to 0.0 (disabled) -- only enable this with a documented
justification, since an undocumented shaping term is worse for the report
than not having one."""

SHOT_NO_TARGET_RADIUS: float = 150.0
"""Distance threshold (in arena world units) beyond which the nearest enemy
is not considered a valid target for R_SHOOT_WHILE_NO_TARGET purposes.
A shot fired when the nearest enemy is farther than this value sets
"shot_fired_with_no_target" in step_events (see rewards.py).

The arena is 960×680 (core_env.ARENA_WIDTH/HEIGHT), diagonal ≈ 1173 world
units. 150.0 is therefore ~13% of the diagonal -- deliberately tight: a
shot only counts as "on target" when an enemy is fairly close, so the
R_SHOOT_WHILE_NO_TARGET penalty discourages long-range spray without
punishing reasonable mid-range shots. TODO: tune against training behaviour
(and against the real weapon/projectile range in config/arena.json) and
record the final value in report section 3."""
