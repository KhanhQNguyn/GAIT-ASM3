"""Single source of truth for ALL Part II reward constants. rewards.py is
the ONLY module allowed to read these to compute a reward -- no other file
should hardcode a reward number. scripts/generate_report_tables.py (project
root) imports this module directly to keep the report's reward table in
sync with the code.

Capped at <= 8 terms per the architecture principle: 5 required by the spec
+ up to 2 ACTIVE optional shaping terms, each justified below. The 2-slot
guideline is a self-imposed preference, not a rubric rule -- the rubric only
demands that optional shaping be justified -- so the 2026-08-28 rebalance
intentionally takes 4 active slots (R_DAMAGE_DEALT_PER_HP, R_TIME_STEP_PENALTY,
R_SHOOT_TOWARD_ENEMY, R_WALL_PROXIMITY_PER_STEP) because each one targets a
MEASURED failure of a previous training iteration (passivity, no combat
gradient, no spawner discovery, wall-retreat). If you add a new term, you
must also add its one-line justification here (rubric checks for this) and
log it separately to TensorBoard in rewards.py.

REBALANCE (2026-08-28, team decision after diagnosing degenerate policies):
evaluation of the tuned_v1/v2 models showed every policy collapsing to
"camp a corner and soak contact damage" (0 enemy kills in deterministic
rollouts). Root causes, each fixed here or in config/arena.json:
  - fighting was net-negative before aim skill developed (a contact trade
    costs more reward than a kill earns, and a kill needs 2 landed hits);
  - hiding was free (no per-step cost);
  - enemies were far slower than the player, so neither fighting nor
    fleeing was ever forced.
The two ACTIVE shaping terms are now R_DAMAGE_DEALT_PER_HP (dense signal
for landing hits) and R_TIME_STEP_PENALTY (makes passivity costly).
R_APPROACH_NEAREST_ENEMY and R_SHOOT_WHILE_NO_TARGET are retained at 0.0
(inert, documented decisions) so the report's reward table stays stable
and the gating/cap machinery stays tested.
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
"""DECISION: KEPT at -100.0. Strong negative reward on death, on top of
R_DAMAGE_TAKEN_PER_HP for the killing blow -- must dominate any single
episode's positive rewards so the agent reliably prioritizes survival.

Ablation (Task 7): two 100k-timestep style-1 PPO runs
(`--config tuned_v1 --curriculum on --seed 0`), identical except
`--death-penalty -100` vs `--death-penalty -30` (via
`scripts/train.py --death-penalty` and
`scripts/plot_death_penalty_ablation.py`). Honest finding: in BOTH runs
`rollout/ep_len_mean` stayed pinned at 1200 (the episode-length cap) at
every logged checkpoint from ~4k steps onward -- the agent never died in
either run, so R_DEATH never actually fired and the two runs optimized
what was, in practice, an identical reward function (every panel matches,
including `ep_rew_mean` to two decimal places). The ablation is therefore
genuinely inconclusive on which magnitude trains better at this budget;
that is reported honestly here rather than papered over with an invented
preference. See report/figures/death_penalty_ablation_style1.png for the
evidence.

Kept at -100.0 rather than weakened to -30 because (a) the ablation found
no behavioral difference to justify a change, and (b)
tests/test_reward_terms.py requires R_DEATH < -45.0
(test_death_penalty_dominates_a_full_episode_of_positive_reward: 5 enemy
kills x R_KILL_ENEMY=5.0 + 1 spawner kill x R_KILL_SPAWNER=20.0 = +45.0
must still net negative after death) -- -30 would fail that test outright
regardless of the ablation's outcome."""

# --- Optional shaping terms (<= 2 ACTIVE; inert 0.0 terms retained below) ---

R_DAMAGE_DEALT_PER_HP: float = 0.05
"""DECISION (2026-08-28 rebalance): NEW ACTIVE shaping term. Positive reward
per HP of damage player projectiles deal to enemies, before any kill lands.
Justification: with R_KILL_ENEMY alone, combat is unrewarded until a whole
kill lands -- and an enemy needs 2 hits (30 HP vs 25 damage), so early
training gets essentially no gradient toward shooting at all. That is the
primary reason every tuned_v1/v2 policy collapsed to corner-camping with 0
enemy kills (see the module docstring's rebalance note). A per-HP term
gives dense signal on every landed hit (+1.25 per 25-damage hit), keeps
the kill bonus meaningful on top (5.0 + 1.5 = 6.5 for a full 30-HP enemy),
and cannot be farmed by passivity: HP only leaves an enemy via player
projectiles. Not counted as one of the "risky" shaping terms because it
shapes toward the spec's own required behavior (destroying enemies)."""

R_TIME_STEP_PENALTY: float = -0.01
"""DECISION (2026-08-28 rebalance): NEW ACTIVE shaping term. Small fixed
cost per step survived, applied unconditionally every step. Justification:
with no per-step cost, "hide in a corner and soak contact damage" is a
free local optimum -- the agent loses nothing by doing nothing for 1200
steps, which is exactly what the tuned_v1/v2 policies converged to. At
-0.01/step an idle 1200-step episode costs -12, so doing nothing is now
strictly worse than even a single enemy kill (+5) plus its damage-dealt
reward; the agent must earn to offset the clock. Kept small relative to
every event term so it shapes pacing, not behavior, and cannot make death
preferable (1200 * -0.01 = -12 >> -R_DEATH=100)."""

R_WALL_PROXIMITY_PER_STEP: float = -0.05
"""DECISION (2026-09-08: made CORNER-AWARE, originally -0.02 from the
2026-08-28 rebalance, raised to -0.05 same day): ACTIVE shaping term.
Penalty while the player stands within WALL_PROXIMITY_MARGIN of any arena
wall, ramping linearly from 0 at the margin edge to the full value AT the
wall. Since 2026-09-08 the penalty SUMS the two nearest walls' ramps, so
standing in a corner automatically costs ~2x a straight wall -- the
"stuck in a corner" deterrent without a separate harsh penalty or cliff.
Rationale: head-to-head aim-stats evals (scripts/eval_aim_stats.py) of
every trained checkpoint showed wall_frac 0.74-0.94 at -0.02 -- the term
was noise against the kill economy, so every policy hugged walls and
sprayed bullets into the enemy queue (kills land even unaimed because
enemies walk INTO projectiles). At -0.05 a full parked episode costs
~-40, and a corner-parked one ~-80, enough to make open-field kiting the
cheaper strategy; still linear-ramped (steered, not teleported) and still
small next to a death (-100)."""

WALL_PROXIMITY_MARGIN: float = 120.0
"""Distance (arena world units) from any wall at which
R_WALL_PROXIMITY_PER_STEP starts ramping up. 120 units is ~1/8 of the
arena's smaller dimension (680) -- deep enough that the agent has room to
turn around, shallow enough that most of the arena stays penalty-free."""

R_AIMED_HIT_BONUS: float = 1.0
"""DECISION (2026-09-08): ACTIVE shaping term. Extra reward when a player
projectile that was fired TOWARD its objective actually hits an enemy or
spawner: R_AIMED_HIT_BONUS * (graded alignment stamped on the projectile
at fire time, in [0, 1]). Separates INTENDED hits (dead-on: +1.0 on top
of the +1.25 damage-dealt and +5 kill) from LUCKY hits (enemies walking
into spray pay ~0 bonus). Justification: aim-stats evals showed kills
landing with mean shot alignment 0.00-0.16 -- the kill economy paid luck
and spray as well as aim, so the policy never had to learn to aim. This
is the only lever that pays intent directly; it is event-based (paid only
on an actual hit), so it cannot be farmed by parking and firing. Order
kept: R_KILL_SPAWNER (20) > R_KILL_ENEMY (5) > a dead-on aimed hit
(<= 1.0 + 1.25 + 0.12 aim shaping) so phase progression stays dominant."""

R_SHOOT_TOWARD_ENEMY: float = 0.12
"""DECISION (graded AND raised 0.06 -> 0.12 on 2026-09-08, after the
2026-08-28 rebalance): ACTIVE shaping term. Positive reward each time the
player fires a projectile
toward its CURRENT OBJECTIVE: (a) the nearest enemy within
SHOT_NO_TARGET_RADIUS when one exists, or (b) -- when no enemy is in range
-- the nearest active spawner (no distance gate; spawners are static, and
aiming at one is how the agent discovers spawner kills). The payout is now
GRADED: R_SHOOT_TOWARD_ENEMY * max(0, cos(diff)), diff = angle between the
shot direction and the objective. Replaces the old binary ~45-degree
window, which paid nothing until the shot was already inside the window --
zero gradient while approaching alignment. That mattered most for style 2,
where the aim is coupled to the (cardinal-only) movement direction and the
policy must learn to reposition until an enemy lands on an axis, then
charge head-on. Justification: reward only lands on a HIT and hits were too
rare early to provide a gradient, so PPO down-weighted every shoot action;
this term pays at the AIM level (aim +0.12 max, hit +1.25, kill +5.0). The
spawner fallback exists because ZERO spawner kills ever occurred across all
tuning runs -- R_KILL_SPAWNER (+20) and R_PHASE_PROGRESS (+50) were never
sampled without aim-level signal. It cannot be farmed: partial credit
decays as cos(diff), so spamming at 90+ deg off pays nothing."""

R_APPROACH_NEAREST_ENEMY: float = 0.0
"""DECISION (2026-08-28 rebalance): DISABLED, set to 0.0 (was 0.01,
gated + capped). Rationale: the term paid the agent for CLOSING distance
to the nearest enemy -- but enemies already seek the player, so distance
closes itself; in practice the term could be farmed without fighting, and
the diagnosis of the degenerate corner-camping policies showed approach
shaping was irrelevant next to the real problem (no dense combat reward,
no cost to passivity), now addressed by R_DAMAGE_DEALT_PER_HP and
R_TIME_STEP_PENALTY. The gating/cap implementation machinery in
rewards.py is RETAINED and still tested (tests monkeypatch the constant)
so re-enabling or retuning later is a one-line change. Prior team decision
history (Member D kept vs Member C drop) is recorded in
docs/DECISIONS.md; this rebalance supersedes it. Update report section 3."""

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
an enemy is fairly close.

Formerly also served as R_APPROACH_NEAREST_ENEMY's engage-range gate;
since that term is now disabled (see its decision above), this constant's
only remaining role is the (itself disabled) shot-penalty term, so it is
currently inert end to end. RETAINED because both consumers are one-line
re-enables away from needing it, and tests still exercise the machinery.
TODO: tune against the real weapon/projectile range in config/arena.json
if either consumer term is ever re-enabled, and record the final value in
report section 3/4."""
