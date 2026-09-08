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

R_WALL_PROXIMITY_PER_STEP: float = -0.02
"""DECISION (2026-08-28 rebalance): ACTIVE shaping term. Small penalty while
the player stands within WALL_PROXIMITY_MARGIN of any arena wall, ramping
linearly from 0 at the margin edge to the full value AT the wall.
Justification: eval of the tuned_v3 style-2 policy showed it retreating to
walls and getting cornered -- walls leave enemies only ~90-180 degrees of
approach, so wall-hugging is a rational hiding spot even for a policy that
can fight. This term makes open-field positioning the cheaper strategy:
parked at a wall for a full episode costs -24 (comparable to the time
penalty), while the agent's measured combat skill (kills + damage-dealt
rewards) more than offsets staying out. Scales with proximity rather than
being a flat cliff so the agent is gently steered away, not teleported."""

WALL_PROXIMITY_MARGIN: float = 120.0
"""Distance (arena world units) from any wall at which
R_WALL_PROXIMITY_PER_STEP starts ramping up. 120 units is ~1/8 of the
arena's smaller dimension (680) -- deep enough that the agent has room to
turn around, shallow enough that most of the arena stays penalty-free."""

R_SHOOT_TOWARD_ENEMY: float = 0.06
"""DECISION (2026-08-28 rebalance): ACTIVE shaping term. Positive reward
each time the player fires a projectile roughly toward its CURRENT
OBJECTIVE: (a) the nearest enemy within SHOT_NO_TARGET_RADIUS when one
exists, or (b) -- when no enemy is in range -- the nearest active spawner
(no distance gate; spawners are static, and aiming at one is how the agent
discovers spawner kills). Both cases use the same ~45 degree alignment
window. Justification: training curves showed the agent was UNLEARNING
shooting -- reward only lands on a HIT, and hits were too rare early to
provide a gradient, so PPO down-weighted every shoot/approach action. This
term pays out at the AIM level (aim +0.06, hit +1.25, kill +5.0), and the
spawner fallback exists because ZERO spawner kills ever occurred across
all tuning runs -- R_KILL_SPAWNER (+20) and R_PHASE_PROGRESS (+50) were
never sampled, so the biggest reward gradient in the game was unreachable
without aim-level signal. It cannot be farmed: it requires the shot to
actually point at a nearby enemy or at the spawner objective."""

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
