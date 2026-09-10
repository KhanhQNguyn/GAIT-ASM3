"""Single source of truth for ALL Part II reward constants. rewards.py is
the ONLY module allowed to read these to compute a reward -- no other file
should hardcode a reward number. scripts/generate_report_tables.py (project
root) imports this module directly to keep the report's reward table in
sync with the code.

5 spec-required terms + optional shaping, each justified below. The "<= 2
extra shaping terms" line is a self-imposed preference, not a rubric rule --
the spec (assessment_requirements_summary.md line 146) only demands that
optional shaping be justified. The set carries FOUR active shaping terms
(R_DAMAGE_DEALT_PER_HP, R_TIME_STEP_PENALTY, R_WALL_PROXIMITY_PER_STEP,
R_AIMED_HIT_BONUS), each targeting a MEASURED failure of a previous
training iteration (passivity, no combat gradient, wall-retreat,
spray-not-aim). R_SHOOT_TOWARD_ENEMY was retired 2026-09-08d (it paid for
FIRING toward a target, hit or miss -- the direct spray incentive; see
BUG_LESS_ATTACK.md). This is still a Part II-J / originality-justification
point (docs/RULES.md R-REWARD-3) -- the report's section 3 must argue why
4 shaping terms is not reward hacking. If you add a new term, add its
one-line justification here and log it to TensorBoard in rewards.py.

REBALANCE (2026-08-28): tuned_v1/v2 models collapsed to "camp a corner and
soak contact damage" (0 kills). Fixed by non-kamikaze enemies + faster
enemies + R_TIME_STEP_PENALTY (passivity now costs) + R_DAMAGE_DEALT_PER_HP
(dense hit gradient). 2026-09-08 then added R_AIMED_HIT_BONUS /
R_SHOOT_TOWARD_ENEMY (aim) and bumped enemy.contact_damage to 25.

REBALANCE (2026-09-08c "glass-cannon fix", see FIX_PART2.md): the ~1M-step
tuned_v3/v4 models over-corrected into an aggressive suicide policy (dies
~step 600-900 at phase ~3; 73% / 47% death rate) because per-episode
phase_progress (+158) + kills dwarfed R_DEATH (-100) -- surviving to the
step cap was never reward-optimal. Fixed WITHOUT a new shaping term, in two
parts: (1) STRUCTURAL -- the cumulative PROGRESSION reward an episode can
bank (kill_spawner + phase_progress; NOT kill_enemy since 2026-09-08d) is
clamped to PROGRESS_REWARD_EPISODE_CAP = abs(R_DEATH) in rewards.py, so a
death can never be out-earned by rushing to a high phase; (2) TUNING --
damage made more costly while the world stays hostile enough that passivity
still ~= death: R_DAMAGE_TAKEN_PER_HP -0.5 -> -0.7 here, plus
enemy.contact_damage 25 -> 20, enemy_speed_gain_per_phase 0.55 -> 0.5 and
rotate_speed_rad 0.14 -> 0.2 in config/arena.json.

REBALANCE (2026-09-08d "less-attack fix", see BUG_LESS_ATTACK.md): the
2026-09-08c 1M models dodged and sprayed instead of fighting -- exact
per-shot instrumentation showed real aim alignment ~0.3, hit-rate ~0.1,
and kills dropping to EXACTLY 0 for the ~30% of the episode spent past the
cap (the cap zeroed kill_enemy, so post-cap the agent had no reason to
finish anything). Three changes: (A) the cap no longer covers kill_enemy
(defending yourself always pays +5); (B) R_SHOOT_TOWARD_ENEMY 0.12 -> 0.0
(removed the pay-for-facing spray incentive) and R_AIMED_HIT_BONUS 1.0 ->
3.0 (a lined-up hit now clearly beats a spray hit); (D) core_env grew a
real _shots_fired counter so scripts/eval_aim_stats.py stops over-reporting
alignment (it had said 0.87 vs the true ~0.3).

REBALANCE (2026-09-08e "spec-alignment", see FIX_PART2.md): the
PROGRESS_REWARD_EPISODE_CAP itself is now DISABLED (+inf, machinery kept
inert + monkeypatch-tested). It clamped phase_progress and kill_spawner --
two of the five reward terms the spec REQUIRES (section 5) -- for ~70% of
the episode; the trained agent was destroying ~4 spawners/episode and paid
for ~1.4. The section-5 progression economy is now uncapped as intended;
the glass-cannon is shaped only through the spec's own negative terms
(R_DEATH, R_DAMAGE_TAKEN_PER_HP) plus the hostile world.

REBALANCE (2026-09-08f, magnitude tune within section 5): the cap-free 1M
models cleared ~4 phases/episode (good -- spec-optimal spawner-focus) but
died ~50-90% of the time because uncapped R_PHASE_PROGRESS=50 made 4 phases
(+200) worth 2x R_DEATH. R_PHASE_PROGRESS 50 -> 30 and R_DEATH -100 -> -150
(both still section-5-legal: a positive phase reward, and a "strong"
negative death reward) so surviving to keep clearing phases beats rushing
to phase 4 and dying. Not a cap -- every phase still pays, in full.

R_APPROACH_NEAREST_ENEMY / R_SHOOT_WHILE_NO_TARGET (0.0) and
PROGRESS_REWARD_EPISODE_CAP (+inf) are all retained inert -- documented
considered-and-rejected designs; their machinery stays tested.
"""

# --- Required by the spec ---

R_KILL_ENEMY: float = 5.0
"""Destroying a single enemy. Baseline positive signal for offense."""

R_KILL_SPAWNER: float = 20.0
"""Destroying a spawner. Larger than R_KILL_ENEMY because spawners gate
phase progression -- this must dominate enemy-killing in the agent's
incentive structure or it will farm enemies forever instead of advancing."""

R_PHASE_PROGRESS: float = 30.0
"""Reaching a new phase (all active spawners destroyed). The spec's
"positive reward for progressing to the next phase" -- a large single-event
signal (still 6x R_KILL_ENEMY, alongside R_KILL_SPAWNER=20 as the top-tier
event rewards), since phase progression is the closest thing this
environment has to a 'win' condition within an episode.

DECISION (2026-09-08f, lowered 50 -> 30, see FIX_PART2.md): with the
progression cap removed (2026-09-08e), an uncapped +50/phase made "rush to
phase 4 (+200 = 2x R_DEATH), then die" reward-optimal -- the 1M cap-free
models cleared ~4 phases/episode but died ~50-90% of the time, always at
phase 3-5. At +30 four phases is +120 (< abs(R_DEATH)=150), so surviving
and continuing to clear phases beats dying at phase 4. Still satisfies
section 5 ("positive reward for progressing") and stays a top-tier event
reward; this is a magnitude choice, not a structural cap on a required
term."""

R_DAMAGE_TAKEN_PER_HP: float = -0.7
"""Negative reward per HP of damage taken, scaled by HP rather than a flat
per-hit penalty so a single graze and a heavy hit are distinguished.

DECISION (2026-09-08c glass-cannon fix, see FIX_PART2.md): raised -0.5 ->
-0.7. At -0.5 with enemy.contact_damage 25, eating a full 100-HP bar of
contact cost only -50 reward -- less than the ~+6.25 each one-shot kill
pays, so the shipped ~1M-step tuned_v3/v4 policies learned to trade contact
for kills and die around step 600-900 (73% / 47% death rate; per-episode
phase_progress +158 and kills far outweighed R_DEATH -100). A first 300k
validation pass tried -1.0 (with the world also softened): that
over-corrected -- the policy went passive, wall-camped, survived to 1200 but
stopped clearing spawners. -0.7 with enemy.contact_damage 20 (a 5-hit
margin) makes a full bar of contact cost -70, i.e. ~11 kills to offset,
while the world is kept hostile (max_concurrent_enemies back to 18) so
camping still gets the player swarmed and killed. A scale change to an
already-required spec term (damage taken), not a new shaping term."""

R_DEATH: float = -150.0
"""The spec's "strong negative reward on death" term -- one-off, on top of
R_DAMAGE_TAKEN_PER_HP for the killing blow.

DECISION (2026-09-08f, -100 -> -150, see FIX_PART2.md): after the
progression cap was removed (2026-09-08e), the cap-free 1M models under
R_DEATH=-100 cleared ~4 phases/episode but died ~50-90% of the time (all
deaths at phase 3-5). Paired with R_PHASE_PROGRESS 50 -> 30, -150 makes
"survive and keep clearing phases" beat "rush to phase 4, bank the reward,
die": 4 phases now = +120 < 150. Kept below -200: inflating it further
tends to push PPO into the hyper-passive wall-camp (the 2026-08-28
failure). R-REWARD-4 (>= 10x another term -- here 30x R_KILL_ENEMY):
justified quantitatively by the measured death-rate response above rather
than by a fresh ablation; the `scripts/train.py --death-penalty` +
`scripts/plot_death_penalty_ablation.py` machinery is available if a
-100 vs -150 ablation plot is wanted for the report.

Earlier decision (2026-09-08c): NOT raised then, because the glass-cannon
was addressed on the DAMAGE side (R_DAMAGE_TAKEN_PER_HP -0.5 -> -0.7,
enemy.contact_damage 25 -> 20) while the (now-removed) progression cap
carried the rest.

Ablation (Task 7, PRE-lethality-rebalance -- superseded, kept for the
report's honesty): two 100k-timestep style-1 PPO runs
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

Floor: tests/test_reward_terms.py requires |R_DEATH| large enough that a
somewhat successful step (5 enemy kills + 1 spawner kill = +45) that then
dies still nets negative -- any value <= -46 satisfies it; -150 has ample
margin."""

# --- Optional shaping terms (<= 2 ACTIVE; inert 0.0 terms retained below) ---

R_DAMAGE_DEALT_PER_HP: float = 0.05
"""DECISION (2026-08-28 rebalance): NEW ACTIVE shaping term. Positive reward
per HP of damage player projectiles deal to enemies, capped per hit at the
enemy's remaining HP (no overkill credit -- see core_env._resolve_collisions).
Justification: with R_KILL_ENEMY alone, combat gives PPO no gradient until a
full kill lands, and early in training almost every shot misses -- that is
the primary reason every tuned_v1/v2 policy collapsed to corner-camping with
0 enemy kills (see the module docstring's rebalance note). A per-HP term
pays a dense signal on any landed damage. Enemies are one-shot since the
2026-09-08 change (base_health 25 == projectile_damage 25), so a clean hit
earns 25 * 0.05 = 1.25 here plus R_KILL_ENEMY 5.0 = 6.25 total; it cannot be
farmed by passivity because HP only leaves an enemy via player projectiles.
Not counted as one of the "risky" shaping terms because it shapes toward the
spec's own required behavior (destroying enemies)."""

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

R_AIMED_HIT_BONUS: float = 3.0
"""DECISION (2026-09-08, raised 1.0 -> 3.0 on 2026-09-08d, see
BUG_LESS_ATTACK.md): ACTIVE shaping term. Extra reward when a player
projectile that was fired TOWARD its objective actually hits an enemy or
spawner: R_AIMED_HIT_BONUS * (graded alignment stamped on the projectile
at fire time, in [0, 1]). Separates INTENDED hits (dead-on: +3.0 on top
of the +1.25 damage-dealt and +5 kill) from LUCKY hits (enemies walking
into spray pay ~0 bonus).

Raised 1.0 -> 3.0 because exact per-shot instrumentation (BUG_LESS_ATTACK.md)
found real fire-time alignment was only ~0.3 and hit-rate ~0.1 -- the
policy sprayed roughly toward the swarm and let one-shot enemies walk into
bullets, because at 1.0 an aimed hit (~+1) barely beat a spray hit
(+1.25 damage-dealt) plus the now-removed R_SHOOT_TOWARD_ENEMY. At 3.0 an
aimed hit (up to +3) clearly dominates a spray hit, so precise aim is the
better shooting strategy. Still event-based (paid only on an actual hit),
so it cannot be farmed by parking and firing; enemies are one-shot and
spawn-limited, so it cannot be farmed by re-hitting. Order kept:
R_KILL_SPAWNER (20) > R_KILL_ENEMY (5) > a dead-on aimed hit
(<= 3.0 + 1.25) so phase progression stays dominant."""

R_SHOOT_TOWARD_ENEMY: float = 0.0
"""DECISION (2026-09-08d, DISABLED 0.12 -> 0.0, see BUG_LESS_ATTACK.md):
was an ACTIVE graded shaping term -- R_SHOOT_TOWARD_ENEMY * max(0, cos(diff))
per shot, paid for FIRING toward the current objective whether or not the
shot hit. Exact per-shot instrumentation showed this was the direct spray
incentive: at real alignment ~0.3 it paid ~+0.04 per shot for merely
pointing at the swarm and pulling the trigger, so "dodge and spray roughly
toward the crowd" was locally profitable and the policy never had to learn
to actually hit. Removing it means a MISS now pays literally nothing;
shooting income comes only from landed damage (R_DAMAGE_DEALT_PER_HP) and
aimed hits (R_AIMED_HIT_BONUS, raised to 3.0). The graded
`_shot_toward_enemy_flag` machinery in core_env is retained (it still
stamps `aim_alignment` on every projectile for R_AIMED_HIT_BONUS), so
re-enabling this term is a one-line constant change. Spawner discovery is
still covered: R_AIMED_HIT_BONUS pays for aimed hits on the nearest
spawner when no enemy is in range, plus R_KILL_SPAWNER (+20). Drops the
active-shaping count 5 -> 4."""

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
If either consumer term is ever re-enabled, tune this against the real
weapon/projectile range in config/arena.json and record the final value in
report section 3/4."""
