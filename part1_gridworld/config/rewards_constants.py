"""Single source of truth for Part I reward values.

Per the assignment spec, these mechanics/rewards must NOT be altered by any
helper function or shortcut. Every place that computes a reward in
part1_gridworld (env.py, intrinsic.py, tests) must import these constants
rather than hardcoding numbers, so `scripts/generate_report_tables.py` (at
the project root) always reflects the actual values used in training.

Do not add new reward terms here without also updating:
  - config/schema.md (mechanics description)
  - tests/test_env_rules.py (regression test pinning the value)
  - report/report_template.md, section 3 (Reward Design)
"""

# Collecting an apple.
REWARD_APPLE: float = 1.0

# Picking up the key. The spec is explicit that this must be zero -- the
# key's only value is unlocking the chest, it is not a reward source itself.
REWARD_KEY: float = 0.0

# Opening the chest (requires holding the key).
REWARD_CHEST: float = 2.0

# Reward on death (fire or monster contact). REVIEWED AND INTENTIONALLY KEPT
# AT 0.0 -- do not "fix" this into a negative value. The spec's reward list
# (apples/key/chest) has no death term, death is only specified as ending
# the episode, and the spec is explicit that "rewards and mechanics must not
# be altered." Per lesson.md (docs/lesson.md) -- feedback from a previous
# team's assignment -- do not change a spec-defined reward without a
# justification stronger than "it would make a comparison look cleaner."
#
# Task 2 (level1) still gets its required SARSA-vs-Q-learning contrast
# without any penalty: dying ends the episode and forfeits every apple not
# yet collected, so death already carries an implicit, emergent cost from
# the MDP's own structure. SARSA's on-policy target incorporates the
# exploring policy's real (nonzero) chance of stepping into the fire gap,
# while Q-learning's off-policy max assumes optimal play afterward and
# discounts that risk -- exactly the mechanism level1.json's own
# `_design_note` describes. No hand-added penalty is needed to produce it.
REWARD_DEATH: float = 0.0

# No per-step penalty is specified by the spec; kept at 0.0 and exposed here
# (rather than hardcoded in env.py) purely so it stays visible as a
# deliberate "not used" choice, discoverable by generate_report_tables.py.
REWARD_STEP: float = 0.0
