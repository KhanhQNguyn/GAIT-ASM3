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

# Reward on death (fire or monster contact). The spec does not mandate a
# death penalty -- it only specifies that death ends the episode -- so this
# defaults to 0.0 to stay strictly literal to the spec. If the team decides
# to add a death penalty as a deliberate design choice, change this value
# AND document the justification in report/report_template.md section 3,
# since it is not one of the spec-mandated reward terms.
REWARD_DEATH: float = 0.0

# No per-step penalty is specified by the spec; kept at 0.0 and exposed here
# (rather than hardcoded in env.py) purely so it stays visible as a
# deliberate "not used" choice, discoverable by generate_report_tables.py.
REWARD_STEP: float = 0.0
