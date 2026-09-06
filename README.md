# RL Assignment — Gridworld (Part I) + Deep RL Arena (Part II)

Two-part reinforcement learning project (RMIT, 40% assessment):

- **Part I** (`part1_gridworld/`) — classical value-based RL (Q-learning, SARSA,
  and Expected SARSA as a creativity extension) in a visually rendered Pygame
  gridworld, across 7 levels of increasing complexity.
- **Part II** (`part2_arena/`) — deep RL (Stable-Baselines3, PPO/DQN) controlling
  a ship in a real-time Pygame arena, under two distinct control schemes.

See `assessment_requirements_summary.md` (project root, one level up from here
if you're reading this inside `project/`) for the authoritative spec and
rubric, and `RUBRIC_MAP.md` for how each file here maps to rubric points.

## Status

Both parts are fully implemented. `pytest part1_gridworld/tests part2_arena/tests` passes 100%
with zero skips. Part I has real trained Q-tables, CSV episode logs, and comparison-script
figures under `part1_gridworld/`. Part II's `models/`/`logs/` currently only hold short smoke
runs confirming `train.py -> models/ -> eval_style{1,2}.py` connects end-to-end — the full
100k-600k-timestep training runs the report/video need are still outstanding.

## Setup

```
pip install -r requirements.txt
```

Each part also has its own `requirements.txt` if you want isolated envs.

## Layout

```
part1_gridworld/   classical RL gridworld (see part1_gridworld/README below)
part2_arena/        deep RL arena (see part2_arena below)
report/               report template + generated figures/tables
scripts/                cross-part utility scripts (report table generation)
RUBRIC_MAP.md            module -> rubric row -> points
CONTRIBUTIONS.md          student numbers + contribution log
VIDEO_SCRIPT.md            demo video shot list
SUBMISSION_CHECKLIST.md     final submission checklist
```

## Running

Part I:
```
cd part1_gridworld
python main.py
```

Part II training:
```
cd part2_arena
python scripts/train.py --style 1 --algo ppo --timesteps 300000
python scripts/eval_style1.py
python scripts/eval_style2.py
```

## Report table generation

Keep `report/figures/reward_tables.md` in sync with the actual reward
constants (both parts) by running:
```
python scripts/generate_report_tables.py
```
#
