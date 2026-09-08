# RL Assignment - Gridworld (Part I) + Deep RL Arena (Part II)

Two-part reinforcement learning project (RMIT, 40% assessment):

- **Part I** (`part1_gridworld/`) - classical value-based RL (Q-learning, SARSA,
  and Expected SARSA as a creativity extension) in a visually rendered Pygame
  gridworld, across 7 levels of increasing complexity.
- **Part II** (`part2_arena/`) - deep RL (Stable-Baselines3, PPO/DQN) controlling
  a ship in a real-time Pygame arena, under two distinct control schemes.

See `assessment_requirements_summary.md` (project root, one level up from here
if you're reading this inside `project/`) for the authoritative spec and
rubric, and `RUBRIC_MAP.md` for how each file here maps to rubric points.

## Status

Both parts are fully implemented and trained. `pytest part1_gridworld/tests part2_arena/tests`
passes 100% with zero skips (run inside the venv - see Setup). Part I has trained Q-tables,
CSV episode logs, and comparison figures. Part II has full tuned_v3 training runs for both
control styles (models under `part2_arena/models/`, TensorBoard runs under
`part2_arena/logs/`): style 1 learns open-field combat (10-13 kills/episode, positive
returns), style 2 learns positional combat plus phase progression (spawner kills + phase
advances - visible in stochastic evaluation). See `docs/PART2_TUNING_GUIDE.md` for every
tunable parameter and its effect.

## Setup

Recommended (isolated venv):
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1      # PowerShell (Set-ExecutionPolicy RemoteSigned -Scope CurrentUser if blocked)
pip install -r requirements.txt
pip install tbparse pandas        # used by the analysis scripts
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

### Part I - gridworld (interactive Pygame menu)
```
cd part1_gridworld
python main.py
```
Pick a level + algorithm in the menu (arrow keys / clicks), choose
Train + Render or Watch Only (loads the saved policy), press START.
Playback controls in-game: Space pause/resume, `[` / `]` speed, R restart
episode, Esc quit.

### Part II - arena training (headless, no window)
```
cd part2_arena
# ~5 min per 100k steps; style 2 phases reliably from ~500k steps
python scripts/train.py --style 1 --algo ppo --timesteps 100000 --curriculum on --config tuned_v3 --seed 0
python scripts/train.py --style 2 --algo ppo --timesteps 500000 --curriculum on --config tuned_v3 --seed 0
```
- `--algo ppo|dqn`, `--config <preset>` (see `config/hyperparams.json` - add new
  presets rather than editing existing ones), `--curriculum on|off`,
  `--seed N` for reproducibility.
- Ablation flags: `--death-penalty -30`, `--wall-penalty 0` (see
  `docs/PART2_TUNING_GUIDE.md`).

### Part II - arena evaluation (visual, records for the video)
```
cd part2_arena
python scripts/eval_style1.py --config tuned_v3 --curriculum on
python scripts/eval_style2.py --config tuned_v3 --curriculum on
```
- `--sampling stochastic` (default) samples from the PPO policy - shows the
  full learned behaviour including style 2's phase progression;
  `--sampling deterministic` uses the argmax action (smoother, but style 2
  never phases in argmax).
- `--algo ppo|dqn`, `--episodes N`, `--fps N`.
- Viewer controls in-game: Space pause, `[` / `]` playback speed, R restart
  episode, N skip to next episode, Esc quit.

### TensorBoard
```
cd part2_arena
tensorboard --logdir logs
```
Per-reward-term panels live under `reward_terms/*` /
`reward_terms_episode/*` (what feeds `scripts/plot_reward_decomposition.py`).

## Report table generation

Keep `report/figures/reward_tables.md` in sync with the actual reward
constants (both parts) by running:
```
python scripts/generate_report_tables.py
```
#
