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

Both parts are fully implemented. `pytest part1_gridworld/tests part2_arena/tests`
passes 100% with zero skips (run inside the venv - see Setup). Part I has trained
Q-tables, CSV episode logs, and comparison figures.

Part II shipped checkpoints (2026-09-08 aim-fix rebalance: graded aim shaping,
8px projectiles, wall penalty -0.05/step):

- `models/style1_ppo_tuned_v3_curriculum(.zip + _best/)` - style 1, PPO, 300k steps.
- `models/style2_ppo_tuned_v4_curriculum(.zip + _best/)` - style 2, PPO, 300k steps.
  `tuned_v4` = `tuned_v3` with `gamma` 0.995 -> 0.999, so the -100 death penalty is
  visible at episode-scale horizons (at 0.995 a death ~800 steps away discounts to
  ~-1.8 and wall-camping looks optimal).

Headless aim metrics for the shipped style-2 model (5 stochastic episodes,
`scripts/eval_aim_stats.py`): mean shot alignment 0.16 (0.00 pre-fix), shots
inside 30 deg of the enemy bearing 9% (0% pre-fix), wall-hugging step fraction
0.54 (0.94 pre-fix). Phase progression at 300k steps is still rare - longer
runs (500k-600k) improve it; see `docs/PART2_TUNING_GUIDE.md` for every tunable
parameter and its effect. Pre-fix checkpoints are kept under
`models/pre_aimfix_backup_2026-09-08/` for the report's before/after comparison.

## Setup

Recommended (isolated venv):
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1      # PowerShell (Set-ExecutionPolicy RemoteSigned -Scope CurrentUser if blocked)
pip install -r requirements.txt
pip install tbparse pandas        # used by the analysis scripts
pip install ruff                  # lint (rules in pyproject.toml); part1 has known pre-existing E501/I001 findings
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
# ~5-9 min for a 300k-step run on the dev machine (i5-12500H, headless)
python scripts/train.py --style 1 --algo ppo --timesteps 300000 --curriculum on --config tuned_v3 --seed 0
python scripts/train.py --style 2 --algo ppo --timesteps 300000 --curriculum on --config tuned_v4 --seed 0
```
- The two commands above are the shipped configuration (models under
  `models/`). `tuned_v4` is `tuned_v3` with `gamma` 0.999 - the 2026-09-08
  fix for style 2's wall-camping local optimum (see Status).
- `--algo ppo|dqn`, `--config <preset>` (see `config/hyperparams.json` - add new
  presets rather than editing existing ones), `--curriculum on|off`,
  `--seed N` for reproducibility.
- Ablation flags: `--death-penalty -30`, `--wall-penalty 0` (see
  `docs/PART2_TUNING_GUIDE.md`).
- Retraining overwrites the same-named model file; old checkpoints live in
  `models/pre_aimfix_backup_2026-09-08/`.

### Part II - arena evaluation (visual, records for the video)
```
cd part2_arena
python scripts/eval_style1.py          # defaults: --config tuned_v3 --curriculum on
python scripts/eval_style2.py          # defaults: --config tuned_v4 --curriculum on
```
- Defaults already load the shipped checkpoints; pass flags to override.
- `--sampling stochastic` (default) samples from the PPO policy - shows the
  full learned behaviour; `--sampling deterministic` uses the argmax action,
  which can collapse to a degenerate mode (walls / spray) - use stochastic
  for the demo video.
- `--algo ppo|dqn`, `--episodes N`, `--fps N`.
- Viewer controls in-game: Space pause, `[` / `]` playback speed, R restart
  episode, N skip to next episode, Esc quit.

### Part II - headless aim stats (quantitative eval, no window)
```
cd part2_arena
python scripts/eval_aim_stats.py --style 2 --config tuned_v4 --curriculum on --episodes 5 --sampling stochastic
python scripts/eval_aim_stats.py --style 1 --config tuned_v3 --curriculum on --episodes 5
```
Per-episode metrics: shots fired, mean graded aim alignment (1.0 = dead-on),
fraction of shots within 30 deg of the enemy bearing, damage dealt/taken,
kills, wall-hugging step fraction, phase reached. Use `--checkpoint best`
for the EvalCallback's best-by-eval-reward save instead of the final model.

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
