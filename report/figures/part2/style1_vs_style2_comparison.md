# Style 1 vs Style 2 comparison (ppo, tuned_v1, curriculum=on, 20 episodes each, seed=0)

| Style | Episodes | Avg return | Avg survival (steps) | Avg phases reached | Avg enemies destroyed | Avg spawners destroyed |
|---|---|---|---|---|---|---|
| 1 | 20 | -13.54 | 1200.0 | 0.00 | 2.00 | 0.00 |
| 2 | 20 | 114.05 | 1200.0 | 2.00 | 0.00 | 3.00 |

*Enemies destroyed = via player projectile only (see this script's module docstring) -- an enemy destroyed by touching the player is not counted here.*
