# Style 1 vs Style 2 comparison (ppo, tuned_v3, curriculum=on, 20 episodes each, seed=0)

| Style | Episodes | Avg return | Avg survival (steps) | Avg phases reached | Avg enemies destroyed | Avg spawners destroyed |
|---|---|---|---|---|---|---|
| 1 | 20 | 98.66 | 895.4 | 1.45 | 15.80 | 2.05 |
| 2 | 20 | 31.06 | 641.8 | 2.30 | 10.50 | 3.20 |

*Enemies destroyed = via player projectile only (see this script's module docstring) -- an enemy destroyed by touching the player is not counted here.*
