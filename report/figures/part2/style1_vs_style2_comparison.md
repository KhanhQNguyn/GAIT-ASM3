# Style 1 vs Style 2 comparison (ppo, tuned_v3, curriculum=on, 20 episodes each, seed=0)

| Style | Episodes | Avg return | Avg survival (steps) | Avg phases reached | Avg enemies destroyed | Avg spawners destroyed |
|---|---|---|---|---|---|---|
| 1 | 20 | 73.18 | 800.2 | 2.00 | 13.60 | 2.75 |
| 2 | 20 | 25.86 | 718.6 | 1.55 | 12.00 | 2.05 |

*Enemies destroyed = via player projectile only (see this script's module docstring) -- an enemy destroyed by touching the player is not counted here.*
