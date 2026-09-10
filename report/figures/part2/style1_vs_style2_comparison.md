# Style 1 vs Style 2 comparison (ppo, tuned_v3, curriculum=on, 20 episodes each, seed=0)

| Style | Episodes | Avg return | Avg survival (steps) | Avg phases reached | Avg enemies destroyed | Avg spawners destroyed |
|---|---|---|---|---|---|---|
| 1 | 20 | 5.12 | 807.6 | 0.85 | 11.75 | 0.95 |
| 2 | 20 | -7.24 | 649.6 | 1.70 | 9.65 | 2.35 |

*Enemies destroyed = via player projectile only (see this script's module docstring) -- an enemy destroyed by touching the player is not counted here.*
