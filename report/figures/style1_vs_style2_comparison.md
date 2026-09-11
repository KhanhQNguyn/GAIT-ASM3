# Style 1 vs Style 2 comparison (ppo, tuned_v3, curriculum=on, 20 episodes each, seed=0; deterministic + stochastic action-selection passes)

| Style | Sampling | Episodes | Avg return | Avg survival (steps) | Avg phases reached | Avg enemies destroyed | Avg spawners destroyed |
|---|---|---|---|---|---|---|---|
| 1 | deterministic | 20 | 157.50 | 972.6 | 1.60 | 17.60 | 2.10 |
| 1 | stochastic | 20 | 238.87 | 892.7 | 3.95 | 18.50 | 7.15 |
| 2 | deterministic | 20 | 83.86 | 648.5 | 2.50 | 12.10 | 3.65 |
| 2 | stochastic | 20 | 363.81 | 930.1 | 4.20 | 23.65 | 7.55 |

*Enemies destroyed = via player projectile only (see this script's module docstring) -- an enemy destroyed by touching the player is not counted here.*
