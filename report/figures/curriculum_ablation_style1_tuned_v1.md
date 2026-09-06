# Curriculum ablation -- Style 1 (tuned_v1)

| Run | Final ep_rew_mean | Steps to reach 3.3 |
|---|---|---|
| curriculum ON  | 4.07 | 196608 |
| curriculum OFF | -6.05 | not reached |

One of the two runs never reached the threshold within its logged timesteps -- see the raw final rewards above instead of a steps-to-threshold claim.
