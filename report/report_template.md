# RL Assignment Report

**STRICT constraint: maximum 10 pages total, including images. No appendix —
anything beyond page 10 will not be considered by markers.** Budget roughly
one page per section below; trim aggressively rather than exceed the limit.

Generated tables/plots referenced below should live in `report/figures/` and
be produced by `scripts/generate_report_tables.py` and the various
`plot_*.py` scripts in each part, so the report never drifts from the code.

---

## 1. Environment Descriptions (~1-1.5 pages)

### 1.1 Part I — Gridworld
TODO: grid size, levels, mechanics, rules (rocks, fire, apples, keys,
chests, monsters), how Pygame rendering/interaction works.

### 1.2 Part II — Arena
TODO: arena layout, player/enemy/spawner/projectile systems, phase system,
what makes this feel like a real-time arena rather than a tile grid.

## 2. Observation Design (~1 page)

TODO: table of every feature in the Part II observation vector, its index,
its meaning, and why it was included. Reference `part2_arena/arena/obs.py`.

## 3. Reward Design (~1 page)

TODO: table of every reward term (Part I: apple/key/chest/death; Part II:
`rewards_config.py` constants) with justification for each, especially any
optional shaping terms. Pull directly from
`report/figures/reward_tables.md` (generated).

## 4. Hyperparameter Exploration (~1-1.5 pages)

TODO: what was tuned (learning rate, gamma, epsilon schedule, PPO/DQN
hyperparameters), what values were tried, and evidence (table or plot) of
the effect. Reference `part1_gridworld/config/training_config.json` and the
`train.py` runs logged to `part2_arena/logs/`.

## 5. Control Scheme Comparison (~1 page)

TODO: Style 1 (rotation+thrust) vs. Style 2 (directional) — training curves,
qualitative behavior differences, screenshots. If the PPO-vs-DQN ablation
was run, summarize it here or in a dedicated creativity subsection.

## 6. Training Evidence (~1-1.5 pages)

TODO: Part I training curves (Q-learning, SARSA, Expected SARSA, monster
levels, intrinsic reward on/off comparison — see
`part1_gridworld/src/plot_results.py` and `compare_algorithms.py`). Part II
TensorBoard screenshots / exported curves.

## 7. Originality Justification (~0.5 page)

TODO: what in this submission goes beyond the minimum spec, and why it's
your own work (reference any reused tutorial/assignment code per the reuse
policy).

## 8. Creativity Extensions (~0.5-1 page)

TODO: summarize whichever of the four extensions were completed —
PPO vs. DQN ablation, reward decomposition dashboard, curriculum learning,
Expected SARSA — with evidence for each.

## 9. Team

- Student numbers + contribution summary: see `CONTRIBUTIONS.md` (copy the
  final table in here).
- Video link: TODO.
