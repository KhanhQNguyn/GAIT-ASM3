# Level Config Schema

Each `levelN.json` describes one gridworld layout. All coordinates are
`[x, y]` with `(0, 0)` at the top-left, `x` increasing right, `y` increasing
down. Grid size is small (10x10 recommended per the feasibility guide) so
tabular Q-learning/SARSA converge quickly.

```jsonc
{
  "level_id": 0,                     // matches the file name (level0.json -> 0)
  "name": "apples_right_side",       // short human-readable identifier
  "grid_size": [10, 10],             // [width, height] in tiles
  "agent_start": [0, 0],             // agent spawn tile
  "max_steps": 200,                  // episode truncation safety net

  "rocks": [[3, 4], [3, 5]],         // impassable tiles; moving into one = no movement
  "fire": [[5, 5]],                  // instant-death tiles on contact

  "apples": [[8, 2], [8, 5], [8, 8]],// each worth REWARD_APPLE (+1), see rewards_constants.py
  "key": null,                       // [x, y] or null if this level has no key
  "chest": null,                     // [x, y] or null; requires the key to open, worth REWARD_CHEST (+2)

  "monsters": [                      // empty list if this level has no monsters
    {
      "start": [6, 6],
      "move_prob": 0.4                // fixed at 0.4 per spec; kept explicit for testability
    }
  ]
}
```

## Invariants env.py must enforce (do not alter these mechanics)

- Moving into a rock or off the grid edge results in no movement (not an
  error, not a reset).
- Moving into fire, or a monster moving into the agent's tile, is immediate
  death and ends the episode.
- Apples give `REWARD_APPLE`; picking one up removes it from the tile.
- The key gives `REWARD_KEY` (0) and is only useful to open the chest.
- The chest gives `REWARD_CHEST` only if the agent currently holds the key;
  opening it removes both the key-held flag and the chest.
- Episode ends when every apple is collected AND (if present) the chest is
  opened, OR the agent dies. A level with no key/chest only requires all
  apples collected.
- After the agent's move, each monster independently has `move_prob` (0.4)
  chance to take one random step among its currently unblocked directions.

## Level -> task mapping

| File | Task | Notes |
|---|---|---|
| level0.json | Task 1 (Q-learning) | apples only, right side |
| level1.json | Task 2 (SARSA) | apples only, different layout, includes hazards for a meaningful conservative-behavior comparison |
| level2.json | Task 3 | multiple apples + key + chest |
| level3.json | Task 3 | multiple apples + key + chest, different layout |
| level4.json | Task 4 | monsters |
| level5.json | Task 4 | monsters, different layout |
| level6.json | Task 5 | intrinsic reward (reuses a level4/5-style layout) |
