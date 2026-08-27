"""Static-validation tests for every config/levelN.json.

These guard against a malformed or unsolvable level being committed -- a
typo that puts the chest behind a solid rock wall would otherwise surface
only as Q-learning/SARSA silently never converging (see docs/AUDIT_main.md
6.2, 6.3). They exercise GridWorldEnv._load_level's validation and a
reachability check; they do NOT train an agent.
"""

from collections import deque

import pytest

from src.env import CONFIG_DIR, GridWorldEnv

LEVEL_IDS = [0, 1, 2, 3, 4, 5, 6]


@pytest.mark.parametrize("level_id", LEVEL_IDS)
def test_level_loads_and_validates(level_id):
    """GridWorldEnv._load_level(config/level{level_id}.json) returns without
    raising -- i.e. required keys present, all coordinates in bounds, no
    unintended tile overlaps.

    TODO: implement once GridWorldEnv._load_level performs its documented
    validation (currently it only json.loads). Construct the path from
    env.CONFIG_DIR and assert no ValueError.
    """
    path = CONFIG_DIR / f"level{level_id}.json"
    GridWorldEnv._load_level(path)  # raises ValueError on any schema violation


@pytest.mark.parametrize("level_id", LEVEL_IDS)
def test_level_is_solvable(level_id):
    """Every apple, the key (if any), and the chest (if any) are reachable
    from agent_start via 4-connected moves that do not pass through rocks or
    off-grid tiles (fire/monster tiles are traversable for the purposes of
    this check -- they are lethal, not blocking). A level failing this is a
    config bug, not an RL problem.

    TODO: implement as a plain BFS over the grid from agent_start, treating
    rocks and out-of-bounds as walls; assert the reachable set contains
    every apple coord, the key coord, and the chest coord. Does not need
    GridWorldEnv -- read the JSON directly.
    """
    import json

    with open(CONFIG_DIR / f"level{level_id}.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    gw, gh = data["grid_size"]
    rocks = {tuple(r) for r in data["rocks"]}
    start = tuple(data["agent_start"])

    reachable = {start}
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nx, ny = x + dx, y + dy
            in_bounds = 0 <= nx < gw and 0 <= ny < gh
            if in_bounds and (nx, ny) not in rocks and (nx, ny) not in reachable:
                reachable.add((nx, ny))
                queue.append((nx, ny))

    for apple in data["apples"]:
        assert tuple(apple) in reachable, f"level{level_id}: apple {apple} unreachable from {start}"
    if data["key"] is not None:
        assert tuple(data["key"]) in reachable, f"level{level_id}: key unreachable from {start}"
    if data["chest"] is not None:
        assert tuple(data["chest"]) in reachable, f"level{level_id}: chest unreachable from {start}"
