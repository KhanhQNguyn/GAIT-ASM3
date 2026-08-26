"""Static-validation tests for every config/levelN.json.

These guard against a malformed or unsolvable level being committed -- a
typo that puts the chest behind a solid rock wall would otherwise surface
only as Q-learning/SARSA silently never converging (see docs/AUDIT_main.md
6.2, 6.3). They exercise GridWorldEnv._load_level's validation and a
reachability check; they do NOT train an agent.
"""

import pytest

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
    pytest.skip("TODO: implement once GridWorldEnv._load_level validates")


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
    pytest.skip("TODO: implement BFS reachability check")
