"""Pure gridworld environment logic for Part I. NO pygame import here --
this module must be fully testable and runnable headless. Rendering lives in
render.py and receives GridWorldEnv's state, never the other way around.

Loads a level from config/levelN.json (see config/schema.md for the exact
schema) and exposes a small, explicit step/reset API used by trainer.py.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from enum import IntEnum

from config.rewards_constants import (
    REWARD_APPLE,
    REWARD_CHEST,
    REWARD_DEATH,
    REWARD_KEY,
    REWARD_STEP,
)

CONFIG_DIR = pathlib.Path(__file__).resolve().parent.parent / "config"


class Action(IntEnum):
    """The 4 movement actions. Order matters -- Q-tables index by this."""

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


ACTION_DELTAS = {
    Action.UP: (0, -1),
    Action.DOWN: (0, 1),
    Action.LEFT: (-1, 0),
    Action.RIGHT: (1, 0),
}


@dataclass
class Monster:
    """A single monster's runtime state (position + its fixed move probability)."""

    position: tuple[int, int]
    move_prob: float = 0.4


@dataclass
class StepResult:
    """Return value of GridWorldEnv.step()."""

    state: tuple
    reward: float
    done: bool
    info: dict = field(default_factory=dict)


class GridWorldEnv:
    """Headless gridworld environment for one level.

    State representation (DECIDED -- reset()/step() must return exactly this
    tuple, and algorithms.py's QTable / save_qtable / load_qtable depend on
    it being hashable and stable):

        (agent_x, agent_y, apples_bitmask, has_key, chest_open, monsters)

      - agent_x, agent_y : int tile coords.
      - apples_bitmask   : int; bit i (0-indexed) SET means apple i -- in the
                           level JSON's "apples" list order -- is still
                           uncollected. All-collected => 0.
      - has_key          : bool; False for levels with no key.
      - chest_open       : bool; False for levels with no chest. (Opening the
                           chest also flips has_key back to False -- the key
                           is consumed -- but the state still carries has_key
                           so pre-open states stay distinct.)
      - monsters         : tuple(sorted((mx, my) for each monster)); () when
                           the level has no monsters. Sorted so an identical
                           set of monster positions always hashes equal.

    Feasibility note: for level4/level6 (2 monsters, open grid) this state
    space is large. The lever, since level layouts are fixed, is the
    per-level `episodes` override in config/training_config.json -- raise it
    if the training curve has not plateaued. Do NOT drop `monsters` from the
    tuple to shrink the space: that makes the environment non-Markov and
    breaks Task 4's "learn to avoid monsters" requirement.

    Mechanics this class must enforce EXACTLY per config/schema.md, and must
    NOT be altered by any helper function elsewhere in the codebase:
      - Moving into a rock or off-grid: no movement (not an error).
      - Moving into fire: death, episode ends immediately.
      - A monster moving into the agent's tile: death, episode ends.
      - The agent moving onto a monster's current tile: also death, episode
        ends immediately. This is an independent, equally valid death path —
        not an edge case, not prevented by step ordering. Both directions of
        occupancy collision must be handled identically.
      - Apple pickup: REWARD_APPLE, tile cleared.
      - Key pickup: REWARD_KEY (0), sets an internal "has_key" flag.
      - Chest open: only if has_key, REWARD_CHEST, consumes the key.
      - Episode ends when all apples collected AND (chest opened if present),
        OR the agent dies.
      - After the agent's action, each monster independently has move_prob
        (0.4) chance to take one random step among currently unblocked
        directions (rocks/grid edges block monsters too).
    """

    def __init__(self, level_path: str | pathlib.Path):
        self.level_path = pathlib.Path(level_path)
        self.level = self._load_level(self.level_path)
        # TODO: initialize agent position, remaining apples, key/chest state,
        # monster list, step counter, and RNG from self.level.

    @staticmethod
    def _load_level(path: pathlib.Path) -> dict:
        """Load and validate a levelN.json file against the schema documented
        in config/schema.md.

        TODO: Implement as follows (do NOT use bare assertions -- raise
        ValueError with a message that names the level file and the specific
        problem so malformed levels fail loudly at load time, not as a
        confusing KeyError/IndexError mid-training):

          1. json.load the file.
          2. Check all required top-level keys are present:
               level_id, name, grid_size, agent_start, max_steps,
               rocks, fire, apples, key, chest, monsters
             Raise ValueError naming any missing key.
          3. Check all coordinate values are within bounds:
               [0, grid_size[0]) x [0, grid_size[1])
             for: agent_start, every rock coordinate, every fire coordinate,
             every apple coordinate, key (if not null), chest (if not null),
             and every monster's "start" field.
             Raise ValueError naming the out-of-bounds coordinate and field.
          4. Check no two of {rocks, fire, apples, key, chest, monster starts}
             occupy the same tile unless that is an intentional per-level
             design choice (if so, note it explicitly in the level JSON's
             "_design_note" field and skip the overlap check for that tile).
             Raise ValueError naming the overlapping tile and fields.
          5. Return the validated dict.

        Solvability (every apple / key / chest actually reachable from
        agent_start given the rocks) is deliberately NOT checked here -- it
        needs a BFS and is a config-authoring check, not a hot-path load
        check. It lives in tests/test_level_configs.py::test_level_is_solvable
        instead.
        """
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


    def reset(self) -> tuple:
        """Reset to the level's initial state and return the initial state
        tuple used to index the Q-table.

        TODO: implement.
        """
        raise NotImplementedError

    def step(self, action: Action) -> StepResult:
        """Apply one agent action, then resolve monster movement, then
        compute reward and termination.

        Order of operations (must match the spec exactly):
          1. Move the agent (blocked by rocks/edges -> no movement).
          2. If the agent stepped onto fire -> death, episode ends here
             (monsters do not get to move this turn).
          3. If the agent's new tile is occupied by a monster -> death,
             episode ends here (monsters do not get to move this turn).
             This is a normal, expected death path — the agent walked into
             the monster. Do not skip this check or fold it into step 6.
          4. Resolve pickups (apple / key / chest) at the agent's new tile.
          5. Check for a "collected everything" win condition.
          6. If not already done, move each monster with probability
             move_prob (0.4) in a random unblocked direction.
          7. If a monster now occupies the agent's tile (the monster moved
             onto the agent this turn) -> death, episode ends. (The reverse
             case — agent moved onto monster — was already caught in step 3.)
          8. Increment step counter; truncate if max_steps reached.

        TODO: implement, returning a StepResult.
        """
        raise NotImplementedError

    def _resolve_monster_moves(self) -> None:
        """Give each monster its 0.4 chance to move one tile in a random
        currently-unblocked direction (rocks/edges block monsters).

        Monsters do NOT block each other: two monsters may occupy the same
        tile, and a monster's set of unblocked directions is computed from
        rocks and grid edges only, ignoring other monsters. (This is the
        simplest rule consistent with the spec, which only names rocks and
        edges as blockers; documented here and in config/schema.md so it is
        not silently decided differently at implementation time.)

        If a monster has zero unblocked directions available (i.e. it is
        fully surrounded by rocks and/or grid edges -- a reachable situation
        given level4-6's 2x2 rock clusters near monster start positions), it
        does not move that turn. This counts as the monster "not moving" for
        the 0.4 roll -- do NOT crash on an empty choices list, and do NOT
        retry the roll or move in a blocked direction.

        Kept as its own method so tests/test_monster_stochastic.py can
        call it directly with a seeded RNG and check the empirical move
        rate converges to 0.4 over many trials.

        TODO: implement.
        """
        raise NotImplementedError

    @property
    def action_space_n(self) -> int:
        return len(Action)
