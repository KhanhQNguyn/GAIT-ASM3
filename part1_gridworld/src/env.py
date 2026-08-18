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

    State representation: TODO -- decide a hashable tuple representation
    for Q-table indexing, e.g. (agent_x, agent_y, frozenset(apples_left),
    has_key, chest_open, tuple(monster_positions)). Document the final
    choice here once implemented, since algorithms.py's QTable depends on
    this being hashable and reasonably small.

    Mechanics this class must enforce EXACTLY per config/schema.md, and must
    NOT be altered by any helper function elsewhere in the codebase:
      - Moving into a rock or off-grid: no movement (not an error).
      - Moving into fire, or a monster moving into the agent: death, episode ends.
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
        """Load and lightly validate a levelN.json file against the schema
        documented in config/schema.md.

        TODO: json.load, validate required keys, return dict.
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
          3. Resolve pickups (apple / key / chest) at the agent's new tile.
          4. Check for a "collected everything" win condition.
          5. If not already done, move each monster with probability
             move_prob (0.4) in a random unblocked direction.
          6. If a monster is now on the agent's tile (either it moved onto
             the agent, or -- should not happen given step 1 -- the agent
             moved onto it) -> death, episode ends.
          7. Increment step counter; truncate if max_steps reached.

        TODO: implement, returning a StepResult.
        """
        raise NotImplementedError

    def _resolve_monster_moves(self) -> None:
        """Give each monster its 0.4 chance to move one tile in a random
        currently-unblocked direction (rocks/edges block monsters).

        Kept as its own method so tests/test_monster_stochastic.py can
        call it directly with a seeded RNG and check the empirical move
        rate converges to 0.4 over many trials.

        TODO: implement.
        """
        raise NotImplementedError

    @property
    def action_space_n(self) -> int:
        return len(Action)
