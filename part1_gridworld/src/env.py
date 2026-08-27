"""Pure gridworld environment logic for Part I. NO pygame import here --
this module must be fully testable and runnable headless. Rendering lives in
render.py and receives GridWorldEnv's state, never the other way around.

Loads a level from config/levelN.json (see config/schema.md for the exact
schema) and exposes a small, explicit step/reset API used by trainer.py.

Known limitations / design notes
---------------------------------
State representation chosen:
    (agent_x, agent_y, apples_bitmask, has_key, chest_open, monsters_tuple)

- apples_bitmask is an int where bit i (0-indexed) = 1 means apple i (in
  level JSON order) is still uncollected. All-zeros => all collected.
  This is compact (O(1) hash, O(n) bits vs frozenset overhead) and fully
  hashable for a dict-keyed Q-table.

- monsters_tuple is a tuple of (x, y) pairs, sorted by (x, y) so that the
  same set of monster positions always hashes to the same value regardless
  of iteration order.

- For level4/level6 (2 monsters, open 10x10 grid), this state space is
  large (~10*10 * 2^3 * 2 * 2 * 10*10 per monster) but bounded and still
  convergeable given the per-level episode overrides in training_config.json.
  Do NOT remove monsters from the tuple; that makes the env non-Markov and
  breaks Task 4's "learn to avoid monsters" requirement.
"""

from __future__ import annotations

import json
import pathlib
import random
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Tuple

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
        """Load and validate the level JSON, then initialise all runtime state.

        Args:
            level_path: Path to a levelN.json file. Relative paths are
                resolved from the current working directory.
        """
        self.level_path = pathlib.Path(level_path)
        self.level = self._load_level(self.level_path)

        # Fixed layout data (never mutated after init)
        self._grid_w: int = self.level["grid_size"][0]
        self._grid_h: int = self.level["grid_size"][1]
        self._max_steps: int = self.level["max_steps"]
        self._rocks: frozenset[tuple[int, int]] = frozenset(
            tuple(r) for r in self.level["rocks"]
        )
        self._fire: frozenset[tuple[int, int]] = frozenset(
            tuple(f) for f in self.level["fire"]
        )
        # Store apple initial positions as a list (order = bitmask index)
        self._apple_positions: List[Tuple[int, int]] = [
            tuple(a) for a in self.level["apples"]
        ]
        self._key_pos: tuple[int, int] | None = (
            tuple(self.level["key"]) if self.level["key"] is not None else None
        )
        self._chest_pos: tuple[int, int] | None = (
            tuple(self.level["chest"]) if self.level["chest"] is not None else None
        )
        self._monster_starts: List[Tuple[int, int]] = [
            tuple(m["start"]) for m in self.level["monsters"]
        ]
        self._monster_move_probs: List[float] = [
            m["move_prob"] for m in self.level["monsters"]
        ]
        self._agent_start: tuple[int, int] = tuple(self.level["agent_start"])

        # Mutable runtime state (reset by reset())
        self._agent_pos: tuple[int, int] = self._agent_start
        self._apples_bitmask: int = (1 << len(self._apple_positions)) - 1
        self._has_key: bool = False
        self._chest_open: bool = False
        self._monsters: List[Monster] = []
        self._step_count: int = 0
        self._done: bool = False

        # Module-level RNG for monster decisions (seeded externally via seed_utils)
        self._rng = random.Random()

        # Initialise to starting state
        self.reset()

    @staticmethod
    def _load_level(path: pathlib.Path) -> dict:
        """Load and validate a levelN.json file against the schema documented
        in config/schema.md.

        Raises ValueError (never bare assertions) with a message that names
        the level file and the specific problem so malformed levels fail
        loudly at load time, not as a confusing KeyError/IndexError mid-training.

        Steps:
          1. json.load the file.
          2. Check all required top-level keys are present.
          3. Check all coordinate values are within bounds.
          4. Check no two of {rocks, fire, apples, key, chest, monster starts}
             occupy the same tile (unless _design_note exempts it).
          5. Return the validated dict.
        """
        if not path.exists():
            raise ValueError(f"Level file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 1. Required keys
        required_keys = [
            "level_id", "name", "grid_size", "agent_start", "max_steps",
            "rocks", "fire", "apples", "key", "chest", "monsters",
        ]
        for key in required_keys:
            if key not in data:
                raise ValueError(
                    f"[{path.name}] Missing required key: '{key}'"
                )

        gw, gh = data["grid_size"][0], data["grid_size"][1]

        def _check_coord(coord, field_name: str) -> None:
            x, y = coord[0], coord[1]
            if not (0 <= x < gw and 0 <= y < gh):
                raise ValueError(
                    f"[{path.name}] Coordinate {coord} in '{field_name}' is "
                    f"out of bounds for grid {gw}x{gh}."
                )

        # 2. Check agent_start
        _check_coord(data["agent_start"], "agent_start")

        # 3. Check all coordinate fields
        for rock in data["rocks"]:
            _check_coord(rock, "rocks")
        for fire in data["fire"]:
            _check_coord(fire, "fire")
        for apple in data["apples"]:
            _check_coord(apple, "apples")
        if data["key"] is not None:
            _check_coord(data["key"], "key")
        if data["chest"] is not None:
            _check_coord(data["chest"], "chest")
        for m in data["monsters"]:
            _check_coord(m["start"], "monster start")

        # 4. Overlap check (skip if _design_note present — intentional overlap)
        if "_design_note" not in data:
            all_tiles: dict[tuple, str] = {}
            for rock in data["rocks"]:
                t = tuple(rock)
                if t in all_tiles:
                    raise ValueError(
                        f"[{path.name}] Tile {t} overlaps 'rocks' and '{all_tiles[t]}'"
                    )
                all_tiles[t] = "rocks"
            for fire in data["fire"]:
                t = tuple(fire)
                if t in all_tiles:
                    raise ValueError(
                        f"[{path.name}] Tile {t} overlaps 'fire' and '{all_tiles[t]}'"
                    )
                all_tiles[t] = "fire"
            for apple in data["apples"]:
                t = tuple(apple)
                if t in all_tiles:
                    raise ValueError(
                        f"[{path.name}] Tile {t} overlaps 'apples' and '{all_tiles[t]}'"
                    )
                all_tiles[t] = "apples"
            if data["key"] is not None:
                t = tuple(data["key"])
                if t in all_tiles:
                    raise ValueError(
                        f"[{path.name}] Tile {t} overlaps 'key' and '{all_tiles[t]}'"
                    )
                all_tiles[t] = "key"
            if data["chest"] is not None:
                t = tuple(data["chest"])
                if t in all_tiles:
                    raise ValueError(
                        f"[{path.name}] Tile {t} overlaps 'chest' and '{all_tiles[t]}'"
                    )
                all_tiles[t] = "chest"
            for m in data["monsters"]:
                t = tuple(m["start"])
                if t in all_tiles:
                    raise ValueError(
                        f"[{path.name}] Tile {t} overlaps 'monster start' and '{all_tiles[t]}'"
                    )
                # Monsters may share start tiles with each other — only track
                # vs static objects, not vs other monsters.

        return data

    def reset(self) -> tuple:
        """Reset to the level's initial state and return the initial state
        tuple used to index the Q-table.

        Returns:
            The initial state tuple:
            (agent_x, agent_y, apples_bitmask, has_key, chest_open, monsters_tuple)
        """
        self._agent_pos = self._agent_start
        # All apples present: bits 0..N-1 all set
        self._apples_bitmask = (1 << len(self._apple_positions)) - 1
        self._has_key = False
        self._chest_open = False
        self._monsters = [
            Monster(position=pos, move_prob=prob)
            for pos, prob in zip(self._monster_starts, self._monster_move_probs)
        ]
        self._step_count = 0
        self._done = False
        return self._get_state()

    def _get_state(self) -> tuple:
        """Return the current hashable state tuple."""
        monsters_tuple = tuple(
            sorted(m.position for m in self._monsters)
        )
        return (
            self._agent_pos[0],
            self._agent_pos[1],
            self._apples_bitmask,
            self._has_key,
            self._chest_open,
            monsters_tuple,
        )

    def _is_blocked(self, x: int, y: int) -> bool:
        """Return True if tile (x, y) is impassable (off-grid or rock)."""
        if x < 0 or x >= self._grid_w or y < 0 or y >= self._grid_h:
            return True
        if (x, y) in self._rocks:
            return True
        return False

    def _monster_positions_set(self) -> set[tuple[int, int]]:
        """Return the set of all current monster positions (for O(1) lookup)."""
        return {m.position for m in self._monsters}

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

        Args:
            action: One of Action.UP / DOWN / LEFT / RIGHT.

        Returns:
            StepResult(state, reward, done, info)
        """
        if self._done:
            # Episode already ended — return current state without side-effects
            return StepResult(
                state=self._get_state(),
                reward=0.0,
                done=True,
                info={"error": "step() called after episode is done"},
            )

        reward = REWARD_STEP  # Per-step reward (0.0 per spec)
        info: dict = {}

        # 1. Move the agent
        dx, dy = ACTION_DELTAS[action]
        nx, ny = self._agent_pos[0] + dx, self._agent_pos[1] + dy
        if not self._is_blocked(nx, ny):
            self._agent_pos = (nx, ny)
        # If blocked: stay in place (no movement, not an error)

        ax, ay = self._agent_pos

        # 2. Fire check — immediate death, skip monster resolution
        if (ax, ay) in self._fire:
            reward += REWARD_DEATH
            self._done = True
            info["cause"] = "fire"
            return StepResult(state=self._get_state(), reward=reward, done=True, info=info)

        # 3. Agent-into-monster check — immediate death, skip monster resolution
        if (ax, ay) in self._monster_positions_set():
            reward += REWARD_DEATH
            self._done = True
            info["cause"] = "agent_into_monster"
            return StepResult(state=self._get_state(), reward=reward, done=True, info=info)

        # 4. Pickups at agent's new tile
        # Apple pickup
        for i, apple_pos in enumerate(self._apple_positions):
            if (ax, ay) == apple_pos and (self._apples_bitmask >> i) & 1:
                reward += REWARD_APPLE
                self._apples_bitmask &= ~(1 << i)  # consume the apple
                break

        # Key pickup (key tile is consumed once picked up)
        if self._key_pos is not None and (ax, ay) == self._key_pos and not self._has_key and not self._chest_open:
            reward += REWARD_KEY  # 0.0 per spec — imported, not hardcoded
            self._has_key = True

        # Chest interaction (only opens if has_key; nothing happens otherwise)
        if self._chest_pos is not None and (ax, ay) == self._chest_pos and not self._chest_open:
            if self._has_key:
                reward += REWARD_CHEST
                self._chest_open = True
                self._has_key = False  # Key is consumed

        # 5. Win condition: all apples collected AND chest opened (if present)
        all_apples_done = (self._apples_bitmask == 0)
        chest_done = (self._chest_pos is None) or self._chest_open
        if all_apples_done and chest_done:
            self._done = True
            info["cause"] = "win"
            return StepResult(state=self._get_state(), reward=reward, done=True, info=info)

        # 6. Monster movement (only if episode not yet over)
        self._resolve_monster_moves()

        # 7. Monster-into-agent check
        if (ax, ay) in self._monster_positions_set():
            reward += REWARD_DEATH
            self._done = True
            info["cause"] = "monster_into_agent"
            return StepResult(state=self._get_state(), reward=reward, done=True, info=info)

        # 8. Increment step counter; truncate if max_steps reached
        self._step_count += 1
        if self._step_count >= self._max_steps:
            self._done = True
            info["cause"] = "truncated"
            return StepResult(state=self._get_state(), reward=reward, done=True, info=info)

        return StepResult(state=self._get_state(), reward=reward, done=False, info=info)

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
        """
        for monster in self._monsters:
            # Each monster independently draws a Bernoulli(move_prob) decision
            if self._rng.random() >= monster.move_prob:
                continue  # This monster doesn't move this turn

            # Find all currently unblocked directions
            mx, my = monster.position
            unblocked = []
            for dx, dy in ACTION_DELTAS.values():
                nx, ny = mx + dx, my + dy
                if not self._is_blocked(nx, ny):
                    unblocked.append((nx, ny))

            if not unblocked:
                # Fully surrounded by rocks/edges — does not move
                continue

            # Choose uniformly at random from unblocked directions
            monster.position = self._rng.choice(unblocked)

    def get_state_snapshot(self) -> dict:
        """Return a plain-dict snapshot of the current environment state for
        the renderer. Renderer must never receive the GridWorldEnv object itself.

        Returns a dict with all necessary keys for GridWorldRenderer.draw():
            - grid_w, grid_h: grid dimensions
            - agent_pos: (x, y)
            - rocks: list of (x, y)
            - fire: list of (x, y)
            - apples: list of (x, y) for remaining (uncollected) apples
            - key_pos: (x, y) or None (None if level has no key or key collected)
            - chest_pos: (x, y) or None (None if level has no chest or chest opened)
            - monsters: list of (x, y)
            - has_key: bool
            - chest_open: bool
            - step_count: int
            - max_steps: int
            - done: bool
        """
        remaining_apples = [
            pos for i, pos in enumerate(self._apple_positions)
            if (self._apples_bitmask >> i) & 1
        ]
        # Key is visible on map only if it hasn't been picked up yet
        key_on_map = self._key_pos if (self._key_pos is not None and not self._has_key and not self._chest_open) else None
        # Chest is visible only if not opened yet
        chest_on_map = self._chest_pos if (self._chest_pos is not None and not self._chest_open) else None

        return {
            "grid_w": self._grid_w,
            "grid_h": self._grid_h,
            "agent_pos": self._agent_pos,
            "rocks": list(self._rocks),
            "fire": list(self._fire),
            "apples": remaining_apples,
            "key_pos": key_on_map,
            "chest_pos": chest_on_map,
            "monsters": [m.position for m in self._monsters],
            "has_key": self._has_key,
            "chest_open": self._chest_open,
            "step_count": self._step_count,
            "max_steps": self._max_steps,
            "done": self._done,
        }

    @property
    def action_space_n(self) -> int:
        return len(Action)

    @property
    def grid_size(self) -> tuple[int, int]:
        """Return (width, height) of the grid."""
        return (self._grid_w, self._grid_h)
