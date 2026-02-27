"""
Grid world environments with incrementally complex transition rules.

Each grid world defines:
- A grid of cells (some walkable, some walls)
- A start position and goal position(s)
- Transition rules: what happens when the agent takes an action
- Rewards: positive for reaching goals, negative for walls/steps

The agent does NOT know the rules — it must discover them through exploration.

Complexity levels:
1. SimpleMaze: walls + goal, deterministic movement
2. PortalMaze: adds teleporters (non-local state transitions)
3. KeyDoorMaze: adds state-dependent transitions (keys unlock doors)
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Actions and cell types
# ---------------------------------------------------------------------------

class Action(IntEnum):
    """The four cardinal directions."""
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    def delta(self) -> Tuple[int, int]:
        """Row, col displacement for this action."""
        return {
            Action.UP: (-1, 0),
            Action.RIGHT: (0, 1),
            Action.DOWN: (1, 0),
            Action.LEFT: (0, -1),
        }[self]

    @staticmethod
    def all() -> List["Action"]:
        return [Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT]


class CellType(IntEnum):
    """What occupies a grid cell."""
    EMPTY = 0
    WALL = 1
    GOAL = 2
    PORTAL = 3   # Teleports agent to a linked cell
    KEY = 4      # Picks up a key
    DOOR = 5     # Passable only with the matching key
    START = 6


@dataclass
class Observation:
    """What the agent sees after taking an action."""
    position: Tuple[int, int]       # Current (row, col)
    reward: float                   # Reward received
    done: bool                      # Episode over?
    inventory: frozenset            # Items held (for key-door worlds)
    visible_cells: Optional[Dict[Tuple[int, int], CellType]] = None  # Local view

    def __repr__(self) -> str:
        return (f"Obs(pos={self.position}, reward={self.reward:.1f}, "
                f"done={self.done})")


@dataclass
class GridConfig:
    """Configuration for building a grid world."""
    rows: int = 5
    cols: int = 5
    walls: List[Tuple[int, int]] = field(default_factory=list)
    start: Tuple[int, int] = (0, 0)
    goals: List[Tuple[int, int]] = field(default_factory=lambda: [(4, 4)])
    portals: Dict[Tuple[int, int], Tuple[int, int]] = field(default_factory=dict)
    keys: Dict[Tuple[int, int], str] = field(default_factory=dict)  # pos -> key_id
    doors: Dict[Tuple[int, int], str] = field(default_factory=dict)  # pos -> key_id needed
    step_reward: float = -0.1
    goal_reward: float = 10.0
    wall_penalty: float = -0.5
    max_steps: int = 100
    visibility_radius: int = -1  # -1 = full observability


class GridWorld:
    """
    A configurable grid world environment.

    The agent interacts via step(action) and receives Observations.
    The transition rules are hidden — the agent must discover them.
    """

    def __init__(self, config: GridConfig):
        self.config = config
        self._build_grid()
        self.reset()

    def _build_grid(self) -> None:
        """Construct the internal grid representation."""
        c = self.config
        self.grid = np.full((c.rows, c.cols), CellType.EMPTY, dtype=int)

        for pos in c.walls:
            self.grid[pos] = CellType.WALL
        for pos in c.goals:
            self.grid[pos] = CellType.GOAL
        for pos in c.portals:
            self.grid[pos] = CellType.PORTAL
        for pos in c.keys:
            self.grid[pos] = CellType.KEY
        for pos in c.doors:
            self.grid[pos] = CellType.DOOR

        self.grid[c.start] = CellType.START

    def reset(self) -> Observation:
        """Reset the environment to the start state."""
        self.position = self.config.start
        self.inventory: Set[str] = set()
        self.steps = 0
        self.done = False
        self.total_reward = 0.0
        self._collected_keys: Set[Tuple[int, int]] = set()
        return self._make_observation(0.0, False)

    def step(self, action: Action) -> Observation:
        """
        Take an action and return the resulting observation.

        This is the ONLY interface the agent has with the world.
        All rules must be discovered through this interaction.
        """
        if self.done:
            return self._make_observation(0.0, True)

        self.steps += 1
        row, col = self.position
        dr, dc = action.delta()
        new_row, new_col = row + dr, col + dc

        # --- Apply transition rules ---

        # Boundary check
        if not self._in_bounds(new_row, new_col):
            reward = self.config.wall_penalty
            self.total_reward += reward
            return self._make_observation(reward, False)

        cell = CellType(self.grid[new_row, new_col])

        # Wall collision
        if cell == CellType.WALL:
            reward = self.config.wall_penalty
            self.total_reward += reward
            return self._make_observation(reward, False)

        # Door check (needs matching key)
        if cell == CellType.DOOR:
            needed_key = self.config.doors.get((new_row, new_col))
            if needed_key and needed_key not in self.inventory:
                # Door is locked — treat like wall
                reward = self.config.wall_penalty
                self.total_reward += reward
                return self._make_observation(reward, False)

        # Move succeeds
        self.position = (new_row, new_col)

        # Portal teleportation
        if cell == CellType.PORTAL:
            destination = self.config.portals.get((new_row, new_col))
            if destination:
                self.position = destination

        # Key pickup
        if cell == CellType.KEY and (new_row, new_col) not in self._collected_keys:
            key_id = self.config.keys.get((new_row, new_col))
            if key_id:
                self.inventory.add(key_id)
                self._collected_keys.add((new_row, new_col))

        # Goal check
        if cell == CellType.GOAL:
            reward = self.config.goal_reward
            self.done = True
            self.total_reward += reward
            return self._make_observation(reward, True)

        # Normal step
        reward = self.config.step_reward

        # Max steps check
        if self.steps >= self.config.max_steps:
            self.done = True

        self.total_reward += reward
        return self._make_observation(reward, self.done)

    def _in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.config.rows and 0 <= col < self.config.cols

    def _make_observation(self, reward: float, done: bool) -> Observation:
        """Construct an observation from current state."""
        visible = None
        if self.config.visibility_radius >= 0:
            visible = self._get_visible_cells()
        else:
            # Full observability — show entire grid
            visible = {}
            for r in range(self.config.rows):
                for c in range(self.config.cols):
                    visible[(r, c)] = CellType(self.grid[r, c])

        return Observation(
            position=self.position,
            reward=reward,
            done=done,
            inventory=frozenset(self.inventory),
            visible_cells=visible,
        )

    def _get_visible_cells(self) -> Dict[Tuple[int, int], CellType]:
        """Get cells within visibility radius."""
        r0, c0 = self.position
        radius = self.config.visibility_radius
        visible = {}
        for r in range(max(0, r0 - radius), min(self.config.rows, r0 + radius + 1)):
            for c in range(max(0, c0 - radius), min(self.config.cols, c0 + radius + 1)):
                visible[(r, c)] = CellType(self.grid[r, c])
        return visible

    def render(self) -> str:
        """ASCII rendering of the grid for debugging."""
        symbols = {
            CellType.EMPTY: ".",
            CellType.WALL: "#",
            CellType.GOAL: "G",
            CellType.PORTAL: "P",
            CellType.KEY: "K",
            CellType.DOOR: "D",
            CellType.START: "S",
        }
        lines = []
        for r in range(self.config.rows):
            row_str = ""
            for c in range(self.config.cols):
                if (r, c) == self.position:
                    row_str += "A"  # Agent
                else:
                    row_str += symbols.get(CellType(self.grid[r, c]), "?")
            lines.append(row_str)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pre-built environments of increasing complexity
# ---------------------------------------------------------------------------

def make_simple_maze() -> GridWorld:
    """
    Level 1: Simple 5x5 maze with walls.
    Rules to discover: walls block movement, reaching goal ends episode.

        S . . . .
        . # # . .
        . . . . .
        . . # # .
        . . . . G
    """
    config = GridConfig(
        rows=5, cols=5,
        walls=[(1, 1), (1, 2), (3, 2), (3, 3)],
        start=(0, 0),
        goals=[(4, 4)],
    )
    return GridWorld(config)


def make_portal_maze() -> GridWorld:
    """
    Level 2: 6x6 maze with a portal that teleports the agent.
    Rules to discover: portals cause non-local transitions.

        S . # . . .
        . . # . . .
        . . # P . .
        . . # . . .
        . . . . # .
        . . . . # G

    Portal at (2,3) teleports to (4,0).
    """
    config = GridConfig(
        rows=6, cols=6,
        walls=[(0, 2), (1, 2), (2, 2), (3, 2), (4, 4), (5, 4)],
        start=(0, 0),
        goals=[(5, 5)],
        portals={(2, 3): (4, 0)},
    )
    return GridWorld(config)


def make_key_door_maze() -> GridWorld:
    """
    Level 3: 6x6 maze with a key and a locked door.
    Rules to discover: keys are collected on visit, doors require keys.

        S . . . . .
        . # # # D .
        . . . . . .
        . . K . . .
        . # # # # .
        . . . . . G

    Key 'red' at (3,2), Door at (1,4) needs 'red'.
    """
    config = GridConfig(
        rows=6, cols=6,
        walls=[(1, 1), (1, 2), (1, 3), (4, 1), (4, 2), (4, 3), (4, 4)],
        start=(0, 0),
        goals=[(5, 5)],
        keys={(3, 2): "red"},
        doors={(1, 4): "red"},
    )
    return GridWorld(config)
