"""
Phase 2 Demo: Exploring grid worlds to discover compositional rules.

The Explorer agent interacts with grid worlds through the step() interface,
building a world model from observations. It discovers rules like:
- "Moving UP decreases row by 1"
- "Wall at (1,1) blocks movement"
- "Portal at (2,3) teleports to (4,0)"
- "Key at (3,2) adds 'red' to inventory"
"""

from agi_composer.worlds.grid_env import (
    make_simple_maze, make_portal_maze, make_key_door_maze,
)
from agi_composer.worlds.explorer import Explorer, ExplorerConfig


def main():
    print("=" * 60)
    print("  AGI Composer — Phase 2: Grid World Exploration")
    print("=" * 60)

    # --- Level 1: Simple Maze ---
    print("\n--- Level 1: Simple Maze ---\n")
    env = make_simple_maze()
    print("Grid:")
    print(env.render())
    print()

    explorer = Explorer(ExplorerConfig(seed=42, episodes_per_level=50))
    result = explorer.explore(env, verbose=True)
    print()
    print(result.summary())

    # --- Level 2: Portal Maze ---
    print("\n--- Level 2: Portal Maze ---\n")
    env = make_portal_maze()
    print("Grid:")
    print(env.render())
    print()

    explorer = Explorer(ExplorerConfig(
        seed=42, episodes_per_level=60, initial_curiosity=0.9,
    ))
    result = explorer.explore(env, verbose=True)
    print()
    print(result.summary())

    # --- Level 3: Key-Door Maze ---
    print("\n--- Level 3: Key-Door Maze ---\n")
    env = make_key_door_maze()
    print("Grid:")
    print(env.render())
    print()

    explorer = Explorer(ExplorerConfig(
        seed=42, episodes_per_level=80, initial_curiosity=0.95,
    ))
    result = explorer.explore(env, verbose=True)
    print()
    print(result.summary())


if __name__ == "__main__":
    main()
