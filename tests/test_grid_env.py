"""Tests for grid world environments."""

import unittest

from agi_composer.worlds.grid_env import (
    Action, CellType, GridConfig, GridWorld,
    make_simple_maze, make_portal_maze, make_key_door_maze,
)


class TestAction(unittest.TestCase):
    """Test action mechanics."""

    def test_action_deltas(self):
        self.assertEqual(Action.UP.delta(), (-1, 0))
        self.assertEqual(Action.DOWN.delta(), (1, 0))
        self.assertEqual(Action.LEFT.delta(), (0, -1))
        self.assertEqual(Action.RIGHT.delta(), (0, 1))

    def test_all_actions(self):
        self.assertEqual(len(Action.all()), 4)


class TestGridWorld(unittest.TestCase):
    """Test basic grid world mechanics."""

    def test_reset_returns_start(self):
        env = make_simple_maze()
        obs = env.reset()
        self.assertEqual(obs.position, (0, 0))
        self.assertFalse(obs.done)

    def test_move_to_empty(self):
        env = make_simple_maze()
        env.reset()
        obs = env.step(Action.RIGHT)
        self.assertEqual(obs.position, (0, 1))
        self.assertFalse(obs.done)

    def test_wall_collision(self):
        env = make_simple_maze()
        env.reset()
        # Move to (1,0), then try RIGHT into wall at (1,1)
        env.step(Action.DOWN)
        obs = env.step(Action.RIGHT)
        self.assertEqual(obs.position, (1, 0))  # Stayed in place
        self.assertLess(obs.reward, 0)  # Penalty

    def test_boundary_collision(self):
        env = make_simple_maze()
        env.reset()
        obs = env.step(Action.UP)  # Try to go above row 0
        self.assertEqual(obs.position, (0, 0))
        self.assertLess(obs.reward, 0)

    def test_reach_goal(self):
        config = GridConfig(rows=2, cols=2, start=(0, 0), goals=[(0, 1)])
        env = GridWorld(config)
        env.reset()
        obs = env.step(Action.RIGHT)
        self.assertEqual(obs.position, (0, 1))
        self.assertTrue(obs.done)
        self.assertGreater(obs.reward, 0)

    def test_max_steps(self):
        config = GridConfig(rows=3, cols=3, start=(0, 0),
                            goals=[(2, 2)], max_steps=5)
        env = GridWorld(config)
        env.reset()
        # Move in a circle to burn steps without reaching goal
        actions = [Action.DOWN, Action.UP] * 5
        for a in actions:
            obs = env.step(a)
            if obs.done:
                break
        self.assertTrue(obs.done)

    def test_render(self):
        env = make_simple_maze()
        env.reset()
        rendered = env.render()
        self.assertIn("A", rendered)  # Agent marker
        self.assertIn("#", rendered)  # Wall marker


class TestPortalMaze(unittest.TestCase):
    """Test portal teleportation mechanics."""

    def test_portal_teleports(self):
        env = make_portal_maze()
        env.reset()
        # Navigate to portal at (2,3)
        # From (0,0): DOWN, DOWN to (2,0), then RIGHT to (2,1)
        # But (2,2) is a wall, so we need to go around
        # Actually let's just build a minimal portal test
        config = GridConfig(
            rows=3, cols=3,
            start=(0, 0),
            goals=[(2, 2)],
            portals={(0, 1): (2, 0)},
        )
        env = GridWorld(config)
        env.reset()
        obs = env.step(Action.RIGHT)  # Step onto portal at (0,1)
        self.assertEqual(obs.position, (2, 0))  # Teleported!


class TestKeyDoorMaze(unittest.TestCase):
    """Test key-door mechanics."""

    def test_door_blocks_without_key(self):
        config = GridConfig(
            rows=3, cols=3,
            start=(0, 0),
            goals=[(0, 2)],
            doors={(0, 1): "red"},
        )
        env = GridWorld(config)
        env.reset()
        obs = env.step(Action.RIGHT)  # Try door without key
        self.assertEqual(obs.position, (0, 0))  # Blocked
        self.assertLess(obs.reward, 0)

    def test_door_opens_with_key(self):
        config = GridConfig(
            rows=3, cols=3,
            start=(0, 0),
            goals=[(0, 2)],
            keys={(1, 0): "red"},
            doors={(0, 1): "red"},
        )
        env = GridWorld(config)
        env.reset()
        # Pick up key
        obs = env.step(Action.DOWN)  # Move to (1,0) — key
        self.assertIn("red", obs.inventory)
        # Go back up
        env.step(Action.UP)
        # Now door should open
        obs = env.step(Action.RIGHT)
        self.assertEqual(obs.position, (0, 1))  # Passed through door


if __name__ == "__main__":
    unittest.main()
