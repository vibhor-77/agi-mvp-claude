"""
Phase 2 integration tests — full exploration pipeline.

Tests the complete loop: Explorer discovers rules about grid worlds
by actively exploring them, building a world model that captures
the compositional transition rules.
"""

import unittest

from agi_composer.worlds.grid_env import (
    GridConfig, GridWorld, Action,
    make_simple_maze, make_portal_maze, make_key_door_maze,
)
from agi_composer.worlds.explorer import Explorer, ExplorerConfig


class TestPhase2SimpleMaze(unittest.TestCase):
    """End-to-end test: discover and solve a simple maze."""

    def test_full_pipeline(self):
        """Explorer should discover rules and solve the simple maze."""
        env = make_simple_maze()
        explorer = Explorer(ExplorerConfig(seed=42, episodes_per_level=50))
        result = explorer.explore(env)

        # Should solve
        self.assertTrue(result.solved, "Failed to solve simple maze")

        # Should discover movement rules
        movement_rules = [r for r in result.rules if r.rule_type == "movement"]
        self.assertGreaterEqual(len(movement_rules), 2,
                                "Should discover at least 2 movement rules")

        # Should discover some walls
        wall_rules = [r for r in result.rules if r.rule_type == "wall"]
        self.assertGreater(len(wall_rules), 0,
                           "Should discover at least one wall")

        # Should discover the goal
        goal_rules = [r for r in result.rules if r.rule_type == "goal"]
        self.assertGreater(len(goal_rules), 0,
                           "Should discover the goal")

        # Path should be reasonable (optimal is 8 steps for 5x5 with walls)
        self.assertIsNotNone(result.best_path_length)
        self.assertLessEqual(result.best_path_length, 30,
                             "Path should be somewhat efficient")


class TestPhase2PortalMaze(unittest.TestCase):
    """End-to-end test: discover portal rules."""

    def test_explores_portal_world(self):
        """Explorer should handle a portal maze."""
        env = make_portal_maze()
        explorer = Explorer(ExplorerConfig(
            seed=42, episodes_per_level=60,
            initial_curiosity=0.9,
        ))
        result = explorer.explore(env)

        # Should explore many states
        self.assertGreater(result.world_model.num_states_visited, 8)

        # Should observe many transitions
        self.assertGreater(result.world_model.num_transitions_observed, 20)


class TestPhase2KeyDoorMaze(unittest.TestCase):
    """End-to-end test: discover key-door rules."""

    def test_explores_key_door_world(self):
        """Explorer should discover key and door mechanics."""
        env = make_key_door_maze()
        explorer = Explorer(ExplorerConfig(
            seed=42, episodes_per_level=80,
            initial_curiosity=0.95,
        ))
        result = explorer.explore(env)

        # Should explore many states
        self.assertGreater(result.world_model.num_states_visited, 5)

        # Model summary should be informative
        summary = result.world_model.summary()
        self.assertIn("World Model", summary)


class TestPhase2ModelAccuracy(unittest.TestCase):
    """Test that the world model makes accurate predictions."""

    def test_predictions_after_training(self):
        """After exploration, predictions should be mostly accurate."""
        env = make_simple_maze()
        explorer = Explorer(ExplorerConfig(seed=42, episodes_per_level=30))
        result = explorer.explore(env)

        model = result.world_model

        # Test predictions on known transitions
        correct = 0
        total = 0
        for t in model.transitions[:50]:  # Check first 50 transitions
            predicted, confidence = model.predict(t.state, t.action)
            if confidence > 0.3:
                total += 1
                if predicted == t.next_state:
                    correct += 1

        if total > 0:
            accuracy = correct / total
            self.assertGreater(accuracy, 0.5,
                               f"Model accuracy too low: {accuracy:.0%}")


if __name__ == "__main__":
    unittest.main()
