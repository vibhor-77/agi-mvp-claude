"""Tests for world model learning."""

import unittest

from agi_composer.worlds.grid_env import Action, GridConfig, GridWorld
from agi_composer.worlds.world_model import WorldModel


class TestWorldModel(unittest.TestCase):
    """Test world model learning from observations."""

    def setUp(self):
        self.model = WorldModel()
        config = GridConfig(rows=3, cols=3, start=(0, 0), goals=[(2, 2)])
        self.env = GridWorld(config)

    def test_learns_movement(self):
        """Model should learn that DOWN adds (1,0) to position."""
        obs = self.env.reset()
        state = obs.position
        obs = self.env.step(Action.DOWN)
        self.model.observe(state, Action.DOWN, obs)

        # Model should have learned DOWN = (+1, 0)
        predicted, confidence = self.model.predict((0, 0), Action.DOWN)
        self.assertEqual(predicted, (1, 0))
        self.assertGreater(confidence, 0)

    def test_learns_wall(self):
        """Model should detect walls from failed movements."""
        config = GridConfig(
            rows=3, cols=3, start=(0, 0), goals=[(2, 2)],
            walls=[(0, 1)],
        )
        env = GridWorld(config)
        obs = env.reset()
        state = obs.position
        obs = env.step(Action.RIGHT)  # Hits wall
        self.model.observe(state, Action.RIGHT, obs)

        self.assertIn((0, 1), self.model._wall_positions)

    def test_learns_portal(self):
        """Model should detect non-local transitions as portals."""
        config = GridConfig(
            rows=3, cols=3, start=(0, 0), goals=[(2, 2)],
            portals={(0, 1): (2, 0)},
        )
        env = GridWorld(config)
        obs = env.reset()
        state = obs.position
        obs = env.step(Action.RIGHT)  # Portal teleport
        self.model.observe(state, Action.RIGHT, obs)

        self.assertTrue(len(self.model._portal_map) > 0)

    def test_learns_goal(self):
        """Model should detect goals from positive terminal rewards."""
        config = GridConfig(rows=2, cols=2, start=(0, 0), goals=[(0, 1)])
        env = GridWorld(config)
        obs = env.reset()
        state = obs.position
        obs = env.step(Action.RIGHT)
        self.model.observe(state, Action.RIGHT, obs)

        self.assertIn((0, 1), self.model._goal_positions)

    def test_prediction_with_no_data(self):
        """Prediction with no observations should have zero confidence."""
        _, confidence = self.model.predict((5, 5), Action.UP)
        self.assertEqual(confidence, 0.0)

    def test_uncertainty_decreases(self):
        """Uncertainty should decrease with observations."""
        u_before = self.model.get_uncertainty((0, 0), Action.DOWN)
        self.assertEqual(u_before, 1.0)

        obs = self.env.reset()
        obs = self.env.step(Action.DOWN)
        self.model.observe((0, 0), Action.DOWN, obs)

        u_after = self.model.get_uncertainty((0, 0), Action.DOWN)
        self.assertLess(u_after, u_before)

    def test_summary_returns_string(self):
        summary = self.model.summary()
        self.assertIsInstance(summary, str)
        self.assertIn("World Model", summary)

    def test_discovered_rules_list(self):
        """After some observations, rules should be discovered."""
        obs = self.env.reset()
        for action in [Action.DOWN, Action.RIGHT, Action.DOWN, Action.RIGHT]:
            state = obs.position if not obs.done else (0, 0)
            if obs.done:
                break
            inv = obs.inventory
            obs = self.env.step(action)
            self.model.observe(state, action, obs, inv)

        rules = self.model.get_discovered_rules()
        self.assertGreater(len(rules), 0)


if __name__ == "__main__":
    unittest.main()
