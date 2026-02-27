"""
Phase 2: Grid worlds with agency and exploration.

The agent must discover the compositional rules governing a grid world
by actively exploring it, building a world model from observations,
and using that model to plan actions that maximize reward.

This tests all 4 pillars:
- Composability: World rules are compositions of discrete primitives
- Feedback loops: Predictions feed back into action selection
- Approximability: The model refines as more data is gathered
- Exploration: The agent actively seeks informative actions
"""

from agi_composer.worlds.grid_env import GridWorld, GridConfig
from agi_composer.worlds.world_model import WorldModel, TransitionRule
from agi_composer.worlds.explorer import Explorer, ExplorerConfig, ExplorationResult

__all__ = [
    "GridWorld",
    "GridConfig",
    "WorldModel",
    "TransitionRule",
    "Explorer",
    "ExplorerConfig",
    "ExplorationResult",
]
