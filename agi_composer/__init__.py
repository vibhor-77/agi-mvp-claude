"""
AGI Composer: Compositional Function Discovery via Energy Minimization.

Discovers the symbolic generating function behind a stream of numbers
by composing primitive mathematical operations, guided by energy-based
search that minimizes description length and prediction error.
"""

from agi_composer.primitives import Primitive, PRIMITIVE_REGISTRY
from agi_composer.expression import Expression, ExpressionNode
from agi_composer.energy import EnergyFunction
from agi_composer.search import AnnealingSearch
from agi_composer.composer import Composer, FitResult

__version__ = "0.1.0"
__all__ = [
    "Primitive",
    "PRIMITIVE_REGISTRY",
    "Expression",
    "ExpressionNode",
    "EnergyFunction",
    "AnnealingSearch",
    "Composer",
    "FitResult",
]
