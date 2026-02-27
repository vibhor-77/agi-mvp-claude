"""Tests for the energy function."""

import unittest
import numpy as np

from agi_composer.energy import EnergyFunction
from agi_composer.expression import (
    make_expression,
    make_variable,
    make_constant,
    make_binary,
)
from agi_composer.primitives import PRIMITIVE_REGISTRY


class TestEnergyFunction(unittest.TestCase):
    """Test energy computation properties."""

    def setUp(self):
        self.x = np.linspace(0, 5, 100)
        self.y = 2.0 * self.x + 3.0
        self.energy_fn = EnergyFunction(alpha=1.0, beta=0.01)

    def test_perfect_fit_has_low_energy(self):
        """An exact expression should have near-zero prediction error."""
        root = make_binary(
            PRIMITIVE_REGISTRY["add"],
            make_binary(
                PRIMITIVE_REGISTRY["mul"],
                make_constant(2.0),
                make_variable(),
            ),
            make_constant(3.0),
        )
        expr = make_expression(root)
        result = self.energy_fn.compute(expr, self.x, self.y)

        self.assertLess(result.prediction_error, 1e-8)
        self.assertGreater(result.r_squared, 0.9999)

    def test_bad_fit_has_high_energy(self):
        """A constant expression should have high prediction error."""
        expr = make_expression(make_constant(0.0))
        result = self.energy_fn.compute(expr, self.x, self.y)

        self.assertGreater(result.prediction_error, 0.5)
        self.assertLess(result.r_squared, 0.1)

    def test_simpler_expression_has_lower_complexity(self):
        """x should have lower complexity than sin(x² + 3)."""
        simple = make_expression(make_variable())
        complex_root = make_binary(
            PRIMITIVE_REGISTRY["add"],
            make_binary(
                PRIMITIVE_REGISTRY["mul"],
                make_variable(),
                make_variable(),
            ),
            make_constant(3.0),
        )
        complex_expr = make_expression(complex_root)

        r_simple = self.energy_fn.compute(simple, self.x, self.y)
        r_complex = self.energy_fn.compute(complex_expr, self.x, self.y)

        self.assertLess(r_simple.complexity_cost, r_complex.complexity_cost)

    def test_energy_is_nonnegative(self):
        """Energy should always be non-negative."""
        expr = make_expression(make_variable())
        result = self.energy_fn.compute(expr, self.x, self.y)
        self.assertGreaterEqual(result.total_energy, 0)

    def test_beta_zero_ignores_complexity(self):
        """With beta=0, energy is purely prediction error."""
        fn = EnergyFunction(alpha=1.0, beta=0.0)
        expr = make_expression(make_variable())
        result = fn.compute(expr, self.x, self.y)
        self.assertAlmostEqual(result.total_energy, result.prediction_error)

    def test_alpha_zero_ignores_error(self):
        """With alpha=0, energy is purely complexity."""
        fn = EnergyFunction(alpha=0.0, beta=1.0)
        expr = make_expression(make_variable())
        result = fn.compute(expr, self.x, self.y)
        self.assertAlmostEqual(result.total_energy, result.complexity_cost)


if __name__ == "__main__":
    unittest.main()
