"""Tests for expression tree construction, evaluation, and optimization."""

import unittest
import numpy as np

from agi_composer.expression import (
    Expression,
    ExpressionNode,
    make_variable,
    make_constant,
    make_unary,
    make_binary,
    make_expression,
)
from agi_composer.primitives import PRIMITIVE_REGISTRY


class TestExpressionNode(unittest.TestCase):
    """Test basic node properties."""

    def test_variable_is_leaf(self):
        node = make_variable()
        self.assertTrue(node.is_leaf)
        self.assertTrue(node.is_variable)
        self.assertFalse(node.is_constant)

    def test_constant_is_leaf(self):
        node = make_constant(3.14)
        self.assertTrue(node.is_leaf)
        self.assertTrue(node.is_constant)
        self.assertAlmostEqual(node.constant_value, 3.14)

    def test_depth_leaf(self):
        self.assertEqual(make_variable().depth, 0)

    def test_depth_nested(self):
        # sin(x) has depth 1
        sin_x = make_unary(PRIMITIVE_REGISTRY["sin"], make_variable())
        self.assertEqual(sin_x.depth, 1)

        # sin(x + 3) has depth 2
        x_plus_3 = make_binary(
            PRIMITIVE_REGISTRY["add"], make_variable(), make_constant(3.0)
        )
        sin_expr = make_unary(PRIMITIVE_REGISTRY["sin"], x_plus_3)
        self.assertEqual(sin_expr.depth, 2)

    def test_size(self):
        # x: 1 node
        self.assertEqual(make_variable().size, 1)
        # x + 3: 3 nodes
        node = make_binary(
            PRIMITIVE_REGISTRY["add"], make_variable(), make_constant(3.0)
        )
        self.assertEqual(node.size, 3)

    def test_complexity_small_integer_cheaper(self):
        small = make_constant(2.0)
        large = make_constant(123.456)
        self.assertLess(small.complexity, large.complexity)


class TestExpressionEvaluation(unittest.TestCase):
    """Test expression evaluation on numpy arrays."""

    def test_variable_returns_input(self):
        x = np.array([1.0, 2.0, 3.0])
        expr = make_expression(make_variable())
        np.testing.assert_array_equal(expr.evaluate(x), x)

    def test_constant_returns_constant(self):
        x = np.array([1.0, 2.0, 3.0])
        expr = make_expression(make_constant(5.0))
        np.testing.assert_array_equal(expr.evaluate(x), [5.0, 5.0, 5.0])

    def test_sin_x(self):
        x = np.array([0.0, np.pi / 2, np.pi])
        root = make_unary(PRIMITIVE_REGISTRY["sin"], make_variable())
        expr = make_expression(root)
        expected = np.sin(x)
        np.testing.assert_allclose(expr.evaluate(x), expected, atol=1e-10)

    def test_x_plus_3(self):
        x = np.array([1.0, 2.0, 3.0])
        root = make_binary(
            PRIMITIVE_REGISTRY["add"], make_variable(), make_constant(3.0)
        )
        expr = make_expression(root)
        np.testing.assert_allclose(expr.evaluate(x), [4.0, 5.0, 6.0])

    def test_x_squared(self):
        x = np.array([2.0, 3.0, 4.0])
        root = make_unary(PRIMITIVE_REGISTRY["square"], make_variable())
        expr = make_expression(root)
        np.testing.assert_allclose(expr.evaluate(x), [4.0, 9.0, 16.0])

    def test_sin_x_squared_plus_3x(self):
        """Test composition: sin(x²) + 3*x."""
        x = np.linspace(0, 2, 50)
        expected = np.sin(x**2) + 3 * x

        x_squared = make_unary(PRIMITIVE_REGISTRY["square"], make_variable())
        sin_x2 = make_unary(PRIMITIVE_REGISTRY["sin"], x_squared)
        three_x = make_binary(
            PRIMITIVE_REGISTRY["mul"], make_constant(3.0), make_variable()
        )
        root = make_binary(PRIMITIVE_REGISTRY["add"], sin_x2, three_x)
        expr = make_expression(root)

        np.testing.assert_allclose(expr.evaluate(x), expected, atol=1e-10)

    def test_evaluation_produces_finite(self):
        """Any expression should produce finite results on reasonable input."""
        x = np.linspace(0.1, 5, 100)
        root = make_unary(
            PRIMITIVE_REGISTRY["log"],
            make_unary(PRIMITIVE_REGISTRY["sin"], make_variable()),
        )
        expr = make_expression(root)
        result = expr.evaluate(x)
        self.assertTrue(np.all(np.isfinite(result)))


class TestConstantOptimization(unittest.TestCase):
    """Test L-BFGS constant optimization."""

    def test_optimize_linear_coefficients(self):
        """Discover y = a*x + b given the structure add(mul(const, x), const)."""
        x = np.linspace(0, 10, 100)
        y = 2.5 * x + 7.0

        # Structure: const * x + const (with wrong initial constants)
        ax = make_binary(
            PRIMITIVE_REGISTRY["mul"], make_constant(1.0), make_variable()
        )
        root = make_binary(PRIMITIVE_REGISTRY["add"], ax, make_constant(0.0))
        expr = make_expression(root)

        mse = expr.optimize_constants(x, y)
        self.assertLess(mse, 0.01, "Should optimize constants close to 2.5 and 7.0")

    def test_no_constants_to_optimize(self):
        """Expression with no constants should return MSE without crashing."""
        x = np.array([1.0, 2.0, 3.0])
        y = x * 2
        expr = make_expression(make_variable())
        mse = expr.optimize_constants(x, y)
        self.assertTrue(np.isfinite(mse))


class TestExpressionString(unittest.TestCase):
    """Test human-readable string representation."""

    def test_variable_str(self):
        self.assertEqual(str(make_variable()), "x")

    def test_constant_str(self):
        self.assertEqual(str(make_constant(3.0)), "3")

    def test_sin_x_str(self):
        node = make_unary(PRIMITIVE_REGISTRY["sin"], make_variable())
        self.assertEqual(str(node), "sin(x)")

    def test_binary_str(self):
        node = make_binary(
            PRIMITIVE_REGISTRY["add"], make_variable(), make_constant(3.0)
        )
        self.assertEqual(str(node), "(x + 3)")


class TestExpressionCopy(unittest.TestCase):
    """Test deep copy independence."""

    def test_copy_is_independent(self):
        root = make_binary(
            PRIMITIVE_REGISTRY["add"], make_variable(), make_constant(5.0)
        )
        expr1 = make_expression(root)
        expr2 = expr1.copy()

        # Modify the copy
        expr2.root.children[1].constant_value = 99.0

        # Original should be unchanged
        self.assertAlmostEqual(expr1.root.children[1].constant_value, 5.0)


if __name__ == "__main__":
    unittest.main()
