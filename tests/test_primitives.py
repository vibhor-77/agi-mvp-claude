"""Tests for primitive mathematical operations."""

import unittest
import numpy as np

from agi_composer.primitives import (
    PRIMITIVE_REGISTRY,
    UNARY_PRIMITIVES,
    BINARY_PRIMITIVES,
    _safe_div,
    _safe_log,
    _safe_sqrt,
)


class TestPrimitiveRegistry(unittest.TestCase):
    """Verify the primitive registry is well-formed."""

    def test_registry_not_empty(self):
        self.assertGreater(len(PRIMITIVE_REGISTRY), 0)

    def test_all_primitives_have_required_fields(self):
        for name, prim in PRIMITIVE_REGISTRY.items():
            self.assertEqual(prim.name, name)
            self.assertIn(prim.arity, (0, 1, 2))
            self.assertGreater(prim.complexity, 0)
            self.assertIsInstance(prim.symbol, str)

    def test_unary_primitives_have_arity_1(self):
        for p in UNARY_PRIMITIVES:
            self.assertEqual(p.arity, 1)

    def test_binary_primitives_have_arity_2(self):
        for p in BINARY_PRIMITIVES:
            self.assertEqual(p.arity, 2)


class TestSafeOperations(unittest.TestCase):
    """Verify safe numeric operations handle edge cases."""

    def test_safe_div_normal(self):
        a = np.array([6.0, 10.0])
        b = np.array([2.0, 5.0])
        result = _safe_div(a, b)
        np.testing.assert_allclose(result, [3.0, 2.0])

    def test_safe_div_by_zero(self):
        a = np.array([1.0, 2.0])
        b = np.array([0.0, 0.0])
        result = _safe_div(a, b)
        self.assertTrue(np.all(np.isfinite(result)))

    def test_safe_log_positive(self):
        x = np.array([1.0, np.e, np.e**2])
        result = _safe_log(x)
        np.testing.assert_allclose(result, [0.0, 1.0, 2.0], atol=1e-10)

    def test_safe_log_non_positive(self):
        x = np.array([-1.0, 0.0, -100.0])
        result = _safe_log(x)
        self.assertTrue(np.all(np.isfinite(result)))

    def test_safe_sqrt_negative(self):
        x = np.array([-4.0, 4.0, 0.0])
        result = _safe_sqrt(x)
        self.assertTrue(np.all(np.isfinite(result)))
        self.assertAlmostEqual(result[1], 2.0)

    def test_unary_primitives_handle_arrays(self):
        x = np.linspace(-2, 2, 50)
        for prim in UNARY_PRIMITIVES:
            result = prim(x)
            self.assertEqual(result.shape, x.shape)
            self.assertTrue(np.all(np.isfinite(result)),
                            f"{prim.name} produced non-finite values")

    def test_binary_primitives_handle_arrays(self):
        a = np.linspace(-2, 2, 50)
        b = np.linspace(0.1, 3, 50)
        for prim in BINARY_PRIMITIVES:
            result = prim(a, b)
            self.assertEqual(result.shape, a.shape)
            self.assertTrue(np.all(np.isfinite(result)),
                            f"{prim.name} produced non-finite values")


if __name__ == "__main__":
    unittest.main()
