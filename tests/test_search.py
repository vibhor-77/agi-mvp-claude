"""Tests for the simulated annealing search."""

import unittest
import numpy as np

from agi_composer.energy import EnergyFunction
from agi_composer.search import AnnealingSearch


class TestAnnealingSearch(unittest.TestCase):
    """Test the search algorithm on known functions."""

    def test_discover_linear(self):
        """Should discover y = 2x + 1 (or equivalent)."""
        x = np.linspace(0, 5, 80)
        y = 2.0 * x + 1.0

        energy_fn = EnergyFunction(alpha=1.0, beta=0.01)
        search = AnnealingSearch(
            energy_fn, population_size=15, seed=42,
            initial_temperature=1.5, cooling_rate=0.999,
        )
        best, history = search.search(x, y, max_iterations=3000)

        self.assertLess(best.energy_result.raw_mse, 0.1,
                        f"Failed to discover linear: got {best.expression}")
        self.assertGreater(best.energy_result.r_squared, 0.99)

    def test_discover_quadratic(self):
        """Should discover y = x² (or close approximation)."""
        x = np.linspace(-3, 3, 80)
        y = x ** 2

        energy_fn = EnergyFunction(alpha=1.0, beta=0.005)
        search = AnnealingSearch(
            energy_fn, population_size=20, seed=123,
            initial_temperature=2.0, cooling_rate=0.9995,
        )
        best, history = search.search(x, y, max_iterations=5000)

        self.assertGreater(best.energy_result.r_squared, 0.95,
                           f"Failed quadratic: {best.expression}, "
                           f"R²={best.energy_result.r_squared:.4f}")

    def test_history_records_progress(self):
        """Search history should record the trajectory."""
        x = np.linspace(0, 3, 50)
        y = x + 1.0

        energy_fn = EnergyFunction()
        search = AnnealingSearch(energy_fn, population_size=5, seed=0)
        _, history = search.search(x, y, max_iterations=500)

        self.assertGreater(len(history.iterations), 0)
        self.assertGreater(len(history.best_energies), 0)
        # Energy should generally decrease over time
        self.assertLessEqual(history.best_energies[-1], history.best_energies[0])

    def test_early_stopping(self):
        """Should stop early when target error is reached."""
        x = np.linspace(0, 5, 50)
        y = x * 3.0  # Very simple target

        energy_fn = EnergyFunction()
        search = AnnealingSearch(energy_fn, population_size=10, seed=42)
        best, history = search.search(
            x, y, max_iterations=10000, target_error=0.01
        )

        # Should have stopped well before 10000
        if history.iterations:
            self.assertLess(history.iterations[-1], 10000)

    def test_deterministic_with_seed(self):
        """Same seed should produce same results."""
        x = np.linspace(0, 3, 50)
        y = x ** 2

        energy_fn = EnergyFunction()

        search1 = AnnealingSearch(energy_fn, population_size=5, seed=999)
        best1, _ = search1.search(x, y, max_iterations=200)

        search2 = AnnealingSearch(energy_fn, population_size=5, seed=999)
        best2, _ = search2.search(x, y, max_iterations=200)

        self.assertAlmostEqual(best1.energy, best2.energy, places=6)


if __name__ == "__main__":
    unittest.main()
