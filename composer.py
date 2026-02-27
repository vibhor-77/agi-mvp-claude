"""
Composer — the high-level orchestrator for compositional function discovery.

This is the main entry point for users. It wraps the energy function and
search algorithm behind a simple fit/predict API, analogous to scikit-learn.

Usage:
    composer = Composer(seed=42)
    result = composer.fit(x, y, max_iterations=5000)
    predictions = result.expression.evaluate(new_x)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from agi_composer.energy import EnergyFunction, EnergyResult
from agi_composer.expression import Expression
from agi_composer.search import AnnealingSearch, SearchCandidate, SearchHistory


@dataclass
class FitResult:
    """Result of a compositional discovery run."""
    expression: Expression          # The discovered symbolic expression
    energy: float                   # Final energy value
    prediction_error: float         # Final prediction error (normalized RMSE)
    r_squared: float                # Coefficient of determination
    complexity: float               # MDL complexity of the expression
    raw_mse: float                  # Raw mean squared error
    history: SearchHistory          # Full search trajectory
    iterations_used: int            # How many iterations were run

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the discovered expression on new data."""
        return self.expression.evaluate(x)

    def summary(self) -> str:
        """Human-readable summary of the discovery result."""
        lines = [
            "═" * 60,
            "  AGI Composer — Discovery Result",
            "═" * 60,
            f"  Expression:  {self.expression}",
            f"  Energy:      {self.energy:.8f}",
            f"  R²:          {self.r_squared:.8f}",
            f"  RMSE:        {np.sqrt(self.raw_mse):.8f}",
            f"  Complexity:  {self.complexity:.2f}",
            f"  Tree size:   {self.expression.size} nodes",
            f"  Tree depth:  {self.expression.depth}",
            f"  Iterations:  {self.iterations_used}",
            "═" * 60,
        ]
        return "\n".join(lines)


class Composer:
    """
    High-level API for compositional function discovery.

    Parameters
    ----------
    alpha : float
        Weight on prediction error in energy function. Default 1.0.
    beta : float
        Weight on complexity cost in energy function. Default 0.01.
    population_size : int
        Number of parallel search chains. Default 20.
    initial_temperature : float
        Starting annealing temperature. Default 2.0.
    cooling_rate : float
        Temperature decay per iteration. Default 0.9995.
    seed : Optional[int]
        Random seed for reproducibility.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.01,
        population_size: int = 20,
        initial_temperature: float = 2.0,
        cooling_rate: float = 0.9995,
        seed: Optional[int] = None,
    ):
        self.energy_fn = EnergyFunction(alpha=alpha, beta=beta)
        self.search = AnnealingSearch(
            energy_fn=self.energy_fn,
            population_size=population_size,
            initial_temperature=initial_temperature,
            cooling_rate=cooling_rate,
            seed=seed,
        )
        self.seed = seed

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        max_iterations: int = 5000,
        target_error: float = 1e-6,
        verbose: bool = False,
    ) -> FitResult:
        """
        Discover the symbolic expression that generates the data.

        Parameters
        ----------
        x : np.ndarray
            Input values (1D array).
        y : np.ndarray
            Target output values (1D array, same length as x).
        max_iterations : int
            Maximum search iterations. Default 5000.
        target_error : float
            Stop early if normalized RMSE falls below this. Default 1e-6.
        verbose : bool
            Print progress during search. Default False.

        Returns
        -------
        FitResult
            The discovered expression, its accuracy metrics, and search history.
        """
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        assert len(x) == len(y), "x and y must have the same length"

        best, history = self.search.search(
            x, y,
            max_iterations=max_iterations,
            target_error=target_error,
            verbose=verbose,
        )

        er = best.energy_result
        iterations_used = history.iterations[-1] if history.iterations else 0

        return FitResult(
            expression=best.expression,
            energy=er.total_energy,
            prediction_error=er.prediction_error,
            r_squared=er.r_squared,
            complexity=er.complexity_cost,
            raw_mse=er.raw_mse,
            history=history,
            iterations_used=iterations_used,
        )
