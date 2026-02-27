"""
Energy-based search via Simulated Annealing over symbolic expressions.

The search algorithm explores the space of possible compositions by:
1. Starting with a population of random candidate expressions.
2. At each step, mutating candidates and evaluating their energy.
3. Accepting improvements always; accepting worse solutions with probability
   that decreases with temperature (Boltzmann acceptance).
4. Gradually cooling the temperature to shift from exploration to exploitation.
5. Periodically optimizing constants within the best candidates.

This directly implements the 4 pillars:
- **Feedback loop**: Energy signal guides which candidates survive.
- **Approximability**: Constant optimization refines coefficients.
- **Composability**: Mutations compose/decompose primitive operations.
- **Exploration**: Temperature-controlled random mutations explore the space.

The cooling schedule mirrors entropy minimization: at high temperature,
the system explores freely (high entropy); at low temperature, it
converges on the minimum energy state (low entropy).
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from agi_composer.expression import Expression, ExpressionNode
from agi_composer.energy import EnergyFunction, EnergyResult
from agi_composer.utils import random_expression, mutate


@dataclass
class SearchCandidate:
    """A candidate expression with its cached energy."""
    expression: Expression
    energy_result: Optional[EnergyResult] = None

    @property
    def energy(self) -> float:
        return self.energy_result.total_energy if self.energy_result else float("inf")

    def copy(self) -> SearchCandidate:
        return SearchCandidate(
            expression=self.expression.copy(),
            energy_result=self.energy_result,
        )


@dataclass
class SearchHistory:
    """Records the search trajectory for analysis."""
    iterations: List[int] = field(default_factory=list)
    best_energies: List[float] = field(default_factory=list)
    temperatures: List[float] = field(default_factory=list)
    best_expressions: List[str] = field(default_factory=list)
    acceptance_rates: List[float] = field(default_factory=list)


class AnnealingSearch:
    """
    Simulated annealing search over the space of symbolic expressions.

    Uses a population of candidates (parallel chains) with periodic
    cross-pollination of the best solutions.

    Parameters
    ----------
    energy_fn : EnergyFunction
        The energy function to minimize.
    population_size : int
        Number of parallel search chains. Default 20.
    initial_temperature : float
        Starting temperature for annealing. Default 2.0.
    cooling_rate : float
        Multiplicative cooling factor per iteration. Default 0.9995.
    min_temperature : float
        Floor temperature (never cool below this). Default 0.001.
    optimize_constants_every : int
        How often to run L-BFGS on constants. Default 100.
    restart_stale_after : int
        Restart a chain if it hasn't improved in this many iterations. Default 500.
    max_depth : int
        Maximum depth for initial random expressions. Default 4.
    seed : Optional[int]
        Random seed for reproducibility.
    """

    def __init__(
        self,
        energy_fn: EnergyFunction,
        population_size: int = 20,
        initial_temperature: float = 2.0,
        cooling_rate: float = 0.9995,
        min_temperature: float = 0.001,
        optimize_constants_every: int = 100,
        restart_stale_after: int = 500,
        max_depth: int = 4,
        seed: Optional[int] = None,
    ):
        self.energy_fn = energy_fn
        self.population_size = population_size
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.optimize_constants_every = optimize_constants_every
        self.restart_stale_after = restart_stale_after
        self.max_depth = max_depth
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

    def _init_population(self, x: np.ndarray,
                         y: np.ndarray) -> List[SearchCandidate]:
        """Create initial population of random expressions."""
        population = []
        for _ in range(self.population_size):
            root = random_expression(self.rng, max_depth=self.max_depth)
            expr = Expression(root)
            # Initial constant optimization
            expr.optimize_constants(x, y, max_iter=20)
            result = self.energy_fn.compute(expr, x, y)
            population.append(SearchCandidate(expr, result))
        return population

    def _boltzmann_accept(self, delta_energy: float,
                          temperature: float) -> bool:
        """
        Metropolis-Hastings acceptance criterion.

        Always accept improvements. Accept worse solutions with probability
        exp(-delta_E / T), enabling escape from local minima.
        """
        if delta_energy <= 0:
            return True
        if temperature < 1e-10:
            return False
        prob = math.exp(-delta_energy / temperature)
        return self.rng.random() < prob

    def search(
        self,
        x: np.ndarray,
        y: np.ndarray,
        max_iterations: int = 5000,
        target_error: float = 1e-6,
        callback: Optional[Callable[[int, SearchCandidate, float], None]] = None,
        verbose: bool = False,
    ) -> "tuple":
        """
        Run the annealing search.

        Parameters
        ----------
        x : np.ndarray
            Input values.
        y : np.ndarray
            Target output values.
        max_iterations : int
            Maximum number of annealing iterations.
        target_error : float
            Stop early if prediction error falls below this.
        callback : Callable, optional
            Called each iteration with (iteration, best_candidate, temperature).
        verbose : bool
            Print progress every 500 iterations.

        Returns
        -------
        Tuple of (SearchCandidate, SearchHistory)
            The best candidate found and the search history.
        """
        # Initialize
        population = self._init_population(x, y)
        best_overall = min(population, key=lambda c: c.energy).copy()
        temperature = self.initial_temperature
        history = SearchHistory()
        stale_counters = [0] * self.population_size
        accepted_count = 0
        total_proposals = 0

        for iteration in range(max_iterations):
            # --- Anneal each chain ---
            for i, candidate in enumerate(population):
                # Mutate
                new_root = mutate(
                    copy.deepcopy(candidate.expression.root),
                    self.rng,
                    temperature=temperature,
                )
                new_expr = Expression(new_root)

                # Optimize constants periodically (not every step — expensive)
                if iteration % self.optimize_constants_every == 0:
                    new_expr.optimize_constants(x, y, max_iter=30)

                new_result = self.energy_fn.compute(new_expr, x, y)
                delta = new_result.total_energy - candidate.energy
                total_proposals += 1

                if self._boltzmann_accept(delta, temperature):
                    population[i] = SearchCandidate(new_expr, new_result)
                    accepted_count += 1
                    stale_counters[i] = 0

                    # Update global best
                    if new_result.total_energy < best_overall.energy:
                        best_overall = SearchCandidate(
                            new_expr.copy(), new_result
                        )
                else:
                    stale_counters[i] += 1

                # Restart stale chains
                if stale_counters[i] >= self.restart_stale_after:
                    root = random_expression(self.rng, max_depth=self.max_depth)
                    expr = Expression(root)
                    expr.optimize_constants(x, y, max_iter=20)
                    result = self.energy_fn.compute(expr, x, y)
                    population[i] = SearchCandidate(expr, result)
                    stale_counters[i] = 0

            # --- Cross-pollination: inject best into worst chain ---
            if iteration % 200 == 0 and iteration > 0:
                worst_idx = max(range(len(population)),
                                key=lambda j: population[j].energy)
                population[worst_idx] = best_overall.copy()

            # --- Cool temperature ---
            temperature = max(
                self.min_temperature,
                temperature * self.cooling_rate,
            )

            # --- Record history ---
            if iteration % 50 == 0:
                rate = accepted_count / max(total_proposals, 1)
                history.iterations.append(iteration)
                history.best_energies.append(best_overall.energy)
                history.temperatures.append(temperature)
                history.best_expressions.append(str(best_overall.expression))
                history.acceptance_rates.append(rate)

            # --- Verbose output ---
            if verbose and iteration % 500 == 0:
                print(
                    f"[iter {iteration:5d}] "
                    f"T={temperature:.4f}  "
                    f"best_energy={best_overall.energy:.6f}  "
                    f"R²={best_overall.energy_result.r_squared:.6f}  "
                    f"expr={best_overall.expression}"
                )

            # --- Callback ---
            if callback:
                callback(iteration, best_overall, temperature)

            # --- Early stopping ---
            if (best_overall.energy_result and
                    best_overall.energy_result.prediction_error < target_error):
                if verbose:
                    print(f"[iter {iteration}] Target error reached. Stopping.")
                break

        # Final constant optimization on the best solution
        best_overall.expression.optimize_constants(x, y, max_iter=200)
        best_overall.energy_result = self.energy_fn.compute(
            best_overall.expression, x, y
        )

        return best_overall, history
