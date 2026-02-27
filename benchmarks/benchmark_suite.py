"""
Benchmark suite for AGI Composer.

Tests the system on a set of known generating functions of increasing
complexity, measuring:
- Discovery accuracy (R², RMSE)
- Discovered expression (symbolic form)
- Computational cost (iterations, time)
- Expression complexity (tree size)

These benchmarks are inspired by the Feynman Symbolic Regression Benchmark
and provide a clear measure of the system's compositional discovery ability.
"""

import time
import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple

from agi_composer import Composer


@dataclass
class BenchmarkProblem:
    """A benchmark problem: discover the generating function from data."""
    name: str
    true_expression: str  # Human-readable ground truth
    generate: Callable[[np.ndarray], np.ndarray]
    x_range: Tuple[float, float]
    n_points: int = 100
    difficulty: str = "easy"  # easy, medium, hard


# ---------------------------------------------------------------------------
# Benchmark problems — ordered by difficulty
# ---------------------------------------------------------------------------

BENCHMARKS = [
    # --- Easy: single operations ---
    BenchmarkProblem(
        "linear", "2*x + 1",
        lambda x: 2.0 * x + 1.0,
        (0, 10), difficulty="easy",
    ),
    BenchmarkProblem(
        "quadratic", "x²",
        lambda x: x ** 2,
        (-5, 5), difficulty="easy",
    ),
    BenchmarkProblem(
        "cubic", "x³ - 2x",
        lambda x: x ** 3 - 2 * x,
        (-3, 3), difficulty="easy",
    ),

    # --- Medium: single transcendental or two-term compositions ---
    BenchmarkProblem(
        "sine", "sin(x)",
        lambda x: np.sin(x),
        (0, 2 * np.pi), difficulty="medium",
    ),
    BenchmarkProblem(
        "exp_decay", "exp(-x/3)",
        lambda x: np.exp(-x / 3),
        (0, 10), difficulty="medium",
    ),
    BenchmarkProblem(
        "log_linear", "log(x) + 2x",
        lambda x: np.log(x + 0.1) + 2 * x,
        (0.1, 10), difficulty="medium",
    ),
    BenchmarkProblem(
        "sqrt_quad", "sqrt(x² + 1)",
        lambda x: np.sqrt(x ** 2 + 1),
        (-5, 5), difficulty="medium",
    ),

    # --- Hard: deeper compositions ---
    BenchmarkProblem(
        "sin_squared", "sin(x²)",
        lambda x: np.sin(x ** 2),
        (0, 3), difficulty="hard",
    ),
    BenchmarkProblem(
        "composed", "sin(x) + x²/5",
        lambda x: np.sin(x) + x ** 2 / 5,
        (0, 6), difficulty="hard",
    ),
    BenchmarkProblem(
        "damped_sine", "exp(-x/5) * sin(3x)",
        lambda x: np.exp(-x / 5) * np.sin(3 * x),
        (0, 10), difficulty="hard",
    ),
]


def run_benchmark(problem: BenchmarkProblem,
                  max_iterations: int = 5000,
                  seed: int = 42,
                  verbose: bool = False) -> dict:
    """Run a single benchmark problem."""
    x = np.linspace(problem.x_range[0], problem.x_range[1], problem.n_points)
    y = problem.generate(x)

    composer = Composer(
        seed=seed,
        beta=0.005,
        population_size=20,
        initial_temperature=2.0,
        cooling_rate=0.9995,
    )

    t0 = time.time()
    result = composer.fit(x, y, max_iterations=max_iterations, verbose=verbose)
    elapsed = time.time() - t0

    # Test generalization on held-out data
    x_test = np.linspace(
        problem.x_range[0], problem.x_range[1], problem.n_points * 2
    )
    y_test = problem.generate(x_test)
    y_pred = result.predict(x_test)
    test_mse = float(np.mean((y_pred - y_test) ** 2))

    return {
        "name": problem.name,
        "difficulty": problem.difficulty,
        "true_expr": problem.true_expression,
        "discovered_expr": str(result.expression),
        "train_r2": result.r_squared,
        "train_rmse": np.sqrt(result.raw_mse),
        "test_mse": test_mse,
        "complexity": result.complexity,
        "tree_size": result.expression.size,
        "iterations": result.iterations_used,
        "time_sec": elapsed,
    }


def run_all_benchmarks(max_iterations: int = 5000, seed: int = 42,
                       verbose: bool = True):
    """Run all benchmark problems and print a summary table."""
    print("=" * 90)
    print("  AGI Composer — Benchmark Suite")
    print("=" * 90)
    print()

    results = []
    for problem in BENCHMARKS:
        if verbose:
            print(f"  [{problem.difficulty:6s}] {problem.name:15s} = {problem.true_expression}")
        r = run_benchmark(problem, max_iterations=max_iterations,
                          seed=seed, verbose=False)
        results.append(r)
        if verbose:
            status = "✓" if r["train_r2"] > 0.95 else "✗"
            print(f"           {status} R²={r['train_r2']:.4f}  "
                  f"RMSE={r['train_rmse']:.6f}  "
                  f"size={r['tree_size']:2d}  "
                  f"time={r['time_sec']:.1f}s  "
                  f"→ {r['discovered_expr']}")
            print()

    # Summary
    solved = sum(1 for r in results if r["train_r2"] > 0.95)
    total = len(results)
    print("=" * 90)
    print(f"  Solved: {solved}/{total} (R² > 0.95)")

    by_difficulty = {}
    for r in results:
        d = r["difficulty"]
        if d not in by_difficulty:
            by_difficulty[d] = {"solved": 0, "total": 0}
        by_difficulty[d]["total"] += 1
        if r["train_r2"] > 0.95:
            by_difficulty[d]["solved"] += 1

    for d in ["easy", "medium", "hard"]:
        if d in by_difficulty:
            s = by_difficulty[d]
            print(f"    {d:8s}: {s['solved']}/{s['total']}")

    print("=" * 90)

    return results


if __name__ == "__main__":
    run_all_benchmarks()
