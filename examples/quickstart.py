"""
Quick start example for AGI Composer.

Demonstrates the core workflow:
1. Generate data from a known function
2. Use the Composer to discover the generating function
3. Inspect the result and verify generalization
"""

import numpy as np
from agi_composer import Composer


def main():
    # --- Generate mystery data ---
    # The "true" function is: y = sin(x²) + 3x
    # The Composer doesn't know this — it must discover it.
    x = np.linspace(0, 4, 100)
    y = np.sin(x ** 2) + 3 * x

    print("AGI Composer — Quick Start")
    print("=" * 50)
    print(f"Data: 100 points from x=0 to x=4")
    print(f"True function: sin(x²) + 3x")
    print()

    # --- Discover the function ---
    composer = Composer(
        seed=42,
        beta=0.005,       # Low complexity penalty for hard targets
        population_size=25,
    )

    print("Searching...")
    result = composer.fit(x, y, max_iterations=8000, verbose=True)

    # --- Print results ---
    print()
    print(result.summary())

    # --- Verify generalization ---
    x_test = np.linspace(4, 6, 50)  # Extrapolation beyond training range
    y_test = np.sin(x_test ** 2) + 3 * x_test
    y_pred = result.predict(x_test)

    extrap_mse = float(np.mean((y_pred - y_test) ** 2))
    print(f"\n  Extrapolation MSE (x=4..6): {extrap_mse:.6f}")
    print(f"  (True compositional models should extrapolate well)")


if __name__ == "__main__":
    main()
