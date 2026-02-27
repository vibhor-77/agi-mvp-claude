"""
Energy function — the optimization objective for compositional discovery.

Inspired by thermodynamic free energy and the Minimum Description Length (MDL)
principle, the energy of a candidate expression balances two competing forces:

    E(candidate) = α · prediction_error + β · complexity_cost

- **Prediction error** measures how well the expression fits the observed data.
  Lower is better. We use normalized RMSE for scale-invariance.

- **Complexity cost** measures how simple the symbolic expression is (Occam's razor).
  Simpler expressions are preferred because they are more likely to capture the
  true generating function rather than overfitting noise.

The balance between α and β controls the exploration/exploitation tradeoff:
- High α: prioritize accuracy (risk overfitting)
- High β: prioritize simplicity (risk underfitting)

This mirrors the free energy principle in physics:
    F = U - TS  (free energy = internal energy - temperature × entropy)

Where accuracy plays the role of internal energy, and complexity plays the
role of entropy (measuring the "information cost" of the model).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agi_composer.expression import Expression


@dataclass
class EnergyResult:
    """Detailed breakdown of energy computation."""
    total_energy: float
    prediction_error: float  # Normalized RMSE
    complexity_cost: float   # MDL complexity of expression
    raw_mse: float           # Un-normalized mean squared error
    r_squared: float         # Coefficient of determination

    def __repr__(self) -> str:
        return (
            f"Energy(total={self.total_energy:.6f}, "
            f"error={self.prediction_error:.6f}, "
            f"complexity={self.complexity_cost:.2f}, "
            f"R²={self.r_squared:.6f})"
        )


class EnergyFunction:
    """
    Computes the energy of a candidate expression given observed data.

    Parameters
    ----------
    alpha : float
        Weight on prediction error. Default 1.0.
    beta : float
        Weight on complexity cost. Default 0.01.
        Small default because complexity is typically much larger than RMSE.
    error_transform : str
        How to transform MSE before weighting. Options:
        - "rmse": square root of MSE (default)
        - "log": log(1 + MSE), more tolerant of large errors
        - "raw": plain MSE
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.01,
                 error_transform: str = "rmse"):
        assert alpha >= 0, "alpha must be non-negative"
        assert beta >= 0, "beta must be non-negative"
        assert error_transform in ("rmse", "log", "raw")
        self.alpha = alpha
        self.beta = beta
        self.error_transform = error_transform

    def compute(self, expression: Expression, x: np.ndarray,
                y: np.ndarray) -> EnergyResult:
        """
        Compute the energy of a candidate expression.

        Parameters
        ----------
        expression : Expression
            The candidate symbolic expression.
        x : np.ndarray
            Input values.
        y : np.ndarray
            Target output values.

        Returns
        -------
        EnergyResult
            Detailed energy breakdown.
        """
        pred = expression.evaluate(x)

        # Mean squared error
        residuals = pred - y
        mse = float(np.mean(residuals ** 2))

        # Normalized prediction error (scale-invariant)
        y_var = float(np.var(y))
        if y_var > 1e-10:
            normalized_mse = mse / y_var
        else:
            # Target is nearly constant — just use absolute MSE
            normalized_mse = mse

        # Transform the error metric
        if self.error_transform == "rmse":
            prediction_error = float(np.sqrt(normalized_mse))
        elif self.error_transform == "log":
            prediction_error = float(np.log1p(normalized_mse))
        else:
            prediction_error = normalized_mse

        # Complexity cost (MDL)
        complexity_cost = expression.complexity

        # Total energy
        total = self.alpha * prediction_error + self.beta * complexity_cost

        # R² for interpretability
        if y_var > 1e-10:
            r_squared = 1.0 - mse / y_var
        else:
            r_squared = 1.0 if mse < 1e-10 else 0.0

        return EnergyResult(
            total_energy=total,
            prediction_error=prediction_error,
            complexity_cost=complexity_cost,
            raw_mse=mse,
            r_squared=r_squared,
        )

    def __repr__(self) -> str:
        return (
            f"EnergyFunction(α={self.alpha}, β={self.beta}, "
            f"transform={self.error_transform})"
        )
