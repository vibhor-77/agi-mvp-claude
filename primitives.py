"""
Primitive mathematical functions — the atomic building blocks of compositions.

Each primitive has:
- A callable that operates on numpy arrays
- An arity (1 = unary like sin, 2 = binary like add)
- A human-readable name
- A complexity cost (simpler = lower cost, following MDL principle)

The set of primitives defines the "vocabulary" from which the system composes
candidate explanations. This is analogous to how a baby might have innate
primitives for spatial reasoning, and composes them to understand the world.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True, slots=True)
class Primitive:
    """An atomic mathematical operation."""

    name: str
    arity: int  # 1 = unary (sin, log), 2 = binary (+, *), 0 = constant/variable
    func: Callable
    complexity: float  # MDL cost of using this primitive
    symbol: str  # For pretty-printing expressions

    def __call__(self, *args: np.ndarray) -> np.ndarray:
        return self.func(*args)

    def __repr__(self) -> str:
        return f"Primitive({self.name}, arity={self.arity})"


# ---------------------------------------------------------------------------
# Safe numeric operations (avoid NaN/Inf propagation)
# ---------------------------------------------------------------------------

def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Division that returns 0 where denominator is near-zero."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(np.abs(b) > 1e-10, a / b, 0.0)
    return np.clip(result, -1e6, 1e6)


def _safe_log(x: np.ndarray) -> np.ndarray:
    """Natural log, returns 0 for non-positive inputs."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(x > 1e-10, np.log(x), 0.0)
    return np.clip(result, -1e6, 1e6)


def _safe_sqrt(x: np.ndarray) -> np.ndarray:
    """Square root, returns 0 for negative inputs."""
    return np.sqrt(np.maximum(x, 0.0))


def _safe_pow(base: np.ndarray, exp: np.ndarray) -> np.ndarray:
    """Power that clips to avoid overflow."""
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.power(np.abs(base) + 1e-10, np.clip(exp, -5, 5))
    return np.clip(result, -1e6, 1e6)


def _safe_exp(x: np.ndarray) -> np.ndarray:
    """Exponential with overflow protection."""
    return np.exp(np.clip(x, -50, 50))


# ---------------------------------------------------------------------------
# Primitive registry — the function vocabulary
# ---------------------------------------------------------------------------

# Nullary (arity 0): input variable and learned constants
_INPUT = Primitive("x", 0, None, 1.0, "x")  # func is unused; handled specially

# Unary (arity 1): single-argument transformations
_SIN = Primitive("sin", 1, np.sin, 2.0, "sin")
_COS = Primitive("cos", 1, np.cos, 2.0, "cos")
_EXP = Primitive("exp", 1, _safe_exp, 3.0, "exp")
_LOG = Primitive("log", 1, _safe_log, 3.0, "log")
_NEG = Primitive("neg", 1, np.negative, 1.0, "-")
_ABS = Primitive("abs", 1, np.abs, 1.5, "abs")
_SQRT = Primitive("sqrt", 1, _safe_sqrt, 2.0, "sqrt")
_SQUARE = Primitive("square", 1, np.square, 1.5, "²")

# Binary (arity 2): two-argument operations
_ADD = Primitive("add", 2, np.add, 1.0, "+")
_SUB = Primitive("sub", 2, np.subtract, 1.0, "-")
_MUL = Primitive("mul", 2, np.multiply, 1.5, "*")
_DIV = Primitive("div", 2, _safe_div, 2.0, "/")
_POW = Primitive("pow", 2, _safe_pow, 3.0, "^")

# Organized registry for easy access
PRIMITIVE_REGISTRY: dict[str, Primitive] = {
    # Nullary
    "x": _INPUT,
    # Unary
    "sin": _SIN,
    "cos": _COS,
    "exp": _EXP,
    "log": _LOG,
    "neg": _NEG,
    "abs": _ABS,
    "sqrt": _SQRT,
    "square": _SQUARE,
    # Binary
    "add": _ADD,
    "sub": _SUB,
    "mul": _MUL,
    "div": _DIV,
    "pow": _POW,
}

# Convenience groupings
UNARY_PRIMITIVES = [p for p in PRIMITIVE_REGISTRY.values() if p.arity == 1]
BINARY_PRIMITIVES = [p for p in PRIMITIVE_REGISTRY.values() if p.arity == 2]
TERMINAL_PRIMITIVES = [p for p in PRIMITIVE_REGISTRY.values() if p.arity == 0]
