"""
Expression trees — symbolic representations of composed functions.

An expression is a tree where:
- Leaf nodes are either the input variable `x` or a learned constant.
- Internal nodes are primitive operations applied to their children.

Example: sin(x² + 3) is represented as:
    sin
     |
    add
   /   \
  square  const(3)
   |
   x

The tree structure naturally supports:
- Evaluation: recursive bottom-up computation
- Complexity measurement: sum of node costs (MDL)
- Pretty printing: infix notation for readability
- Mutation: swap subtrees for search/evolution
- Coefficient optimization: extract and tune constants
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import minimize as scipy_minimize

from agi_composer.primitives import (
    Primitive,
    PRIMITIVE_REGISTRY,
    UNARY_PRIMITIVES,
    BINARY_PRIMITIVES,
)


@dataclass
class ExpressionNode:
    """A single node in an expression tree."""

    primitive: Primitive
    children: list[ExpressionNode] = field(default_factory=list)
    constant_value: Optional[float] = None  # Only for constant leaf nodes

    @property
    def is_leaf(self) -> bool:
        return self.primitive.arity == 0

    @property
    def is_constant(self) -> bool:
        return self.primitive.name == "const"

    @property
    def is_variable(self) -> bool:
        return self.primitive.name == "x"

    @property
    def depth(self) -> int:
        """Maximum depth of the subtree rooted at this node."""
        if self.is_leaf:
            return 0
        return 1 + max(child.depth for child in self.children)

    @property
    def size(self) -> int:
        """Total number of nodes in the subtree."""
        return 1 + sum(child.size for child in self.children)

    @property
    def complexity(self) -> float:
        """Total MDL complexity cost of the subtree."""
        cost = self.primitive.complexity
        if self.is_constant:
            # Constants have additional cost proportional to their information content.
            # Small integers are "cheaper" than arbitrary floats.
            if self.constant_value is not None:
                v = abs(self.constant_value)
                if v == 0:
                    cost += 0.5
                elif v == round(v) and v <= 10:
                    cost += 1.0
                else:
                    cost += 2.0 + 0.1 * np.log1p(v)
        return cost + sum(child.complexity for child in self.children)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate this expression on input array x."""
        if self.is_variable:
            return x.copy()
        if self.is_constant:
            return np.full_like(x, self.constant_value, dtype=float)
        child_values = [child.evaluate(x) for child in self.children]
        result = self.primitive(*child_values)
        # Safety: clip to prevent downstream NaN/Inf cascades
        return np.clip(np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6), -1e6, 1e6)

    def get_constants(self) -> list[float]:
        """Collect all constant values in the tree (depth-first order)."""
        constants = []
        if self.is_constant:
            constants.append(self.constant_value)
        for child in self.children:
            constants.extend(child.get_constants())
        return constants

    def set_constants(self, values: list[float]) -> int:
        """Set constant values from a flat list. Returns number consumed."""
        consumed = 0
        if self.is_constant:
            self.constant_value = values[consumed]
            consumed += 1
        for child in self.children:
            consumed += child.set_constants(values[consumed:])
        return consumed

    def __str__(self) -> str:
        """Human-readable infix notation."""
        if self.is_variable:
            return "x"
        if self.is_constant:
            v = self.constant_value
            if v is not None and v == int(v) and abs(v) < 1000:
                return str(int(v))
            return f"{v:.4g}" if v is not None else "?"

        name = self.primitive.symbol

        if self.primitive.arity == 1:
            child_str = str(self.children[0])
            if self.primitive.name == "neg":
                return f"(-{child_str})"
            if self.primitive.name == "square":
                return f"({child_str})²"
            return f"{name}({child_str})"

        if self.primitive.arity == 2:
            left = str(self.children[0])
            right = str(self.children[1])
            if self.primitive.name == "pow":
                return f"({left})^({right})"
            return f"({left} {name} {right})"

        return f"{name}(?)"

    def __repr__(self) -> str:
        return f"Node({self})"


# ---------------------------------------------------------------------------
# Constant primitive (not in global registry — created dynamically)
# ---------------------------------------------------------------------------

CONST_PRIMITIVE = Primitive("const", 0, None, 1.5, "c")


# ---------------------------------------------------------------------------
# Expression: wraps a tree with convenience methods
# ---------------------------------------------------------------------------

class Expression:
    """
    A complete symbolic expression with evaluation and optimization support.

    Wraps an ExpressionNode tree and provides:
    - Evaluation on numpy arrays
    - Constant optimization via L-BFGS
    - Deep copy for safe mutation
    - String representation
    """

    def __init__(self, root: ExpressionNode):
        self.root = root

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the expression on input data."""
        return self.root.evaluate(x)

    @property
    def complexity(self) -> float:
        return self.root.complexity

    @property
    def depth(self) -> int:
        return self.root.depth

    @property
    def size(self) -> int:
        return self.root.size

    def optimize_constants(self, x: np.ndarray, y: np.ndarray,
                           max_iter: int = 50) -> float:
        """
        Optimize constant values via L-BFGS to minimize MSE.

        This is the "approximation" pillar: given a fixed symbolic structure,
        find the best coefficients. Returns the optimized MSE.
        """
        constants = self.root.get_constants()
        if not constants:
            # No constants to optimize
            pred = self.evaluate(x)
            return float(np.mean((pred - y) ** 2))

        def objective(params: np.ndarray) -> float:
            self.root.set_constants(params.tolist())
            pred = self.evaluate(x)
            mse = float(np.mean((pred - y) ** 2))
            return mse if np.isfinite(mse) else 1e10

        initial = np.array(constants, dtype=float)
        try:
            result = scipy_minimize(
                objective,
                initial,
                method="L-BFGS-B",
                options={"maxiter": max_iter, "ftol": 1e-10},
            )
            self.root.set_constants(result.x.tolist())
            return float(result.fun)
        except Exception:
            # If optimization fails, restore original constants
            self.root.set_constants(constants)
            pred = self.evaluate(x)
            return float(np.mean((pred - y) ** 2))

    def copy(self) -> Expression:
        """Deep copy for safe mutation."""
        return Expression(copy.deepcopy(self.root))

    def __str__(self) -> str:
        return str(self.root)

    def __repr__(self) -> str:
        return f"Expression({self.root})"


# ---------------------------------------------------------------------------
# Factory functions for building expressions
# ---------------------------------------------------------------------------

def make_variable() -> ExpressionNode:
    """Create an input variable node (x)."""
    return ExpressionNode(PRIMITIVE_REGISTRY["x"])


def make_constant(value: float = 1.0) -> ExpressionNode:
    """Create a constant node with a given value."""
    return ExpressionNode(CONST_PRIMITIVE, constant_value=value)


def make_unary(op: Primitive, child: ExpressionNode) -> ExpressionNode:
    """Create a unary operation node."""
    assert op.arity == 1, f"{op.name} is not unary"
    return ExpressionNode(op, children=[child])


def make_binary(op: Primitive, left: ExpressionNode,
                right: ExpressionNode) -> ExpressionNode:
    """Create a binary operation node."""
    assert op.arity == 2, f"{op.name} is not binary"
    return ExpressionNode(op, children=[left, right])


def make_expression(root: ExpressionNode) -> Expression:
    """Wrap a node tree into an Expression."""
    return Expression(root)
