"""
Utility functions for expression generation, mutation, and numeric safety.

This module provides the random expression generators and mutation operators
used by the search algorithm. These are the "moves" in the energy landscape
that the simulated annealing explorer uses to traverse the space of symbolic
expressions.
"""

from __future__ import annotations

import copy
import random
from typing import Optional

from typing import Optional, List

import numpy as np

from agi_composer.primitives import (
    PRIMITIVE_REGISTRY,
    UNARY_PRIMITIVES,
    BINARY_PRIMITIVES,
)
from agi_composer.expression import (
    ExpressionNode,
    Expression,
    CONST_PRIMITIVE,
    make_variable,
    make_constant,
    make_unary,
    make_binary,
)


# ---------------------------------------------------------------------------
# Random expression generation
# ---------------------------------------------------------------------------

def random_leaf(rng: random.Random) -> ExpressionNode:
    """Generate a random leaf node (variable or small constant)."""
    if rng.random() < 0.6:
        return make_variable()
    else:
        # Bias toward small integers and simple constants
        choices = [0.0, 1.0, 2.0, 3.0, -1.0, 0.5, np.pi, np.e]
        return make_constant(rng.choice(choices))


def random_expression(rng: random.Random, max_depth: int = 4,
                      current_depth: int = 0) -> ExpressionNode:
    """
    Generate a random expression tree using the grow method.

    At each level, randomly choose between:
    - A leaf node (probability increases with depth)
    - A unary operation on a random subtree
    - A binary operation on two random subtrees
    """
    # Probability of generating a leaf increases with depth
    leaf_prob = 0.3 + 0.7 * (current_depth / max_depth) if max_depth > 0 else 1.0

    if current_depth >= max_depth or rng.random() < leaf_prob:
        return random_leaf(rng)

    # Choose unary vs binary
    if rng.random() < 0.4:
        op = rng.choice(UNARY_PRIMITIVES)
        child = random_expression(rng, max_depth, current_depth + 1)
        return make_unary(op, child)
    else:
        op = rng.choice(BINARY_PRIMITIVES)
        left = random_expression(rng, max_depth, current_depth + 1)
        right = random_expression(rng, max_depth, current_depth + 1)
        return make_binary(op, left, right)


# ---------------------------------------------------------------------------
# Mutation operators
# ---------------------------------------------------------------------------

def _collect_nodes(node: ExpressionNode,
                   path: Optional[List[int]] = None
                   ) -> List[tuple]:
    """Collect all (path, node) pairs in the tree."""
    if path is None:
        path = []
    result = [(list(path), node)]
    for i, child in enumerate(node.children):
        result.extend(_collect_nodes(child, path + [i]))
    return result


def _get_node_at_path(root: ExpressionNode,
                      path: List[int]) -> ExpressionNode:
    """Navigate to a node at a given path."""
    node = root
    for idx in path:
        node = node.children[idx]
    return node


def _replace_at_path(root: ExpressionNode, path: List[int],
                     new_node: ExpressionNode) -> ExpressionNode:
    """Replace the node at path with new_node. Returns modified root."""
    if not path:
        return new_node
    root = copy.deepcopy(root)
    parent = root
    for idx in path[:-1]:
        parent = parent.children[idx]
    parent.children[path[-1]] = new_node
    return root


def mutate_subtree(root: ExpressionNode, rng: random.Random,
                   max_depth: int = 3) -> ExpressionNode:
    """Replace a random subtree with a new random expression."""
    nodes = _collect_nodes(root)
    path, _ = rng.choice(nodes)
    new_subtree = random_expression(rng, max_depth=max_depth)
    return _replace_at_path(root, path, new_subtree)


def mutate_point(root: ExpressionNode,
                 rng: random.Random) -> ExpressionNode:
    """Change a single node's operation (preserving arity and children)."""
    root = copy.deepcopy(root)
    nodes = _collect_nodes(root)
    path, node = rng.choice(nodes)

    if node.is_leaf:
        # Swap variable <-> constant, or change constant value
        if node.is_variable:
            new_node = make_constant(rng.choice([1.0, 2.0, 3.0, -1.0, 0.5]))
        elif node.is_constant:
            if rng.random() < 0.5:
                new_node = make_variable()
            else:
                new_node = make_constant(
                    node.constant_value + rng.gauss(0, 0.5)
                )
        else:
            return root
    elif node.primitive.arity == 1:
        new_op = rng.choice(UNARY_PRIMITIVES)
        new_node = ExpressionNode(new_op, children=node.children)
    elif node.primitive.arity == 2:
        new_op = rng.choice(BINARY_PRIMITIVES)
        new_node = ExpressionNode(new_op, children=node.children)
    else:
        return root

    return _replace_at_path(root, path, new_node)


def mutate_constant(root: ExpressionNode,
                    rng: random.Random, scale: float = 0.5) -> ExpressionNode:
    """Perturb a random constant in the tree."""
    root = copy.deepcopy(root)
    nodes = _collect_nodes(root)
    const_nodes = [(p, n) for p, n in nodes if n.is_constant]
    if not const_nodes:
        return root
    path, node = rng.choice(const_nodes)
    node.constant_value += rng.gauss(0, scale)
    return root


def mutate_insert(root: ExpressionNode,
                  rng: random.Random) -> ExpressionNode:
    """Insert a new operation above a random node."""
    nodes = _collect_nodes(root)
    path, target = rng.choice(nodes)

    if rng.random() < 0.5:
        # Wrap in unary
        op = rng.choice(UNARY_PRIMITIVES)
        new_node = make_unary(op, copy.deepcopy(target))
    else:
        # Make target a child of a new binary with a random other child
        op = rng.choice(BINARY_PRIMITIVES)
        other = random_leaf(rng)
        if rng.random() < 0.5:
            new_node = make_binary(op, copy.deepcopy(target), other)
        else:
            new_node = make_binary(op, other, copy.deepcopy(target))

    return _replace_at_path(root, path, new_node)


def mutate_simplify(root: ExpressionNode,
                    rng: random.Random) -> ExpressionNode:
    """Replace a random subtree with a simpler expression (shrink)."""
    nodes = _collect_nodes(root)
    # Only simplify non-root internal nodes
    internal = [(p, n) for p, n in nodes if not n.is_leaf and len(p) > 0]
    if not internal:
        return root
    path, _ = rng.choice(internal)
    return _replace_at_path(root, path, random_leaf(rng))


def mutate(root: ExpressionNode, rng: random.Random,
           temperature: float = 1.0) -> ExpressionNode:
    """
    Apply a random mutation, biased by temperature.

    At high temperature: prefer larger structural changes (exploration).
    At low temperature: prefer small tweaks (exploitation).
    """
    # Mutation weights depend on temperature
    if temperature > 0.5:
        # Exploratory regime: bigger structural changes
        weights = {
            "subtree": 0.30,
            "point": 0.20,
            "constant": 0.10,
            "insert": 0.25,
            "simplify": 0.15,
        }
    else:
        # Exploitative regime: fine-tuning
        weights = {
            "subtree": 0.10,
            "point": 0.25,
            "constant": 0.40,
            "insert": 0.05,
            "simplify": 0.20,
        }

    mutation_type = rng.choices(
        list(weights.keys()),
        weights=list(weights.values()),
        k=1
    )[0]

    mutators = {
        "subtree": lambda r: mutate_subtree(r, rng),
        "point": lambda r: mutate_point(r, rng),
        "constant": lambda r: mutate_constant(r, rng),
        "insert": lambda r: mutate_insert(r, rng),
        "simplify": lambda r: mutate_simplify(r, rng),
    }

    result = mutators[mutation_type](root)

    # Safety: prevent trees from growing too large
    expr = Expression(result)
    if expr.depth > 10 or expr.size > 50:
        return mutate_simplify(root, rng)

    return result
