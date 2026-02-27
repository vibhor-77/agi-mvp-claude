# Architecture — AGI Composer

## Design Philosophy

AGI Composer is built on the premise that **true generalization requires
discovering the causal structure behind data**, not merely fitting curves to it.

Where deep learning asks "what weights minimize loss on this dataset?", we ask
"what is the simplest composition of known operations that explains this data?"

This is a fundamentally different question. The answer is a **symbolic program**,
not a weight matrix — and it can be inspected, verified, and composed further.

## Core Abstraction: Expression Trees

The central data structure is an **expression tree** where:

```
        add
       /   \
    sin     mul
     |     /   \
   square  3    x
     |
     x
```

Represents: `sin(x²) + 3*x`

Each node is either:
- A **terminal** (input variable `x` or a learned constant)
- A **unary operation** (sin, cos, exp, log, sqrt, square, neg, abs)
- A **binary operation** (+, -, ×, ÷, ^)

This gives us composability: any expression the system discovers is a composition
of known primitives, and can be further composed into larger expressions.

## Energy Function (Objective)

The energy function implements the **Minimum Description Length** principle:

```
E(expr) = α · normalized_prediction_error + β · complexity_cost
```

- **Prediction error**: How wrong is the expression? (RMSE normalized by target variance)
- **Complexity cost**: How many bits does the expression require? (sum of node costs)

This ensures the system finds the **simplest accurate explanation** — Occam's razor
formalized as an optimization objective.

### Connection to Physics

This mirrors the **free energy principle**:
```
F = U - TS
```
- Internal energy U ↔ prediction error (how much the model disagrees with reality)
- Entropy S ↔ model complexity (how much information the model encodes)
- Temperature T ↔ exploration-exploitation balance

At high temperature, the system tolerates disorder (high complexity, exploring).
At low temperature, it converges on minimum energy (simplest accurate model).

## Search Algorithm: Simulated Annealing

The search operates over the **combinatorial space of expression trees**:

1. **Initialize**: Random population of expression trees
2. **Mutate**: Apply random structural changes (swap subtree, change operation, etc.)
3. **Evaluate**: Compute energy of the new candidate
4. **Accept/Reject**: Metropolis-Hastings criterion — always accept improvements,
   accept worse solutions with probability `exp(-ΔE/T)`
5. **Cool**: Reduce temperature following a schedule
6. **Optimize**: Periodically tune constants via L-BFGS within the best structures

### Mutation Operators

| Operator | Effect | When Used |
|---|---|---|
| `subtree` | Replace random subtree with new random tree | High temperature (exploration) |
| `point` | Change one node's operation | Medium temperature |
| `constant` | Perturb a constant value | Low temperature (fine-tuning) |
| `insert` | Wrap a node in a new operation | High temperature |
| `simplify` | Replace subtree with a leaf | Any (Occam pressure) |

The probability of each mutation shifts with temperature, naturally transitioning
from exploration to exploitation.

## Mapping to the 4 Pillars

| Pillar | Implementation |
|---|---|
| **Feedback Loops** | Energy signal → acceptance criterion → mutation bias |
| **Approximability** | L-BFGS constant optimization within fixed structures |
| **Abstraction & Composability** | Expression trees of primitive operations |
| **Exploration** | Temperature-controlled random mutations + restarts |

## Phase Roadmap

### Phase 1 (Current): Mathematical Function Discovery
- Input: `(x, y)` pairs from a generating function
- Output: Symbolic expression that explains the data
- Key test: Can it discover `sin(x²) + 3x` from 100 data points?

### Phase 2: Agency and Grid Worlds
- Input: Observable state stream + available actions
- Output: Compositional world model + action policy
- Key extension: Primitives expand to include state transitions

### Phase 3: ARC-AGI
- Input: Grid transformation examples (input → output pairs)
- Output: Compositional transformation rule
- Key extension: Primitives expand to spatial operations (rotate, fill, mirror)

### Phase 4: Zork
- Input: Text observations + action space
- Output: Compositional world model + exploration strategy
- Key extension: Primitives expand to logical/linguistic operations
