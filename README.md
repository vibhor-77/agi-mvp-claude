# AGI Composer: Compositional Function Discovery via Energy Minimization

**Phase 1 — Compositional Discovery Engine**

A system that discovers the generating function behind a stream of numbers
by composing primitive mathematical operations, guided by an energy-based
search that minimizes description length (entropy) and prediction error.

## Core Idea

Instead of fitting millions of neural network parameters to data, AGI Composer
maintains a population of *candidate symbolic expressions* — trees of composed
mathematical functions — and evolves them toward the simplest accurate explanation
of the observed data.

The energy function balances two forces:
- **Prediction accuracy**: How well does the candidate explain the observations?
- **Complexity cost**: How simple is the symbolic expression? (Occam's razor / MDL)

```
E(candidate) = α · prediction_error + β · complexity_cost
```

This is inspired by the **Minimum Description Length** principle and connects to
thermodynamic free energy minimization in physics.

## The 4 Pillars of True General Learning

| Pillar | Role in Phase 1 |
|---|---|
| **Feedback Loops** | Energy signal feeds back to guide search toward better candidates |
| **Approximability** | Candidates approximate the true function with quantified error |
| **Abstraction & Composability** | Primitives compose into arbitrarily complex expressions |
| **Exploration** | Simulated annealing explores the space of possible compositions |

## Project Structure

```
agi-composer/
├── agi_composer/
│   ├── __init__.py
│   ├── primitives.py      # Atomic functions (sin, log, poly, etc.)
│   ├── expression.py      # Expression tree representation
│   ├── energy.py           # Energy function (prediction error + complexity)
│   ├── search.py           # Energy-based search (simulated annealing)
│   ├── composer.py         # Main orchestrator
│   └── utils.py            # Helpers and numeric utilities
├── tests/
│   ├── test_primitives.py
│   ├── test_expression.py
│   ├── test_energy.py
│   ├── test_search.py
│   └── test_integration.py
├── benchmarks/
│   ├── benchmark_suite.py  # Benchmark runner
│   └── feynman.py          # Feynman symbolic regression problems
├── examples/
│   └── quickstart.py       # Quick demo script
├── docs/
│   └── ARCHITECTURE.md     # Detailed design document
├── pyproject.toml
└── README.md
```

## Quick Start

```python
from agi_composer import Composer

# Define a mystery stream (secretly: sin(x**2) + 3*x)
import numpy as np
xs = np.linspace(0, 5, 100)
ys = np.sin(xs**2) + 3*xs

# Discover the generating function
composer = Composer(seed=42)
result = composer.fit(xs, ys, max_iterations=5000)

print(f"Discovered: {result.expression}")
print(f"Energy:     {result.energy:.6f}")
print(f"Error:      {result.prediction_error:.6f}")
```

## Running Tests

```bash
# With pytest (recommended)
pytest tests/ -v

# Without pytest (stdlib fallback)
python -m unittest discover tests/ -v
```

## Running Benchmarks

```bash
python benchmarks/benchmark_suite.py
```

## Installation

```bash
pip install -e .
```

Requires: Python 3.10+, NumPy, SciPy

## Roadmap

- **Phase 1** (current): Compositional discovery engine for mathematical functions
- **Phase 2**: Add exploration/agency — grid worlds with agent actions
- **Phase 3**: ARC-AGI subset — discover geometric transformation rules
- **Phase 4**: Zork — full compositional world model discovery

## License

MIT
