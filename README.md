# CCF Scheduler

> **Converging Computing Framework (CCF)** —
> a four-route potential minimisation for computing power network scheduling,
> with a two-stage scheme (inter-layer quota + intra-cabin SOR relaxation).
>
> Companion code for the paper *"A Converging Computing Framework of
> Intelligent Computing and Supercomputing for Computing Power Networks"*.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)]()

---

## What is CCF?

CCF models computing power network scheduling as the minimisation of a
**four-component potential**

```
Phi(i, j) = lambda1 * phi_rep      # representation / knowledge-graph distance
            + lambda2 * phi_load  # behaviour-predicted load
            + lambda3 * phi_cab    # cross-cabin / cross-layer transfer
            + lambda4 * phi_sec    # security weight
```

subject to per-node CPU/GPU/MEM capacity constraints, and solves it with
a two-stage scheme:

1. **Stage 1** — inter-layer coarse allocation (each task pushed to a
   target layer with smallest marginal potential, while respecting a
   per-layer CPU-share quota);
2. **Stage 2** — intra-cabin SOR-style relaxation (overloaded nodes shed
   tasks to neighbours with smallest potential, accepting the move iff
   `Phi_new <= omega * Phi_old`).

The package exposes **nine variants** for ablation:

| Variant        | Description                                            |
|----------------|--------------------------------------------------------|
| `Ours`         | full four-route CCF                                    |
| `NoRep`        | drop `phi_rep` (no knowledge-graph term)               |
| `NoPred`       | drop the predicted-load term (`l_j -> 0`)             |
| `NoCabin`      | drop the cabin term (`rho = 1`)                       |
| `NoSec`        | drop the security term                                 |
| `FlatCCF`      | skip Stage 1 (no inter-layer quota)                   |
| `GreedyByLoad` | external LPT-style least-loaded heuristic              |
| `FirstFit`     | external smallest-index-fits heuristic                 |
| `Random`       | uniformly random feasible assignment                    |

---

## Project layout

```
ccf-scheduler/
├── ccf/                       # core package
│   ├── __init__.py            # public API
│   ├── types.py               # Instance, CCFConfig, layer constants
│   ├── instances.py           # data generation + feasibility helpers
│   ├── potential.py           # Phi matrix + decomposition
│   ├── algorithm.py           # stage1, stage2, solve, metrics
│   └── baselines.py           # Random, GreedyByLoad, FirstFit
├── experiments/               # entry-point scripts
│   ├── main_experiment.py     # 9-scale x 9-variant x 5-seed grid
│   └── sensitivity.py         # convergence + omega sweep + figures
├── tests/                     # pytest unit tests
│   ├── test_invariants.py
│   ├── test_variants.py
│   └── test_baselines.py
├── results/                   # generated CSVs + figures
│   ├── raw.csv
│   ├── scales.csv
│   ├── convergence.csv
│   ├── omega_sensitivity.csv
│   └── figures/
├── pyproject.toml             # PEP 517/518 build config
├── requirements.txt           # pinned runtime deps
├── Makefile                   # convenience targets
├── LICENSE
└── README.md
```

---

## Installation

```bash
# Option 1: pip install (editable)
pip install -e ".[dev]"

# Option 2: just runtime deps (no test deps)
pip install -r requirements.txt
```

Requires **Python 3.10+** and the following packages:
- `numpy >= 1.24`
- `matplotlib >= 3.7`
- `pandas >= 2.0` (optional, for ad-hoc analysis)

---

## Quick start

```python
import ccf

# Build a reproducible instance
inst = ccf.make_instance(N=50, W=10, seed=42)

# Run the full CCF solver
cfg = ccf.CCFConfig(variant="Ours", omega=0.7, T_max=200)
res = ccf.solve(inst, cfg)

print(f"Phi_total:   {res['phi_total']:.2f}")
print(f"co-cabin:     {res['cross_cabin_ratio']:.2f}")
print(f"cross-layer: {res['cross_layer_ratio']:.2f}")
print(f"decomp:       rep={res['phi_rep']:.1f} load={res['phi_load']:.1f} "
      f"cab={res['phi_cab']:.1f} sec={res['phi_sec']:.1f}")
print(f"solve time:   {res['solve_ms']:.1f} ms")
```

To reproduce all paper figures, run the experiment scripts:

```bash
# Full grid (9 scales x 9 variants x 5 seeds ~ a few minutes)
python -m experiments.main_experiment

# SOR convergence + omega sensitivity + figures
python -m experiments.sensitivity
```

Or via Make:

```bash
make install   # editable + dev deps
make test      # pytest
make smoke     # import sanity check
make grid      # 9-scale x 9-variant x 5-seed
make sensitivity  # convergence / omega / figures
make all       # grid + sensitivity
make clean     # remove generated artifacts
```

---

## Public API

The package exports the following public API (`from ccf import ...`):

```python
# Data structures
CCFConfig      # algorithm hyperparameters (variant, omega, T_max, ...)
Instance       # N tasks, W nodes, deterministic given seed
LAYER_CLOUD, LAYER_REGION, LAYER_EDGE
LAYER_NAMES, LAYER_TEMPLATE, CLOUD_INDEX, CLOUD_CABIN

# Synthetic data
make_instance(N, W, seed=0)

# Predicates
feasibility(inst)             # (N, W) bool
sigma(pi)                     # security weight
delta_layer(...)              # layer-distance matrix
cabin_repr_layer(inst, c)     # representative layer of a cabin

# Potential
build_potential(inst, cfg)    # (N, W) Phi matrix
decompose_potential(inst, a)  # {phi_rep, phi_load, phi_cab, phi_sec}

# Core algorithm
solve(inst, cfg, record=False)  # full result dict
stage1_assign(inst, Phi)        # inter-layer target layer
stage2_relax(inst, Phi, target, cfg)  # intra-cabin SOR

# Baselines (external heuristics)
random_assign(inst, seed=0)
greedy_by_load(inst, cfg, record=False)
first_fit(inst, cfg, record=False)
```

---

## Testing

```bash
pytest                          # 14 tests across 3 modules
pytest -v --tb=short            # verbose output
pytest --cov=ccf tests/         # with coverage (requires pytest-cov)
```

The tests check:
- deterministic instance generation,
- feasibility / potential shape and finiteness,
- the decomposition sums to the integrated potential,
- all 9 variants run end-to-end without errors,
- `solve()` is deterministic,
- the `record=True` history is non-increasing for under-relaxation,
- baseline sanity (feasibility, capacity).

---

## License

[MIT](./LICENSE) © 2026 Wingspeg.

Originally authored by Liying Wang and Cheng Wang (Tongji University).
