"""
ccf -- Converging Computing Framework Scheduling (CCF) package.

A four-component potential minimisation with a two-stage scheme:

  Stage 1 : inter-layer coarse allocation
  Stage 2 : intra-cabin SOR-style relaxation

Variants:
  * Ours / NoRep / NoPred / NoCabin / NoSec : four-route ablation
  * FlatCCF  : skips Stage 1 (no inter-layer quota)
  * GreedyByLoad / FirstFit : external heuristics
  * Random   : uniformly random feasible assignment

The integrated CCF potential is
    Phi(i, j) = lambda1 * phi_rep + lambda2 * phi_load + lambda3 * phi_cab + lambda4 * phi_sec
"""
from .types import (
    CCFConfig,
    CLOUD_CABIN,
    CLOUD_INDEX,
    Instance,
    LAYER_CLOUD,
    LAYER_EDGE,
    LAYER_NAMES,
    LAYER_REGION,
    LAYER_TEMPLATE,
)
from .instances import (
    cabin_repr_layer,
    delta_layer,
    feasibility,
    make_instance,
    sigma,
)
from .potential import build_potential, decompose_potential
from .algorithm import (
    _cross_metrics,
    _is_overloaded,
    _neighbours,
    _per_node_used,
    _sla_penalty,
    _try_place,
    solve,
    stage1_assign,
    stage2_relax,
)
from .baselines import (
    first_fit,
    greedy_by_load,
    random_assign,
)

__version__ = "0.1.0"
__all__ = [
    # types
    "CCFConfig", "Instance",
    "LAYER_CLOUD", "LAYER_REGION", "LAYER_EDGE",
    "LAYER_NAMES", "LAYER_TEMPLATE", "CLOUD_INDEX", "CLOUD_CABIN",
    # instances
    "make_instance", "feasibility", "sigma", "delta_layer",
    "cabin_repr_layer",
    # potential
    "build_potential", "decompose_potential",
    # algorithm
    "solve", "stage1_assign", "stage2_relax",
    "_neighbours", "_per_node_used", "_is_overloaded", "_try_place",
    "_cross_metrics", "_sla_penalty",
    # baselines
    "random_assign", "greedy_by_load", "first_fit",
    # version
    "__version__",
]
