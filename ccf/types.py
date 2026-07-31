"""
ccf.types -- Data structures for the Converging Computing Framework.

Three layers: cloud (central) / region / edge.  One cloud node is always
present; regions and edges are partitioned into cabins (each region
plus a balanced share of its edge nodes).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

# ----------------------------------------------------------------------
# Layer constants
# ----------------------------------------------------------------------
LAYER_CLOUD, LAYER_REGION, LAYER_EDGE = 0, 1, 2
LAYER_NAMES = {
    LAYER_CLOUD:  "cloud",
    LAYER_REGION: "region",
    LAYER_EDGE:   "edge",
}

# Per-layer node template: (cpu, gpu, mem_GB, bandwidth_MBps,
#                            speed_GFLOPs, protection)
LAYER_TEMPLATE = {
    LAYER_CLOUD:  dict(cpu=32, gpu=4, mem=256, bw=200.0, v=1.0, prot=0.90),
    LAYER_REGION: dict(cpu=16, gpu=2, mem=128, bw=80.0,  v=0.5, prot=0.55),
    LAYER_EDGE:   dict(cpu=8,  gpu=0, mem=32,  bw=20.0,  v=0.2, prot=0.20),
}

# Cloud is always node 0 of the instance; cabin id -1 (degenerate cabin).
CLOUD_INDEX = 0
CLOUD_CABIN = -1


# ----------------------------------------------------------------------
# Public data classes
# ----------------------------------------------------------------------
@dataclass
class Instance:
    """A single scheduling instance: W nodes, N tasks, deterministic given seed."""
    N: int                                  # number of tasks
    W: int                                  # number of nodes
    nodes_layer:  np.ndarray                # (W,) int in {0, 1, 2}
    nodes_cabin:  np.ndarray                # (W,) int cabin id; cloud is its own cabin (-1)
    nodes_cpu:    np.ndarray                # (W,) float
    nodes_gpu:    np.ndarray                # (W,) float
    nodes_mem:    np.ndarray                # (W,) float
    nodes_bw:     np.ndarray                # (W,) float MB/s
    nodes_v:      np.ndarray                # (W,) float speed
    nodes_prot:   np.ndarray                # (W,) float protection
    tasks_w:      np.ndarray                # (N,) workload seconds
    tasks_cpu:    np.ndarray                # (N,)
    tasks_gpu:    np.ndarray                # (N,)
    tasks_mem:    np.ndarray                # (N,)
    tasks_data:   np.ndarray                # (N,) input MB
    tasks_sec:    np.ndarray                # (N,) int in {1, 2, 3}
    tasks_cabin:  np.ndarray                # (N,) source cabin id
    tasks_pred:   np.ndarray                # (N,) per-task noise for load term
    node_pred:    np.ndarray                # (W,) predicted load per node


@dataclass
class CCFConfig:
    """Algorithm hyperparameters; ``variant`` selects which components are active."""
    variant: str = "Ours"                   # Ours / NoRep / NoPred / NoCabin / NoSec / FlatCCF / GreedyByLoad / FirstFit / Random
    lam: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)  # lambda1..4
    rho: float = 1.5                        # cross-cabin amplification
    omega: float = 0.7                      # SOR relaxation factor
    eps: float = 1e-4                       # convergence threshold on |delta Phi|
    T_max: int = 200                        # max iterations
    cabin_radius: int = 2                   # 1 = intra-cabin only; 2 = same cabin OR same layer
    normalize: bool = True                  # rescale each term to a comparable magnitude
    random_init: bool = False               # if True, initialise stage 2 with a random
                                            # feasible assignment (used to make the
                                            # SOR convergence curve visible in plots)
    init_seed: int = 0                      # seed for random_init mode
