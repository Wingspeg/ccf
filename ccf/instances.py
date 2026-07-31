"""
ccf.instances -- Synthetic data generation and helper predicates.

A CCF instance is a fully synthetic workload with three layers (cloud /
region / edge) and a cabin structure (each region cabin plus a balanced
share of the edge nodes).  All randomness is seeded so the same
(N, W, seed) always produces the same instance.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from .types import (
    CLOUD_INDEX,
    Instance,
    LAYER_CLOUD,
    LAYER_EDGE,
    LAYER_REGION,
    LAYER_TEMPLATE,
)


# ----------------------------------------------------------------------
# Instance generation
# ----------------------------------------------------------------------
def _node_layout(W: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Build the (W,) layer and (W,) cabin arrays.

    Cabin structure: 1 central cloud (cabin -1) + W_r regions + W_e edges.
    Each cabin is one region plus a balanced split of the edge nodes.
    """
    n_cloud = 1
    rest = W - 1
    W_r = max(1, int(round(rest * 0.40)))
    W_e = rest - W_r
    layer = np.concatenate([
        np.full(n_cloud, LAYER_CLOUD, dtype=int),
        np.full(W_r, LAYER_REGION, dtype=int),
        np.full(W_e, LAYER_EDGE,   dtype=int),
    ])

    # Cabins: cloud is its own; regions each form a cabin with floor(W_e/W_r)
    # edge nodes; leftover edges attach to the last cabin.
    cabin = np.full(W, -1, dtype=int)
    cabin[CLOUD_INDEX] = -1
    if W_r > 0:
        per = max(1, W_e // W_r)
        idx = 1 + W_r
        for k in range(W_r):
            cabin[1 + k] = k
            cabin[idx: idx + per] = k
            idx += per
        if idx < W:
            cabin[idx:] = W_r - 1
    return layer, cabin


def make_instance(N: int, W: int, seed: int = 0) -> Instance:
    """Build a reproducible synthetic instance with the given N (tasks)
    and W (nodes).  All knobs are deterministic given the seed.
    """
    rng = np.random.default_rng(seed)
    layer, cabin = _node_layout(W, rng)

    # Node capacities / properties from the per-layer template
    nodes_cpu  = np.array([LAYER_TEMPLATE[l]["cpu"]  for l in layer], dtype=float)
    nodes_gpu  = np.array([LAYER_TEMPLATE[l]["gpu"]  for l in layer], dtype=float)
    nodes_mem  = np.array([LAYER_TEMPLATE[l]["mem"]  for l in layer], dtype=float)
    nodes_bw   = np.array([LAYER_TEMPLATE[l]["bw"]   for l in layer], dtype=float)
    nodes_v    = np.array([LAYER_TEMPLATE[l]["v"]    for l in layer], dtype=float)
    nodes_prot = np.array([LAYER_TEMPLATE[l]["prot"] for l in layer], dtype=float)

    # Tasks
    tasks_w    = rng.uniform(50.0, 500.0, size=N)              # seconds
    tasks_cpu  = rng.uniform(0.5, 8.0,  size=N)
    tasks_gpu  = rng.choice([0.0, 0.5, 1.0, 1.5, 2.0], size=N)  # GPU requirement
    tasks_mem  = rng.uniform(1.0, 32.0, size=N)
    tasks_data = 10 ** rng.uniform(0.0, 3.0, size=N)            # 1 MB - 1 GB
    sec_draw   = rng.random(size=N)
    tasks_sec  = np.where(sec_draw < 0.40, 1,
                  np.where(sec_draw < 0.80, 2, 3))
    # Source cabin: among actual cabins (>= 0)
    n_cab = int(cabin.max()) + 1 if (cabin >= 0).any() else 0
    tasks_cabin = rng.integers(0, max(1, n_cab), size=N)

    # Per-node predicted load (Sec. 4.2): l_j ~ N(0.3, 0.15^2) clip [0.05, 0.95]
    lpred = rng.normal(0.3, 0.15, size=W)
    lpred = np.clip(lpred, 0.05, 0.95)
    # Per-task "workload factor" used by the load term; small extra noise
    # so tasks with the same w are not identical.
    tpred = rng.normal(0.0, 0.05, size=N)

    return Instance(
        N=N, W=W,
        nodes_layer=layer, nodes_cabin=cabin,
        nodes_cpu=nodes_cpu, nodes_gpu=nodes_gpu, nodes_mem=nodes_mem,
        nodes_bw=nodes_bw, nodes_v=nodes_v, nodes_prot=nodes_prot,
        tasks_w=tasks_w, tasks_cpu=tasks_cpu, tasks_gpu=tasks_gpu,
        tasks_mem=tasks_mem, tasks_data=tasks_data, tasks_sec=tasks_sec,
        tasks_cabin=tasks_cabin,
        tasks_pred=tpred, node_pred=lpred,
    )


# ----------------------------------------------------------------------
# Feasibility & helpers
# ----------------------------------------------------------------------
def feasibility(inst: Instance) -> np.ndarray:
    """Return the (N, W) boolean feasibility matrix."""
    return ((inst.tasks_cpu[:, None] <= inst.nodes_cpu[None, :]) &
            (inst.tasks_gpu[:, None] <= inst.nodes_gpu[None, :]) &
            (inst.tasks_mem[:, None] <= inst.nodes_mem[None, :]))


def sigma(pi: np.ndarray) -> np.ndarray:
    """Security weight sigma(pi) in {1, 2, 3} for low / med / high."""
    return pi.astype(float)


def delta_layer(s_cabin: np.ndarray, node_cabin: np.ndarray,
                node_layer: np.ndarray) -> np.ndarray:
    """Layer-distance delta(s_i, j) used by the cabin potential.

    Returns an (N, W) array with values in {0, 1, 2}:
      0 = same cabin
      1 = same layer, different cabin
      2 = different layer
    """
    N = s_cabin.shape[0]
    W = node_cabin.shape[0]
    same_cabin = (s_cabin[:, None] == node_cabin[None, :])

    # We approximate the source layer as the *layer of the source cabin's
    # first region node* (i.e. the lower-indexed node of that cabin).
    src_layer = np.zeros(N, dtype=int)
    for c in np.unique(s_cabin):
        idxs = np.where(node_cabin == c)[0]
        if len(idxs) == 0:
            src_layer[:] = LAYER_EDGE
        else:
            src_layer[s_cabin == c] = node_layer[idxs[0]]

    delta = np.zeros((N, W), dtype=int)
    for j in range(W):
        if node_layer[j] == LAYER_CLOUD:
            # cloud node: source layer never cloud (we generated tasks only
            # with source cabin in [0, n_cab-1] which are non-cloud)
            delta[:, j] = 2
        else:
            same = (src_layer == node_layer[j])
            delta[:, j] = np.where(same_cabin[:, j], 0,
                            np.where(same, 1, 2))
    return delta


def cabin_repr_layer(inst: Instance, cabin_id: int) -> int:
    """Pick the layer of the *region* node of the given cabin."""
    if cabin_id < 0:
        return LAYER_CLOUD
    idxs = np.where(inst.nodes_cabin == cabin_id)[0]
    if len(idxs) == 0:
        return LAYER_EDGE
    layers = inst.nodes_layer[idxs]
    if (layers == LAYER_REGION).any():
        return LAYER_REGION
    return int(layers[0])
