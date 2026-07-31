"""
ccf.baselines -- External and ablation baselines for CCF.

Four baselines, all in this module:

  random_assign     Uniformly random feasible assignment (lower bound).
  greedy_by_load    LPT-style: each task goes to the currently least-loaded
                    feasible node (ignores the four-route potential).
  first_fit         Greedy by index: each task goes to the smallest-index
                    feasible node that can host it.
  flat_ccfsched     Flat version of CCF (no inter-layer quota); uses the
                    full potential but skips Stage 1.

These are not ablations of the four routes -- they are independent
heuristics that do not use the four-component potential (except
flat_ccfsched, which uses it without the cabin structure).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .types import CCFConfig, Instance
from .instances import feasibility
from .potential import build_potential


# ----------------------------------------------------------------------
# Random baseline
# ----------------------------------------------------------------------
def random_assign(inst: Instance, seed: int = 0) -> Tuple[np.ndarray, float]:
    """Uniformly random feasible assignment, used as a lower-bound baseline."""
    rng = np.random.default_rng(seed + 7919)
    N, W = inst.N, inst.W
    feas = feasibility(inst)
    assign = np.full(N, -1, dtype=int)
    for i in range(N):
        cand = np.where(feas[i])[0]
        if len(cand) == 0:
            assign[i] = 0   # last-resort (will be infeasible)
        else:
            assign[i] = rng.choice(cand)
    return assign, float("nan")


# ----------------------------------------------------------------------
# Greedy-by-Load baseline
# ----------------------------------------------------------------------
def greedy_by_load(inst: Instance, cfg: CCFConfig,
                   record: bool = False
                   ) -> Tuple[np.ndarray, float, List[float]]:
    """Each task goes to the currently least-loaded feasible node.

    Tasks are processed in descending workload order (LPT-style ordering
    to mimic a typical heuristic). The "load" used for ranking is the
    cumulative workload seconds already placed on the node; this ignores
    the four-component potential on purpose -- it is an external baseline
    with no knowledge of the four routes.
    """
    Phi = build_potential(inst, cfg)
    feas = feasibility(inst)
    Phi_g = np.where(feas, Phi, np.inf)
    order = np.argsort(-inst.tasks_w)  # largest workload first
    assign = np.zeros(inst.N, dtype=int)
    node_load = np.zeros(inst.W)        # cumulative workload seconds
    for i in order:
        feas_row = np.where(np.isfinite(Phi_g[i]))[0]
        if len(feas_row) == 0:
            assign[i] = 0
            continue
        # pick the feasible node with the smallest current load (tie-break by index)
        j = int(feas_row[np.argmin(node_load[feas_row])])
        assign[i] = j
        node_load[j] += float(inst.tasks_w[i])
    phi_total = float(Phi_g[np.arange(inst.N), assign].sum())
    if np.isinf(phi_total):
        # any infeasible row -> evaluate on Ours potential as 0
        Phi_ours = np.where(feas, build_potential(inst, CCFConfig(variant="Ours")), 0.0)
        phi_total = float(Phi_ours[np.arange(inst.N), assign].sum())
    history = [phi_total]
    return assign, phi_total, history


# ----------------------------------------------------------------------
# First-Fit baseline
# ----------------------------------------------------------------------
def first_fit(inst: Instance, cfg: CCFConfig,
              record: bool = False
              ) -> Tuple[np.ndarray, float, List[float]]:
    """First node (by index) that has enough residual capacity.

    Tasks are processed in their natural index order; each task is placed
    at the smallest-index feasible node whose remaining CPU/GPU/MEM can
    accommodate the task. An external baseline with no knowledge of the
    four routes.
    """
    Phi = build_potential(inst, cfg)
    feas = feasibility(inst)
    Phi_g = np.where(feas, Phi, np.inf)
    assign = np.zeros(inst.N, dtype=int)
    used_cpu = np.zeros(inst.W)
    used_gpu = np.zeros(inst.W)
    used_mem = np.zeros(inst.W)
    for i in range(inst.N):
        placed = False
        for j in range(inst.W):
            if not np.isfinite(Phi_g[i, j]):
                continue
            if (used_cpu[j] + inst.tasks_cpu[i] <= inst.nodes_cpu[j] + 1e-6 and
                used_gpu[j] + inst.tasks_gpu[i] <= inst.nodes_gpu[j] + 1e-6 and
                used_mem[j] + inst.tasks_mem[i] <= inst.nodes_mem[j] + 1e-6):
                assign[i] = j
                used_cpu[j] += float(inst.tasks_cpu[i])
                used_gpu[j] += float(inst.tasks_gpu[i])
                used_mem[j] += float(inst.tasks_mem[i])
                placed = True
                break
        if not placed:
            # fallback: any feasible row (ignoring capacity), 0 otherwise
            feas_row = np.where(np.isfinite(Phi_g[i]))[0]
            assign[i] = int(feas_row[0]) if len(feas_row) else 0
    phi_total = float(Phi_g[np.arange(inst.N), assign].sum())
    if np.isinf(phi_total):
        Phi_ours = np.where(feas, build_potential(inst, CCFConfig(variant="Ours")), 0.0)
        phi_total = float(Phi_ours[np.arange(inst.N), assign].sum())
    history = [phi_total]
    return assign, phi_total, history
