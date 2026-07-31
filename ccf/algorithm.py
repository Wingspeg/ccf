"""
ccf.algorithm -- Stage 1 (inter-layer) and Stage 2 (intra-cabin SOR)
plus the high-level solve() entry point and the cross-metric / SLA
helpers.
"""
from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np

from .types import CCFConfig, Instance, LAYER_CLOUD, LAYER_EDGE, LAYER_REGION
from .instances import (
    cabin_repr_layer,
    feasibility,
)
from .potential import build_potential, decompose_potential


# ----------------------------------------------------------------------
# Stage 1 : inter-layer coarse allocation
# ----------------------------------------------------------------------
def stage1_assign(inst: Instance, Phi: np.ndarray) -> np.ndarray:
    """Return (N,) target-layer assignment l*[i] in {0, 1, 2}.

    For each task compute Psi[i, l] = min over feasible (i, j) with L(j)=l.
    The per-layer share of tasks is proportional to its total CPU capacity.
    Tasks are pushed to the layer with smallest Psi while respecting the
    quota, and to a feasible layer (skip layers where Psi is inf).
    """
    N, W = inst.N, inst.W
    layers = inst.nodes_layer
    feas = feasibility(inst)
    Psi = np.full((N, 3), np.inf)
    for l in (LAYER_CLOUD, LAYER_REGION, LAYER_EDGE):
        layer_mask = (layers == l)
        if not layer_mask.any():
            continue
        # feasible pairs in this layer
        for i in range(N):
            cand = np.where(layer_mask & feas[i])[0]
            if len(cand):
                Psi[i, l] = Phi[i, cand].min()
    # Capacity share per layer (CPU), only over layers that can host at
    # least one task
    cap = np.array([inst.nodes_cpu[layers == l].sum() for l in (0, 1, 2)])
    cap = np.maximum(cap, 1.0)
    share = cap / cap.sum()
    target_n = np.floor(share * N).astype(int)
    target_n[LAYER_CLOUD] += N - target_n.sum()
    # Sort tasks by their best (finite) layer index (lowest Psi first)
    best_finite = np.where(np.isfinite(Psi.min(axis=1)),
                           Psi.min(axis=1),
                           np.inf)
    order = np.argsort(best_finite)
    assign = np.full(N, -1, dtype=int)
    for i in order:
        for cand in np.argsort(Psi[i]):
            if not np.isfinite(Psi[i, cand]):
                break
            if target_n[cand] > 0:
                assign[i] = cand
                target_n[cand] -= 1
                break
        if assign[i] == -1:
            # last resort: any feasible layer, prefer region then edge then cloud
            for cand in [LAYER_REGION, LAYER_EDGE, LAYER_CLOUD]:
                layer_mask = (layers == cand)
                if layer_mask.any() and (feas[i] & layer_mask).any():
                    assign[i] = cand
                    break
            if assign[i] == -1:
                # truly infeasible: put on cloud, SOR will deal with it
                assign[i] = LAYER_CLOUD
    return assign


# ----------------------------------------------------------------------
# Stage 2 helpers
# ----------------------------------------------------------------------
def _neighbours(inst: Instance, j: int, radius: int) -> np.ndarray:
    """Return node indices that task-migration can consider from node j.

    radius=1 : same cabin (strict intra-cabin relaxation)
    radius=2 : same cabin OR same layer (intra-cabin preferred, same-layer
               fallback when the cabin is saturated)
    radius=3 : any node (no relaxation, debug only)

    Cloud nodes are always mapped to their own layer because they live in
    a degenerate "cabin" of size 1 -- otherwise no migration is possible.
    """
    cab = inst.nodes_cabin[j]
    layer = inst.nodes_layer[j]
    if cab < 0:
        return np.where(inst.nodes_layer == layer)[0]
    if radius <= 1:
        return np.where(inst.nodes_cabin == cab)[0]
    if radius == 2:
        mask = (inst.nodes_cabin == cab) | (inst.nodes_layer == layer)
        return np.where(mask)[0]
    return np.arange(inst.W)


def _per_node_used(inst: Instance, assign: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (used_cpu, used_gpu, used_mem) per node for the current assignment."""
    N = inst.N
    W = inst.W
    used_cpu = np.zeros(W); used_gpu = np.zeros(W); used_mem = np.zeros(W)
    if N == 0:
        return used_cpu, used_gpu, used_mem
    # vectorised scatter-add
    np.add.at(used_cpu, assign, inst.tasks_cpu)
    np.add.at(used_gpu, assign, inst.tasks_gpu)
    np.add.at(used_mem, assign, inst.tasks_mem)
    return used_cpu, used_gpu, used_mem


def _is_overloaded(inst: Instance, used_cpu, used_gpu, used_mem, tol: float = 1e-6) -> np.ndarray:
    return (used_cpu - inst.nodes_cpu > tol) | \
           (used_gpu - inst.nodes_gpu > tol) | \
           (used_mem - inst.nodes_mem > tol)


def _try_place(used_cpu, used_gpu, used_mem, j, inst, i):
    """Return True if task i can be placed on node j given current usage."""
    return (used_cpu[j] + inst.tasks_cpu[i] <= inst.nodes_cpu[j] + 1e-6 and
            used_gpu[j] + inst.tasks_gpu[i] <= inst.nodes_gpu[j] + 1e-6 and
            used_mem[j] + inst.tasks_mem[i] <= inst.nodes_mem[j] + 1e-6)


# ----------------------------------------------------------------------
# Stage 2 : intra-cabin SOR relaxation
# ----------------------------------------------------------------------
def stage2_relax(inst: Instance, Phi: np.ndarray, target_layer: np.ndarray,
                 cfg: CCFConfig, record: bool = False
                 ) -> Tuple[np.ndarray, float, List[float]]:
    """Two-stage SOR relaxation.  Returns (assign, phi_total, history).

    If ``record`` is True, ``history`` holds Phi_total after each iteration
    (length = iterations+1 including the initial greedy solution).
    """
    N, W = inst.N, inst.W
    feas = feasibility(inst)
    # restrict the potential to (tasks -> nodes) pairs that respect target_layer
    mask = np.zeros((N, W), dtype=bool)
    for l in (LAYER_CLOUD, LAYER_REGION, LAYER_EDGE):
        layer_mask = (inst.nodes_layer == l)
        task_mask  = (target_layer == l)
        if task_mask.any() and layer_mask.any():
            mask[np.ix_(task_mask, layer_mask)] = True
    # Allowed (i, j) pairs: target_layer respects AND feasible
    allowed = mask & feas
    Phi_r = np.where(allowed, Phi, np.inf)
    # --- initial assignment ---
    if cfg.random_init:
        # uniformly random over ALL feasible (i, j) pairs (ignoring the
        # target_layer mask).  This produces a deliberately-bad starting
        # point so the SOR relaxation has visible work to do, which is
        # what we want for the convergence-curve figure.
        assign = np.zeros(N, dtype=int)
        rng = np.random.default_rng(cfg.init_seed)
        Phi_global = np.where(feas, Phi, np.inf)
        for i in range(N):
            cand = np.where(np.isfinite(Phi_global[i]))[0]
            if len(cand):
                assign[i] = int(rng.choice(cand))
            else:
                feas_row = np.where(feas[i])[0]
                assign[i] = int(feas_row[0]) if len(feas_row) else 0
    else:
        # greedy: within the allowed (i, j) pairs, choose the
        # min-potential node.  Infeasible cells get +inf so they are
        # never chosen unless absolutely necessary.
        assign = np.zeros(N, dtype=int)
        for i in range(N):
            row = Phi_r[i]
            if np.isfinite(row).any():
                assign[i] = int(np.argmin(row))
            else:
                feas_row = np.where(feas[i])[0]
                assign[i] = int(feas_row[0]) if len(feas_row) else 0

    def total_phi(a: np.ndarray) -> float:
        """Total potential of assignment ``a`` evaluated against the
        *target-layer-restricted* matrix.  Infeasible cells (outside the
        target layer or with infeasible resources) contribute 0 here so
        the relaxation can make progress even when a random initialiser
        picks an off-layer node.  The *true* potential used for the
        reported metric is always recomputed in ``solve()`` via
        ``_eval_full`` on the Ours potential.
        """
        Phi_safe = np.where(feas, Phi, 0.0)
        return float(Phi_safe[np.arange(N), a].sum())

    history: List[float] = []
    phi_total = total_phi(assign)
    if record:
        history.append(phi_total)

    # --- relaxation loop ---
    for it in range(cfg.T_max):
        moved = False
        used_cpu, used_gpu, used_mem = _per_node_used(inst, assign)
        overloaded = _is_overloaded(inst, used_cpu, used_gpu, used_mem)
        if not overloaded.any():
            break
        # process each overloaded node
        for j in np.where(overloaded)[0]:
            # tasks currently on j, ordered by their potential (highest first)
            on_j = np.where(assign == j)[0]
            if len(on_j) == 0:
                continue
            order = on_j[np.argsort(-Phi_r[on_j, j])]   # worst first
            for i in order:
                # if j is no longer overloaded, stop processing j
                if not (used_cpu[j] > inst.nodes_cpu[j] + 1e-6 or
                        used_gpu[j] > inst.nodes_gpu[j] + 1e-6 or
                        used_mem[j] > inst.nodes_mem[j] + 1e-6):
                    break
                # find neighbour that can host task i with minimum potential
                neigh = _neighbours(inst, j, cfg.cabin_radius)
                cand_idx = []
                for j2 in neigh:
                    if j2 == j:
                        continue
                    if _try_place(used_cpu, used_gpu, used_mem, j2, inst, i):
                        cand_idx.append(j2)
                if not cand_idx:
                    continue
                cand_idx = np.array(cand_idx, dtype=int)
                best_j = int(cand_idx[Phi_r[i, cand_idx].argmin()])
                # SOR acceptance: accept the migration iff
                #     Phi[i, best_j] <= omega * Phi[i, j]
                if not (Phi_r[i, best_j] <= cfg.omega * Phi_r[i, j] + 1e-12):
                    continue
                # update assignment and used vectors
                assign[i] = best_j
                used_cpu[j]  -= inst.tasks_cpu[i]
                used_cpu[best_j] += inst.tasks_cpu[i]
                used_gpu[j]  -= inst.tasks_gpu[i]
                used_gpu[best_j] += inst.tasks_gpu[i]
                used_mem[j]  -= inst.tasks_mem[i]
                used_mem[best_j] += inst.tasks_mem[i]
                moved = True
                if record:
                    history.append(total_phi(assign))
        phi_new = total_phi(assign)
        if record and not moved:
            history.append(phi_new)
        if abs(phi_new - phi_total) < cfg.eps:
            phi_total = phi_new
            break
        phi_total = phi_new
        if not moved:
            break
    return assign, phi_total, history


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def _cross_metrics(inst: Instance, assign: np.ndarray) -> Tuple[float, float]:
    """Return (cross_cabin_ratio, cross_layer_ratio) for the assignment.

    cross_cabin  = fraction of tasks placed on a node whose cabin differs
                   from the task's source cabin.
    cross_layer  = fraction of tasks placed on a different layer.
    """
    N = inst.N
    if N == 0:
        return 0.0, 0.0
    src_cabin = inst.tasks_cabin
    dst_cabin = inst.nodes_cabin[assign]
    src_layer = np.array([cabin_repr_layer(inst, c) for c in src_cabin])
    dst_layer = inst.nodes_layer[assign]
    cc = float((src_cabin != dst_cabin).mean())
    cl = float((src_layer != dst_layer).mean())
    return cc, cl


def _sla_penalty(inst: Instance, assign: np.ndarray) -> float:
    """Return SLA penalty = sum_i max(0, ECT_i - deadline_i).

    We define
        ECT_i = w_i / v_{assign[i]} + (data_i / bw_{assign[i]})  (seconds)
        deadline_i = 2 * w_i  (a generous, single scalar deadline).
    """
    if inst.N == 0:
        return 0.0
    v = inst.nodes_v[assign]
    bw = inst.nodes_bw[assign]
    ect = inst.tasks_w / v + inst.tasks_data / bw
    dl  = 2.0 * inst.tasks_w
    return float(np.maximum(0.0, ect - dl).sum())


# ----------------------------------------------------------------------
# High-level solver entry
# ----------------------------------------------------------------------
def solve(inst: Instance, cfg: CCFConfig, record: bool = False
          ) -> Dict[str, object]:
    """Run the configured variant and return a result dict.

    ``phi_total`` is always evaluated under the *full* Ours potential
    (all four terms), so the ablation compares solutions on a common
    scale -- a higher phi_total for an ablated variant means its
    solution is worse on the integrated objective.  The four component
    totals are also reported separately for component-level ablation.
    """
    t0 = time.perf_counter()
    phi_total = float("nan")
    history: List[float] = []
    feas = feasibility(inst)
    Phi_ours = np.where(feas, build_potential(inst, CCFConfig(variant="Ours")), 0.0)

    def _eval_full(a: np.ndarray) -> float:
        return float(Phi_ours[np.arange(inst.N), a].sum())

    if cfg.variant == "FlatCCF":
        # run stage 2 over the *full* potential (no layer restriction)
        Phi = build_potential(inst, cfg)
        N, W = inst.N, inst.W
        feas = feasibility(inst)
        Phi_g = np.where(feas, Phi, np.inf)
        # Manual flat relaxation (skip stage 1)
        assign_g = Phi_g.argmin(axis=1)
        # fallback for infeasible rows
        for i in range(N):
            if not np.isfinite(Phi_g[i, assign_g[i]]):
                feas_row = np.where(feas[i])[0]
                assign_g[i] = int(feas_row[0]) if len(feas_row) else 0
        used_cpu, used_gpu, used_mem = _per_node_used(inst, assign_g)
        overloaded = _is_overloaded(inst, used_cpu, used_gpu, used_mem)
        history = [float(Phi_g[np.arange(N), assign_g].sum())]
        for it in range(cfg.T_max):
            moved = False
            for j in np.where(overloaded)[0]:
                on_j = np.where(assign_g == j)[0]
                if len(on_j) == 0:
                    continue
                for i in on_j[np.argsort(-Phi_g[on_j, j])]:
                    if not (used_cpu[j] > inst.nodes_cpu[j] + 1e-6 or
                            used_gpu[j] > inst.nodes_gpu[j] + 1e-6 or
                            used_mem[j] > inst.nodes_mem[j] + 1e-6):
                        break
                    cand_idx = []
                    for j2 in range(W):
                        if j2 == j:
                            continue
                        if _try_place(used_cpu, used_gpu, used_mem, j2, inst, i):
                            cand_idx.append(j2)
                    if not cand_idx:
                        continue
                    cand_idx = np.array(cand_idx, dtype=int)
                    best_j = int(cand_idx[Phi_g[i, cand_idx].argmin()])
                    if Phi_g[i, best_j] >= Phi_g[i, j]:
                        continue
                    assign_g[i] = best_j
                    used_cpu[j]  -= inst.tasks_cpu[i]; used_cpu[best_j]  += inst.tasks_cpu[i]
                    used_gpu[j]  -= inst.tasks_gpu[i]; used_gpu[best_j]  += inst.tasks_gpu[i]
                    used_mem[j]  -= inst.tasks_mem[i]; used_mem[best_j]  += inst.tasks_mem[i]
                    moved = True
            overloaded = _is_overloaded(inst, used_cpu, used_gpu, used_mem)
            history.append(float(Phi_g[np.arange(N), assign_g].sum()))
            if not moved:
                break
        assign = assign_g
        phi_total = _eval_full(assign)
        if not record:
            history = []
    elif cfg.variant == "GreedyByLoad":
        # Imported lazily to avoid a circular import (baselines imports solve)
        from .baselines import greedy_by_load
        assign, phi_total, history = greedy_by_load(inst, cfg, record=record)
    elif cfg.variant == "FirstFit":
        from .baselines import first_fit
        assign, phi_total, history = first_fit(inst, cfg, record=record)
    elif cfg.variant == "Random":
        from .baselines import random_assign
        assign, _ = random_assign(inst, seed=int(t0 * 1e6) & 0xFFFF)
        phi_total = _eval_full(assign)
    else:
        Phi = build_potential(inst, cfg)
        target = stage1_assign(inst, Phi)
        assign, _, history = stage2_relax(inst, Phi, target, cfg, record=record)
        phi_total = _eval_full(assign)
        if not record:
            history = []
    t1 = time.perf_counter()
    solve_ms = (t1 - t0) * 1000.0

    # ---- compute metrics ----
    cross_cabin, cross_layer = _cross_metrics(inst, assign)
    sla = _sla_penalty(inst, assign)
    decomp = decompose_potential(inst, assign)
    # infeasibility rate: how many tasks were placed on a node that cannot
    # accommodate the task's CPU/GPU/memory requirement.
    feas = feasibility(inst)
    infeas_count = int((~feas[np.arange(inst.N), assign]).sum())
    infeas_rate = infeas_count / max(inst.N, 1)
    return {
        "assign": assign,
        "phi_total": float(phi_total),
        "solve_ms": float(solve_ms),
        "cross_cabin_ratio": cross_cabin,
        "cross_layer_ratio": cross_layer,
        "sla": sla,
        "infeas_rate": infeas_rate,
        "phi_rep":  decomp["phi_rep"],
        "phi_load": decomp["phi_load"],
        "phi_cab":  decomp["phi_cab"],
        "phi_sec":  decomp["phi_sec"],
        "history": history,
    }
