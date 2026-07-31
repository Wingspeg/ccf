"""
ccf.potential -- Four-component potential matrix and decomposition.

The integrated CCF potential is

    Phi(i, j) = lambda1 * phi_rep + lambda2 * phi_load + lambda3 * phi_cab + lambda4 * phi_sec

where each term captures one route in the framework:
  * phi_rep  : representation mismatch (knowledge-graph distance, 1 - m_ij)
  * phi_load : behaviour-predicted load (w / v) * (1 + l_j)
  * phi_cab  : cross-cabin / cross-layer data transfer (data / bw) * rho^delta
  * phi_sec  : security weight * (1 - protection)
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from .types import CCFConfig, Instance
from .instances import cabin_repr_layer, delta_layer, feasibility, sigma


# ----------------------------------------------------------------------
# Empirical scale factors
# ----------------------------------------------------------------------
# Chosen so each term's typical magnitude on a 50x500 instance is ~1.0
# when lambda_i = 1.  The rep term is already in [0, 1]; the others have
# been calibrated to the data generator in ccf.instances.
_SCALE = {
    "rep":  1.0,        # 1 - m, m in [0, 1]
    "load": 1000.0,     # w in [50, 500], v in [0.2, 1.0], (1+l) ~ 1.3
    "cab":  50.0,       # d in [1, 1000], b in [20, 200], rho^delta in [1, 2.25]
    "sec":  1.0,        # sigma in {1, 2, 3}, (1-prot) in [0.1, 0.8]
}


# ----------------------------------------------------------------------
# Potential matrix
# ----------------------------------------------------------------------
def build_potential(inst: Instance, cfg: CCFConfig) -> np.ndarray:
    """Return the (N, W) potential matrix Phi for the given variant.

    Variants:
      Ours      : full four-term potential
      NoRep     : phi_rep = 0
      NoPred    : predicted load term dropped (l_j -> 0)
      NoCabin   : rho = 1  (no amplification)
      NoSec     : phi_sec = 0
    """
    N, W = inst.N, inst.W
    Phi = np.zeros((N, W), dtype=float)

    scale_rep  = _SCALE["rep"]  if cfg.normalize else 1.0
    scale_load = _SCALE["load"] if cfg.normalize else 1.0
    scale_cab  = _SCALE["cab"]  if cfg.normalize else 1.0
    scale_sec  = _SCALE["sec"]  if cfg.normalize else 1.0

    feas = feasibility(inst)

    # ----- representation potential (KG matching) -----
    # m_ij = match between (tasks_cpu, tasks_gpu, tasks_mem) and node template.
    # Use a normalised cosine-like score in [0, 1]: 0 means infeasible.
    t_req = np.stack([inst.tasks_cpu, inst.tasks_gpu, inst.tasks_mem], axis=1)  # (N, 3)
    n_cap = np.stack([inst.nodes_cpu,  inst.nodes_gpu,  inst.nodes_mem ], axis=1)  # (W, 3)
    n_cap_safe = np.where(n_cap == 0, 1.0, n_cap)
    ratio = t_req[:, None, :] / n_cap_safe[None, :, :]                  # (N, W, 3)
    match = np.clip(ratio, 0.0, 1.0).mean(axis=2)                        # (N, W) in [0, 1]
    match = np.where(feas, match, 0.0)
    if cfg.variant != "NoRep":
        # lambda1 * (1 - m_ij) / scale; m_ij = 0 -> +inf potential
        phi_rep = cfg.lam[0] * (1.0 - match) / scale_rep
        phi_rep[~feas] = 1e18
    else:
        phi_rep = np.zeros((N, W))
    Phi += phi_rep

    # ----- load potential (behaviour-predicted load) -----
    lj = inst.node_pred if cfg.variant != "NoPred" else np.zeros(W)
    phi_load = (cfg.lam[1] * (inst.tasks_w[:, None] / inst.nodes_v[None, :])
                          * (1.0 + lj[None, :]) / scale_load)
    Phi += phi_load

    # ----- cabin potential (cross-cabin / cross-layer) -----
    if cfg.variant != "NoCabin":
        rho = cfg.rho
        delta = delta_layer(inst.tasks_cabin, inst.nodes_cabin, inst.nodes_layer)  # (N, W) in {0,1,2}
        bw_safe = np.maximum(inst.nodes_bw[None, :], 1e-3)
        phi_cab = (cfg.lam[2] * (inst.tasks_data[:, None] / bw_safe)
                            * (rho ** delta) / scale_cab)
        Phi += phi_cab

    # ----- security potential -----
    if cfg.variant != "NoSec":
        sig = sigma(inst.tasks_sec)[:, None]                               # (N, 1)
        prot_j = inst.nodes_prot[None, :]                                  # (1, W)
        phi_sec = (cfg.lam[3] * sig * (1.0 - prot_j) / scale_sec)
        Phi += phi_sec

    # Mask infeasible (high-potential inf) cells to a finite but huge value
    # so downstream numpy ops stay finite.
    Phi = np.where(np.isfinite(Phi), Phi, 1e18)
    return Phi


# ----------------------------------------------------------------------
# Per-component decomposition
# ----------------------------------------------------------------------
def decompose_potential(inst: Instance, a: np.ndarray) -> Dict[str, float]:
    """Evaluate an assignment and return each of the four potential
    terms separately (rep / load / cab / sec), summed across tasks.
    Infeasible cells contribute 0 for the corresponding term.
    """
    N = inst.N
    feas = feasibility(inst)
    cfg = CCFConfig(variant="Ours")
    scale_rep  = _SCALE["rep"]  if cfg.normalize else 1.0
    scale_load = _SCALE["load"] if cfg.normalize else 1.0
    scale_cab  = _SCALE["cab"]  if cfg.normalize else 1.0
    scale_sec  = _SCALE["sec"]  if cfg.normalize else 1.0
    lj = inst.node_pred
    t_req = np.stack([inst.tasks_cpu, inst.tasks_gpu, inst.tasks_mem], axis=1)
    n_cap = np.stack([inst.nodes_cpu,  inst.nodes_gpu,  inst.nodes_mem ], axis=1)
    n_cap_safe = np.where(n_cap == 0, 1.0, n_cap)
    ratio = t_req[:, None, :] / n_cap_safe[None, :, :]
    match = np.clip(ratio, 0.0, 1.0).mean(axis=2)
    match = np.where(feas, match, 0.0)
    phi_rep = np.where(feas, cfg.lam[0] * (1.0 - match) / scale_rep, 0.0)
    phi_load = np.where(feas,
                        cfg.lam[1] * (inst.tasks_w[:, None] / inst.nodes_v[None, :])
                                 * (1.0 + lj[None, :]) / scale_load,
                        0.0)
    rho = cfg.rho
    delta = delta_layer(inst.tasks_cabin, inst.nodes_cabin, inst.nodes_layer)
    bw_safe = np.maximum(inst.nodes_bw[None, :], 1e-3)
    phi_cab = np.where(feas,
                       cfg.lam[2] * (inst.tasks_data[:, None] / bw_safe)
                                * (rho ** delta) / scale_cab, 0.0)
    sig = sigma(inst.tasks_sec)[:, None]
    phi_sec = np.where(feas,
                       cfg.lam[3] * sig * (1.0 - inst.nodes_prot[None, :]) / scale_sec,
                       0.0)
    rows = np.arange(N)
    return {
        "phi_rep":  float(phi_rep[rows, a].sum()),
        "phi_load": float(phi_load[rows, a].sum()),
        "phi_cab":  float(phi_cab[rows, a].sum()),
        "phi_sec":  float(phi_sec[rows, a].sum()),
    }
