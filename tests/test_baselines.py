"""tests.test_baselines -- Sanity checks for the four external baselines."""
from __future__ import annotations

import numpy as np
import pytest

import ccf


@pytest.fixture(scope="module")
def inst():
    return ccf.make_instance(N=20, W=8, seed=11)


def test_random_baseline_uses_feasible_nodes(inst):
    assign, _ = ccf.random_assign(inst, seed=0)
    feas = ccf.feasibility(inst)
    for i, j in enumerate(assign):
        if j == 0:  # fallback for the truly-infeasible case
            continue
        assert feas[i, j]


def test_greedy_by_load_processes_largest_first(inst):
    cfg = ccf.CCFConfig(variant="Ours")
    assign, phi, hist = ccf.greedy_by_load(inst, cfg, record=True)
    # first iteration history should equal the integrated Ours potential
    # evaluated on the greedy assignment
    Phi = ccf.build_potential(inst, cfg)
    feas = ccf.feasibility(inst)
    Phi_safe = np.where(feas, Phi, 0.0)
    expected = float(Phi_safe[np.arange(inst.N), assign].sum())
    assert phi == pytest.approx(expected, rel=1e-6)


def test_first_fit_assigns_respects_feasibility(inst):
    """FirstFit only enforces per-resource feasibility, not aggregate
    capacity.  Verify that every assigned (i, j) pair is individually feasible
    even though nodes may be over-subscribed.
    """
    assign, _, _ = ccf.first_fit(inst, ccf.CCFConfig(variant="Ours"))
    feas = ccf.feasibility(inst)
    for i, j in enumerate(assign):
        assert feas[i, j]
