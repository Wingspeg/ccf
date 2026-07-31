"""tests.test_invariants -- Sanity checks on data and potential."""
from __future__ import annotations

import numpy as np
import pytest

import ccf


def test_make_instance_is_deterministic():
    a = ccf.make_instance(N=10, W=5, seed=42)
    b = ccf.make_instance(N=10, W=5, seed=42)
    assert a.N == b.N == 10
    assert a.W == b.W == 5
    assert np.array_equal(a.tasks_w, b.tasks_w)
    assert np.array_equal(a.nodes_layer, b.nodes_layer)
    assert np.array_equal(a.nodes_cabin, b.nodes_cabin)
    # Different seed -> different instance
    c = ccf.make_instance(N=10, W=5, seed=43)
    assert not np.array_equal(a.tasks_w, c.tasks_w)


def test_instance_layer_layout():
    inst = ccf.make_instance(N=10, W=10, seed=1)
    # Always one cloud node at index 0
    assert inst.nodes_layer[0] == ccf.LAYER_CLOUD
    # Cloud is its own cabin (-1)
    assert inst.nodes_cabin[0] == ccf.CLOUD_CABIN
    # Only one cloud node per instance
    assert (inst.nodes_layer == ccf.LAYER_CLOUD).sum() == 1


def test_feasibility_is_boolean():
    inst = ccf.make_instance(N=8, W=5, seed=7)
    feas = ccf.feasibility(inst)
    assert feas.dtype == bool
    assert feas.shape == (inst.N, inst.W)


def test_build_potential_shape_and_finite():
    inst = ccf.make_instance(N=8, W=5, seed=7)
    cfg = ccf.CCFConfig(variant="Ours")
    Phi = ccf.build_potential(inst, cfg)
    assert Phi.shape == (inst.N, inst.W)
    # No inf / nan -- the package uses 1e18 as the infeasible sentinel
    assert np.all(np.isfinite(Phi))


def test_infeasible_cells_have_high_potential():
    """For an infeasible (i, j) pair, the potential should be much larger
    than for a feasible pair on the same instance."""
    inst = ccf.make_instance(N=8, W=5, seed=7)
    cfg = ccf.CCFConfig(variant="Ours")
    Phi = ccf.build_potential(inst, cfg)
    feas = ccf.feasibility(inst)
    # A clearly infeasible cell (e.g. tiny node 4 vs any task with cpu > 8)
    big_task = np.argmax(inst.tasks_cpu)
    infeas_cells = np.where(~feas[big_task])[0]
    if len(infeas_cells):
        for j in infeas_cells:
            # 1e18 sentinel -- much higher than any feasible entry
            assert Phi[big_task, j] >= 1e17


def test_decompose_potential_sums_to_ours():
    """The four decomposition components should sum to the integrated
    Ours potential (within numerical tolerance)."""
    inst = ccf.make_instance(N=20, W=8, seed=11)
    cfg = ccf.CCFConfig(variant="Ours")
    Phi = ccf.build_potential(inst, cfg)
    feas = ccf.feasibility(inst)
    Phi_safe = np.where(feas, Phi, 0.0)
    # greedy assignment
    assign = np.argmin(np.where(feas, Phi, np.inf), axis=1)
    decomp = ccf.decompose_potential(inst, assign)
    rowsum = decomp["phi_rep"] + decomp["phi_load"] + decomp["phi_cab"] + decomp["phi_sec"]
    expected = float(Phi_safe[np.arange(inst.N), assign].sum())
    assert rowsum == pytest.approx(expected, rel=1e-6)
