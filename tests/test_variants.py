"""tests.test_variants -- Check that the nine variants behave as expected."""
from __future__ import annotations

import numpy as np
import pytest

import ccf


VARIANTS = ["Ours", "NoRep", "NoPred", "NoCabin", "NoSec",
            "FlatCCF", "GreedyByLoad", "FirstFit", "Random"]


@pytest.fixture(scope="module")
def inst():
    return ccf.make_instance(N=20, W=8, seed=11)


def test_all_variants_run(inst):
    for v in VARIANTS:
        cfg = ccf.CCFConfig(variant=v, omega=0.7, T_max=20)
        res = ccf.solve(inst, cfg)
        assert res["phi_total"] >= 0
        assert 0.0 <= res["cross_cabin_ratio"] <= 1.0
        assert 0.0 <= res["cross_layer_ratio"] <= 1.0
        assert res["infeas_rate"] >= 0.0


def test_flat_ccf_collapses_locality(inst):
    """FlatCCF should put more tasks cross-cabin / cross-layer than Ours."""
    cfg_ours = ccf.CCFConfig(variant="Ours", omega=0.7, T_max=50)
    cfg_flat = ccf.CCFConfig(variant="FlatCCF", omega=0.7, T_max=50)
    res_ours = ccf.solve(inst, cfg_ours)
    res_flat = ccf.solve(inst, cfg_flat)
    # Flat removes the cabin structure -> higher cross_cabin ratio
    assert res_flat["cross_cabin_ratio"] >= res_ours["cross_cabin_ratio"]


def test_solve_is_deterministic(inst):
    cfg = ccf.CCFConfig(variant="Ours", omega=0.7, T_max=20)
    a = ccf.solve(inst, cfg)
    b = ccf.solve(inst, cfg)
    assert a["phi_total"] == pytest.approx(b["phi_total"], rel=1e-9)
    assert np.array_equal(a["assign"], b["assign"])


def test_solve_returns_all_keys(inst):
    cfg = ccf.CCFConfig(variant="Ours")
    res = ccf.solve(inst, cfg)
    expected_keys = {
        "assign", "phi_total", "solve_ms",
        "cross_cabin_ratio", "cross_layer_ratio", "sla",
        "infeas_rate",
        "phi_rep", "phi_load", "phi_cab", "phi_sec",
        "history",
    }
    assert expected_keys <= set(res.keys())


def test_record_flag_returns_history(inst):
    cfg = ccf.CCFConfig(variant="Ours", omega=0.7, T_max=20, random_init=True, init_seed=42)
    res = ccf.solve(inst, cfg, record=True)
    assert isinstance(res["history"], list)
    assert len(res["history"]) >= 1
    # History is non-increasing for under-relaxation (omega <= 1)
    for prev, cur in zip(res["history"], res["history"][1:]):
        assert cur <= prev + 1e-6
