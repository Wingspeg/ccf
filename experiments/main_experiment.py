"""
main_experiment -- 9-scale x 9-variant x 5-seed grid.

Runs the full ablation grid and writes the per-cell numbers to CSV.
This is the entry point for Sec. 5.4 of the paper.

Usage:
    python -m experiments.main_experiment
    # or:
    python experiments/main_experiment.py
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

import ccf

# ----------------------------------------------------------------------
# Grid configuration (matches Sec. 5.4 of the paper)
# ----------------------------------------------------------------------
VARIANTS = ["Ours", "NoRep", "NoPred", "NoCabin", "NoSec",
            "FlatCCF", "GreedyByLoad", "FirstFit", "Random"]
SCALES = [(W, N) for W in (10, 20, 50) for N in (50, 200, 500)]
SEEDS = list(range(5))

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------
def run_grid() -> list[dict]:
    rows: list[dict] = []
    for (W, N) in SCALES:
        for seed in SEEDS:
            inst = ccf.make_instance(N=N, W=W, seed=seed)
            for v in VARIANTS:
                cfg = ccf.CCFConfig(variant=v, omega=0.7, T_max=200)
                res = ccf.solve(inst, cfg, record=False)
                rows.append({
                    "W": W, "N": N, "seed": seed, "variant": v,
                    "phi_total": res["phi_total"],
                    "solve_ms":  res["solve_ms"],
                    "cross_cabin_ratio": res["cross_cabin_ratio"],
                    "cross_layer_ratio": res["cross_layer_ratio"],
                    "sla":        res["sla"],
                    "infeas_rate": res["infeas_rate"],
                    "phi_rep":  res["phi_rep"],
                    "phi_load": res["phi_load"],
                    "phi_cab":  res["phi_cab"],
                    "phi_sec":  res["phi_sec"],
                })
                print(f"W={W:>2} N={N:>3} seed={seed} {v:13s}  "
                      f"Phi={res['phi_total']:10.1f}  "
                      f"rep={res['phi_rep']:7.1f} "
                      f"load={res['phi_load']:7.1f} "
                      f"cab={res['phi_cab']:6.1f} "
                      f"sec={res['phi_sec']:5.1f}  "
                      f"cc={res['cross_cabin_ratio']:.2f}  "
                      f"ms={res['solve_ms']:6.1f}")
    return rows


def write_scales_csv(rows: list[dict]) -> None:
    """Write the 9-scale comparison table aggregated across seeds (mean)."""
    keys_to_agg = ["phi", "ms", "cc", "cl", "sla", "infeas",
                   "phi_rep", "phi_load", "phi_cab", "phi_sec"]
    agg: dict[tuple[int, int, str], dict[str, list[float]]] = {}
    for r in rows:
        key = (r["W"], r["N"], r["variant"])
        slot = agg.setdefault(key, {k: [] for k in keys_to_agg})
        slot["phi"].append(r["phi_total"])
        slot["ms"].append(r["solve_ms"])
        slot["cc"].append(r["cross_cabin_ratio"])
        slot["cl"].append(r["cross_layer_ratio"])
        slot["sla"].append(r["sla"])
        slot["infeas"].append(r["infeas_rate"])
        slot["phi_rep"].append(r["phi_rep"])
        slot["phi_load"].append(r["phi_load"])
        slot["phi_cab"].append(r["phi_cab"])
        slot["phi_sec"].append(r["phi_sec"])

    out = OUT / "scales.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["W", "N", "variant",
                    "phi_mean", "phi_std",
                    "phi_rep_mean", "phi_load_mean", "phi_cab_mean", "phi_sec_mean",
                    "solve_ms_mean", "cc_mean", "cl_mean", "sla_mean", "infeas_mean"])
        for (W, N, v), d in sorted(agg.items()):
            w.writerow([W, N, v,
                        f"{np.mean(d['phi']):.2f}",
                        f"{np.std(d['phi']):.2f}",
                        f"{np.mean(d['phi_rep']):.2f}",
                        f"{np.mean(d['phi_load']):.2f}",
                        f"{np.mean(d['phi_cab']):.2f}",
                        f"{np.mean(d['phi_sec']):.2f}",
                        f"{np.mean(d['ms']):.2f}",
                        f"{np.mean(d['cc']):.3f}",
                        f"{np.mean(d['cl']):.3f}",
                        f"{np.mean(d['sla']):.2f}",
                        f"{np.mean(d['infeas']):.3f}"])
    print(f"Wrote {out}")


def write_raw_csv(rows: list[dict]) -> None:
    out = OUT / "raw.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out}")


def main() -> None:
    t0 = time.perf_counter()
    rows = run_grid()
    print(f"Total time: {time.perf_counter() - t0:.1f}s")
    write_raw_csv(rows)
    write_scales_csv(rows)


if __name__ == "__main__":
    main()
