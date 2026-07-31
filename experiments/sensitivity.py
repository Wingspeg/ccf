"""
sensitivity -- SOR convergence, omega sensitivity, solvetime / components.

Outputs:
  results/convergence.csv        : Phi_total vs iteration for all variants
  results/omega_sensitivity.csv  : Phi_total & iter for omega in [0.5, 1.8]
  results/figures/convergence.png
  results/figures/omega_sensitivity.png
  results/figures/solvetime.png
  results/figures/ablation_components.png

Usage:
    python -m experiments.sensitivity
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import ccf

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "results"
FIG  = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
CONV_SCALE   = (20, 200)        # smaller scale -> more visible SOR relaxation curve
CONV_SEEDS   = list(range(3))
OMEGAS       = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 1.8]
OMEGA_SEEDS  = list(range(3))
OMEGA_SCALE  = (50, 500)        # size used for the omega-sensitivity figure


# ----------------------------------------------------------------------
# Convergence
# ----------------------------------------------------------------------
def run_convergence() -> list[dict]:
    """Return per-iteration Phi_total for Ours / NoRep / NoPred / NoCabin / NoSec / FlatCCF.

    Uses ``random_init=True`` so the SOR relaxation has work to do and
    the convergence curve is visible in the figure.
    """
    W, N = CONV_SCALE
    rows: list[dict] = []
    for variant in ["Ours", "NoRep", "NoPred", "NoCabin", "NoSec", "FlatCCF", "GreedyByLoad", "FirstFit"]:
        histories: list[list[float]] = []
        for seed in CONV_SEEDS:
            inst = ccf.make_instance(N=N, W=W, seed=seed)
            cfg = ccf.CCFConfig(variant=variant, omega=0.7, T_max=200,
                                 random_init=True, init_seed=seed * 1000 + 7)
            res = ccf.solve(inst, cfg, record=True)
            histories.append(res["history"])
        # pad to the same length
        L = max(len(h) for h in histories)
        for h in histories:
            h += [h[-1]] * (L - len(h))
        mean = np.mean(histories, axis=0)
        std  = np.std(histories, axis=0)
        for i, (m, s) in enumerate(zip(mean, std)):
            rows.append({"variant": variant, "iter": i, "phi_mean": float(m), "phi_std": float(s)})
    out = OUT / "convergence.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "iter", "phi_mean", "phi_std"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out}")
    return rows


# ----------------------------------------------------------------------
# Omega sensitivity
# ----------------------------------------------------------------------
def run_omega_sensitivity() -> list[dict]:
    """Return Phi_total and iteration count for omega in OMEGAS."""
    W, N = OMEGA_SCALE
    rows: list[dict] = []
    for omega in OMEGAS:
        phis: list[float] = []
        iters: list[int] = []
        for seed in OMEGA_SEEDS:
            inst = ccf.make_instance(N=N, W=W, seed=seed)
            cfg = ccf.CCFConfig(variant="Ours", omega=omega, T_max=200)
            res = ccf.solve(inst, cfg, record=True)
            phis.append(res["phi_total"])
            iters.append(len(res["history"]))
        rows.append({
            "omega": omega,
            "phi_mean": float(np.mean(phis)),
            "phi_std":  float(np.std(phis)),
            "iter_mean": float(np.mean(iters)),
            "iter_std":  float(np.std(iters)),
        })
        print(f"  omega={omega:.2f}  Phi={np.mean(phis):.2f}  iters={np.mean(iters):.1f}")
    out = OUT / "omega_sensitivity.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["omega", "phi_mean", "phi_std", "iter_mean", "iter_std"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out}")
    return rows


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
def fig_convergence(rows: list[dict]) -> None:
    by_var: dict[str, tuple[list, list, list]] = {}
    for r in rows:
        by_var.setdefault(r["variant"], ([], [], []))
        by_var[r["variant"]][0].append(r["iter"])
        by_var[r["variant"]][1].append(r["phi_mean"])
        by_var[r["variant"]][2].append(r["phi_std"])
    fig, ax = plt.subplots(figsize=(6, 4.2))
    colors = {
        "Ours":     "tab:red",
        "NoRep":    "tab:gray",
        "NoPred":   "tab:blue",
        "NoCabin":  "tab:orange",
        "NoSec":    "tab:green",
        "FlatCCF": "tab:purple",
        "GreedyByLoad": "tab:brown",
        "FirstFit": "tab:pink",
    }
    markers = {
        "Ours": "o", "NoRep": "s", "NoPred": "^",
        "NoCabin": "D", "NoSec": "v", "FlatCCF": "*",
        "GreedyByLoad": "P", "FirstFit": "X",
    }
    for v, (it, m, s) in by_var.items():
        it = np.array(it); m = np.array(m); s = np.array(s)
        ax.plot(it, m, label=v, color=colors[v], marker=markers[v],
                markevery=max(1, len(it)//12), markersize=4, linewidth=1.2)
        ax.fill_between(it, m - s, m + s, color=colors[v], alpha=0.15)
    ax.set_xlabel("SOR iteration")
    ax.set_ylabel(r"$\Phi_{\mathrm{total}}$ (normalised)")
    ax.set_xlim(0, max(max(it) for it, _, _ in by_var.values()))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=4, fontsize=9, frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "convergence.png", dpi=200, bbox_inches="tight")
    print(f"Wrote {FIG / 'convergence.png'}")
    plt.close(fig)


def fig_omega(rows: list[dict]) -> None:
    om  = np.array([r["omega"]  for r in rows])
    phi = np.array([r["phi_mean"] for r in rows])
    phis = np.array([r["phi_std"]  for r in rows])
    itr = np.array([r["iter_mean"] for r in rows])
    itrs = np.array([r["iter_std"]  for r in rows])
    fig, ax1 = plt.subplots(figsize=(6, 3.4))
    color_phi = "tab:red"
    ax1.plot(om, phi, "o-", color=color_phi, label=r"$\Phi_{\mathrm{total}}$", markersize=4)
    ax1.fill_between(om, phi - phis, phi + phis, color=color_phi, alpha=0.15)
    ax1.set_xlabel(r"relaxation factor $\omega$")
    ax1.set_ylabel(r"$\Phi_{\mathrm{total}}$", color=color_phi)
    ax1.tick_params(axis="y", labelcolor=color_phi)
    ax1.axvline(1.0, color="gray", linestyle=":", alpha=0.5)
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    color_it = "tab:blue"
    ax2.plot(om, itr, "s--", color=color_it, label="iterations", markersize=4)
    ax2.fill_between(om, itr - itrs, itr + itrs, color=color_it, alpha=0.15)
    ax2.set_ylabel("SOR iterations", color=color_it)
    ax2.tick_params(axis="y", labelcolor=color_it)
    fig.tight_layout()
    fig.savefig(FIG / "omega_sensitivity.png", dpi=200)
    print(f"Wrote {FIG / 'omega_sensitivity.png'}")
    plt.close(fig)


def fig_solvetime(scales_csv: Path) -> None:
    rows = list(csv.DictReader(open(scales_csv)))
    sizes = sorted({(int(r["W"]), int(r["N"])) for r in rows},
                   key=lambda x: x[0] * x[1])
    fig, ax = plt.subplots(figsize=(6, 4.2))
    markers = {"Ours": "o", "NoRep": "s", "NoPred": "^",
               "NoCabin": "D", "NoSec": "v", "FlatCCF": "*",
               "GreedyByLoad": "P", "FirstFit": "X", "Random": "x"}
    for variant in ["Ours", "NoPred", "NoCabin", "FlatCCF", "GreedyByLoad", "FirstFit", "Random"]:
        ms = [np.mean([float(r["solve_ms_mean"]) for r in rows
                       if r["variant"] == variant
                       and int(r["W"]) == W and int(r["N"]) == N])
              for W, N in sizes]
        size_labels = [f"{W}×{N}" for W, N in sizes]
        ax.plot(size_labels, ms, marker=markers.get(variant, "o"),
                label=variant, linewidth=1.2, markersize=5)
    ax.set_yscale("log")
    ax.set_xlabel("instance scale  (W × N)")
    ax.set_ylabel("solve time  (ms, log)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=4, fontsize=9, frameon=False)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIG / "solvetime.png", dpi=200, bbox_inches="tight")
    print(f"Wrote {FIG / 'solvetime.png'}")
    plt.close(fig)


def fig_components(scales_csv: Path) -> None:
    """Stacked bar chart of the four potential components at the 50x500 scale."""
    rows = list(csv.DictReader(open(scales_csv)))
    target = [(50, 500)]
    variants = ["Ours", "NoRep", "NoPred", "NoCabin", "NoSec",
                "FlatCCF", "GreedyByLoad", "FirstFit", "Random"]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    x = np.arange(len(variants))
    width = 0.6
    bottoms = np.zeros(len(variants))
    for term, color in [("phi_rep", "tab:blue"),
                        ("phi_load", "tab:orange"),
                        ("phi_cab", "tab:green"),
                        ("phi_sec", "tab:red")]:
        vals = []
        for v in variants:
            rs = [r for r in rows if r["variant"] == v
                  and (int(r["W"]), int(r["N"])) in target]
            vals.append(np.mean([float(r[f"{term}_mean"]) for r in rs]))
        ax.bar(x, vals, width, bottom=bottoms, label=term.replace("phi_", r"$\phi^{"),
               color=color, edgecolor="white", linewidth=0.4)
        bottoms += np.array(vals)
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=20)
    ax.set_ylabel(r"$\Phi_{\mathrm{total}}$  (component stack)")
    handles, _ = ax.get_legend_handles_labels()
    new_labels = [r"$\phi^{\mathrm{rep}}$", r"$\phi^{\mathrm{load}}$",
                  r"$\phi^{\mathrm{cab}}$", r"$\phi^{\mathrm{sec}}$"]
    ax.legend(handles, new_labels,
              loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=4, fontsize=9, frameon=False)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "ablation_components.png", dpi=200, bbox_inches="tight")
    print(f"Wrote {FIG / 'ablation_components.png'}")
    plt.close(fig)


def main() -> None:
    print("--- convergence ---")
    conv = run_convergence()
    fig_convergence(conv)
    print("--- omega sensitivity ---")
    omega = run_omega_sensitivity()
    fig_omega(omega)
    print("--- solvetime / components ---")
    fig_solvetime(OUT / "scales.csv")
    fig_components(OUT / "scales.csv")


if __name__ == "__main__":
    main()
