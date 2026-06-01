"""Task #7 — Sensitivity sweep méta-paramètres Ising.

Test robustesse argmax Ξ_tc à Tc en fonction de :
  (a) ε : largeur edge mask
  (b) subset_size : taille des subsets pour O-info / TC

block_size fixé à 2 (contrainte : 2^N bins joint entropy).
Variation de subset_size via random-subset sur grille brute (subset_size ∈ {2,3,4,6}).

Si argmax reste à Tc sous variations larges → observable robuste.
Si décale ou disparaît → sensibilité méta-param → cadre fragile.
"""
import sys
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from systems.ising2d import simulate
from xi.edge import susceptibility, edge_mask
from xi.interaction import o_information, total_correlation
from xi.hierarchy import rg_blocks


def random_subset_tc_omega(snapshots, subset_size=4, n_subsets=400, rng=None):
    """Tire n_subsets sous-ensembles aléatoires de subset_size spins sur la grille.
    Convertit en {0,1}, calcule TC et Ω moyens."""
    if rng is None:
        rng = np.random.default_rng(0)
    L = snapshots[0].shape[0]
    N_spins = L * L
    flat_snaps = np.stack([s.flatten() for s in snapshots], axis=0)  # (T_snap, N_spins)
    flat_snaps01 = ((flat_snaps + 1) // 2).astype(np.int8)
    tcs, oms = [], []
    for _ in range(n_subsets):
        idx = rng.choice(N_spins, size=subset_size, replace=False)
        sub = flat_snaps01[:, idx]
        tcs.append(total_correlation(sub))
        oms.append(o_information(sub))
    return np.mean(tcs), np.mean(oms), np.std(oms)


def run(
    L=32,
    T_values=None,
    n_equil=1200,
    n_sample=250,
    sample_gap=4,
    epsilons=(0.05, 0.10, 0.20, 0.30, 0.50, 0.70),
    subset_sizes=(2, 3, 4, 6),
    n_subsets=400,
    seed=42,
    outdir=None,
):
    if outdir is None:
        outdir = os.path.dirname(os.path.abspath(__file__))
    if T_values is None:
        T_values = np.linspace(1.8, 3.0, 19)
    Tc = 2.0 / np.log(1.0 + np.sqrt(2.0))
    n_T = len(T_values)

    print(f"L={L} | {n_T} T values | Tc={Tc:.4f}")
    print(f"ε sweep: {epsilons}")
    print(f"subset_size sweep: {subset_sizes}\n")

    # Phase 1 : simuler tous les T une seule fois, partager entre méta-params
    snaps_per_T = {}
    chi_T = np.zeros(n_T)
    print("Simuler tous T...")
    t0 = time.time()
    for i, T in enumerate(T_values):
        snaps = simulate(L=L, T=float(T),
                         n_equil=n_equil, n_sample=n_sample,
                         sample_gap=sample_gap, seed=seed)
        snaps_per_T[i] = snaps
        mags = np.array([s.mean() for s in snaps])
        chi_T[i] = susceptibility(mags, T, L * L)
        if i % 5 == 0:
            print(f"  T={T:.3f} χ={chi_T[i]:.2f}")
    print(f"sim {time.time()-t0:.1f}s")

    # Phase 2 : pour chaque (ε, subset_size), calculer Ξ et noter argmax
    print("\nSensitivity (ε, subset_size)...")
    results = {}
    rng = np.random.default_rng(seed)

    # Pour chaque subset_size : calculer TC(T), Ω(T) une fois (indep de ε)
    tc_by_subsize = {}
    om_by_subsize = {}
    omstd_by_subsize = {}
    for ss in subset_sizes:
        tc_T = np.zeros(n_T)
        om_T = np.zeros(n_T)
        omstd_T = np.zeros(n_T)
        for i in range(n_T):
            tc, om, omstd = random_subset_tc_omega(
                snaps_per_T[i], subset_size=ss, n_subsets=n_subsets, rng=rng,
            )
            tc_T[i] = tc
            om_T[i] = om
            omstd_T[i] = omstd
        tc_by_subsize[ss] = tc_T
        om_by_subsize[ss] = om_T
        omstd_by_subsize[ss] = omstd_T
        peak_T = T_values[np.argmax(tc_T)]
        print(f"  subset_size={ss}: peak TC T={peak_T:.3f} (Δ={peak_T-Tc:+.3f})")

    # Phase 3 : pour chaque ε, build masks et compute Ξ_tc(T) per subset_size
    xi_grid = np.zeros((len(epsilons), len(subset_sizes), n_T))
    argmax_grid = np.zeros((len(epsilons), len(subset_sizes)))
    for ei, eps in enumerate(epsilons):
        mask = edge_mask(chi_T, epsilon=eps)
        for si, ss in enumerate(subset_sizes):
            xi_t = tc_by_subsize[ss] * mask
            xi_grid[ei, si, :] = xi_t
            if mask.any() and xi_t.max() > 0:
                argmax_grid[ei, si] = T_values[np.argmax(xi_t)]
            else:
                argmax_grid[ei, si] = np.nan

    # Print synth table
    print(f"\nargmax(Ξ_tc) [T] grid (lignes=ε, colonnes=subset_size):")
    hdr = "ε / ss"
    print(f"{hdr:>10} | " + " | ".join([f"{ss:>6}" for ss in subset_sizes]))
    print("-" * (12 + 9 * len(subset_sizes)))
    for ei, eps in enumerate(epsilons):
        row = f"{eps:>10.2f} | " + " | ".join([f"{argmax_grid[ei,si]:>6.3f}"
                                                 for si in range(len(subset_sizes))])
        print(row)

    # Robustness metric : std des argmax across grid
    valid = ~np.isnan(argmax_grid)
    if valid.any():
        argmax_mean = np.nanmean(argmax_grid)
        argmax_std = np.nanstd(argmax_grid)
        max_dev = np.nanmax(np.abs(argmax_grid - Tc))
        print(f"\nargmax mean = {argmax_mean:.4f}  std = {argmax_std:.4f}")
        print(f"max |argmax - Tc| = {max_dev:.4f}")
        if argmax_std < 0.05 and max_dev < 0.1:
            verdict = "ROBUSTE"
        elif argmax_std < 0.1:
            verdict = "MODÉRÉMENT ROBUSTE"
        else:
            verdict = "FRAGILE"
        print(f"Verdict : {verdict}")

    # Plot heatmap argmax + lignes Ξ_tc par ε
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    im = ax.imshow(argmax_grid, aspect="auto", cmap="plasma",
                   extent=[subset_sizes[0]-0.5, subset_sizes[-1]+0.5,
                           epsilons[-1]+0.05, epsilons[0]-0.05])
    ax.set_xlabel("subset_size", color="white")
    ax.set_ylabel("ε", color="white")
    ax.set_title(f"argmax(Ξ_tc) — Tc={Tc:.3f} | std={argmax_std:.3f}", color="white")
    cb = fig.colorbar(im, ax=ax)
    cb.ax.tick_params(colors="white")
    # annoter cellules
    for ei, eps in enumerate(epsilons):
        for si, ss in enumerate(subset_sizes):
            val = argmax_grid[ei, si]
            color = "black" if abs(val - Tc) < 0.1 else "white"
            ax.text(ss, eps, f"{val:.2f}", ha="center", va="center",
                    color=color, fontsize=8)

    ax = axes[0, 1]
    cmap_s = plt.get_cmap("viridis", len(subset_sizes))
    for si, ss in enumerate(subset_sizes):
        ax.plot(T_values, tc_by_subsize[ss], "o-", color=cmap_s(si),
                label=f"ss={ss}", lw=1.4, ms=4)
    ax.axvline(Tc, color="#ff3355", ls="--", lw=1, label=f"Tc={Tc:.3f}")
    ax.set_xlabel("T", color="white")
    ax.set_ylabel("TC (random subsets)", color="white")
    ax.legend(facecolor="black", labelcolor="white", edgecolor="white", fontsize=7)
    ax.set_title("TC(T) par subset_size", color="white")

    ax = axes[1, 0]
    cmap_e = plt.get_cmap("plasma", len(epsilons))
    for ei, eps in enumerate(epsilons):
        ax.plot(T_values, xi_grid[ei, 2, :], "o-", color=cmap_e(ei),
                label=f"ε={eps:.2f}", lw=1.4, ms=4)
    ax.axvline(Tc, color="#ff3355", ls="--", lw=1)
    ax.set_xlabel("T", color="white")
    ax.set_ylabel(f"Ξ_tc (ss={subset_sizes[2]})", color="white")
    ax.legend(facecolor="black", labelcolor="white", edgecolor="white", fontsize=7)
    ax.set_title(f"Ξ_tc(T) par ε (subset_size={subset_sizes[2]})", color="white")

    ax = axes[1, 1]
    for si, ss in enumerate(subset_sizes):
        ax.plot(T_values, omstd_by_subsize[ss], "o-", color=cmap_s(si),
                label=f"ss={ss}", lw=1.4, ms=4)
    ax.axvline(Tc, color="#ff3355", ls="--", lw=1)
    ax.set_xlabel("T", color="white")
    ax.set_ylabel("std(Ω) susceptibilité info", color="white")
    ax.legend(facecolor="black", labelcolor="white", edgecolor="white", fontsize=7)
    ax.set_title("std(Ω) par subset_size", color="white")

    for ax in axes.flat:
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_color("white")
    fig.patch.set_facecolor("black")
    fig.suptitle(f"Ξ Sensitivity (ε, subset_size) — Ising L={L}", color="white", y=0.995)
    plt.tight_layout()
    png = os.path.join(outdir, "ising_sensitivity.png")
    plt.savefig(png, dpi=130, facecolor="black", bbox_inches="tight")
    print(f"\nsaved {png}")

    out = {
        "L": L, "Tc": Tc, "T_values": T_values.tolist(),
        "epsilons": list(epsilons), "subset_sizes": list(subset_sizes),
        "chi_T": chi_T.tolist(),
        "tc_by_subsize": {str(ss): tc_by_subsize[ss].tolist() for ss in subset_sizes},
        "om_by_subsize": {str(ss): om_by_subsize[ss].tolist() for ss in subset_sizes},
        "omstd_by_subsize": {str(ss): omstd_by_subsize[ss].tolist() for ss in subset_sizes},
        "argmax_grid": argmax_grid.tolist(),
        "xi_grid": xi_grid.tolist(),
        "argmax_mean": float(argmax_mean),
        "argmax_std": float(argmax_std),
        "max_dev_Tc": float(max_dev),
        "verdict": verdict,
    }
    with open(os.path.join(outdir, "ising_sensitivity.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {os.path.join(outdir, 'ising_sensitivity.json')}")
    return out


if __name__ == "__main__":
    run()
