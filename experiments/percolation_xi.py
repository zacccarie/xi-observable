"""Task #13 — Percolation 2D : test universalité χ_info hors classe Ising.

Site percolation L×L, sweep p ∈ [0.4, 0.8] autour p_c=0.5927.
Mesures par valeur de p :
  - P_∞ (taille plus grand cluster / N) = order parameter
  - χ_perco (moments tailles clusters finis) = susceptibilité géométrique
  - Ω et TC sur subsets aléatoires
  - mean(Ω), std(Ω), CV(Ω) sur n_real réalisations indépendantes

Question : CV(Ω) pique-t-il à p_c=0.5927 (classe random cluster), comme à
Tc Ising (classe conforme), K_c RBN (dynamique discrète) et K_c Kuramoto 2D
(synchronisation) ?

Si OUI → χ_info universel transversal aux classes universalité.
Si NON → spécifique aux classes thermodynamiques / dynamiques.
"""
import sys
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from systems.percolation import sample_ensemble, cluster_stats
from xi.interaction import o_information, total_correlation


def random_subset_omega(ensemble, subset_size=4, n_subsets=400, rng=None):
    """ensemble (n_samples, L, L). Tire subsets sur grille aplatie."""
    if rng is None:
        rng = np.random.default_rng(0)
    n_s, L, _ = ensemble.shape
    flat = ensemble.reshape(n_s, L * L)
    tcs, oms = [], []
    for _ in range(n_subsets):
        idx = rng.choice(L * L, size=subset_size, replace=False)
        sub = flat[:, idx].astype(np.int8)
        tcs.append(total_correlation(sub))
        oms.append(o_information(sub))
    return np.mean(tcs), np.mean(oms)


def run(
    L=32,
    p_values=None,
    n_real=12,
    n_samples_per_real=400,
    subset_size=4,
    n_subsets=400,
    seed0=11,
    outdir=None,
):
    if outdir is None:
        outdir = os.path.dirname(os.path.abspath(__file__))
    if p_values is None:
        p_values = np.concatenate([
            np.linspace(0.40, 0.55, 4),
            np.linspace(0.56, 0.62, 13),  # zoom autour p_c
            np.linspace(0.65, 0.80, 4),
        ])
    p_c = 0.5927  # site percolation 2D
    n_p = len(p_values)
    rng_master = np.random.default_rng(seed0)

    print(f"Percolation 2D L={L} | {n_p} p values | p_c={p_c:.4f}")
    print(f"n_real={n_real} | n_samples/real={n_samples_per_real}\n")

    P_inf = np.zeros(n_p)
    chi_perco = np.zeros(n_p)
    om_mean = np.zeros(n_p)
    om_std = np.zeros(n_p)
    tc_mean = np.zeros(n_p)

    t0 = time.time()
    for i, p in enumerate(p_values):
        Pinf_list, chip_list, om_list, tc_list = [], [], [], []
        for r in range(n_real):
            seed = int(rng_master.integers(0, 2**31))
            ens = sample_ensemble(L, p, n_samples_per_real, seed=seed)
            # P_inf et χ moyennés sur snapshots
            Pinfs, chis = [], []
            for snap in ens:
                _, _, Pinf, chi = cluster_stats(snap)
                Pinfs.append(Pinf)
                chis.append(chi)
            Pinf_list.append(np.mean(Pinfs))
            chip_list.append(np.mean(chis))
            tc, om = random_subset_omega(ens, subset_size=subset_size,
                                          n_subsets=n_subsets,
                                          rng=np.random.default_rng(seed + 7))
            om_list.append(om)
            tc_list.append(tc)
        P_inf[i] = np.mean(Pinf_list)
        chi_perco[i] = np.mean(chip_list)
        om_mean[i] = np.mean(om_list)
        om_std[i] = np.std(om_list)
        tc_mean[i] = np.mean(tc_list)
        print(f"  p={p:.4f}  P∞={P_inf[i]:.3f}  χ_perco={chi_perco[i]:.2f}  "
              f"⟨Ω⟩={om_mean[i]:+.4f}  std(Ω)={om_std[i]:.4f}  TC={tc_mean[i]:.4f}")
    print(f"sim {time.time()-t0:.1f}s")

    cv_omega = om_std / (np.abs(om_mean) + 1e-9)

    peak_chi = p_values[np.argmax(chi_perco)]
    peak_tc = p_values[np.argmax(tc_mean)]
    peak_ostd = p_values[np.argmax(om_std)]
    peak_cv = p_values[np.argmax(cv_omega)]

    print()
    print(f"p_c analytique  = {p_c:.4f}")
    print(f"peak χ_perco    = {peak_chi:.4f}  (Δ={peak_chi-p_c:+.4f})")
    print(f"peak TC         = {peak_tc:.4f}  (Δ={peak_tc-p_c:+.4f})")
    print(f"peak std(Ω)     = {peak_ostd:.4f}  (Δ={peak_ostd-p_c:+.4f})")
    print(f"peak CV(Ω)      = {peak_cv:.4f}  (Δ={peak_cv-p_c:+.4f})")

    best = min([
        ("χ_perco", abs(peak_chi - p_c)),
        ("TC", abs(peak_tc - p_c)),
        ("std(Ω)", abs(peak_ostd - p_c)),
        ("CV(Ω)", abs(peak_cv - p_c)),
    ], key=lambda x: x[1])
    print(f"\nMeilleur localisateur p_c : {best[0]} (Δ={best[1]:.4f})")
    universal = abs(peak_cv - p_c) < 0.03
    print(f"χ_info universel sur percolation : "
          f"{'✓ CV(Ω) localise p_c' if universal else '✗ CV(Ω) raté'}")

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    panels = [
        ("P∞ order param", P_inf, "cyan", peak_chi, "χ_perco"),
        ("χ_perco susceptibility", chi_perco, "#ff8800", peak_chi, "χ_perco"),
        ("TC random subsets", tc_mean, "yellow", peak_tc, "TC"),
        ("⟨Ω⟩", om_mean, "#aaff00", None, None),
        ("std(Ω) suscept info", om_std, "magenta", peak_ostd, "std(Ω)"),
        ("CV(Ω) = std/|mean|", cv_omega, "#00ff88", peak_cv, "CV(Ω)"),
    ]
    for ax, (lab, vec, col, pk, pk_lab) in zip(axes.flat, panels):
        ax.plot(p_values, vec, "o-", color=col, lw=1.5, ms=5)
        ax.axvline(p_c, color="#ff3355", ls="--", lw=1, label=f"p_c={p_c:.3f}")
        if pk is not None:
            ax.axvline(pk, color=col, ls=":", lw=1, label=f"peak={pk:.3f}")
        if "⟨Ω⟩" in lab:
            ax.axhline(0, color="white", ls=":", lw=0.5)
        ax.set_xlabel("p", color="white")
        ax.set_ylabel(lab, color="white")
        ax.legend(facecolor="black", labelcolor="white", edgecolor="white", fontsize=7)
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_color("white")

    fig.patch.set_facecolor("black")
    fig.suptitle(f"Percolation 2D L={L} — universalité χ_info hors classe Ising",
                 color="white", y=0.995)
    plt.tight_layout()
    png = os.path.join(outdir, "percolation_xi.png")
    plt.savefig(png, dpi=130, facecolor="black", bbox_inches="tight")
    print(f"\nsaved {png}")

    out = {
        "L": L, "p_c": p_c, "p_values": p_values.tolist(),
        "P_inf": P_inf.tolist(), "chi_perco": chi_perco.tolist(),
        "om_mean": om_mean.tolist(), "om_std": om_std.tolist(),
        "tc_mean": tc_mean.tolist(), "cv_omega": cv_omega.tolist(),
        "peak_chi": float(peak_chi), "peak_tc": float(peak_tc),
        "peak_ostd": float(peak_ostd), "peak_cv": float(peak_cv),
        "universal_cv": bool(universal), "best_locator": best,
    }
    with open(os.path.join(outdir, "percolation_xi.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {os.path.join(outdir, 'percolation_xi.json')}")
    return out


if __name__ == "__main__":
    run()
