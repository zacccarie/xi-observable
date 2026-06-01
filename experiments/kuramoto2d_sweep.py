"""Task #10 — Kuramoto 2D local : test conséquence #5 (dualité CG↔C).

C local (4-voisins) → CG spatial doit fonctionner (vs all-to-all qui échoue).
Pipeline xi_hierarchical (RG blocks) applicable directement.

Attendu : γ_K(K) minimum à K_c effectif (transition synchronisation 2D).
Aussi : test χ_info = std(Ω) inter-réalisations pour localiser K_c (Conséquence #7).
"""
import sys
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from systems.kuramoto_2d import simulate as kura2d_sim, chi_from_r
from xi.observable import xi_hierarchical
from xi.interaction import o_information, total_correlation
from xi.hierarchy import rg_blocks


def random_subset_omega(snapshots, subset_size=4, n_subsets=200, rng=None):
    """Sur snapshots catégoriels (L,L), tire subsets aléatoires de subset_size cellules."""
    if rng is None:
        rng = np.random.default_rng(0)
    L = snapshots[0].shape[0]
    N = L * L
    flat = np.stack([s.flatten() for s in snapshots], axis=0)
    oms, tcs = [], []
    for _ in range(n_subsets):
        idx = rng.choice(N, size=subset_size, replace=False)
        sub = flat[:, idx].astype(np.int8)
        oms.append(o_information(sub))
        tcs.append(total_correlation(sub))
    return np.mean(tcs), np.mean(oms), np.std(oms)


def run(
    L=16,
    sigma_omega=1.0,
    K_values=None,
    n_realizations=5,
    k_max=2,
    epsilon=0.3,
    seed0=11,
    outdir=None,
):
    if outdir is None:
        outdir = os.path.dirname(os.path.abspath(__file__))
    if K_values is None:
        K_values = np.array([0.3, 0.5, 0.7, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 4.0, 5.0])
    n_K = len(K_values)
    rng_master = np.random.default_rng(seed0)

    print(f"L={L} ({L*L} osc) | σ_ω={sigma_omega} | {n_K} K | "
          f"{n_realizations} réalisations")

    # Phase 1 : sweep K avec n_realizations seeds. Collect r̄, χ, Ω stats.
    r_mean = np.zeros(n_K)
    chi_arr = np.zeros(n_K)
    om_mean = np.zeros(n_K)
    om_std = np.zeros(n_K)
    tc_mean = np.zeros(n_K)
    snaps_first_real = {}  # garde 1 réal/K pour Stade B hier

    t0 = time.time()
    for i, K in enumerate(K_values):
        om_list, tc_list, chi_list, r_list = [], [], [], []
        for r in range(n_realizations):
            seed = int(rng_master.integers(0, 2**31))
            snaps, r_traj = kura2d_sim(
                L=L, K=float(K), sigma_omega=sigma_omega,
                dt=0.05, t_equil=60.0, t_sample=120.0, sample_gap=1.0,
                n_bins=4, seed=seed,
            )
            chi_list.append(chi_from_r(r_traj, L * L))
            r_list.append(r_traj.mean())
            tc, om, _ = random_subset_omega(snaps, subset_size=4,
                                             n_subsets=200,
                                             rng=np.random.default_rng(seed + 7))
            om_list.append(om)
            tc_list.append(tc)
            if r == 0:
                snaps_first_real[float(K)] = snaps
        r_mean[i] = np.mean(r_list)
        chi_arr[i] = np.mean(chi_list)
        om_mean[i] = np.mean(om_list)
        om_std[i] = np.std(om_list)
        tc_mean[i] = np.mean(tc_list)
        print(f"  K={K:.2f}  r̄={r_mean[i]:.3f}  χ={chi_arr[i]:.3f}  "
              f"⟨Ω⟩={om_mean[i]:+.4f}  std(Ω)={om_std[i]:.4f}  "
              f"TC={tc_mean[i]:.4f}")
    print(f"sim {time.time()-t0:.1f}s")

    # Phase 2 : Stade B hier sur 1 réal/K (test γ→0 avec C local)
    res_hier = xi_hierarchical(
        snaps_first_real, K_values, L=L,
        k_max=k_max, block_size=2, epsilon=epsilon,
        encode="categorical", cg_rule="mode", chi_provided=chi_arr,
    )

    # CV(Ω) coefficient de variation
    cv_omega = om_std / (np.abs(om_mean) + 1e-9)

    peak_chi = K_values[np.argmax(chi_arr)]
    peak_tc = K_values[np.argmax(tc_mean)]
    peak_ostd = K_values[np.argmax(om_std)]
    peak_cv = K_values[np.argmax(cv_omega)]
    # γ minimum |·|
    valid_g = ~np.isnan(res_hier["gamma_T"])
    if valid_g.any():
        peak_gamma_idx = np.argmin(np.abs(res_hier["gamma_T"]))
        peak_gamma_K = K_values[peak_gamma_idx]
    else:
        peak_gamma_K = np.nan

    print(f"\npeak χ        K={peak_chi:.2f}")
    print(f"peak TC       K={peak_tc:.2f}")
    print(f"peak std(Ω)   K={peak_ostd:.2f}")
    print(f"peak CV(Ω)    K={peak_cv:.2f}")
    print(f"min |γ_K|     K={peak_gamma_K:.2f}  (γ={res_hier['gamma_T'][peak_gamma_idx]:+.3f})")

    # Plot
    fig, axes = plt.subplots(3, 2, figsize=(13, 11))

    ax = axes[0, 0]
    ax.plot(K_values, r_mean, "o-", color="cyan", lw=1.5, ms=5)
    ax.set_ylabel("r̄", color="white")
    ax.set_title("Order parameter (Kuramoto 2D local)", color="white", fontsize=10)

    ax = axes[0, 1]
    ax.plot(K_values, chi_arr, "o-", color="#ff8800", lw=1.5, ms=5)
    ax.set_ylabel("χ = N·Var(r)", color="white")
    ax.axvline(peak_chi, color="#ff8800", ls=":", lw=1, label=f"peak={peak_chi:.2f}")
    ax.legend(facecolor="black", labelcolor="white", edgecolor="white", fontsize=8)

    ax = axes[1, 0]
    ax.plot(K_values, om_mean, "o-", color="yellow", lw=1.5, ms=5, label="⟨Ω⟩")
    ax.axhline(0, color="white", ls=":", lw=0.5)
    ax.set_ylabel("⟨Ω⟩", color="white")
    ax.legend(facecolor="black", labelcolor="white", edgecolor="white", fontsize=8)

    ax = axes[1, 1]
    ax.plot(K_values, om_std, "s-", color="magenta", lw=1.5, ms=5, label="std(Ω)")
    ax.plot(K_values, cv_omega, "^-", color="#00ff88", lw=1.2, ms=4, label="CV(Ω)")
    ax.set_ylabel("susceptibilité info", color="white")
    ax.axvline(peak_ostd, color="magenta", ls=":", lw=1, label=f"std peak={peak_ostd:.2f}")
    ax.axvline(peak_cv, color="#00ff88", ls=":", lw=1, label=f"CV peak={peak_cv:.2f}")
    ax.legend(facecolor="black", labelcolor="white", edgecolor="white", fontsize=8)

    ax = axes[2, 0]
    ax.plot(K_values, res_hier["gamma_T"], "o-", color="#00ffff", lw=1.5, ms=5)
    ax.axhline(0, color="white", ls=":", lw=0.6)
    ax.axvline(peak_gamma_K, color="#00ffff", ls=":", lw=1,
               label=f"min|γ|={peak_gamma_K:.2f}")
    ax.set_ylabel("γ_K (slope log TC vs k)", color="white")
    ax.set_xlabel("K", color="white")
    ax.legend(facecolor="black", labelcolor="white", edgecolor="white", fontsize=8)
    ax.set_title("Self-similarity hier (C local → CG spatial valide ?)",
                 color="white", fontsize=10)

    ax = axes[2, 1]
    cmap = plt.get_cmap("plasma", k_max + 1)
    for k in range(k_max + 1):
        ax.plot(K_values, res_hier["tc_hier"][:, k], "o-",
                color=cmap(k), label=f"k={k}", lw=1.4, ms=4)
    ax.set_xlabel("K", color="white")
    ax.set_ylabel("TC(K, k)", color="white")
    ax.legend(facecolor="black", labelcolor="white", edgecolor="white", fontsize=7)

    for ax in axes.flat:
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_color("white")

    fig.patch.set_facecolor("black")
    fig.suptitle(f"Kuramoto 2D LOCAL N={L*L} — test C local + CG spatial",
                 color="white", y=0.995)
    plt.tight_layout()
    png = os.path.join(outdir, "kuramoto2d_sweep.png")
    plt.savefig(png, dpi=130, facecolor="black", bbox_inches="tight")
    print(f"\nsaved {png}")

    out = {
        "L": L, "sigma_omega": sigma_omega, "K_values": K_values.tolist(),
        "r_mean": r_mean.tolist(), "chi": chi_arr.tolist(),
        "om_mean": om_mean.tolist(), "om_std": om_std.tolist(),
        "tc_mean": tc_mean.tolist(), "cv_omega": cv_omega.tolist(),
        "gamma_T": res_hier["gamma_T"].tolist(),
        "tc_hier": res_hier["tc_hier"].tolist(),
        "peak_chi": float(peak_chi), "peak_tc": float(peak_tc),
        "peak_ostd": float(peak_ostd), "peak_cv": float(peak_cv),
        "peak_gamma_K": float(peak_gamma_K) if not np.isnan(peak_gamma_K) else None,
    }
    with open(os.path.join(outdir, "kuramoto2d_sweep.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {os.path.join(outdir, 'kuramoto2d_sweep.json')}")
    return out


if __name__ == "__main__":
    run()
