"""#8 — χ_info = Var_réalisations(Ω) sur Kuramoto : test décisif.

Kuramoto était le système problématique : artefact CG temporel (7×), χ pique
K=1.40 et τ_corr K=1.50, tous deux sous K_c=1.596 (finite-size).

Question : Var inter-réalisations de Ω (différents tirages ω quenched) pique-t-elle
PROPREMENT à K_c, là où χ et τ_corr décalaient ?

PAS de coarse-graining temporel (leçon artefact). Ω calculé directement sur les
snapshots quadrantisés : samples = instants, variables = oscillateurs.
Analogue RBN : RBN varie le réseau (wiring+fonctions), Kuramoto varie ω.
"""
import sys
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from systems.kuramoto import simulate, chi_from_r
from xi.temporal import tc_random_subsets


def run(
    N=64,
    K_min=0.8,
    K_max=2.4,
    n_K=17,
    n_real=16,
    sigma_omega=1.0,
    dt=0.05,
    t_equil=80.0,
    t_sample=150.0,
    sample_gap=0.25,
    n_bins=4,
    subset_size=4,
    n_subsets=300,
    seed0=23,
    outdir=None,
):
    if outdir is None:
        outdir = os.path.dirname(os.path.abspath(__file__))
    Kc = 2.0 / (np.pi * (1.0 / (sigma_omega * np.sqrt(2 * np.pi))))
    K_values = np.linspace(K_min, K_max, n_K)
    rng_master = np.random.default_rng(seed0)

    print(f"Kuramoto χ_info | N={N} | {n_real} réal/K | Kc={Kc:.4f}")
    print("(snapshots bruts quadrantisés, AUCUN CG temporel)\n")

    chi = np.zeros(n_K)
    r_mean = np.zeros(n_K)
    o_mean = np.zeros(n_K)
    o_std = np.zeros(n_K)
    abs_o_mean = np.zeros(n_K)

    t0 = time.time()
    for i, K in enumerate(K_values):
        o_list, chi_list, r_list, ao_list = [], [], [], []
        for r in range(n_real):
            seed = int(rng_master.integers(0, 2 ** 31))
            snaps_grid, r_traj = simulate(
                N=N, K=float(K), sigma_omega=sigma_omega, dt=dt,
                t_equil=t_equil, t_sample=t_sample, sample_gap=sample_gap,
                n_bins=n_bins, seed=seed,
            )
            samples = np.stack([g.flatten() for g in snaps_grid], axis=0)  # (n_snap, N)
            _, abs_o, o, _, _ = tc_random_subsets(
                samples, subset_size=subset_size,
                n_subsets=n_subsets, rng=np.random.default_rng(seed),
            )
            o_list.append(o)
            ao_list.append(abs_o)
            chi_list.append(chi_from_r(r_traj, N))
            r_list.append(r_traj.mean())
        chi[i] = np.mean(chi_list)
        r_mean[i] = np.mean(r_list)
        o_mean[i] = np.mean(o_list)
        o_std[i] = np.std(o_list)
        abs_o_mean[i] = np.mean(ao_list)
        print(f"  K={K:.3f} | r={r_mean[i]:.3f} | χ={chi[i]:>7.3f} | "
              f"Ω={o_mean[i]:+.4f} | std(Ω)={o_std[i]:.4f}")
    print(f"\nsim {time.time()-t0:.1f}s")

    peak_chi = K_values[np.argmax(chi)]
    peak_ostd = K_values[np.argmax(o_std)]
    peak_abso = K_values[np.argmax(abs_o_mean)]
    print(f"\nKc analytic   = {Kc:.4f}")
    print(f"peak χ        K={peak_chi:.3f} (Δ={peak_chi-Kc:+.3f})")
    print(f"peak |Ω|      K={peak_abso:.3f} (Δ={peak_abso-Kc:+.3f})")
    print(f"peak std(Ω)   K={peak_ostd:.3f} (Δ={peak_ostd-Kc:+.3f})")
    better = abs(peak_ostd - Kc) < abs(peak_chi - Kc)
    print(f"\nχ_info localise Kc mieux que χ : "
          f"{'✓ OUI — χ_info sauve Kuramoto' if better else '✗ non'}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (lab, vec, col, peak) in zip(axes, [
        ("χ = N·Var(r)", chi, "#00ffff", peak_chi),
        ("std(Ω) = χ_info", o_std, "#ff00ff", peak_ostd),
        ("r̄ order param", r_mean, "#00ff88", None),
    ]):
        ax.plot(K_values, vec, "o-", color=col, lw=2)
        ax.axvline(Kc, color="#ff3355", ls="--", label=f"Kc={Kc:.3f}")
        if peak is not None:
            ax.axvline(peak, color=col, ls=":", lw=1, label=f"peak={peak:.2f}")
        ax.set_xlabel("K", color="white")
        ax.set_ylabel(lab, color="white")
        ax.legend(facecolor="black", labelcolor="white", edgecolor="white", fontsize=8)
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_color("white")
    fig.patch.set_facecolor("black")
    fig.suptitle(f"Kuramoto — χ_info=Var(Ω) (N={N}) : test localisation Kc", color="white")
    plt.tight_layout()
    png = os.path.join(outdir, "kuramoto_suscept_info.png")
    plt.savefig(png, dpi=130, facecolor="black", bbox_inches="tight")
    print(f"saved {png}")

    out = {
        "N": N, "Kc": Kc, "K": K_values.tolist(),
        "chi": chi.tolist(), "r_mean": r_mean.tolist(),
        "o_mean": o_mean.tolist(), "o_std": o_std.tolist(),
        "abs_o_mean": abs_o_mean.tolist(),
        "peak_chi": float(peak_chi), "peak_o_std": float(peak_ostd),
        "better_than_chi": bool(better),
    }
    with open(os.path.join(outdir, "kuramoto_suscept_info.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {os.path.join(outdir, 'kuramoto_suscept_info.json')}")
    return out


if __name__ == "__main__":
    run()
