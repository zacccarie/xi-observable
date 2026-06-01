"""Conséquence testable #7 : la susceptibilité informationnelle est-elle universelle ?

Finding RBN : std inter-réseaux de Ω pique à l'edge of chaos (336× vs chaos).
Question : Var_réalisations(Ω) pique-t-elle aussi à Tc dans Ising ?

Si OUI → la dispersion d'un observable de régime au point critique est une
signature de criticalité TRANSVERSALE (Ising thermo + RBN dynamique), pas
spécifique RBN. Unifierait les deux systèmes.

À chaque T : n_real simulations indépendantes (seeds distincts), chacune donne
un Ω(blocs 2×2). On mesure mean(Ω), std(Ω), et χ (référence).
"""
import sys
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from systems.ising2d import simulate
from xi.interaction import o_information, total_correlation
from xi.hierarchy import rg_blocks
from xi.edge import susceptibility


def run(
    L=32,
    T_min=1.8,
    T_max=3.0,
    n_T=19,
    n_real=20,
    n_equil=1200,
    n_sample=150,
    sample_gap=4,
    block_size=2,
    seed0=11,
    outdir=None,
):
    if outdir is None:
        outdir = os.path.dirname(os.path.abspath(__file__))
    Tc = 2.0 / np.log(1.0 + np.sqrt(2.0))
    T_values = np.linspace(T_min, T_max, n_T)
    rng_master = np.random.default_rng(seed0)

    print(f"Ising susceptibilité info | L={L} | {n_real} réal/T | Tc={Tc:.4f}\n")
    chi = np.zeros(n_T)
    o_mean = np.zeros(n_T)
    o_std = np.zeros(n_T)
    tc_mean = np.zeros(n_T)

    t0 = time.time()
    for i, T in enumerate(T_values):
        o_list, tc_list, chi_list = [], [], []
        for r in range(n_real):
            seed = int(rng_master.integers(0, 2 ** 31))
            snaps = simulate(L=L, T=float(T), n_equil=n_equil,
                             n_sample=n_sample, sample_gap=sample_gap, seed=seed)
            mags = np.array([s.mean() for s in snaps])
            chi_list.append(susceptibility(mags, T, L * L))
            blocks = np.concatenate([rg_blocks(s, block_size) for s in snaps], axis=0)
            b01 = ((blocks + 1) // 2).astype(np.int8)
            o_list.append(o_information(b01))
            tc_list.append(total_correlation(b01))
        chi[i] = np.mean(chi_list)
        o_mean[i] = np.mean(o_list)
        o_std[i] = np.std(o_list)
        tc_mean[i] = np.mean(tc_list)
        print(f"  T={T:.3f} | χ={chi[i]:>8.2f} | Ω={o_mean[i]:+.4f} | "
              f"std(Ω)={o_std[i]:.4f} | TC={tc_mean[i]:.4f}")
    print(f"\nsim {time.time()-t0:.1f}s")

    peak_chi = T_values[np.argmax(chi)]
    peak_ostd = T_values[np.argmax(o_std)]
    peak_tc = T_values[np.argmax(tc_mean)]
    print(f"\npeak χ        T={peak_chi:.3f} (Δ={peak_chi-Tc:+.3f})")
    print(f"peak std(Ω)   T={peak_ostd:.3f} (Δ={peak_ostd-Tc:+.3f})")
    print(f"peak TC       T={peak_tc:.3f} (Δ={peak_tc-Tc:+.3f})")
    universal = abs(peak_ostd - Tc) < 0.15
    print(f"\nSusceptibilité info universelle : "
          f"{'✓ std(Ω) pique à Tc (Ising + RBN)' if universal else '✗ pas à Tc'}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (lab, vec, col) in zip(axes, [
        ("χ = N·Var(m)", chi, "#00ffff"),
        ("std(Ω) susceptibilité info", o_std, "#ff00ff"),
        ("mean Ω", o_mean, "#ffff00"),
    ]):
        ax.plot(T_values, vec, "o-", color=col, lw=2)
        ax.axvline(Tc, color="#ff3355", ls="--", label=f"Tc={Tc:.3f}")
        ax.set_xlabel("T", color="white")
        ax.set_ylabel(lab, color="white")
        ax.legend(facecolor="black", labelcolor="white", edgecolor="white", fontsize=8)
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_color("white")
    fig.patch.set_facecolor("black")
    fig.suptitle(f"Ising — susceptibilité informationnelle std(Ω) (L={L})", color="white")
    plt.tight_layout()
    png = os.path.join(outdir, "ising_suscept_info.png")
    plt.savefig(png, dpi=130, facecolor="black", bbox_inches="tight")
    print(f"saved {png}")

    out = {
        "L": L, "Tc": Tc, "T": T_values.tolist(),
        "chi": chi.tolist(), "o_mean": o_mean.tolist(),
        "o_std": o_std.tolist(), "tc_mean": tc_mean.tolist(),
        "peak_chi": float(peak_chi), "peak_o_std": float(peak_ostd),
        "universal": bool(universal),
    }
    with open(os.path.join(outdir, "ising_suscept_info.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {os.path.join(outdir, 'ising_suscept_info.json')}")
    return out


if __name__ == "__main__":
    run()
