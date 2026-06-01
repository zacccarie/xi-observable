"""Stade B : Ξ multi-échelle via coarse-graining RG.

Hypothèses testées :
- TC(k) décroît avec k partout SAUF à Tc où corrélations longue portée persistent
- γ_T = pente log TC(k) vs k → moins négatif à Tc (plateau) qu'hors criticalité (chute rapide)
- saturation_k(T) → max à Tc (saturation tardive)
- Ξ_hier = somme pondérée gatée → pique à Tc avec contraste plus net que Stade A
"""
import sys
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from systems.ising2d import simulate
from xi.observable import xi_hierarchical


def run(
    L=32,
    T_min=1.5,
    T_max=3.5,
    n_T=21,
    n_equil=1500,
    n_sample=400,
    sample_gap=4,
    k_max=3,
    block_size=2,
    epsilon=0.3,
    alpha=0.0,
    seed=42,
    outdir=None,
):
    if outdir is None:
        outdir = os.path.dirname(os.path.abspath(__file__))
    Tc = 2.0 / np.log(1.0 + np.sqrt(2.0))
    T_values = np.linspace(T_min, T_max, n_T)

    # L=32, k_max=3 → tailles 32, 16, 8, 4 → blocs 2x2 jusqu'à k=3 (grille 4x4 fait 4 blocs)
    print(f"L={L} | k_max={k_max} | block_size={block_size} | α={alpha}")
    print(f"T sweep [{T_min},{T_max}] n_T={n_T} | n_sample={n_sample}")

    t0 = time.time()
    snapshots = {}
    for T in T_values:
        ts = time.time()
        snapshots[float(T)] = simulate(
            L=L, T=float(T),
            n_equil=n_equil, n_sample=n_sample, sample_gap=sample_gap,
            seed=seed,
        )
        print(f"  T={T:.3f} | {time.time()-ts:.1f}s")
    print(f"sim total {time.time()-t0:.1f}s")

    res = xi_hierarchical(
        snapshots, T_values, L=L,
        k_max=k_max, block_size=block_size,
        alpha=alpha, epsilon=epsilon,
    )

    # Indices points clés
    iTc = np.argmin(np.abs(T_values - Tc))
    print()
    print(f"Tc = {Tc:.4f}")
    print(f"\nTC(T, k) matrix :")
    print(f"{'T':>6} | " + "  ".join(f"k={k}" for k in range(k_max + 1))
          + f"  |  γ_T  | sat_k")
    print("-" * 70)
    for i, T in enumerate(T_values):
        vals = "  ".join(f"{res['tc_hier'][i,k]:>6.3f}" for k in range(k_max + 1))
        g = res["gamma_T"][i]
        sk = res["saturation_k"][i]
        g_str = f"{g:>+.3f}" if not np.isnan(g) else "  nan"
        sk_str = f"{int(sk):>3d}" if not np.isnan(sk) else "nan"
        marker = "  <-- Tc" if i == iTc else ""
        print(f"{T:>6.3f} | {vals}  | {g_str} | {sk_str}{marker}")

    print(f"\npeak χ          T={T_values[np.argmax(res['chi'])]:.3f}")
    print(f"peak Ξ_hier     T={T_values[np.argmax(res['xi_hier'])]:.3f}")
    print(f"peak γ_T (max)  T={T_values[np.nanargmax(res['gamma_T'])]:.3f}  "
          f"(plateau = γ proche 0)")
    print(f"max saturation_k T={T_values[np.nanargmax(res['saturation_k'])]:.3f}")

    # Plots dark
    fig = plt.figure(figsize=(13, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)

    # TC(k) curves per T (focus near Tc + cold + hot)
    ax1 = fig.add_subplot(gs[0, 0])
    T_show = [T_values[0], T_values[iTc - 2], T_values[iTc], T_values[iTc + 2], T_values[-1]]
    colors_curve = ["#00ff88", "#ffff00", "#ff3355", "#ff00ff", "#00ffff"]
    for T, col in zip(T_show, colors_curve):
        i = np.argmin(np.abs(T_values - T))
        ax1.semilogy(range(k_max + 1), res["tc_hier"][i],
                     "o-", color=col, lw=1.5, label=f"T={T:.2f}")
    ax1.set_xlabel("k (échelle RG)", color="white")
    ax1.set_ylabel("TC(T, k)", color="white")
    ax1.set_title("TC vs échelle hiérarchique", color="white")
    ax1.legend(facecolor="black", edgecolor="white", labelcolor="white", fontsize=8)

    # Heatmap TC(T, k)
    ax2 = fig.add_subplot(gs[0, 1])
    im = ax2.imshow(
        res["tc_hier"].T,
        aspect="auto", origin="lower",
        extent=[T_values[0], T_values[-1], -0.5, k_max + 0.5],
        cmap="magma",
    )
    ax2.axvline(Tc, color="#00ffff", ls="--", lw=1, label=f"Tc")
    ax2.set_xlabel("T", color="white")
    ax2.set_ylabel("k", color="white")
    ax2.set_title("heatmap TC(T, k)", color="white")
    cb = plt.colorbar(im, ax=ax2)
    cb.ax.tick_params(colors="white")
    ax2.legend(facecolor="black", labelcolor="white", edgecolor="white")

    # γ_T
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(T_values, res["gamma_T"], "o-", color="#ff00ff", lw=1.5)
    ax3.axvline(Tc, color="#ff3355", ls="--", label=f"Tc")
    ax3.axhline(0, color="white", ls=":", lw=0.5)
    ax3.set_xlabel("T", color="white")
    ax3.set_ylabel("γ_T (pente log TC vs k)", color="white")
    ax3.set_title("scaling exponent — plateau (γ→0) à Tc attendu", color="white")
    ax3.legend(facecolor="black", labelcolor="white", edgecolor="white")

    # saturation_k
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(T_values, res["saturation_k"], "s-", color="#00ff88", lw=1.5)
    ax4.axvline(Tc, color="#ff3355", ls="--")
    ax4.set_xlabel("T", color="white")
    ax4.set_ylabel("saturation_k", color="white")
    ax4.set_title("échelle de saturation TC", color="white")

    # Ξ_hier vs Stade A reference
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(T_values, res["xi_hier"], "o-", color="#ffff00", lw=1.5,
             label=f"Ξ_hier (α={alpha})")
    # ref Stade A si dispo
    sa_path = os.path.join(outdir, "ising_xi_stadeA.json")
    if os.path.exists(sa_path):
        sa = json.load(open(sa_path))
        sa_T = np.array(sa["T"])
        sa_xi = np.array(sa["xi_tc"])
        if sa_xi.max() > 0 and res["xi_hier"].max() > 0:
            ax5.plot(sa_T, sa_xi * res["xi_hier"].max() / sa_xi.max(),
                     "s--", color="#888888", lw=1, label="Ξ_A (Stade A, normé)")
    ax5.axvline(Tc, color="#ff3355", ls="--")
    ax5.set_xlabel("T", color="white")
    ax5.set_ylabel("Ξ", color="white")
    ax5.legend(facecolor="black", labelcolor="white", edgecolor="white")
    ax5.set_title("Ξ_hier vs Ξ_A — contraste critique", color="white")

    # χ pour repère
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(T_values, res["chi"], "o-", color="#00ffff", lw=1.5)
    ax6.axvline(Tc, color="#ff3355", ls="--")
    ax6.set_xlabel("T", color="white")
    ax6.set_ylabel("χ", color="white")
    ax6.set_title("susceptibility (référence)", color="white")

    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_color("white")

    fig.patch.set_facecolor("black")
    fig.suptitle(f"Ξ Stade B — hiérarchie RG (L={L}, k_max={k_max})",
                 color="white", y=0.99, fontsize=13)

    png = os.path.join(outdir, "ising_xi_stadeB.png")
    plt.savefig(png, dpi=130, facecolor="black", bbox_inches="tight")
    print(f"\nsaved {png}")

    out = {
        "L": L, "Tc": Tc, "k_max": k_max, "alpha": alpha, "epsilon": epsilon,
        "T": T_values.tolist(),
        "chi": res["chi"].tolist(),
        "tc_hier": res["tc_hier"].tolist(),
        "o_hier": res["o_hier"].tolist(),
        "abs_o_hier": res["abs_o_hier"].tolist(),
        "gamma_T": res["gamma_T"].tolist(),
        "saturation_k": res["saturation_k"].tolist(),
        "xi_hier": res["xi_hier"].tolist(),
        "mask": res["mask"].tolist(),
        "weights": res["weights"].tolist(),
    }
    jpath = os.path.join(outdir, "ising_xi_stadeB.json")
    with open(jpath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {jpath}")
    return res


if __name__ == "__main__":
    run()
