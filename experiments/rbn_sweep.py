"""Ξ sur Random Boolean Networks — premier test de SYNERGIE en dynamique réelle.

RBN Kauffman N-K, biais p=0.5 → edge of chaos à K_c=2 (Derrida = 2Kp(1-p) = 1).

Questions :
1. Derrida slope localise-t-il K_c=2 ? (edge detector natif)
2. mean Ω(K) : la synergie (Ω<0) apparaît-elle ? À quel K ?
3. Synergie / interaction pique-t-elle à l'edge of chaos K=2 ?
4. Ξ_RBN = mask_Derrida × interaction discrimine-t-il les régimes ?

Système discret natif → pas d'artefact CG (leçon Kuramoto).
Mesure μ = ensemble d'attracteurs depuis CI aléatoires (mesure stationnaire).
"""
import sys
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from systems.rbn import (
    make_rbn, derrida_slope, attractor_ensemble,
    frozen_fraction, activity,
)
from xi.temporal import tc_random_subsets
from xi.edge import edge_mask


def run(
    N=60,
    K_list=(1, 2, 3, 4, 5),
    p=0.5,
    n_real=12,
    n_init=3000,
    t_equil=300,
    subset_size=4,
    n_subsets=300,
    epsilon=0.25,
    seed0=0,
    outdir=None,
):
    if outdir is None:
        outdir = os.path.dirname(os.path.abspath(__file__))

    print(f"RBN N={N} p={p} | K={list(K_list)} | {n_real} réalisations/K")
    print(f"K_c attendu = 2 (Derrida 2Kp(1-p)=1)\n")

    rng_master = np.random.default_rng(seed0)
    K_arr = np.array(K_list, dtype=float)
    derrida = np.zeros(len(K_list))
    derrida_std = np.zeros(len(K_list))
    tc_mean = np.zeros(len(K_list))
    o_mean = np.zeros(len(K_list))
    abs_o_mean = np.zeros(len(K_list))
    frozen = np.zeros(len(K_list))
    act = np.zeros(len(K_list))

    t0 = time.time()
    for ki, K in enumerate(K_list):
        d_list, tc_list, o_list, ao_list, fr_list, ac_list = [], [], [], [], [], []
        for r in range(n_real):
            seed = int(rng_master.integers(0, 2 ** 31))
            inputs, tables = make_rbn(N, K, p=p, seed=seed)
            d = derrida_slope(inputs, tables, K, N, n_pairs=1500, seed=seed)
            ens = attractor_ensemble(inputs, tables, K, N,
                                     n_init=n_init, t_equil=t_equil, seed=seed)
            tc, abs_o, o, _, _ = tc_random_subsets(
                ens, subset_size=subset_size, n_subsets=n_subsets,
                rng=np.random.default_rng(seed),
            )
            d_list.append(d)
            tc_list.append(tc)
            o_list.append(o)
            ao_list.append(abs_o)
            fr_list.append(frozen_fraction(ens))
            ac_list.append(activity(ens))
        derrida[ki] = np.mean(d_list)
        derrida_std[ki] = np.std(d_list)
        tc_mean[ki] = np.mean(tc_list)
        o_mean[ki] = np.mean(o_list)
        abs_o_mean[ki] = np.mean(ao_list)
        frozen[ki] = np.mean(fr_list)
        act[ki] = np.mean(ac_list)
        regime = "SYN" if o_mean[ki] < -1e-3 else ("red" if o_mean[ki] > 1e-3 else "neu")
        print(f"  K={K} | Derrida={derrida[ki]:.3f}±{derrida_std[ki]:.3f} | "
              f"TC={tc_mean[ki]:.4f} | Ω={o_mean[ki]:+.4f} [{regime}] | "
              f"frozen={frozen[ki]:.3f} | act={act[ki]:.3f}")
    print(f"\nsim total {time.time()-t0:.1f}s")

    # Edge mask : |Derrida - 1| minimal → proche edge of chaos
    dist_to_edge = np.abs(derrida - 1.0)
    proximity = 1.0 / (dist_to_edge + 1e-3)          # pic à Derrida=1
    mask = edge_mask(proximity, epsilon=epsilon)
    xi_tc = tc_mean * mask
    xi_abs = abs_o_mean * mask

    print(f"\nDerrida le plus proche de 1 : K={K_arr[np.argmin(dist_to_edge)]:.0f} "
          f"(Derrida={derrida[np.argmin(dist_to_edge)]:.3f})")
    print(f"peak TC          : K={K_arr[np.argmax(tc_mean)]:.0f}")
    print(f"min Ω (synergie) : K={K_arr[np.argmin(o_mean)]:.0f} (Ω={o_mean.min():+.4f})")
    print(f"peak |Ω|         : K={K_arr[np.argmax(abs_o_mean)]:.0f}")
    print(f"edge mask actif  : K={K_arr[mask.astype(bool)].tolist()}")

    synergy_seen = o_mean.min() < -1e-3
    print(f"\nSYNERGIE observée : {'✓ OUI' if synergy_seen else '✗ NON (redondance partout)'}")

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    Kc = 2.0

    ax = axes[0, 0]
    ax.errorbar(K_arr, derrida, yerr=derrida_std, fmt="o-", color="#00ffff", lw=2, capsize=3)
    ax.axhline(1.0, color="#ff3355", ls="--", label="edge (Derrida=1)")
    ax.axvline(Kc, color="#ffaa00", ls=":", label="K_c=2")
    ax.set_xlabel("K", color="white")
    ax.set_ylabel("Derrida slope", color="white")
    ax.set_title("edge detector natif", color="white")
    ax.legend(facecolor="black", labelcolor="white", edgecolor="white", fontsize=8)

    ax = axes[0, 1]
    ax.plot(K_arr, tc_mean, "o-", color="#ffff00", lw=2)
    ax.axvline(Kc, color="#ffaa00", ls=":")
    ax.set_xlabel("K", color="white")
    ax.set_ylabel("TC", color="white")
    ax.set_title("total correlation", color="white")

    ax = axes[0, 2]
    ax.plot(K_arr, o_mean, "o-", color="#ff00ff", lw=2)
    ax.axhline(0, color="white", ls=":", lw=0.8)
    ax.axvline(Kc, color="#ffaa00", ls=":")
    ax.set_xlabel("K", color="white")
    ax.set_ylabel("Ω (− = synergie)", color="white")
    ax.set_title("régime O-information", color="white")

    ax = axes[1, 0]
    ax.plot(K_arr, frozen, "o-", color="#00ff88", lw=2, label="frozen core")
    ax.plot(K_arr, act, "s-", color="#ff8800", lw=2, label="activity")
    ax.axvline(Kc, color="#ffaa00", ls=":")
    ax.set_xlabel("K", color="white")
    ax.set_ylabel("fraction", color="white")
    ax.set_title("ordre ↔ chaos", color="white")
    ax.legend(facecolor="black", labelcolor="white", edgecolor="white", fontsize=8)

    ax = axes[1, 1]
    ax.plot(K_arr, abs_o_mean, "o-", color="#ff5555", lw=2)
    ax.axvline(Kc, color="#ffaa00", ls=":")
    ax.set_xlabel("K", color="white")
    ax.set_ylabel("|Ω|", color="white")
    ax.set_title("intensité interaction", color="white")

    ax = axes[1, 2]
    ax.plot(K_arr, xi_tc, "o-", color="#ffff00", lw=2, label="Ξ_tc")
    ax.plot(K_arr, xi_abs, "s-", color="#ff5555", lw=2, label="Ξ_|Ω|")
    ax.axvline(Kc, color="#ffaa00", ls=":")
    ax.set_xlabel("K", color="white")
    ax.set_ylabel("Ξ (mask Derrida)", color="white")
    ax.set_title("Ξ_RBN gaté edge", color="white")
    ax.legend(facecolor="black", labelcolor="white", edgecolor="white", fontsize=8)

    for ax in axes.flat:
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_color("white")
    fig.patch.set_facecolor("black")
    fig.suptitle(f"Ξ RBN Kauffman N={N} — edge of chaos K_c=2", color="white", y=0.99)
    plt.tight_layout()
    png = os.path.join(outdir, "rbn_xi.png")
    plt.savefig(png, dpi=130, facecolor="black", bbox_inches="tight")
    print(f"\nsaved {png}")

    out = {
        "N": N, "p": p, "K": K_arr.tolist(), "n_real": n_real,
        "derrida": derrida.tolist(), "derrida_std": derrida_std.tolist(),
        "tc": tc_mean.tolist(), "o": o_mean.tolist(), "abs_o": abs_o_mean.tolist(),
        "frozen": frozen.tolist(), "activity": act.tolist(),
        "mask": mask.astype(int).tolist(),
        "xi_tc": xi_tc.tolist(), "xi_abs": xi_abs.tolist(),
        "synergy_seen": bool(synergy_seen),
    }
    with open(os.path.join(outdir, "rbn_xi.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {os.path.join(outdir, 'rbn_xi.json')}")
    return out


if __name__ == "__main__":
    run()
