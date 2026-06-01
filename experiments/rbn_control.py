"""Solidification findings RBN : surrogate data (shuffle colonnes).

Le pic |Ω| à K=2 et le zéro-crossing de Ω sont-ils vrais, ou bruit d'estimation ?

Contrôle correct = shuffle indépendant de chaque colonne (nœud) à travers les
échantillons. PRÉSERVE : marginales, fraction gelée, activity. DÉTRUIT :
corrélations inter-nœuds. Si |Ω|_real > |Ω|_shuffle → vraie interaction.

+ distribution des Ω par réseau à chaque K : bimodale à K=2 (vrai mélange
synergie/redondance) ou unimodale centrée 0 (bruit) ?
"""
import sys
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from systems.rbn import make_rbn, attractor_ensemble, frozen_fraction
from xi.temporal import tc_random_subsets


def shuffle_columns(ensemble, rng):
    out = ensemble.copy()
    for j in range(out.shape[1]):
        perm = rng.permutation(out.shape[0])
        out[:, j] = out[perm, j]
    return out


def run(
    N=60,
    K_list=(1, 2, 3, 4, 5),
    p=0.5,
    n_real=20,
    n_init=2500,
    t_equil=300,
    subset_size=4,
    n_subsets=400,
    seed0=7,
    outdir=None,
):
    if outdir is None:
        outdir = os.path.dirname(os.path.abspath(__file__))

    print(f"RBN control N={N} | K={list(K_list)} | {n_real} réseaux/K | shuffle surrogate\n")
    rng_master = np.random.default_rng(seed0)

    K_arr = np.array(K_list, dtype=float)
    omega_per_net = {K: [] for K in K_list}      # mean(Ω) par réseau
    omega_shuf_per_net = {K: [] for K in K_list}
    abs_real = np.zeros(len(K_list))
    abs_shuf = np.zeros(len(K_list))
    o_real = np.zeros(len(K_list))
    frac_syn = np.zeros(len(K_list))             # fraction réseaux Ω<0

    t0 = time.time()
    for ki, K in enumerate(K_list):
        ar, ash, orr = [], [], []
        for r in range(n_real):
            seed = int(rng_master.integers(0, 2 ** 31))
            inputs, tables = make_rbn(N, K, p=p, seed=seed)
            ens = attractor_ensemble(inputs, tables, K, N,
                                     n_init=n_init, t_equil=t_equil, seed=seed)
            rng_sub = np.random.default_rng(seed)
            _, abs_o, o, _, _ = tc_random_subsets(ens, subset_size, n_subsets, rng_sub)
            ens_sh = shuffle_columns(ens, np.random.default_rng(seed + 1))
            rng_sub2 = np.random.default_rng(seed)
            _, abs_o_sh, o_sh, _, _ = tc_random_subsets(ens_sh, subset_size, n_subsets, rng_sub2)
            omega_per_net[K].append(o)
            omega_shuf_per_net[K].append(o_sh)
            ar.append(abs_o)
            ash.append(abs_o_sh)
            orr.append(o)
        abs_real[ki] = np.mean(ar)
        abs_shuf[ki] = np.mean(ash)
        o_real[ki] = np.mean(orr)
        frac_syn[ki] = np.mean(np.array(orr) < 0)
        excess = abs_real[ki] - abs_shuf[ki]
        print(f"  K={K} | mean Ω={o_real[ki]:+.4f} | "
              f"|Ω|_real={abs_real[ki]:.4f} |Ω|_shuf={abs_shuf[ki]:.4f} "
              f"excess={excess:+.4f} | frac_syn={frac_syn[ki]:.2f}")
    print(f"\nsim {time.time()-t0:.1f}s")

    # Verdicts
    print("\n" + "=" * 60)
    ki2 = list(K_list).index(2) if 2 in K_list else 1
    excess2 = abs_real[ki2] - abs_shuf[ki2]
    print(f"|Ω| excess à K=2 : {excess2:+.4f} "
          f"({'✓ signal réel' if excess2 > abs_shuf[ki2]*0.2 else '✗ ≈ bruit'})")
    om2 = np.array(omega_per_net[2]) if 2 in K_list else np.array(omega_per_net[K_list[1]])
    print(f"K=2 distribution Ω : min={om2.min():+.4f} max={om2.max():+.4f} "
          f"mean={om2.mean():+.4f} | {np.sum(om2<0)}/{len(om2)} réseaux synergie")
    bimodal = (om2.min() < -1e-3) and (om2.max() > 1e-3)
    print(f"  → {'✓ BIMODAL (vrai mélange syn/red)' if bimodal else '✗ unimodal'}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ax = axes[0]
    ax.plot(K_arr, abs_real, "o-", color="#ff5555", lw=2, label="|Ω| réel")
    ax.plot(K_arr, abs_shuf, "s--", color="#888888", lw=2, label="|Ω| shuffle (bruit)")
    ax.axvline(2, color="#ffaa00", ls=":")
    ax.set_xlabel("K", color="white")
    ax.set_ylabel("|Ω|", color="white")
    ax.set_title("signal vs surrogate", color="white")
    ax.legend(facecolor="black", labelcolor="white", edgecolor="white", fontsize=9)

    ax = axes[1]
    ax.plot(K_arr, o_real, "o-", color="#ff00ff", lw=2)
    ax.axhline(0, color="white", ls=":", lw=0.8)
    ax.axvline(2, color="#ffaa00", ls=":", label="K_c=2")
    ax.set_xlabel("K", color="white")
    ax.set_ylabel("mean Ω", color="white")
    ax.set_title("zéro-crossing régime", color="white")
    ax.legend(facecolor="black", labelcolor="white", edgecolor="white", fontsize=9)

    ax = axes[2]
    for K in K_list:
        om = np.array(omega_per_net[K])
        ax.scatter([K] * len(om), om, alpha=0.6, s=25)
    ax.axhline(0, color="white", ls=":", lw=0.8)
    ax.axvline(2, color="#ffaa00", ls=":")
    ax.set_xlabel("K", color="white")
    ax.set_ylabel("Ω par réseau", color="white")
    ax.set_title("distribution inter-réseaux", color="white")

    for ax in axes:
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_color("white")
    fig.patch.set_facecolor("black")
    fig.suptitle("RBN — solidification (surrogate shuffle + distribution Ω)",
                 color="white")
    plt.tight_layout()
    png = os.path.join(outdir, "rbn_control.png")
    plt.savefig(png, dpi=130, facecolor="black", bbox_inches="tight")
    print(f"\nsaved {png}")

    out = {
        "N": N, "K": K_arr.tolist(),
        "abs_real": abs_real.tolist(), "abs_shuf": abs_shuf.tolist(),
        "o_real": o_real.tolist(), "frac_syn": frac_syn.tolist(),
        "omega_per_net": {str(K): list(map(float, omega_per_net[K])) for K in K_list},
        "bimodal_K2": bool(bimodal),
        "excess_K2": float(excess2),
    }
    with open(os.path.join(outdir, "rbn_control.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {os.path.join(outdir, 'rbn_control.json')}")
    return out


if __name__ == "__main__":
    run()
