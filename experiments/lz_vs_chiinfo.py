"""Test conjecture : χ_info ~ Var(LZ) inter-réalisations.

LZ-complexity = proxy calculable de Kolmogorov K. Si la "susceptibilité
informationnelle" Var(Ω) mesure réellement l'incompressibilité différentielle
des réalisations, alors Var(LZ) doit piquer aux mêmes points critiques.

Test sur RBN (discret natif, K_c=2 connu). Pour chaque K :
  - n_real réseaux indépendants
  - Pour chaque réseau : ensemble 3000 attracteurs → LZ(concat aplatissement)
  - mean(LZ), std(LZ) inter-réseaux
  - Comparer position des pics : std(Ω) vs std(LZ)

Si corrélation forte (Pearson > 0.7) ET argmax std(LZ) ≈ K_c → pont
info-théorie/computation validé. χ_info = mesure opérationnelle K_conditional.
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
from xi.complexity import lz76, lz_normalized
from xi.interaction import o_information, total_correlation


def lz_of_ensemble(ensemble):
    """Concat all attractor states, return LZ_normalized of resulting bit string."""
    flat = ensemble.flatten().astype(np.int8)
    return lz_normalized(flat)


def omega_random_subsets(ensemble, subset_size=4, n_subsets=200, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    M, N = ensemble.shape
    oms = []
    for _ in range(n_subsets):
        idx = rng.choice(N, size=subset_size, replace=False)
        sub = ensemble[:, idx].astype(np.int8)
        oms.append(o_information(sub))
    return np.mean(oms), np.std(oms)


def run(
    N=48,
    K_values=(1, 2, 3, 4, 5),
    p=0.5,
    n_networks=12,
    n_init=600,
    t_equil=150,
    subset_size=4,
    n_subsets=200,
    seed=42,
    outdir=None,
):
    if outdir is None:
        outdir = os.path.dirname(os.path.abspath(__file__))

    print(f"N={N}  p={p}  K_values={K_values}")
    print(f"n_networks={n_networks}  n_init={n_init}\n")

    lz_by_K = {K: [] for K in K_values}
    om_by_K = {K: [] for K in K_values}
    om_std_by_K = {K: [] for K in K_values}
    derrida_by_K = {K: [] for K in K_values}
    frozen_by_K = {K: [] for K in K_values}

    t0 = time.time()
    rng_master = np.random.default_rng(seed)
    for K in K_values:
        ts = time.time()
        for net in range(n_networks):
            net_seed = seed * 1000 + K * 100 + net
            inputs, tables = make_rbn(N, K, p, seed=net_seed)
            dslope = derrida_slope(inputs, tables, K, N,
                                   n_pairs=800, seed=net_seed + 1)
            ens = attractor_ensemble(inputs, tables, K, N,
                                     n_init=n_init, t_equil=t_equil,
                                     seed=net_seed + 7)
            ff = frozen_fraction(ens)
            lz = lz_of_ensemble(ens)
            om_mean, om_std_sub = omega_random_subsets(
                ens, subset_size=subset_size, n_subsets=n_subsets,
                rng=np.random.default_rng(net_seed + 13),
            )
            lz_by_K[K].append(lz)
            om_by_K[K].append(om_mean)
            om_std_by_K[K].append(om_std_sub)
            derrida_by_K[K].append(dslope)
            frozen_by_K[K].append(ff)
        elapsed = time.time() - ts
        print(f"  K={K}  mean(LZ)={np.mean(lz_by_K[K]):.4f} "
              f"std(LZ)_nets={np.std(lz_by_K[K]):.4f} "
              f"mean(Ω)={np.mean(om_by_K[K]):+.4f} "
              f"std(Ω)_nets={np.std(om_by_K[K]):.4f}  ({elapsed:.1f}s)")
    print(f"total {time.time()-t0:.1f}s\n")

    # Aggrégation
    Ks = np.array(K_values)
    mean_lz = np.array([np.mean(lz_by_K[K]) for K in K_values])
    std_lz = np.array([np.std(lz_by_K[K]) for K in K_values])
    mean_om = np.array([np.mean(om_by_K[K]) for K in K_values])
    std_om_nets = np.array([np.std(om_by_K[K]) for K in K_values])  # χ_info
    abs_om = np.abs(mean_om)
    cv_om = std_om_nets / (abs_om + 1e-9)
    mean_derrida = np.array([np.mean(derrida_by_K[K]) for K in K_values])
    mean_frozen = np.array([np.mean(frozen_by_K[K]) for K in K_values])

    # Corrélations
    pearson_chiinfo_lz_std = np.corrcoef(std_om_nets, std_lz)[0, 1]
    pearson_chiinfo_lz_mean = np.corrcoef(std_om_nets, mean_lz)[0, 1]
    pearson_omega_lz_mean = np.corrcoef(np.abs(mean_om), mean_lz)[0, 1]

    K_peak_chiinfo = Ks[np.argmax(std_om_nets)]
    K_peak_stdlz = Ks[np.argmax(std_lz)]
    K_peak_meanlz = Ks[np.argmax(mean_lz)]

    print("=" * 60)
    print(f"K_c analytique           = 2")
    print(f"peak std(Ω) = χ_info     = K={K_peak_chiinfo}")
    print(f"peak std(LZ) inter-nets  = K={K_peak_stdlz}")
    print(f"peak mean(LZ)            = K={K_peak_meanlz}")
    print(f"Pearson(std(Ω), std(LZ)) = {pearson_chiinfo_lz_std:+.3f}")
    print(f"Pearson(std(Ω), mean LZ) = {pearson_chiinfo_lz_mean:+.3f}")
    print(f"Pearson(|mean Ω|, mean LZ) = {pearson_omega_lz_mean:+.3f}")
    print("=" * 60)

    same_peak = (K_peak_chiinfo == K_peak_stdlz)
    high_corr = pearson_chiinfo_lz_std > 0.7
    if same_peak and high_corr:
        verdict = "PONT VALIDÉ"
    elif same_peak or high_corr:
        verdict = "PONT PARTIEL"
    else:
        verdict = "PAS DE PONT"
    print(f"\nVerdict : {verdict}")

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    ax = axes[0, 0]
    ax.plot(Ks, mean_derrida, "o-", color="cyan", lw=1.5, ms=5)
    ax.axhline(1.0, color="#ff3355", ls="--", label="K_c (Derrida=1)")
    ax.set_ylabel("Derrida slope", color="white")
    ax.set_title("Order parameter (RBN)", color="white", fontsize=10)
    ax.legend(facecolor="black", labelcolor="white", edgecolor="white")

    ax = axes[0, 1]
    ax.plot(Ks, mean_lz, "o-", color="yellow", lw=1.5, ms=5, label="mean(LZ_n)")
    ax.plot(Ks, mean_frozen, "s--", color="#aaaaff", lw=1.0, label="frozen fraction")
    ax.set_ylabel("complexité moyenne", color="white")
    ax.legend(facecolor="black", labelcolor="white", edgecolor="white", fontsize=8)
    ax.set_title("Mean LZ vs frozen (monotone K croissant)", color="white", fontsize=10)

    ax = axes[0, 2]
    ax.plot(Ks, std_lz, "o-", color="magenta", lw=1.5, ms=5, label="std(LZ) inter-réseaux")
    ax.axvline(K_peak_stdlz, color="magenta", ls=":", label=f"peak K={K_peak_stdlz}")
    ax.set_ylabel("std(LZ) inter-réseaux", color="white")
    ax.legend(facecolor="black", labelcolor="white", edgecolor="white", fontsize=8)
    ax.set_title("Var(LZ) — incompressibilité différentielle", color="white", fontsize=10)

    ax = axes[1, 0]
    ax.plot(Ks, std_om_nets, "o-", color="#00ff88", lw=1.5, ms=5,
            label="std(Ω) = χ_info")
    ax.axvline(K_peak_chiinfo, color="#00ff88", ls=":",
               label=f"peak K={K_peak_chiinfo}")
    ax.set_xlabel("K", color="white")
    ax.set_ylabel("χ_info = std(Ω) inter-réseaux", color="white")
    ax.legend(facecolor="black", labelcolor="white", edgecolor="white", fontsize=8)

    ax = axes[1, 1]
    # Scatter std(Ω) vs std(LZ)
    ax.scatter(std_om_nets, std_lz, c=Ks, cmap="plasma", s=80, edgecolor="white")
    for i, K in enumerate(Ks):
        ax.annotate(f"K={K}", (std_om_nets[i], std_lz[i]),
                    color="white", fontsize=9, xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("std(Ω) = χ_info", color="white")
    ax.set_ylabel("std(LZ) inter-réseaux", color="white")
    ax.set_title(f"Pearson = {pearson_chiinfo_lz_std:+.3f}",
                 color="white", fontsize=10)

    ax = axes[1, 2]
    # Normalised overlay
    def norm(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-12)
    ax.plot(Ks, norm(std_om_nets), "o-", color="#00ff88", lw=1.5, ms=5,
            label="std(Ω) / max")
    ax.plot(Ks, norm(std_lz), "s-", color="magenta", lw=1.5, ms=5,
            label="std(LZ) / max")
    ax.plot(Ks, norm(mean_lz), "^--", color="yellow", lw=1.0, ms=4,
            alpha=0.6, label="mean(LZ) / max")
    ax.set_xlabel("K", color="white")
    ax.set_ylabel("normalisé", color="white")
    ax.legend(facecolor="black", labelcolor="white", edgecolor="white", fontsize=8)
    ax.set_title(f"Comparaison normalisée — {verdict}", color="white", fontsize=10)

    for ax in axes.flat:
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_color("white")

    fig.patch.set_facecolor("black")
    fig.suptitle(f"χ_info vs LZ-complexity — RBN N={N} | "
                 f"pont info-théorie/computation",
                 color="white", y=0.995)
    plt.tight_layout()
    png = os.path.join(outdir, "lz_vs_chiinfo.png")
    plt.savefig(png, dpi=130, facecolor="black", bbox_inches="tight")
    print(f"\nsaved {png}")

    out = {
        "N": N, "p": p, "K_values": list(K_values),
        "mean_lz": mean_lz.tolist(), "std_lz": std_lz.tolist(),
        "mean_omega": mean_om.tolist(),
        "chi_info_std_omega": std_om_nets.tolist(),
        "abs_mean_omega": abs_om.tolist(), "cv_omega": cv_om.tolist(),
        "mean_derrida": mean_derrida.tolist(),
        "mean_frozen": mean_frozen.tolist(),
        "pearson_chiinfo_stdlz": float(pearson_chiinfo_lz_std),
        "pearson_chiinfo_meanlz": float(pearson_chiinfo_lz_mean),
        "pearson_absmean_omega_meanlz": float(pearson_omega_lz_mean),
        "K_peak_chiinfo": int(K_peak_chiinfo),
        "K_peak_stdlz": int(K_peak_stdlz),
        "K_peak_meanlz": int(K_peak_meanlz),
        "verdict": verdict,
        "lz_per_network": {str(K): lz_by_K[K] for K in K_values},
        "omega_per_network": {str(K): om_by_K[K] for K in K_values},
    }
    with open(os.path.join(outdir, "lz_vs_chiinfo.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {os.path.join(outdir, 'lz_vs_chiinfo.json')}")
    return out


if __name__ == "__main__":
    run()
