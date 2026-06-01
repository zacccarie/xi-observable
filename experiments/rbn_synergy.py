"""RBN Kauffman — test si Ω < 0 existe.

Sweep K ∈ {1..5} à p=0.5. K_c = 2 (Derrida slope = 1).
Plusieurs réalisations de réseau par K → moyennes.
Sur attractor ensemble : tire subsets aléatoires de subset_size nœuds,
calcule TC et Ω. Cherche régime Ω < 0 (synergie-dominée).

Falsification : si Ω > 0 partout comme Ising → cadre redondance-universel.
Validation : si Ω < 0 dans certains K → régime synergie réel = tétrade
permet d'identifier deux phases informationnelles distinctes.
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
from xi.interaction import o_information, total_correlation


def tc_omega_random_subsets(ensemble, subset_size=4, n_subsets=300, rng=None):
    """Renvoie TC, Ω moyens (et std) sur n_subsets sous-ensembles aléatoires."""
    if rng is None:
        rng = np.random.default_rng(0)
    M, N = ensemble.shape
    if subset_size > N:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    tcs, omegas = [], []
    for _ in range(n_subsets):
        idx = rng.choice(N, size=subset_size, replace=False)
        sub = ensemble[:, idx].astype(np.int8)
        tcs.append(total_correlation(sub))
        omegas.append(o_information(sub))
    tcs = np.array(tcs)
    omegas = np.array(omegas)
    return (tcs.mean(), tcs.std(),
            omegas.mean(), omegas.std(),
            (omegas < 0).mean())  # fraction de subsets synergie


def run(
    N=64,
    K_values=(1, 2, 3, 4, 5),
    p=0.5,
    n_networks=8,
    n_init=3000,
    t_equil=300,
    subset_size=4,
    n_subsets=400,
    seed=42,
    outdir=None,
):
    if outdir is None:
        outdir = os.path.dirname(os.path.abspath(__file__))

    print(f"N={N}  p={p}  K_values={K_values}")
    print(f"n_networks={n_networks}  n_init={n_init}  subset_size={subset_size}")
    print()

    results = {K: {"derrida": [], "frozen_frac": [], "activity": [],
                   "tc_mean": [], "tc_std": [],
                   "omega_mean": [], "omega_std": [],
                   "frac_synergy": []}
               for K in K_values}

    t0 = time.time()
    rng = np.random.default_rng(seed)
    for K in K_values:
        ts = time.time()
        for net in range(n_networks):
            inputs, tables = make_rbn(N, K, p, seed=seed * 1000 + K * 100 + net)
            dslope = derrida_slope(inputs, tables, K, N,
                                   n_pairs=1500, seed=seed + net)
            ens = attractor_ensemble(inputs, tables, K, N,
                                     n_init=n_init, t_equil=t_equil,
                                     seed=seed + net + 7)
            ff = frozen_fraction(ens)
            act = activity(ens)
            tc_m, tc_s, om_m, om_s, frac_syn = tc_omega_random_subsets(
                ens, subset_size=subset_size, n_subsets=n_subsets, rng=rng,
            )
            results[K]["derrida"].append(dslope)
            results[K]["frozen_frac"].append(ff)
            results[K]["activity"].append(act)
            results[K]["tc_mean"].append(tc_m)
            results[K]["tc_std"].append(tc_s)
            results[K]["omega_mean"].append(om_m)
            results[K]["omega_std"].append(om_s)
            results[K]["frac_synergy"].append(frac_syn)
        elapsed = time.time() - ts
        d_mean = np.mean(results[K]["derrida"])
        ff_mean = np.mean(results[K]["frozen_frac"])
        om_mean = np.mean(results[K]["omega_mean"])
        tc_mean = np.mean(results[K]["tc_mean"])
        fsyn = np.mean(results[K]["frac_synergy"])
        print(f"  K={K}  derrida={d_mean:.3f}  frozen={ff_mean:.3f}  "
              f"TC={tc_mean:.4f}  Ω={om_mean:+.4f}  frac_synergy={fsyn:.2%}  "
              f"({elapsed:.1f}s)")
    print(f"\ntotal {time.time()-t0:.1f}s")

    # Synthèse
    print()
    print(f"{'K':>4} | {'derrida':>9} | {'frozen':>8} | {'act':>6} | "
          f"{'TC':>8} | {'Ω':>9} | {'frac_syn':>8}")
    print("-" * 70)
    summary = {}
    for K in K_values:
        d = np.mean(results[K]["derrida"])
        d_std = np.std(results[K]["derrida"])
        ff = np.mean(results[K]["frozen_frac"])
        act = np.mean(results[K]["activity"])
        tc = np.mean(results[K]["tc_mean"])
        om = np.mean(results[K]["omega_mean"])
        om_std = np.std(results[K]["omega_mean"])
        fsyn = np.mean(results[K]["frac_synergy"])
        print(f"{K:>4} | {d:.3f}±{d_std:.2f} | {ff:>8.3f} | {act:>6.3f} | "
              f"{tc:>8.4f} | {om:+8.4f} | {fsyn:>7.2%}")
        summary[K] = {"derrida": float(d), "derrida_std": float(d_std),
                      "frozen": float(ff), "activity": float(act),
                      "tc": float(tc), "omega": float(om),
                      "omega_std": float(om_std), "frac_synergy": float(fsyn)}

    # Verdict
    print()
    print("=" * 60)
    any_synergy = any(s["omega"] < -0.01 for s in summary.values())
    any_high_synfrac = any(s["frac_synergy"] > 0.3 for s in summary.values())
    if any_synergy:
        print("✓ RÉGIME SYNERGIE EXISTE : Ω moyen < 0 trouvé sur au moins un K.")
    elif any_high_synfrac:
        print("⚠ Synergie partielle : Ω moyen ≥ 0 mais frac_synergy > 30% à certains K.")
    else:
        print("✗ Redondance pure partout (comme Ising) — tétrade régime unique.")
    print("=" * 60)

    # Plot
    Ks = list(K_values)
    derrida_mean = [summary[K]["derrida"] for K in Ks]
    derrida_std = [summary[K]["derrida_std"] for K in Ks]
    tc_mean = [summary[K]["tc"] for K in Ks]
    omega_mean = [summary[K]["omega"] for K in Ks]
    omega_std = [summary[K]["omega_std"] for K in Ks]
    frozen = [summary[K]["frozen"] for K in Ks]
    fsyn = [summary[K]["frac_synergy"] for K in Ks]

    fig, axes = plt.subplots(4, 1, figsize=(8, 11))

    ax = axes[0]
    ax.errorbar(Ks, derrida_mean, yerr=derrida_std,
                marker="o", color="cyan", capsize=3, lw=1.5)
    ax.axhline(1.0, color="#ff3355", ls="--", label="K_c (slope=1)")
    ax.set_ylabel("Derrida slope", color="white")
    ax.set_title("Order parameter — ordered/chaotic transition", color="white", fontsize=10)
    ax.legend(facecolor="black", edgecolor="white", labelcolor="white")

    ax = axes[1]
    ax.plot(Ks, frozen, "o-", color="#00ff88", lw=1.5, label="frozen fraction")
    ax.set_ylabel("frozen fraction", color="white")
    ax.legend(facecolor="black", edgecolor="white", labelcolor="white")

    ax = axes[2]
    ax.errorbar(Ks, omega_mean, yerr=omega_std,
                marker="s", color="magenta", capsize=3, lw=1.5, label="Ω (O-info)")
    ax.plot(Ks, tc_mean, "^-", color="yellow", lw=1.2, label="TC")
    ax.axhline(0, color="white", ls=":", lw=0.6)
    ax.set_ylabel("interaction (Ω, TC)", color="white")
    ax.legend(facecolor="black", edgecolor="white", labelcolor="white")
    ax.set_title("Régime synergie (Ω<0) vs redondance (Ω>0)", color="white", fontsize=10)

    ax = axes[3]
    ax.plot(Ks, fsyn, "o-", color="#ff8800", lw=1.5)
    ax.axhline(0.5, color="white", ls=":", lw=0.6, label="50%")
    ax.set_ylabel("fraction subsets Ω<0", color="white")
    ax.set_xlabel("K", color="white")
    ax.legend(facecolor="black", edgecolor="white", labelcolor="white")
    ax.set_ylim(0, 1)

    for ax in axes:
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        for s in ax.spines.values():
            s.set_color("white")

    fig.patch.set_facecolor("black")
    fig.suptitle(f"RBN Kauffman N={N} p={p} — test régime synergie",
                 color="white", y=0.995)
    plt.tight_layout()
    png = os.path.join(outdir, "rbn_synergy.png")
    plt.savefig(png, dpi=130, facecolor="black", bbox_inches="tight")
    print(f"\nsaved {png}")

    out = {"N": N, "p": p, "K_values": list(K_values), "summary": summary,
           "raw": {str(K): results[K] for K in K_values}}
    jpath = os.path.join(outdir, "rbn_synergy.json")
    with open(jpath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {jpath}")
    return out


if __name__ == "__main__":
    run()
