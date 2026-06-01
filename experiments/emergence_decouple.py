"""Test décisif : l'émergence (Ω<0) vient-elle de C ou de la criticalité ?

Découple monotonie de C (synergy_frac : parité vs canalisante) et criticalité
(Derrida slope, mesuré) dans RBN à fonctions contrôlées.

Régression Ω ~ a·Derrida + b·synergy_frac :
- b domine → émergence vient de C (hypothèse H, R2)
- a domine → émergence vient de la criticalité
- forte covariance Derrida↔synergy_frac → inséparables (finding profond)
"""
import sys
import os
import json
import time
import itertools
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from systems.rbn import (
    make_rbn_controlled, derrida_slope, attractor_ensemble,
)
from xi.temporal import tc_random_subsets


def run(
    N=60,
    K_list=(3,),
    synergy_fracs=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    n_real=14,
    n_init=2500,
    t_equil=300,
    subset_size=4,
    n_subsets=400,
    seed0=101,
    outdir=None,
):
    if outdir is None:
        outdir = os.path.dirname(os.path.abspath(__file__))

    print(f"Découplage émergence | N={N} | K={list(K_list)} | "
          f"synergy_frac={list(synergy_fracs)} | {n_real} réseaux/cellule\n")
    rng_master = np.random.default_rng(seed0)

    rows = []  # (K, sf_target, sf_real, derrida, omega, abs_o)
    t0 = time.time()
    for K, sf in itertools.product(K_list, synergy_fracs):
        for r in range(n_real):
            seed = int(rng_master.integers(0, 2 ** 31))
            inputs, tables, sf_real = make_rbn_controlled(N, K, synergy_frac=sf, seed=seed)
            d = derrida_slope(inputs, tables, K, N, n_pairs=1500, seed=seed)
            ens = attractor_ensemble(inputs, tables, K, N,
                                     n_init=n_init, t_equil=t_equil, seed=seed)
            _, abs_o, o, _, _ = tc_random_subsets(
                ens, subset_size, n_subsets, np.random.default_rng(seed))
            rows.append((K, sf, sf_real, d, o, abs_o))
    print(f"sim {time.time()-t0:.1f}s ({len(rows)} réseaux)\n")

    arr = np.array(rows, dtype=float)
    Kc_, sf_t, sf_r, derr, om, abso = arr.T

    # Agrégat par synergy_frac (K=premier)
    print(f"{'sf':>5} | {'Derrida':>8} | {'Ω':>9} | {'frac_syn(Ω<0)':>13}")
    print("-" * 45)
    for sf in synergy_fracs:
        m = np.isclose(sf_t, sf)
        print(f"{sf:>5.1f} | {derr[m].mean():>8.3f} | {om[m].mean():>+9.4f} | "
              f"{np.mean(om[m] < 0):>13.2f}")

    # Régression Ω ~ a·Derrida + b·synergy_frac (standardisée)
    def z(x):
        return (x - x.mean()) / (x.std() + 1e-12)
    X = np.column_stack([z(derr), z(sf_r), np.ones(len(om))])
    coef, *_ = np.linalg.lstsq(X, z(om), rcond=None)
    a, b, _ = coef
    # corrélations marginales
    r_derr = np.corrcoef(derr, om)[0, 1]
    r_sf = np.corrcoef(sf_r, om)[0, 1]
    r_confound = np.corrcoef(derr, sf_r)[0, 1]

    print("\n" + "=" * 55)
    print("RÉGRESSION  Ω ~ a·Derrida + b·synergy_frac (standardisée)")
    print("=" * 55)
    print(f"  a (Derrida)      = {a:+.3f}   (corr marg Ω~Derrida = {r_derr:+.3f})")
    print(f"  b (synergy_frac) = {b:+.3f}   (corr marg Ω~sf      = {r_sf:+.3f})")
    print(f"  confound Derrida↔sf = {r_confound:+.3f}")
    print()
    if abs(b) > 1.5 * abs(a):
        verdict = "✓ H CONFIRMÉE : émergence vient de C (synergy_frac domine)"
    elif abs(a) > 1.5 * abs(b):
        verdict = "✗ émergence vient de la criticalité (Derrida domine)"
    else:
        verdict = "~ mixte / inséparable (a≈b)"
    print(f"  {verdict}")
    if abs(r_confound) > 0.8:
        print(f"  ⚠ confound fort ({r_confound:+.2f}) : monotonie et criticalité "
              f"covarient — découplage partiel seulement")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    sc = axes[0].scatter(sf_r, om, c=derr, cmap="viridis", s=30)
    axes[0].axhline(0, color="white", ls=":", lw=0.8)
    axes[0].set_xlabel("synergy_frac réel (C)", color="white")
    axes[0].set_ylabel("Ω", color="white")
    axes[0].set_title("Ω vs irréductibilité de C", color="white")
    cb = plt.colorbar(sc, ax=axes[0]); cb.set_label("Derrida", color="white")
    cb.ax.tick_params(colors="white")

    sc2 = axes[1].scatter(derr, om, c=sf_r, cmap="plasma", s=30)
    axes[1].axhline(0, color="white", ls=":", lw=0.8)
    axes[1].axvline(1.0, color="#ff3355", ls="--", lw=1)
    axes[1].set_xlabel("Derrida (criticalité)", color="white")
    axes[1].set_ylabel("Ω", color="white")
    axes[1].set_title("Ω vs criticalité", color="white")
    cb2 = plt.colorbar(sc2, ax=axes[1]); cb2.set_label("synergy_frac", color="white")
    cb2.ax.tick_params(colors="white")

    axes[2].bar(["a\n(Derrida)", "b\n(synergy_frac)"], [abs(a), abs(b)],
                color=["#00ffff", "#ff00ff"])
    axes[2].set_ylabel("|coef| standardisé", color="white")
    axes[2].set_title("qui prédit Ω ?", color="white")

    for ax in axes:
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_color("white")
    fig.patch.set_facecolor("black")
    fig.suptitle("Émergence : vient-elle de C ou de la criticalité ?", color="white")
    plt.tight_layout()
    png = os.path.join(outdir, "emergence_decouple.png")
    plt.savefig(png, dpi=130, facecolor="black", bbox_inches="tight")
    print(f"\nsaved {png}")

    out = {
        "N": N, "K_list": list(K_list), "synergy_fracs": list(synergy_fracs),
        "coef_a_derrida": float(a), "coef_b_synergy": float(b),
        "corr_omega_derrida": float(r_derr), "corr_omega_sf": float(r_sf),
        "confound_derrida_sf": float(r_confound),
        "rows": arr.tolist(),
    }
    with open(os.path.join(outdir, "emergence_decouple.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {os.path.join(outdir, 'emergence_decouple.json')}")
    return out


if __name__ == "__main__":
    run()
