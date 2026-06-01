"""A — synergie ≡ sensibilité au niveau FONCTION booléenne (test exhaustif).

Le découplage RBN (#test décisif) utilisait la parité, maximalement sensible.
L'identification synergie≡chaos est-elle propre à la parité, ou universelle ?

Test indépendant de la dynamique réseau : pour CHAQUE fonction booléenne K=3
(256 énumérables), calculer :
- Ω = O-information de la distribution (x_1..x_K, y) sous inputs uniformes
  → synergie intrinsèque de la fonction (Ω<0 = irréductible)
- s = average sensitivity = Σ_x Σ_i [f(x) ≠ f(x⊕e_i)] / 2^K
  → ingrédient direct du Derrida (criticalité)

Scatter Ω vs s. Coin « Ω<0 (synergie) ET s faible (peu sensible) » :
- VIDE → synergie EXIGE sensibilité → identification universelle (thèse durcie)
- PEUPLÉ → fonctions découplées existent → identification propre à la parité (thèse nuancée)
"""
import sys
import os
import json
from itertools import product
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xi.interaction import o_information


def func_omega(truth_table, K, inputs):
    output = truth_table[:, None]
    samples = np.hstack([inputs, output]).astype(np.int8)
    return o_information(samples)


def avg_sensitivity(truth_table, K):
    flips = 0
    for x in range(2 ** K):
        for i in range(K):
            if truth_table[x] != truth_table[x ^ (1 << i)]:
                flips += 1
    return flips / (2 ** K)


def is_canalizing(truth_table, K):
    """∃ entrée i, valeur v telle que f forcée constante quand x_i=v."""
    for i in range(K):
        for v in (0, 1):
            outs = [truth_table[x] for x in range(2 ** K) if (x >> i) & 1 == v]
            if len(set(outs)) == 1:
                return True
    return False


def is_parity(truth_table, K):
    par = np.array([bin(x).count("1") & 1 for x in range(2 ** K)], dtype=np.int8)
    return np.array_equal(truth_table, par) or np.array_equal(truth_table, par ^ 1)


def run(K=3, outdir=None):
    if outdir is None:
        outdir = os.path.dirname(os.path.abspath(__file__))
    inputs = np.array(list(product([0, 1], repeat=K)), dtype=np.int8)  # (2^K, K)
    n_func = 2 ** (2 ** K)
    print(f"Énumération exhaustive K={K} : {n_func} fonctions booléennes\n")

    omegas, sens, biases, cats = [], [], [], []
    for code in range(n_func):
        tt = np.array([(code >> x) & 1 for x in range(2 ** K)], dtype=np.int8)
        bias = tt.mean()
        # fonctions constantes : Ω indéfini (output déterministe) → skip de l'analyse régime
        om = func_omega(tt, K, inputs)
        s = avg_sensitivity(tt, K)
        if is_parity(tt, K):
            cat = "parity"
        elif is_canalizing(tt, K):
            cat = "canalizing"
        elif bias in (0.0, 1.0):
            cat = "constant"
        else:
            cat = "other"
        omegas.append(om); sens.append(s); biases.append(bias); cats.append(cat)

    omegas = np.array(omegas); sens = np.array(sens)
    biases = np.array(biases); cats = np.array(cats)

    # Coin critique : synergie (Ω<seuil) ET sensibilité faible
    syn_thresh = -0.05
    s_low = np.median(sens)
    corner = (omegas < syn_thresh) & (sens < s_low)
    n_syn = np.sum(omegas < syn_thresh)

    print(f"fonctions synergiques (Ω<{syn_thresh}) : {n_syn}/{n_func}")
    print(f"  dont sensibilité < médiane ({s_low:.2f}) : {corner.sum()} "
          f"← coin découplé")
    print()
    for cat in ["parity", "canalizing", "other", "constant"]:
        m = cats == cat
        if m.sum():
            print(f"  {cat:>11} : n={m.sum():>3} | Ω∈[{omegas[m].min():+.3f},"
                  f"{omegas[m].max():+.3f}] | s∈[{sens[m].min():.2f},{sens[m].max():.2f}]")

    # corrélation Ω vs s sur fonctions non-constantes
    nc = cats != "constant"
    r = np.corrcoef(omegas[nc], sens[nc])[0, 1]
    print(f"\ncorr(Ω, sensibilité) [non-const] = {r:+.3f}")
    if corner.sum() == 0:
        verdict = "✓ coin VIDE — synergie EXIGE sensibilité → identification UNIVERSELLE"
    else:
        verdict = f"✗ coin PEUPLÉ ({corner.sum()}) — synergie sans sensibilité existe → propre à parité"
    print(verdict)

    # Plot
    fig, ax = plt.subplots(figsize=(9, 6))
    colmap = {"parity": "#ff00ff", "canalizing": "#00ffff",
              "other": "#ffff00", "constant": "#666666"}
    for cat in ["constant", "canalizing", "other", "parity"]:
        m = cats == cat
        ax.scatter(sens[m], omegas[m], c=colmap[cat], label=f"{cat} (n={m.sum()})",
                   s=60, alpha=0.7, edgecolors="white", linewidths=0.3)
    ax.axhline(0, color="white", ls=":", lw=0.8)
    ax.axhline(syn_thresh, color="#ff5555", ls="--", lw=1, label=f"seuil synergie")
    ax.axvline(s_low, color="#88ff88", ls="--", lw=1, label=f"médiane sensibilité")
    # zone coin découplé
    ax.axvspan(0, s_low, ymin=0, ymax=0.5, color="#ff0000", alpha=0.08)
    ax.set_xlabel("average sensitivity (→ Derrida / criticalité)", color="white")
    ax.set_ylabel("Ω (synergie si <0)", color="white")
    ax.set_title(f"Synergie vs sensibilité — toutes fonctions K={K} "
                 f"(corr={r:+.2f})", color="white")
    ax.legend(facecolor="black", labelcolor="white", edgecolor="white", fontsize=8)
    ax.set_facecolor("black")
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_color("white")
    fig.patch.set_facecolor("black")
    plt.tight_layout()
    png = os.path.join(outdir, "synergy_sensitivity.png")
    plt.savefig(png, dpi=130, facecolor="black", bbox_inches="tight")
    print(f"\nsaved {png}")

    out = {
        "K": K, "n_func": n_func, "syn_thresh": syn_thresh,
        "n_synergic": int(n_syn), "corner_count": int(corner.sum()),
        "corr_omega_sens": float(r),
        "omega": omegas.tolist(), "sensitivity": sens.tolist(),
        "bias": biases.tolist(), "category": cats.tolist(),
    }
    with open(os.path.join(outdir, "synergy_sensitivity.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {os.path.join(outdir, 'synergy_sensitivity.json')}")
    return out


if __name__ == "__main__":
    run(K=3)
