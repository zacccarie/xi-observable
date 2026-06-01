"""#10 — CV régularisé : un floor sauve-t-il CV(Ω) du parasite basse-activité ?

#9 a montré : CV(Ω) en range serré [2.05,2.55] a un pic parasite à basse T où
⟨|Ω|⟩→0 (division explose). Mais sur range large le parasite est absent.

Test : CV_reg(Ω) = std(Ω) / (⟨|Ω|⟩ + floor). Stocke les Ω individuels pour
recalculer post-hoc avec différents floors (relatifs au max de ⟨|Ω|⟩).

Question : un floor relatif unique élimine-t-il le parasite et colle CV_reg
à Tc pour L=32 ET L=64 (robustesse finite-size) ?
"""
import sys
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from systems.ising2d import simulate
from xi.interaction import o_information
from xi.hierarchy import rg_blocks


def parabolic_peak(x, y):
    y = np.asarray(y)
    i = int(np.argmax(y))
    if i == 0 or i == len(y) - 1:
        return x[i]
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    denom = y0 - 2 * y1 + y2
    if abs(denom) < 1e-12:
        return x[i]
    dx = x[i + 1] - x[i]
    return x[i] - 0.5 * dx * (y2 - y0) / denom


def run(
    L_list=(32, 64),
    T_min=2.0,
    T_max=2.6,
    n_T=25,
    n_real=16,
    block_size=2,
    floors=(0.0, 0.02, 0.05, 0.1, 0.2, 0.3),
    seed0=41,
    outdir=None,
):
    if outdir is None:
        outdir = os.path.dirname(os.path.abspath(__file__))
    Tc = 2.0 / np.log(1.0 + np.sqrt(2.0))
    T_values = np.linspace(T_min, T_max, n_T)
    rng_master = np.random.default_rng(seed0)

    print(f"CV régularisé | L={list(L_list)} | T[{T_min},{T_max}] {n_T}pts | "
          f"{n_real} réal | Tc={Tc:.4f}\n")

    results = {}
    t0 = time.time()
    for L in L_list:
        n_equil = max(1000, 20 * L)
        std_o = np.zeros(n_T)
        absm_o = np.zeros(n_T)
        for i, T in enumerate(T_values):
            o_list = []
            for r in range(n_real):
                seed = int(rng_master.integers(0, 2 ** 31))
                snaps = simulate(L=L, T=float(T), n_equil=n_equil,
                                 n_sample=120, sample_gap=4, seed=seed)
                blocks = np.concatenate([rg_blocks(s, block_size) for s in snaps], axis=0)
                b01 = ((blocks + 1) // 2).astype(np.int8)
                o_list.append(o_information(b01))
            o_arr = np.array(o_list)
            std_o[i] = o_arr.std()
            absm_o[i] = np.abs(o_arr).mean()
        results[L] = {"std": std_o, "absm": absm_o}
        print(f"  L={L} simulé")
    print(f"sim {time.time()-t0:.1f}s\n")

    # Test floors
    print(f"{'L':>4} | floor | peak CV_reg (Δ vs Tc)")
    print("-" * 45)
    peak_table = {L: {} for L in L_list}
    for L in L_list:
        std_o = results[L]["std"]
        absm_o = results[L]["absm"]
        fmax = absm_o.max()
        for fl in floors:
            cv = std_o / (absm_o + fl * fmax + 1e-12)
            pk = parabolic_peak(T_values, cv)
            peak_table[L][fl] = pk
            print(f"{L:>4} | {fl:>5.2f} | {pk:.4f} (Δ={pk-Tc:+.4f})")
        print()

    # Meilleur floor commun : minimise max|Δ| sur les deux L
    best_floor, best_score = None, 1e9
    for fl in floors:
        score = max(abs(peak_table[L][fl] - Tc) for L in L_list)
        if score < best_score:
            best_score, best_floor = score, fl
    print(f"Meilleur floor commun = {best_floor:.2f} "
          f"(max|Δ| = {best_score:.4f} sur L={list(L_list)})")
    saved = best_score < 0.05
    print(f"\nFloor sauve CV en range serré : "
          f"{'✓ OUI — CV_reg robuste, universalité réhabilitée' if saved else '✗ non, reste fragile'}")

    # Plot
    fig, axes = plt.subplots(1, len(L_list) + 1, figsize=(5 * (len(L_list) + 1), 4.5))
    for ax, L in zip(axes[:len(L_list)], L_list):
        std_o = results[L]["std"]; absm_o = results[L]["absm"]; fmax = absm_o.max()
        for fl in [0.0, best_floor]:
            cv = std_o / (absm_o + fl * fmax + 1e-12)
            ax.plot(T_values, cv / cv.max(), "o-", lw=1.5, ms=3,
                    label=f"floor={fl:.2f}")
        ax.axvline(Tc, color="#ff3355", ls="--", label=f"Tc")
        ax.set_xlabel("T", color="white")
        ax.set_ylabel("CV_reg normalisé", color="white")
        ax.set_title(f"L={L}", color="white")
        ax.legend(facecolor="black", labelcolor="white", edgecolor="white", fontsize=8)

    ax = axes[-1]
    for L in L_list:
        dl = [peak_table[L][fl] - Tc for fl in floors]
        ax.plot(floors, dl, "o-", lw=2, label=f"L={L}")
    ax.axhline(0, color="white", ls=":", lw=0.8)
    ax.set_xlabel("floor (× max⟨|Ω|⟩)", color="white")
    ax.set_ylabel("Δ peak vs Tc", color="white")
    ax.set_title("convergence vers Tc", color="white")
    ax.legend(facecolor="black", labelcolor="white", edgecolor="white", fontsize=8)

    for ax in axes:
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_color("white")
    fig.patch.set_facecolor("black")
    fig.suptitle("Ising CV régularisé — floor vs parasite basse-activité", color="white")
    plt.tight_layout()
    png = os.path.join(outdir, "ising_floor_test.png")
    plt.savefig(png, dpi=130, facecolor="black", bbox_inches="tight")
    print(f"saved {png}")

    out = {
        "Tc": Tc, "L_list": list(L_list), "T": T_values.tolist(), "floors": list(floors),
        "std": {str(L): results[L]["std"].tolist() for L in L_list},
        "absm": {str(L): results[L]["absm"].tolist() for L in L_list},
        "peaks": {str(L): {str(fl): float(peak_table[L][fl]) for fl in floors} for L in L_list},
        "best_floor": float(best_floor), "best_score": float(best_score), "saved": bool(saved),
    }
    with open(os.path.join(outdir, "ising_floor_test.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {os.path.join(outdir, 'ising_floor_test.json')}")
    return out


if __name__ == "__main__":
    run()
