"""#9 — Finite-size scaling : CV(Ω) vs χ pour localiser Tc.

Théorie : le pic de χ se déplace T*(L) − Tc ~ L^{−1/ν} (ν=1 Ising 2D), donc
χ converge vers Tc par le haut en ~1/L. Question : CV(Ω) = std(Ω)/⟨|Ω|⟩
converge-t-il PLUS VITE (shift plus petit), validant son avantage finite-size ?

Pour chaque L : sweep T fin autour de Tc, n_real réalisations indépendantes/T.
Pic localisé par interpolation parabolique sub-grille (résolution < spacing).
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
from xi.edge import susceptibility


def parabolic_peak(x, y):
    """Localisation sub-grille du maximum par parabole sur 3 points (spacing uniforme)."""
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
    L_list=(16, 24, 32, 48, 64),
    T_min=2.05,
    T_max=2.55,
    n_T=21,
    n_real=12,
    block_size=2,
    seed0=31,
    outdir=None,
):
    if outdir is None:
        outdir = os.path.dirname(os.path.abspath(__file__))
    Tc = 2.0 / np.log(1.0 + np.sqrt(2.0))
    T_values = np.linspace(T_min, T_max, n_T)

    print(f"Finite-size scaling | L={list(L_list)} | {n_real} réal/T | Tc={Tc:.4f}\n")
    rng_master = np.random.default_rng(seed0)

    peak_chi_L = {}
    peak_cv_L = {}
    curves = {}

    t0 = time.time()
    for L in L_list:
        # équilibration plus longue pour grands L
        n_equil = max(1000, 20 * L)
        n_sample = 120
        sample_gap = 4
        chi = np.zeros(n_T)
        cv = np.zeros(n_T)
        for i, T in enumerate(T_values):
            o_list, chi_list = [], []
            for r in range(n_real):
                seed = int(rng_master.integers(0, 2 ** 31))
                snaps = simulate(L=L, T=float(T), n_equil=n_equil,
                                 n_sample=n_sample, sample_gap=sample_gap, seed=seed)
                mags = np.array([s.mean() for s in snaps])
                chi_list.append(susceptibility(mags, T, L * L))
                blocks = np.concatenate([rg_blocks(s, block_size) for s in snaps], axis=0)
                b01 = ((blocks + 1) // 2).astype(np.int8)
                o_list.append(o_information(b01))
            chi[i] = np.mean(chi_list)
            o_arr = np.array(o_list)
            cv[i] = o_arr.std() / (np.abs(o_arr).mean() + 1e-9)
        pc = parabolic_peak(T_values, chi)
        pcv = parabolic_peak(T_values, cv)
        peak_chi_L[L] = pc
        peak_cv_L[L] = pcv
        curves[L] = {"chi": chi.tolist(), "cv": cv.tolist()}
        print(f"  L={L:>3} | peak χ T*={pc:.4f} (Δ={pc-Tc:+.4f}) | "
              f"peak CV T*={pcv:.4f} (Δ={pcv-Tc:+.4f})")
    print(f"\nsim {time.time()-t0:.1f}s")

    Ls = np.array(L_list, dtype=float)
    dchi = np.array([peak_chi_L[L] - Tc for L in L_list])
    dcv = np.array([peak_cv_L[L] - Tc for L in L_list])

    print(f"\n{'L':>4} | {'Δχ':>8} | {'ΔCV':>8}")
    for L in L_list:
        print(f"{L:>4} | {peak_chi_L[L]-Tc:>+8.4f} | {peak_cv_L[L]-Tc:>+8.4f}")
    print(f"\nmean |Δχ| = {np.abs(dchi).mean():.4f} | mean |ΔCV| = {np.abs(dcv).mean():.4f}")
    better = np.abs(dcv).mean() < np.abs(dchi).mean()
    print(f"CV(Ω) shift plus petit que χ : "
          f"{'✓ OUI — avantage finite-size confirmé' if better else '✗ non'}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.plot(Ls, np.abs(dchi), "o-", color="#00ffff", lw=2, label="|Δ| χ")
    ax.plot(Ls, np.abs(dcv), "s-", color="#ff00ff", lw=2, label="|Δ| CV(Ω)")
    # référence L^-1 (ν=1 Ising)
    ref = np.abs(dchi)[0] * Ls[0] / Ls
    ax.plot(Ls, ref, ":", color="#888888", lw=1, label="∝ L⁻¹ (ν=1)")
    ax.set_xlabel("L", color="white")
    ax.set_ylabel("|T*(L) − Tc|", color="white")
    ax.set_title("shift finite-size du pic", color="white")
    ax.legend(facecolor="black", labelcolor="white", edgecolor="white")
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax = axes[1]
    for L in L_list:
        cvc = np.array(curves[L]["cv"])
        ax.plot(T_values, cvc / cvc.max(), "o-", lw=1.2, ms=3, label=f"L={L}")
    ax.axvline(Tc, color="#ff3355", ls="--", label=f"Tc={Tc:.3f}")
    ax.set_xlabel("T", color="white")
    ax.set_ylabel("CV(Ω) normalisé", color="white")
    ax.set_title("CV(Ω) par L (pic → Tc ?)", color="white")
    ax.legend(facecolor="black", labelcolor="white", edgecolor="white", fontsize=8)

    for ax in axes:
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_color("white")
    fig.patch.set_facecolor("black")
    fig.suptitle("Ising finite-size scaling — CV(Ω) vs χ", color="white")
    plt.tight_layout()
    png = os.path.join(outdir, "ising_scaling_L.png")
    plt.savefig(png, dpi=130, facecolor="black", bbox_inches="tight")
    print(f"saved {png}")

    out = {
        "Tc": Tc, "L_list": list(L_list), "T": T_values.tolist(),
        "peak_chi": {str(L): float(peak_chi_L[L]) for L in L_list},
        "peak_cv": {str(L): float(peak_cv_L[L]) for L in L_list},
        "delta_chi": dchi.tolist(), "delta_cv": dcv.tolist(),
        "curves": {str(L): curves[L] for L in L_list},
        "cv_better": bool(better),
    }
    with open(os.path.join(outdir, "ising_scaling_L.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {os.path.join(outdir, 'ising_scaling_L.json')}")
    return out


if __name__ == "__main__":
    run()
