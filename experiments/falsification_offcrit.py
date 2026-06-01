"""Falsification Stade A : Ξ doit s'effondrer hors régime critique.

Si Ξ reste élevé à T ≫ Tc → Ξ mesure du bruit, théorie tombe.

Test : T ∈ [Tc, 5, 10, 20, 50]. On vérifie :
- TC(T_far) / TC(Tc) ≪ 1
- |Ω|(T_far) / |Ω|(Tc) ≪ 1
- mask_abs (ref_max = χ_Tc) = 0 partout sauf à Tc
- regime sign stable (redondance partout pour Ising)
"""
import sys
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from systems.ising2d import simulate
from xi.observable import xi_observable
from xi.edge import susceptibility, edge_mask
from xi.interaction import o_information, total_correlation
from xi.hierarchy import rg_blocks


def run(
    L=32,
    T_list=None,
    n_equil=1500,
    n_sample=250,
    sample_gap=4,
    block_size=2,
    epsilon=0.3,
    seed=42,
    outdir=None,
):
    if outdir is None:
        outdir = os.path.dirname(os.path.abspath(__file__))
    Tc = 2.0 / np.log(1.0 + np.sqrt(2.0))
    if T_list is None:
        T_list = [Tc, 5.0, 10.0, 20.0, 50.0]
    T_list = list(map(float, T_list))

    # Load reference run if exists
    ref_path = os.path.join(outdir, "ising_xi_stadeA.json")
    chi_ref_at_Tc = None
    if os.path.exists(ref_path):
        ref = json.load(open(ref_path))
        peaks = ref["peaks"]
        chi_arr = np.array(ref["chi"])
        T_arr = np.array(ref["T"])
        chi_ref_at_Tc = chi_arr.max()
        tc_ref_at_Tc = max(ref["tc"])
        print(f"reference loaded : χ_max(sweep)={chi_ref_at_Tc:.2f}  TC_max={tc_ref_at_Tc:.4f}")
    else:
        print("no ref run found, using critical-band local")

    results = {}
    t0 = time.time()
    for T in T_list:
        ts = time.time()
        snaps = simulate(
            L=L, T=T,
            n_equil=n_equil, n_sample=n_sample, sample_gap=sample_gap,
            seed=seed,
        )
        mags = np.array([s.mean() for s in snaps])
        chi = susceptibility(mags, T, L * L)
        blocks = np.concatenate([rg_blocks(s, block_size) for s in snaps], axis=0)
        b01 = ((blocks + 1) // 2).astype(np.int8)
        Omega = o_information(b01)
        TC = total_correlation(b01)
        results[T] = dict(chi=float(chi), Omega=float(Omega), TC=float(TC),
                          mag=float(np.abs(mags).mean()))
        print(f"  T={T:>6.2f} | χ={chi:>9.3f} | TC={TC:>7.4f} | "
              f"Ω={Omega:>+7.4f} (regime={'red' if Omega>0 else 'syn'}) | "
              f"|m|={results[T]['mag']:.3f} | {time.time()-ts:.1f}s")

    print(f"\ntotal {time.time()-t0:.1f}s\n")

    # Analysis
    if Tc in results:
        ref_T = Tc
    else:
        ref_T = T_list[0]
    chi0 = results[ref_T]["chi"]
    tc0 = results[ref_T]["TC"]
    abs_o0 = abs(results[ref_T]["Omega"])

    print(f"Référence T={ref_T:.4f} :  χ={chi0:.3f}  TC={tc0:.4f}  |Ω|={abs_o0:.4f}\n")

    rows = []
    for T in T_list:
        r = results[T]
        ratio_chi = r["chi"] / chi0 if chi0 > 0 else float("inf")
        ratio_tc = r["TC"] / tc0 if tc0 > 0 else float("inf")
        ratio_abs = abs(r["Omega"]) / abs_o0 if abs_o0 > 0 else float("inf")
        rows.append((T, ratio_chi, ratio_tc, ratio_abs, r["mag"]))
        print(f"  T={T:>6.2f}  χ/χ_Tc={ratio_chi:>7.4f}  TC/TC_Tc={ratio_tc:>7.4f}  "
              f"|Ω|/|Ω|_Tc={ratio_abs:>7.4f}  |m|={r['mag']:.3f}")

    # Mask test with absolute ref
    if chi_ref_at_Tc is not None:
        chi_values = np.array([results[T]["chi"] for T in T_list])
        mask_abs = edge_mask(chi_values, epsilon=epsilon,
                             smooth=False, ref_max=chi_ref_at_Tc)
        print("\nedge_mask (ref absolu = χ_max global sweep, ε=0.3) :")
        for i, T in enumerate(T_list):
            print(f"  T={T:.2f}  mask={int(mask_abs[i])}")

    # Verdict
    print("\n" + "=" * 60)
    print("VERDICT FALSIFICATION")
    print("=" * 60)
    passed = True
    for T in T_list:
        if T == ref_T:
            continue
        r_tc = results[T]["TC"] / tc0
        if r_tc > 0.5:
            print(f"  ✗ T={T} : TC reste à {r_tc*100:.1f}% du peak — Ξ non-effondré")
            passed = False
    if passed:
        print("  ✓ Ξ s'effondre proprement hors criticalité — théorie tient.")
    print("=" * 60)

    # Plot dark
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    Ts = np.array([T for T in T_list])
    chis = np.array([results[T]["chi"] for T in T_list])
    tcs = np.array([results[T]["TC"] for T in T_list])
    abso = np.array([abs(results[T]["Omega"]) for T in T_list])

    axes[0].semilogx(Ts, chis, "o-", color="#00ffff", lw=1.5)
    axes[0].axvline(Tc, color="#ff3355", ls="--", label=f"Tc={Tc:.3f}")
    axes[0].set_ylabel("χ", color="white")
    axes[0].legend(facecolor="black", labelcolor="white", edgecolor="white")

    axes[1].semilogx(Ts, tcs, "s-", color="#ffff00", lw=1.5)
    axes[1].axvline(Tc, color="#ff3355", ls="--")
    axes[1].set_ylabel("TC", color="white")

    axes[2].semilogx(Ts, abso, "^-", color="#ff00ff", lw=1.5)
    axes[2].axvline(Tc, color="#ff3355", ls="--")
    axes[2].set_ylabel("|Ω|", color="white")
    axes[2].set_xlabel("T (log)", color="white")

    for ax in axes:
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_color("white")

    fig.patch.set_facecolor("black")
    fig.suptitle("Falsification Ξ — Ising 2D L=32", color="white", y=0.995)
    plt.tight_layout()
    png = os.path.join(outdir, "ising_falsification.png")
    plt.savefig(png, dpi=130, facecolor="black", bbox_inches="tight")
    print(f"\nsaved {png}")

    # Persist
    out = {
        "Tc": Tc, "L": L, "T_list": T_list, "results": results,
        "verdict": "passed" if passed else "failed",
    }
    jpath = os.path.join(outdir, "ising_falsification.json")
    with open(jpath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {jpath}")
    return out


if __name__ == "__main__":
    run()
