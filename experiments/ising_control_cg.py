"""Test décisif : majority-vote spatial est-il neutre ?

Protocole : générer spins L×L i.i.d. (zéro interaction spatiale), passer
dans blocks_at_scale + coarse_grain_stack pour k=0..k_max, mesurer TC(k).

Si TC(k) reste plat (γ≈0) sur du bruit → CG spatial neutre → Stade B Ising
γ→0 à Tc est un VRAI signal de scale-invariance.

Si TC(k) dérive (γ≠0 sur bruit) → CG spatial injecte/détruit structure →
Stade B contaminé comme Kuramoto temporel.

Variantes testées :
  - spins ±1 P(+1)=0.5 (uniform)
  - spins ±1 P(+1)=0.7 (biased, magnétisation moyenne ≠ 0)
"""
import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xi.hierarchy import blocks_at_scale, coarse_grain_stack
from xi.interaction import total_correlation, o_information


def make_random_spins(L, p_plus=0.5, seed=0):
    rng = np.random.default_rng(seed)
    return np.where(rng.random((L, L)) < p_plus, 1, -1).astype(np.int8)


def compute_tc_at_scales(snapshots, k_max, block_size=2):
    tcs = np.full(k_max + 1, np.nan)
    abs_os = np.full(k_max + 1, np.nan)
    n_blocks = np.zeros(k_max + 1, dtype=int)
    for k in range(k_max + 1):
        blocks = blocks_at_scale(snapshots, k=k, block_size=block_size,
                                 rule="majority")
        if blocks is None or blocks.shape[0] < 16:
            continue
        blocks01 = ((blocks + 1) // 2).astype(np.int8)
        tcs[k] = total_correlation(blocks01)
        abs_os[k] = abs(o_information(blocks01))
        n_blocks[k] = blocks.shape[0]
    return tcs, abs_os, n_blocks


def run(L=32, n_snap=250, k_max=3, outdir=None, seed=0):
    if outdir is None:
        outdir = os.path.dirname(os.path.abspath(__file__))

    print(f"L={L} k_max={k_max} n_snap={n_snap}")
    print()

    variants = {
        "uniform_0.5": 0.5,
        "biased_0.7": 0.7,
        "biased_0.3": 0.3,
    }

    results = {}
    for name, p in variants.items():
        snaps = [make_random_spins(L, p_plus=p, seed=seed * 1000 + i)
                 for i in range(n_snap)]
        tcs, abs_os, n_blocks = compute_tc_at_scales(snaps, k_max=k_max)

        # γ régression log TC vs k
        valid = ~np.isnan(tcs) & (tcs > 1e-9)
        if valid.sum() >= 2:
            xs = np.arange(k_max + 1)[valid]
            slope, intercept = np.polyfit(xs, np.log(tcs[valid]), 1)
        else:
            slope = np.nan

        ratio = tcs[-1] / tcs[0] if tcs[0] > 1e-9 else np.inf

        print(f"=== variant {name} (P(+1)={p}) ===")
        for k in range(k_max + 1):
            print(f"  k={k}  TC={tcs[k]:.6f}  |Ω|={abs_os[k]:.6f}  "
                  f"n_blocks={n_blocks[k]}")
        print(f"  γ = {slope:+.3f}    TC(k_max)/TC(0) = {ratio:.2f}")
        print()

        results[name] = {
            "p_plus": p,
            "tc": tcs.tolist(),
            "abs_o": abs_os.tolist(),
            "n_blocks": n_blocks.tolist(),
            "gamma": float(slope),
            "ratio": float(ratio),
        }

    # Verdict
    print("=" * 60)
    max_ratio = max(abs(r["ratio"] - 1.0) for r in results.values())
    max_gamma = max(abs(r["gamma"]) for r in results.values())
    print(f"max |ratio − 1| = {max_ratio:.3f}")
    print(f"max |γ| sur bruit = {max_gamma:.3f}")
    if max_ratio < 0.5 and max_gamma < 0.3:
        verdict = "NEUTRE"
        print(f"✓ {verdict} — CG spatial majority-vote est neutre sur bruit.")
        print("  → Stade B Ising γ→0 à Tc VALIDÉ rétroactivement.")
    elif max_ratio < 2.0 and max_gamma < 1.0:
        verdict = "FAIBLEMENT BIAISÉ"
        print(f"⚠ {verdict} — biais modéré, à comparer avec signal à Tc.")
    else:
        verdict = "CONTAMINÉ"
        print(f"✗ {verdict} — CG injecte structure sur bruit.")
        print("  → Stade B Ising γ→0 à Tc à RÉINTERPRÉTER.")
    print("=" * 60)

    # Comparaison avec signal Tc (chargé depuis Stade B)
    stadeB_path = os.path.join(outdir, "ising_xi_stadeB.json")
    if os.path.exists(stadeB_path):
        stB = json.load(open(stadeB_path))
        T = np.array(stB["T"])
        idx_Tc = np.argmin(np.abs(T - 2.269))
        gamma_Tc = stB["gamma_T"][idx_Tc]
        tc_at_Tc = stB["tc_hier"][idx_Tc]
        print()
        print(f"Comparaison signal Tc (Stade B run précédent):")
        print(f"  γ_T à T≈Tc = {gamma_Tc:+.3f}")
        print(f"  TC à T≈Tc : k=0..3 = {[f'{x:.3f}' for x in tc_at_Tc]}")
        print(f"  TC noise ratio = {max_ratio:.2f}  vs signal TC range "
              f"= {max(tc_at_Tc)/min(tc_at_Tc):.2f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    cmap = ["#00ffff", "#ff00ff", "#ffaa00"]
    ax = axes[0]
    for (name, r), col in zip(results.items(), cmap):
        ax.plot(range(k_max + 1), r["tc"], "o-", color=col, lw=1.5, ms=6,
                label=f"{name} γ={r['gamma']:+.3f}")
    ax.set_xlabel("k (niveau CG spatial)", color="white")
    ax.set_ylabel("TC (spins i.i.d., zéro interaction)", color="white")
    ax.set_title(f"Contrôle CG spatial Ising — verdict: {verdict}", color="white")
    ax.legend(facecolor="black", edgecolor="white", labelcolor="white")
    ax.set_facecolor("black")
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_color("white")

    # Compare with Stade B signal at Tc
    ax = axes[1]
    if os.path.exists(stadeB_path):
        ax.plot(range(k_max + 1), tc_at_Tc, "s-", color="#ff3355", lw=2, ms=8,
                label=f"Ising à T≈Tc (signal, γ={gamma_Tc:+.3f})")
    for (name, r), col in zip(results.items(), cmap):
        ax.plot(range(k_max + 1), r["tc"], "o--", color=col, lw=1.0, ms=5,
                alpha=0.7, label=f"i.i.d. {name}")
    ax.set_xlabel("k", color="white")
    ax.set_ylabel("TC", color="white")
    ax.set_title("Signal Tc vs bruit i.i.d.", color="white")
    ax.legend(facecolor="black", edgecolor="white", labelcolor="white", fontsize=8)
    ax.set_facecolor("black")
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_color("white")

    fig.patch.set_facecolor("black")
    plt.tight_layout()
    png = os.path.join(outdir, "ising_control_cg.png")
    plt.savefig(png, dpi=130, facecolor="black", bbox_inches="tight")
    print(f"\nsaved {png}")

    out = {
        "L": L, "n_snap": n_snap, "k_max": k_max,
        "variants": results,
        "verdict": verdict,
        "max_ratio": max_ratio,
        "max_gamma_noise": max_gamma,
    }
    jpath = os.path.join(outdir, "ising_control_cg.json")
    with open(jpath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {jpath}")
    return out


if __name__ == "__main__":
    run()
