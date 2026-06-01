import sys
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from systems.ising2d import simulate
from xi.observable import xi_hierarchical


def run(
    L=32,
    T_values=None,
    n_equil=1500,
    n_sample=400,
    sample_gap=4,
    k_max=3,
    alpha=0.0,
    epsilon=0.3,
    seed=42,
    outdir=None,
):
    if outdir is None:
        outdir = os.path.dirname(os.path.abspath(__file__))
    if T_values is None:
        # zoom autour Tc + queues
        T_values = np.concatenate([
            np.linspace(1.5, 2.1, 7),
            np.linspace(2.15, 2.45, 13),
            np.linspace(2.5, 3.5, 6),
        ])
    Tc = 2.0 / np.log(1.0 + np.sqrt(2.0))

    print(f"L={L} k_max={k_max} | {len(T_values)} T values | Tc={Tc:.4f}")
    print(f"n_equil={n_equil} n_sample={n_sample}")

    t0 = time.time()
    snaps = {}
    for T in T_values:
        ts = time.time()
        snaps[float(T)] = simulate(
            L=L, T=float(T),
            n_equil=n_equil, n_sample=n_sample,
            sample_gap=sample_gap, seed=seed,
        )
        print(f"  T={T:.3f} | {time.time()-ts:.1f}s")
    print(f"sim total {time.time()-t0:.1f}s")

    res = xi_hierarchical(
        snaps, T_values, L=L,
        k_max=k_max, alpha=alpha, epsilon=epsilon,
    )

    print()
    print(f"{'T':>6} | " + " | ".join([f"TC(k={k})" for k in range(k_max + 1)])
          + " | γ_T   | mask | sat_k")
    print("-" * (6 + 12 * (k_max + 1) + 30))
    for i, T in enumerate(T_values):
        row = f"{T:>6.3f} | "
        row += " | ".join([f"{res['tc_hier'][i,k]:>7.4f}" for k in range(k_max + 1)])
        row += f" | {res['gamma_T'][i]:+.3f} | {res['mask'][i]} | {res['saturation_k'][i]}"
        print(row)

    # Plot
    fig, axes = plt.subplots(4, 1, figsize=(9, 11))
    cmap = plt.get_cmap("viridis", k_max + 1)

    ax = axes[0]
    for k in range(k_max + 1):
        ax.plot(T_values, res["tc_hier"][:, k], "o-",
                color=cmap(k), label=f"k={k}", lw=1.4, ms=4)
    ax.axvline(Tc, color="#ff3355", ls="--", lw=1, label=f"Tc={Tc:.3f}")
    ax.set_ylabel("TC(T, k)", color="white")
    ax.legend(facecolor="black", edgecolor="white", labelcolor="white", fontsize=8, ncol=2)

    ax = axes[1]
    ax.plot(T_values, res["gamma_T"], "o-", color="#ff8800", lw=1.5, ms=5)
    ax.axhline(0, color="white", ls=":", lw=0.6)
    ax.axvline(Tc, color="#ff3355", ls="--", lw=1)
    ax.set_ylabel("γ_T (slope log TC vs k)", color="white")
    ax.set_title("Self-similarity RG : γ → 0 à Tc, γ < 0 hors-Tc", color="white", fontsize=10)

    ax = axes[2]
    ax.plot(T_values, res["xi_hier"], "o-", color="#00ff88", lw=1.5, ms=5,
            label=f"Ξ_hier (α={alpha})")
    ax.plot(T_values, res["chi"] / res["chi"].max() * res["xi_hier"].max(),
            "s--", color="cyan", lw=1, ms=3, label="χ (rescaled)")
    ax.axvline(Tc, color="#ff3355", ls="--", lw=1)
    ax.set_ylabel("Ξ_hier (masked)", color="white")
    ax.legend(facecolor="black", edgecolor="white", labelcolor="white", fontsize=8)

    ax = axes[3]
    sat = res["saturation_k"].copy()
    ax.plot(T_values, sat, "o-", color="#ff00ff", lw=1.5, ms=5)
    ax.axvline(Tc, color="#ff3355", ls="--", lw=1)
    ax.set_ylabel("saturation_k (TC < 10% max)", color="white")
    ax.set_xlabel("T", color="white")

    for ax in axes:
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        for s in ax.spines.values():
            s.set_color("white")

    fig.patch.set_facecolor("black")
    fig.suptitle(f"Ξ Stade B — hiérarchie RG Ising 2D L={L} k_max={k_max}",
                 color="white", y=0.995)
    plt.tight_layout()

    png = os.path.join(outdir, "ising_xi_stadeB.png")
    plt.savefig(png, dpi=130, facecolor="black", bbox_inches="tight")
    print(f"\nsaved {png}")

    out_json = {
        "L": L, "Tc": Tc, "epsilon": epsilon, "alpha": alpha, "k_max": k_max,
        "T": np.asarray(T_values).tolist(),
        "chi": res["chi"].tolist(),
        "mask": res["mask"].tolist(),
        "tc_hier": res["tc_hier"].tolist(),
        "o_hier": res["o_hier"].tolist(),
        "abs_o_hier": res["abs_o_hier"].tolist(),
        "gamma_T": res["gamma_T"].tolist(),
        "xi_hier": res["xi_hier"].tolist(),
        "saturation_k": res["saturation_k"].tolist(),
        "n_blocks_hier": res["n_blocks_hier"].tolist(),
    }
    jpath = os.path.join(outdir, "ising_xi_stadeB.json")
    with open(jpath, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"saved {jpath}")

    return res


if __name__ == "__main__":
    run()
