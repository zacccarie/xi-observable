import sys
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from systems.kuramoto import simulate as kura_sim, chi_from_r
from xi.observable import xi_hierarchical


def run(
    N=64,
    sigma_omega=1.0,
    K_values=None,
    k_max=3,
    alpha=0.0,
    epsilon=0.3,
    seed=42,
    outdir=None,
):
    if outdir is None:
        outdir = os.path.dirname(os.path.abspath(__file__))
    if K_values is None:
        K_values = np.concatenate([
            np.linspace(0.4, 1.2, 5),
            np.linspace(1.3, 2.0, 11),
            np.linspace(2.2, 3.5, 5),
        ])
    Kc_analytic = 2.0 / (np.pi * (1.0 / (sigma_omega * np.sqrt(2 * np.pi))))
    # = 2 sigma sqrt(2π)/π ≈ 1.596 pour σ=1

    L = int(np.sqrt(N))
    print(f"N={N} (grid {L}x{L}) | σ_ω={sigma_omega} | "
          f"K_c analytique ≈ {Kc_analytic:.4f}")
    print(f"{len(K_values)} K values | k_max={k_max}")

    snaps = {}
    chi_arr = np.zeros(len(K_values))
    r_mean = np.zeros(len(K_values))
    t0 = time.time()
    for i, K in enumerate(K_values):
        ts = time.time()
        snaps_g, r = kura_sim(
            N=N, K=float(K), sigma_omega=sigma_omega,
            dt=0.05, t_equil=60.0, t_sample=120.0, sample_gap=1.0,
            n_bins=4, seed=seed,
        )
        snaps[float(K)] = snaps_g
        chi_arr[i] = chi_from_r(r, N)
        r_mean[i] = r.mean()
        print(f"  K={K:.3f}  r̄={r.mean():.3f}  χ={chi_arr[i]:.3f}  "
              f"snaps={len(snaps_g)}  {time.time()-ts:.1f}s")
    print(f"sim total {time.time()-t0:.1f}s")

    res = xi_hierarchical(
        snaps, K_values, L=L,
        k_max=k_max, alpha=alpha, epsilon=epsilon,
        encode="categorical", cg_rule="mode",
        chi_provided=chi_arr,
    )

    print()
    print(f"{'K':>6} | {'r̄':>5} | " + " | ".join([f"TC(k={k})" for k in range(k_max + 1)])
          + " | γ_K   | mask | sat_k")
    print("-" * (6 + 12 * (k_max + 1) + 38))
    for i, K in enumerate(K_values):
        row = f"{K:>6.3f} | {r_mean[i]:>5.3f} | "
        row += " | ".join([f"{res['tc_hier'][i,k]:>7.4f}" for k in range(k_max + 1)])
        row += f" | {res['gamma_T'][i]:+.3f} | {res['mask'][i]} | {res['saturation_k'][i]}"
        print(row)

    # Plot
    fig, axes = plt.subplots(5, 1, figsize=(9, 13))
    cmap = plt.get_cmap("plasma", k_max + 1)

    ax = axes[0]
    ax.plot(K_values, r_mean, "o-", color="cyan", lw=1.5, ms=5, label="r̄")
    ax.axvline(Kc_analytic, color="#ff3355", ls="--", lw=1, label=f"K_c≈{Kc_analytic:.3f}")
    ax.set_ylabel("order parameter r̄", color="white")
    ax.legend(facecolor="black", edgecolor="white", labelcolor="white", fontsize=8)

    ax = axes[1]
    ax.plot(K_values, chi_arr, "o-", color="#ff8800", lw=1.5, ms=5)
    ax.axvline(Kc_analytic, color="#ff3355", ls="--", lw=1)
    ax.set_ylabel("χ = N · Var(r)", color="white")

    ax = axes[2]
    for k in range(k_max + 1):
        ax.plot(K_values, res["tc_hier"][:, k], "o-",
                color=cmap(k), label=f"k={k}", lw=1.4, ms=4)
    ax.axvline(Kc_analytic, color="#ff3355", ls="--", lw=1)
    ax.set_ylabel("TC(K, k)", color="white")
    ax.legend(facecolor="black", edgecolor="white", labelcolor="white", fontsize=8, ncol=2)

    ax = axes[3]
    ax.plot(K_values, res["gamma_T"], "o-", color="#00ff88", lw=1.5, ms=5)
    ax.axhline(0, color="white", ls=":", lw=0.6)
    ax.axvline(Kc_analytic, color="#ff3355", ls="--", lw=1)
    ax.set_ylabel("γ_K (slope log TC vs k)", color="white")
    ax.set_title("Self-similarity Kuramoto : γ → 0 attendu à K_c", color="white", fontsize=10)

    ax = axes[4]
    ax.plot(K_values, res["xi_hier"], "o-", color="magenta", lw=1.5, ms=5,
            label=f"Ξ_hier (α={alpha})")
    ax.axvline(Kc_analytic, color="#ff3355", ls="--", lw=1)
    ax.set_ylabel("Ξ_hier (masked)", color="white")
    ax.set_xlabel("K", color="white")
    ax.legend(facecolor="black", edgecolor="white", labelcolor="white", fontsize=8)

    for ax in axes:
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        for s in ax.spines.values():
            s.set_color("white")

    fig.patch.set_facecolor("black")
    fig.suptitle(f"Ξ universalité — Kuramoto N={N}", color="white", y=0.995)
    plt.tight_layout()

    png = os.path.join(outdir, "kuramoto_xi.png")
    plt.savefig(png, dpi=130, facecolor="black", bbox_inches="tight")
    print(f"\nsaved {png}")

    out_json = {
        "N": N, "Kc_analytic": Kc_analytic, "sigma_omega": sigma_omega,
        "epsilon": epsilon, "alpha": alpha, "k_max": k_max,
        "K": np.asarray(K_values).tolist(),
        "r_mean": r_mean.tolist(),
        "chi": chi_arr.tolist(),
        "mask": res["mask"].tolist(),
        "tc_hier": res["tc_hier"].tolist(),
        "o_hier": res["o_hier"].tolist(),
        "abs_o_hier": res["abs_o_hier"].tolist(),
        "gamma_T": res["gamma_T"].tolist(),
        "xi_hier": res["xi_hier"].tolist(),
        "saturation_k": res["saturation_k"].tolist(),
    }
    jpath = os.path.join(outdir, "kuramoto_xi.json")
    with open(jpath, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"saved {jpath}")

    return res, K_values, r_mean, chi_arr, Kc_analytic


if __name__ == "__main__":
    run()
