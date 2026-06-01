import sys
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from systems.kuramoto import simulate as kura_sim, chi_from_r
from xi.temporal import autocorr, correlation_time
from xi.edge import edge_mask


def run(
    N=64,
    sigma_omega=1.0,
    K_values=None,
    seed=42,
    epsilon=0.3,
    outdir=None,
):
    if outdir is None:
        outdir = os.path.dirname(os.path.abspath(__file__))
    if K_values is None:
        K_values = np.concatenate([
            np.linspace(0.4, 1.0, 4),
            np.linspace(1.1, 2.0, 19),
            np.linspace(2.2, 3.5, 5),
        ])
    Kc_analytic = 2.0 / (np.pi * (1.0 / (sigma_omega * np.sqrt(2 * np.pi))))

    print(f"N={N} | σ_ω={sigma_omega} | K_c ≈ {Kc_analytic:.4f}")
    print(f"{len(K_values)} K values")

    chi_arr = np.zeros(len(K_values))
    r_mean = np.zeros(len(K_values))
    tau_int = np.zeros(len(K_values))   # τ_corr integrated
    tau_e = np.zeros(len(K_values))     # τ_corr 1/e
    tau_fit = np.zeros(len(K_values))   # τ_corr exp fit
    r_var = np.zeros(len(K_values))

    t0 = time.time()
    for i, K in enumerate(K_values):
        ts = time.time()
        _, r, _ = kura_sim(
            N=N, K=float(K), sigma_omega=sigma_omega,
            dt=0.05, t_equil=80.0, t_sample=600.0, sample_gap=0.5,
            n_bins=4, seed=seed, return_phases=True,
        )
        chi_arr[i] = chi_from_r(r, N)
        r_mean[i] = r.mean()
        r_var[i] = r.var()
        tau_int[i] = correlation_time(r, method="integrated", max_lag=400)
        tau_e[i] = correlation_time(r, method="1_over_e", max_lag=400)
        tau_fit[i] = correlation_time(r, method="exp_fit", max_lag=400)
        print(f"  K={K:.3f}  r̄={r_mean[i]:.3f}  χ={chi_arr[i]:.3f}  "
              f"τ_int={tau_int[i]:.1f}  τ_1/e={tau_e[i]:.1f}  τ_fit={tau_fit[i]:.1f}  "
              f"{time.time()-ts:.1f}s")
    print(f"sim total {time.time()-t0:.1f}s")

    mask = edge_mask(chi_arr, epsilon=epsilon)
    # Ξ_kuramoto propre = mask × τ_int (slowing down × edge)
    xi_slowing = tau_int * mask

    # Identifier K* (pic τ)
    K_peak_chi = K_values[np.argmax(chi_arr)]
    K_peak_tau = K_values[np.argmax(tau_int)]

    print()
    print(f"K_c analytic        = {Kc_analytic:.4f}")
    print(f"K peak χ            = {K_peak_chi:.4f}")
    print(f"K peak τ_int        = {K_peak_tau:.4f}")
    print(f"max τ_int           = {tau_int.max():.2f}")
    print(f"τ_int(K_c±0.05)/τ_int(K=3) = "
          f"{tau_int[np.argmin(np.abs(K_values-Kc_analytic))] / max(tau_int[-1],1e-6):.2f}")

    # Plot
    fig, axes = plt.subplots(5, 1, figsize=(9, 13))

    ax = axes[0]
    ax.plot(K_values, r_mean, "o-", color="cyan", lw=1.5, ms=5, label="r̄")
    ax.axvline(Kc_analytic, color="#ff3355", ls="--", lw=1, label=f"K_c≈{Kc_analytic:.3f}")
    ax.set_ylabel("order param r̄", color="white")
    ax.legend(facecolor="black", edgecolor="white", labelcolor="white", fontsize=8)

    ax = axes[1]
    ax.plot(K_values, chi_arr, "o-", color="#ff8800", lw=1.5, ms=5)
    ax.axvline(Kc_analytic, color="#ff3355", ls="--", lw=1)
    ax.set_ylabel("χ = N · Var(r)", color="white")

    ax = axes[2]
    ax.plot(K_values, tau_int, "o-", color="#00ff88", lw=1.5, ms=5, label="τ_integrated")
    ax.plot(K_values, tau_e, "s--", color="#88ff00", lw=1.2, ms=4, label="τ_1/e")
    ax.plot(K_values, tau_fit, "^:", color="yellow", lw=1.0, ms=4, label="τ_exp_fit")
    ax.axvline(Kc_analytic, color="#ff3355", ls="--", lw=1)
    ax.set_ylabel("τ_corr (steps)", color="white")
    ax.set_yscale("log")
    ax.legend(facecolor="black", edgecolor="white", labelcolor="white", fontsize=8)
    ax.set_title("Critical slowing down : τ_corr diverge à K_c", color="white", fontsize=10)

    ax = axes[3]
    # autocorr fonctions à K choisis
    K_show = [K_values[0],
              K_values[np.argmin(np.abs(K_values - Kc_analytic))],
              K_values[-1]]
    K_labels = ["K << K_c", "K ≈ K_c", "K >> K_c"]
    cmap = ["#0088ff", "#ff3355", "#aaaa00"]
    for K, lab, col in zip(K_show, K_labels, cmap):
        _, r, _ = kura_sim(
            N=N, K=float(K), sigma_omega=sigma_omega,
            dt=0.05, t_equil=80.0, t_sample=600.0, sample_gap=0.5,
            n_bins=4, seed=seed, return_phases=True,
        )
        acf = autocorr(r, max_lag=300)
        ax.plot(np.arange(len(acf)), acf, "-",
                color=col, lw=1.4, label=f"{lab} (K={K:.2f})")
    ax.axhline(0, color="white", ls=":", lw=0.5)
    ax.axhline(1.0 / np.e, color="white", ls=":", lw=0.5, alpha=0.6)
    ax.set_ylabel("ACF r(τ)", color="white")
    ax.set_xlabel("τ (samples)", color="white")
    ax.legend(facecolor="black", edgecolor="white", labelcolor="white", fontsize=8)

    ax = axes[4]
    ax.plot(K_values, xi_slowing, "o-", color="magenta", lw=1.6, ms=5,
            label="Ξ_slowing = mask × τ_int")
    ax.axvline(Kc_analytic, color="#ff3355", ls="--", lw=1)
    ax.set_ylabel("Ξ_slowing (Kuramoto-proper)", color="white")
    ax.set_xlabel("K", color="white")
    ax.legend(facecolor="black", edgecolor="white", labelcolor="white", fontsize=8)

    for ax in axes:
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        for s in ax.spines.values():
            s.set_color("white")

    fig.patch.set_facecolor("black")
    fig.suptitle(f"Ξ Critical slowing down — Kuramoto N={N}", color="white", y=0.995)
    plt.tight_layout()

    png = os.path.join(outdir, "kuramoto_slowing.png")
    plt.savefig(png, dpi=130, facecolor="black", bbox_inches="tight")
    print(f"\nsaved {png}")

    out_json = {
        "N": N, "Kc_analytic": Kc_analytic, "sigma_omega": sigma_omega,
        "epsilon": epsilon,
        "K": K_values.tolist(),
        "r_mean": r_mean.tolist(),
        "chi": chi_arr.tolist(),
        "tau_int": tau_int.tolist(),
        "tau_e": tau_e.tolist(),
        "tau_fit": tau_fit.tolist(),
        "mask": mask.astype(int).tolist(),
        "xi_slowing": xi_slowing.tolist(),
        "K_peak_chi": float(K_peak_chi),
        "K_peak_tau": float(K_peak_tau),
    }
    jpath = os.path.join(outdir, "kuramoto_slowing.json")
    with open(jpath, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"saved {jpath}")

    return out_json


if __name__ == "__main__":
    run()
