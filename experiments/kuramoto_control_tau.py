"""Deux tests décisifs sur Kuramoto.

A. CONTRÔLE ARTEFACT — phases i.i.d. uniformes (zéro interaction).
   Passe dans le pipeline temporal (circular_window_mean → quadrantize → TC).
   Si TC(k) croît avec k → le coarse-graining circulaire fabrique de la
   corrélation spurieuse. γ>0 serait alors un artefact, pas un signal.

B. τ_corr — temps d'autocorrélation intégré de r(t).
   Critical slowing down : à K_c, r fluctue lentement → τ_corr pique.
   Observable Kuramoto-natif candidat (vs χ qui pique à K=1.35 ≠ K_c=1.596).
"""
import sys
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from systems.kuramoto import simulate, chi_from_r, order_parameter
from xi.temporal import circular_window_mean, quadrantize_phases, tc_random_subsets


def control_artifact(N=64, T=480, n_bins=4, k_max=4, subset_size=4,
                     n_subsets=300, seed=0):
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0, 2 * np.pi, size=(T, N))
    tcs, n_eff = [], []
    for k in range(k_max + 1):
        window = 2 ** k
        cg = circular_window_mean(phases, window)
        if cg.shape[0] < 16:
            tcs.append(np.nan)
            n_eff.append(0)
            continue
        samples = quadrantize_phases(cg, n_bins=n_bins)
        tc, _, _, _, _ = tc_random_subsets(samples, subset_size, n_subsets, rng)
        tcs.append(tc)
        n_eff.append(cg.shape[0])
    return np.array(tcs), n_eff


def autocorr_time(x, max_lag=None):
    """Temps d'autocorrélation intégré : τ = 1 + 2 Σ_lag C(lag), arrêt au 1er C<0."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    var = x.var()
    if var < 1e-10:
        return 0.0
    n = len(x)
    if max_lag is None:
        max_lag = n // 2
    full = np.correlate(x, x, mode="full")[n - 1:]
    ac = full / (var * n)
    ac = ac[:max_lag]
    tau = 1.0
    for lag in range(1, len(ac)):
        if ac[lag] <= 0:
            break
        tau += 2.0 * ac[lag]
    return tau


def tau_sweep(N=64, sigma_omega=1.0, K_values=None, dt=0.05,
              t_equil=80.0, t_sample=300.0, sample_gap=0.5, seed=0):
    if K_values is None:
        K_values = np.linspace(0.8, 2.4, 17)
    Kc = 2.0 * sigma_omega * np.sqrt(2.0 / np.pi) / 2.0  # = σ·√(2/π) approx
    # Kc exact pour gaussienne g(0)=1/(σ√2π) : Kc = 2/(π g(0)) = 2σ√(2π)/...
    Kc = 2.0 / (np.pi * (1.0 / (sigma_omega * np.sqrt(2 * np.pi))))
    r_mean = np.zeros(len(K_values))
    chi = np.zeros(len(K_values))
    tau = np.zeros(len(K_values))
    for i, K in enumerate(K_values):
        _, r_traj = simulate(
            N=N, K=float(K), sigma_omega=sigma_omega, dt=dt,
            t_equil=t_equil, t_sample=t_sample, sample_gap=sample_gap,
            seed=seed, return_phases=False,
        )
        r_mean[i] = r_traj.mean()
        chi[i] = chi_from_r(r_traj, N)
        tau[i] = autocorr_time(r_traj)
    return K_values, r_mean, chi, tau, Kc


def run(outdir=None):
    if outdir is None:
        outdir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 64)
    print("A. CONTRÔLE ARTEFACT — phases i.i.d. uniformes (zéro interaction)")
    print("=" * 64)
    tcs, n_eff = control_artifact()
    for k, (tc, ne) in enumerate(zip(tcs, n_eff)):
        print(f"  k={k}  window=2^{k}={2**k:>3}  n_samples={ne:>4}  TC={tc:.4f}")
    ratio = tcs[-1] / tcs[0] if tcs[0] > 0 else float("inf")
    print(f"\n  TC(k_max)/TC(0) = {ratio:.2f}")
    if ratio > 2.0:
        print("  ✗ ARTEFACT CONFIRMÉ : TC croît sur données SANS interaction.")
        print("    → circular_window_mean fabrique corrélation spurieuse.")
        print("    → γ_K positif du sweep Kuramoto est largement artefactuel.")
    else:
        print("  ✓ Pas d'artefact majeur : pipeline neutre sur données i.i.d.")
    verdict_artifact = "artifact" if ratio > 2.0 else "clean"

    print()
    print("=" * 64)
    print("B. τ_corr — critical slowing down (observable Kuramoto-natif)")
    print("=" * 64)
    t0 = time.time()
    K_values, r_mean, chi, tau, Kc = tau_sweep()
    print(f"simu {time.time()-t0:.1f}s   Kc≈{Kc:.4f}\n")
    print(f"{'K':>6} | {'r̄':>6} | {'χ':>8} | {'τ_corr':>8}")
    print("-" * 40)
    for i, K in enumerate(K_values):
        print(f"{K:>6.3f} | {r_mean[i]:>6.3f} | {chi[i]:>8.3f} | {tau[i]:>8.3f}")
    peak_chi = K_values[np.argmax(chi)]
    peak_tau = K_values[np.argmax(tau)]
    print(f"\npeak χ      K={peak_chi:.3f}  (Δ vs Kc = {peak_chi-Kc:+.3f})")
    print(f"peak τ_corr K={peak_tau:.3f}  (Δ vs Kc = {peak_tau-Kc:+.3f})")
    if abs(peak_tau - Kc) < abs(peak_chi - Kc):
        print("  ✓ τ_corr localise K_c mieux que χ → critical slowing down validé.")
    else:
        print("  ✗ τ_corr ne bat pas χ pour localiser K_c.")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]
    ax.plot(range(len(tcs)), tcs, "o-", color="#ff5555", lw=2)
    ax.set_xlabel("k (niveau CG temporel)", color="white")
    ax.set_ylabel("TC (phases i.i.d.)", color="white")
    ax.set_title(f"A. Contrôle artefact — ratio={ratio:.1f}× [{verdict_artifact}]",
                 color="white")

    ax = axes[0, 1]
    ax.plot(K_values, r_mean, "o-", color="#00ffff", lw=2)
    ax.axvline(Kc, color="#ff3355", ls="--", label=f"Kc={Kc:.3f}")
    ax.set_xlabel("K", color="white")
    ax.set_ylabel("r̄", color="white")
    ax.set_title("order parameter", color="white")
    ax.legend(facecolor="black", labelcolor="white", edgecolor="white")

    ax = axes[1, 0]
    ax.plot(K_values, chi, "o-", color="#ffaa00", lw=2)
    ax.axvline(Kc, color="#ff3355", ls="--")
    ax.axvline(peak_chi, color="#ffaa00", ls=":", label=f"peak χ={peak_chi:.2f}")
    ax.set_xlabel("K", color="white")
    ax.set_ylabel("χ = N·Var(r)", color="white")
    ax.set_title("susceptibility", color="white")
    ax.legend(facecolor="black", labelcolor="white", edgecolor="white")

    ax = axes[1, 1]
    ax.plot(K_values, tau, "o-", color="#ff00ff", lw=2)
    ax.axvline(Kc, color="#ff3355", ls="--")
    ax.axvline(peak_tau, color="#ff00ff", ls=":", label=f"peak τ={peak_tau:.2f}")
    ax.set_xlabel("K", color="white")
    ax.set_ylabel("τ_corr", color="white")
    ax.set_title("B. critical slowing down", color="white")
    ax.legend(facecolor="black", labelcolor="white", edgecolor="white")

    for ax in axes.flat:
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_color("white")
    fig.patch.set_facecolor("black")
    fig.suptitle("Kuramoto — contrôle artefact + critical slowing down",
                 color="white", y=0.99)
    plt.tight_layout()
    png = os.path.join(outdir, "kuramoto_control_tau.png")
    plt.savefig(png, dpi=130, facecolor="black", bbox_inches="tight")
    print(f"\nsaved {png}")

    out = {
        "control_artifact": {"tc_per_k": tcs.tolist(), "ratio": float(ratio),
                             "verdict": verdict_artifact},
        "tau_sweep": {"K": K_values.tolist(), "r_mean": r_mean.tolist(),
                      "chi": chi.tolist(), "tau": tau.tolist(), "Kc": float(Kc),
                      "peak_chi": float(peak_chi), "peak_tau": float(peak_tau)},
    }
    jpath = os.path.join(outdir, "kuramoto_control_tau.json")
    with open(jpath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {jpath}")
    return out


if __name__ == "__main__":
    run()
