"""Task #19 — Sandpile BTW (SOC) : 5e classe criticalité.

Pas de paramètre contrôle externe. Système s'auto-règle à criticalité.
Mesures :
  - P(s) ~ s^-τ avec τ ≈ 1 (signature SOC 2D BTW universelle)
  - χ_info = std_réalisations(Ω) sur états (différentes seeds → différents états
    SOC) — testable ?
  - LZ-complexity des séries avalanche pour comparer

Question : χ_info marche-t-elle sans paramètre contrôle à varier ?
Hypothèse : oui, parce que SOC = système à criticalité PERMANENTE. χ_info
mesure dispersion entre réalisations indépendantes → reste valide.
Comparaison : χ_info devrait être ÉLEVÉ comme à Tc Ising.
"""
import sys
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from systems.sandpile import simulate, fit_powerlaw_exponent
from xi.interaction import o_information, total_correlation
from xi.complexity import lz_normalized


def omega_random_subsets(states_flat, subset_size=4, n_subsets=200, rng=None):
    """states_flat : (n_states, N) int [0..3]. Tire subsets, calcule Ω moyen."""
    if rng is None:
        rng = np.random.default_rng(0)
    n_states, N = states_flat.shape
    oms, tcs = [], []
    for _ in range(n_subsets):
        idx = rng.choice(N, size=subset_size, replace=False)
        sub = states_flat[:, idx].astype(np.int8)
        oms.append(o_information(sub))
        tcs.append(total_correlation(sub))
    return np.mean(oms), np.mean(tcs)


def run(
    L=32,
    n_real=20,
    n_drops=3000,
    n_warmup=1500,
    seed0=42,
    outdir=None,
):
    if outdir is None:
        outdir = os.path.dirname(os.path.abspath(__file__))

    print(f"Sandpile BTW L={L} | {n_real} réal | n_drops={n_drops}")
    print()

    om_means = np.zeros(n_real)
    tc_means = np.zeros(n_real)
    lz_avalanche = np.zeros(n_real)
    tau_per_real = np.zeros(n_real)
    activity_per_real = np.zeros(n_real)
    all_sizes = []
    all_snapshots = []  # 1 par réalisation
    rng_master = np.random.default_rng(seed0)

    t0 = time.time()
    for r in range(n_real):
        seed = int(rng_master.integers(0, 2**31))
        ts = time.time()
        _, avalanches, snaps = simulate(L=L, n_drops=n_drops,
                                        n_warmup=n_warmup, seed=seed)
        # Stack snapshots flat
        snap_arr = np.stack([s.flatten() for s in snaps], axis=0)
        # Clip états à [0..3] (rarement dépasse 4 après stabilisation)
        snap_clip = np.clip(snap_arr, 0, 3).astype(np.int8)
        om, tc = omega_random_subsets(
            snap_clip, subset_size=4, n_subsets=200,
            rng=np.random.default_rng(seed + 7),
        )
        # LZ sur série avalanche (signature dynamique SOC)
        # Discrétiser tailles avalanche par log binning
        log_sizes = np.log1p(avalanches).astype(np.int8)
        lz_aval = lz_normalized(log_sizes)
        # τ (exposant power-law)
        tau = fit_powerlaw_exponent(avalanches, smin=4)
        # activité moyenne (état stationnaire SOC ~ 2.1 grains/site)
        activity = snap_arr.mean()

        om_means[r] = om
        tc_means[r] = tc
        lz_avalanche[r] = lz_aval
        tau_per_real[r] = tau
        activity_per_real[r] = activity
        all_sizes.append(avalanches)
        if r == 0:
            all_snapshots = snaps[:5]

        print(f"  real{r:>2}: τ={tau:.3f} ⟨z⟩={activity:.2f} "
              f"Ω={om:+.4f} TC={tc:.4f} LZ(aval)={lz_aval:.4f} "
              f"({time.time()-ts:.1f}s)")
    print(f"\ntotal {time.time()-t0:.1f}s")

    # Synthèse
    chi_info = om_means.std()
    abs_om_mean = np.abs(om_means).mean()
    cv = chi_info / (abs_om_mean + 1e-9)
    tau_mean = tau_per_real[~np.isnan(tau_per_real)].mean()
    tau_std = tau_per_real[~np.isnan(tau_per_real)].std()
    activity_mean = activity_per_real.mean()
    activity_std = activity_per_real.std()
    lz_aval_std = lz_avalanche.std()
    lz_aval_mean = lz_avalanche.mean()

    print()
    print("=" * 60)
    print(f"SANDPILE BTW L={L} SYNTHÈSE")
    print(f"τ moyen (P(s)~s^-τ)   = {tau_mean:.3f} ± {tau_std:.3f}  "
          f"(attendu ≈ 1.0 BTW 2D)")
    print(f"⟨z⟩ activité SOC      = {activity_mean:.3f} ± {activity_std:.3f}  "
          f"(attendu ≈ 2.1)")
    print(f"⟨Ω⟩ régime info       = {om_means.mean():+.4f}")
    print(f"sign(Ω) moyen         = {'redondance' if om_means.mean() > 0 else 'synergie'}")
    print(f"χ_info = std(Ω)       = {chi_info:.4f}")
    print(f"CV(Ω)                 = {cv:.4f}")
    print(f"std(LZ_avalanche)     = {lz_aval_std:.4f}")
    print(f"⟨LZ_avalanche⟩        = {lz_aval_mean:.4f}")
    print(f"frac_synergy          = {(om_means < 0).mean():.2%}")
    print("=" * 60)

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    ax = axes[0, 0]
    # Histogramme tailles avalanche, log-log
    all_av = np.concatenate(all_sizes)
    all_av = all_av[all_av > 0]
    bins = np.logspace(0, np.log10(all_av.max() + 1), 30)
    hist, edges = np.histogram(all_av, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    valid = hist > 0
    ax.loglog(centers[valid], hist[valid], "o-", color="cyan", lw=1.2, ms=5)
    if tau_mean and not np.isnan(tau_mean):
        x_fit = np.logspace(0.5, np.log10(centers[valid].max()), 50)
        y_fit = hist[valid].max() * (x_fit / centers[valid][np.argmax(hist[valid])]) ** (-tau_mean)
        ax.loglog(x_fit, y_fit, "--", color="#ff3355", lw=1,
                  label=f"τ={tau_mean:.2f}")
    ax.set_xlabel("avalanche size s", color="white")
    ax.set_ylabel("P(s) count", color="white")
    ax.set_title("Power-law avalanche (SOC signature)", color="white", fontsize=10)
    ax.legend(facecolor="black", labelcolor="white", edgecolor="white")

    ax = axes[0, 1]
    ax.hist(om_means, bins=12, color="yellow", edgecolor="white")
    ax.axvline(0, color="#ff3355", ls="--", lw=1)
    ax.set_xlabel("Ω par réalisation", color="white")
    ax.set_ylabel("count", color="white")
    ax.set_title(f"Distribution Ω inter-réal | χ_info={chi_info:.4f}",
                 color="white", fontsize=10)

    ax = axes[0, 2]
    ax.imshow(all_snapshots[-1] if all_snapshots else np.zeros((L, L)),
              cmap="viridis", vmin=0, vmax=4)
    ax.set_title("État SOC final (1 réal.)", color="white", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[1, 0]
    ax.hist(tau_per_real[~np.isnan(tau_per_real)], bins=10,
            color="magenta", edgecolor="white")
    ax.axvline(1.0, color="#00ff88", ls="--", lw=1, label="τ=1 BTW universel")
    ax.set_xlabel("τ par réalisation", color="white")
    ax.set_title(f"Distribution τ | mean={tau_mean:.3f}",
                 color="white", fontsize=10)
    ax.legend(facecolor="black", labelcolor="white", edgecolor="white")

    ax = axes[1, 1]
    ax.scatter(om_means, lz_avalanche, c=tau_per_real, cmap="plasma",
               s=60, edgecolor="white")
    ax.set_xlabel("Ω par réal", color="white")
    ax.set_ylabel("LZ avalanche série", color="white")
    pearson_om_lz = np.corrcoef(om_means, lz_avalanche)[0, 1]
    ax.set_title(f"Ω vs LZ scatter | Pearson={pearson_om_lz:+.3f}",
                 color="white", fontsize=10)

    ax = axes[1, 2]
    ax.text(0.05, 0.85, f"L = {L}", color="white", fontsize=11,
            transform=ax.transAxes)
    ax.text(0.05, 0.72, f"τ = {tau_mean:.3f} ± {tau_std:.3f}",
            color="cyan", fontsize=11, transform=ax.transAxes)
    ax.text(0.05, 0.59, f"⟨z⟩ = {activity_mean:.3f}", color="yellow",
            fontsize=11, transform=ax.transAxes)
    ax.text(0.05, 0.46, f"⟨Ω⟩ = {om_means.mean():+.4f}", color="magenta",
            fontsize=11, transform=ax.transAxes)
    ax.text(0.05, 0.33, f"χ_info = {chi_info:.4f}", color="#00ff88",
            fontsize=11, transform=ax.transAxes)
    ax.text(0.05, 0.20, f"CV(Ω) = {cv:.4f}", color="#00ff88",
            fontsize=11, transform=ax.transAxes)
    ax.text(0.05, 0.07, f"frac_syn = {(om_means < 0).mean():.0%}",
            color="white", fontsize=11, transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Synthèse SOC", color="white", fontsize=10)

    for ax in axes.flat:
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_color("white")

    fig.patch.set_facecolor("black")
    fig.suptitle(f"Sandpile BTW L={L} — 5e classe criticalité SOC",
                 color="white", y=0.995)
    plt.tight_layout()
    png = os.path.join(outdir, "sandpile_xi.png")
    plt.savefig(png, dpi=130, facecolor="black", bbox_inches="tight")
    print(f"\nsaved {png}")

    out = {
        "L": L, "n_real": n_real, "n_drops": n_drops,
        "tau_mean": float(tau_mean), "tau_std": float(tau_std),
        "activity_mean": float(activity_mean), "activity_std": float(activity_std),
        "om_means": om_means.tolist(),
        "tc_means": tc_means.tolist(),
        "lz_avalanche": lz_avalanche.tolist(),
        "tau_per_real": tau_per_real.tolist(),
        "chi_info": float(chi_info),
        "cv_omega": float(cv),
        "frac_synergy": float((om_means < 0).mean()),
        "pearson_om_lz": float(pearson_om_lz),
    }
    with open(os.path.join(outdir, "sandpile_xi.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {os.path.join(outdir, 'sandpile_xi.json')}")
    return out


if __name__ == "__main__":
    run()
