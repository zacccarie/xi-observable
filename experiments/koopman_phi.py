"""B v0 — Φ = information synergique persistante (premier pas Koopman × irréductibilité).

But original du programme : raffiner Φ. Recadrage : Φ = modes qui émergent
(irréductibles) ET durent (persistants Koopman).

DMD sur trajectoire Kuramoto → opérateur Koopman approché A, spectre λ.
- persistance d'un mode = |λ| (proche 1 = ni amorti ni explosif → dure)
- irréductibilité (proxy v0) = participation ratio PR du mode = (Σ|φ|²)²/Σ|φ|⁴
  PR élevé = mode distribué sur beaucoup d'observables (candidat irréductible) ;
  PR bas = localisé / trivial (redondant)

Φ = modes |λ|≈1 ET PR élevé. Tests :
1. persistance ⊥ PR ? (si oui → Φ filtre 2D non-trivial, pas réductible à un seul critère)
2. |Φ| (taille du noyau) varie-t-elle avec K, pic à l'edge K_c ?

LIMITE v0 : PR est un proxy de l'irréductibilité, pas une O-information de mode.
Le Φ complet (attribution PID spectrale de la synergie) est un chapitre suivant.
"""
import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from systems.kuramoto import simulate


def dmd_spectrum(Y, rank=None, energy=0.999):
    """Exact DMD à SVD TRONQUÉE (sinon modes spurieux |λ|≫1 par pinv rank-deficient).
    Y (T, d) observables en lignes. Retourne (|λ|, modes, λ)."""
    X0 = Y[:-1].T          # (d, m)
    X1 = Y[1:].T           # (d, m)
    U, S, Vh = np.linalg.svd(X0, full_matrices=False)
    if rank is None:
        cum = np.cumsum(S ** 2) / np.sum(S ** 2)
        rank = int(np.searchsorted(cum, energy) + 1)
    rank = max(1, min(rank, len(S)))
    Ur, Sr, Vr = U[:, :rank], S[:rank], Vh[:rank].conj().T
    Atil = Ur.conj().T @ X1 @ Vr @ np.diag(1.0 / Sr)   # (rank, rank) Koopman réduit
    lam, w = np.linalg.eig(Atil)
    modes = X1 @ Vr @ np.diag(1.0 / Sr) @ w            # modes DMD projetés (d, rank)
    return np.abs(lam), modes, lam


def participation_ratio(mode):
    p = np.abs(mode) ** 2
    p = p / (p.sum() + 1e-12)
    return 1.0 / (np.sum(p ** 2) + 1e-12)  # ∈ [1, d], élevé = distribué


def run(
    N=36,
    K_list=(0.6, 1.0, 1.4, 1.6, 1.8, 2.2, 2.8),
    sigma_omega=1.0,
    dt=0.05,
    t_equil=60.0,
    t_sample=120.0,
    sample_gap=0.5,
    seed0=0,
    outdir=None,
):
    if outdir is None:
        outdir = os.path.dirname(os.path.abspath(__file__))
    Kc = 2.0 / (np.pi * (1.0 / (sigma_omega * np.sqrt(2 * np.pi))))
    print(f"Koopman-Φ v0 | Kuramoto N={N} | Kc={Kc:.3f}\n")

    rng = np.random.default_rng(seed0)
    persist_thresh = 0.9
    results = {}
    all_lam, all_pr, all_K = [], [], []

    for K in K_list:
        _, _, phases = simulate(
            N=N, K=float(K), sigma_omega=sigma_omega, dt=dt,
            t_equil=t_equil, t_sample=t_sample, sample_gap=sample_gap,
            seed=int(rng.integers(0, 2 ** 31)), return_phases=True,
        )
        # observables Koopman : cos θ, sin θ
        Y = np.hstack([np.cos(phases), np.sin(phases)])   # (T, 2N)
        absl, vecs, lam = dmd_spectrum(Y)
        prs = np.array([participation_ratio(vecs[:, i]) for i in range(vecs.shape[1])])
        pr_norm = prs / (2 * N)   # ∈ (0,1]

        persist = absl >= persist_thresh
        pr_med = np.median(pr_norm)
        phi = persist & (pr_norm >= pr_med)   # noyau Φ candidat
        results[K] = {
            "n_modes": len(absl), "n_persist": int(persist.sum()),
            "n_phi": int(phi.sum()), "pr_med": float(pr_med),
            "corr_lam_pr": float(np.corrcoef(absl, pr_norm)[0, 1]),
        }
        all_lam.append(absl); all_pr.append(pr_norm); all_K.append(np.full(len(absl), K))
        print(f"  K={K:.2f} | modes={len(absl)} | persistants(|λ|≥{persist_thresh})="
              f"{persist.sum():>3} | Φ={phi.sum():>3} | corr(|λ|,PR)={results[K]['corr_lam_pr']:+.2f}")

    lam_all = np.concatenate(all_lam)
    pr_all = np.concatenate(all_pr)
    K_all = np.concatenate(all_K)
    r_global = np.corrcoef(lam_all, pr_all)[0, 1]
    print(f"\ncorr globale (|λ|, PR) = {r_global:+.3f}  "
          f"→ {'axes ⊥ (Φ filtre 2D non-trivial)' if abs(r_global) < 0.4 else 'axes corrélés'}")

    Ks = np.array(K_list)
    n_phi = np.array([results[K]["n_phi"] for K in K_list])
    peak_phi_K = Ks[np.argmax(n_phi)]
    print(f"|Φ| max à K={peak_phi_K:.2f} (Kc={Kc:.3f})")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sc = axes[0].scatter(lam_all, pr_all, c=K_all, cmap="turbo", s=18, alpha=0.6)
    axes[0].axvline(persist_thresh, color="#ff3355", ls="--", lw=1, label="persistance")
    axes[0].set_xlabel("|λ| (persistance Koopman)", color="white")
    axes[0].set_ylabel("PR normalisé (distribution / irréductibilité proxy)", color="white")
    axes[0].set_title(f"plan Φ — modes (corr={r_global:+.2f})", color="white")
    axes[0].legend(facecolor="black", labelcolor="white", edgecolor="white", fontsize=8)
    cb = plt.colorbar(sc, ax=axes[0]); cb.set_label("K", color="white")
    cb.ax.tick_params(colors="white")

    axes[1].plot(Ks, n_phi, "o-", color="#ff00ff", lw=2, label="|Φ| persistant+distribué")
    axes[1].plot(Ks, [results[K]["n_persist"] for K in K_list], "s--",
                 color="#00ffff", lw=1.5, label="persistants seuls")
    axes[1].axvline(Kc, color="#ff3355", ls="--", label=f"Kc={Kc:.2f}")
    axes[1].set_xlabel("K", color="white")
    axes[1].set_ylabel("# modes", color="white")
    axes[1].set_title("taille du noyau Φ vs couplage", color="white")
    axes[1].legend(facecolor="black", labelcolor="white", edgecolor="white", fontsize=8)

    for ax in axes:
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_color("white")
    fig.patch.set_facecolor("black")
    fig.suptitle("Φ v0 — persistance Koopman × distribution (Kuramoto)", color="white")
    plt.tight_layout()
    png = os.path.join(outdir, "koopman_phi.png")
    plt.savefig(png, dpi=130, facecolor="black", bbox_inches="tight")
    print(f"\nsaved {png}")

    out = {"N": N, "Kc": Kc, "K_list": list(K_list), "persist_thresh": persist_thresh,
           "results": results, "corr_global_lam_pr": float(r_global),
           "peak_phi_K": float(peak_phi_K)}
    with open(os.path.join(outdir, "koopman_phi.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {os.path.join(outdir, 'koopman_phi.json')}")
    return out


if __name__ == "__main__":
    run()
