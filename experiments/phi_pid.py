"""Φ — la synergie vit-elle dans les modes PERSISTANTS ou TRANSITOIRES ?

Tension révélée par A : si émergence ≡ chaos (instabilité), alors « synergie
persistante » (Φ) pourrait être contradictoire — l'émergence serait fugace.

Test décisif sur Coupled Map Lattice (chaotique, structures cohérentes, vraie
synergie via non-linéarité logistique) :
1. DMD → modes Koopman, |λ| (persistance), amplitudes temporelles a_j(t)
2. Split persistants (|λ|≥seuil) vs transitoires
3. Ω(amplitudes persistantes) vs Ω(amplitudes transitoires)

- Ω_persist < 0 (synergie) → Φ existe, émergence DURABLE
- synergie seulement dans transitoires → émergence FUGACE (cohérent A), Φ quasi-vide
"""
import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from systems.cml import simulate_cml, lyapunov_cml
from xi.interaction import o_information
from xi.temporal import tc_random_subsets


def dmd_amplitudes(X, energy=0.999):
    """DMD tronqué. X (T, d). Retourne |λ|, amplitudes temporelles (r, T) réelles."""
    X0 = X[:-1].T
    X1 = X[1:].T
    U, S, Vh = np.linalg.svd(X0, full_matrices=False)
    cum = np.cumsum(S ** 2) / np.sum(S ** 2)
    r = max(2, int(np.searchsorted(cum, energy) + 1))
    r = min(r, len(S))
    Ur, Sr, Vr = U[:, :r], S[:r], Vh[:r].conj().T
    Atil = Ur.conj().T @ X1 @ Vr @ np.diag(1.0 / Sr)
    lam, W = np.linalg.eig(Atil)
    Phi = X1 @ Vr @ np.diag(1.0 / Sr) @ W          # (d, r)
    amps = np.linalg.pinv(Phi) @ X.T               # (r, T) complexe
    return np.abs(lam), np.real(amps), Phi


def quantize_rows(A, n_bins=4):
    """Discrétise chaque ligne (mode) en n_bins quantiles."""
    out = np.empty(A.shape, dtype=np.int8)
    for i in range(A.shape[0]):
        q = np.quantile(A[i], np.linspace(0, 1, n_bins + 1)[1:-1])
        out[i] = np.digitize(A[i], q)
    return out


def group_synergy(amps_disc, idx, subset_size=4, n_subsets=200, rng=None):
    """Ω moyen sur sous-ensembles aléatoires des modes du groupe idx.
    amps_disc (r, T). Échantillons = colonnes (temps), variables = modes."""
    if len(idx) < subset_size:
        if len(idx) < 2:
            return np.nan
        samples = amps_disc[idx].T
        return o_information(samples)
    if rng is None:
        rng = np.random.default_rng(0)
    sub = amps_disc[idx].T                  # (T, n_modes)
    _, _, o, _, _ = tc_random_subsets(sub, subset_size, n_subsets, rng)
    return o


def run(
    N=64,
    regimes=(("ordonné", 3.5, 0.3), ("faible chaos", 3.7, 0.3),
             ("chaos", 3.9, 0.3), ("chaos fort", 4.0, 0.2),
             ("chaos décorrélé", 4.0, 0.05)),
    T=700,
    t_equil=400,
    persist_thresh=0.9,
    n_bins=4,
    seed0=0,
    outdir=None,
):
    if outdir is None:
        outdir = os.path.dirname(os.path.abspath(__file__))
    rng = np.random.default_rng(seed0)

    print(f"Φ PID | CML N={N} | persist |λ|≥{persist_thresh}\n")
    rows = []
    for name, r, eps in regimes:
        seed = int(rng.integers(0, 2 ** 31))
        lyap = lyapunov_cml(N=N, eps=eps, r=r, seed=seed)
        X = simulate_cml(N=N, eps=eps, r=r, T=T, t_equil=t_equil, seed=seed)
        absl, amps, Phi = dmd_amplitudes(X)
        amps_disc = quantize_rows(amps, n_bins=n_bins)
        persist = np.where(absl >= persist_thresh)[0]
        transient = np.where(absl < persist_thresh)[0]
        o_p = group_synergy(amps_disc, persist, rng=np.random.default_rng(seed))
        o_t = group_synergy(amps_disc, transient, rng=np.random.default_rng(seed + 1))
        rows.append((name, r, eps, lyap, len(absl), len(persist), len(transient), o_p, o_t))
        print(f"  {name:<16} r={r} ε={eps} | λ_max={lyap:+.3f} | "
              f"modes={len(absl)} (P={len(persist)},T={len(transient)}) | "
              f"Ω_persist={o_p:+.4f} | Ω_transient={o_t:+.4f}")

    print("\n" + "=" * 60)
    print("VERDICT — où vit la synergie (Ω<0) ?")
    print("=" * 60)
    for name, r, eps, lyap, nm, npe, ntr, o_p, o_t in rows:
        loc = []
        if not np.isnan(o_p) and o_p < -1e-3:
            loc.append("PERSISTANT")
        if not np.isnan(o_t) and o_t < -1e-3:
            loc.append("transitoire")
        loc = " + ".join(loc) if loc else "aucune"
        print(f"  {name:<16} (λ={lyap:+.2f}) : synergie dans {loc}")

    # interprétation tension
    chaotic = [row for row in rows if row[3] > 0]
    pers_syn = [row for row in chaotic if not np.isnan(row[7]) and row[7] < -1e-3]
    print()
    if pers_syn:
        print("⟹ Φ EXISTE : émergence durable (modes persistants synergiques en régime chaotique)")
    else:
        print("⟹ Φ QUASI-VIDE : émergence fugace (synergie dans transitoires seulement) — cohérent A")

    # Plot
    names = [row[0] for row in rows]
    lyaps = [row[3] for row in rows]
    o_ps = [row[7] for row in rows]
    o_ts = [row[8] for row in rows]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    w = 0.38
    ax.bar(x - w / 2, o_ps, w, color="#ff00ff", label="Ω modes persistants (Φ)")
    ax.bar(x + w / 2, o_ts, w, color="#00ffff", label="Ω modes transitoires")
    ax.axhline(0, color="white", lw=0.8)
    for i, ly in enumerate(lyaps):
        ax.text(i, ax.get_ylim()[1] * 0.9, f"λ={ly:+.2f}",
                color="white", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, color="white", rotation=15)
    ax.set_ylabel("Ω (synergie si <0)", color="white")
    ax.set_title("Φ : la synergie vit-elle dans les modes persistants ?", color="white")
    ax.legend(facecolor="black", labelcolor="white", edgecolor="white")
    ax.set_facecolor("black")
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_color("white")
    fig.patch.set_facecolor("black")
    plt.tight_layout()
    png = os.path.join(outdir, "phi_pid.png")
    plt.savefig(png, dpi=130, facecolor="black", bbox_inches="tight")
    print(f"\nsaved {png}")

    out = {"N": N, "persist_thresh": persist_thresh,
           "rows": [{"name": r[0], "r": r[1], "eps": r[2], "lyap": r[3],
                     "n_modes": r[4], "n_persist": r[5], "n_transient": r[6],
                     "omega_persist": r[7], "omega_transient": r[8]} for r in rows]}
    with open(os.path.join(outdir, "phi_pid.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {os.path.join(outdir, 'phi_pid.json')}")
    return out


if __name__ == "__main__":
    run()
