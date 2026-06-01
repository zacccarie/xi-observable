"""Bak-Tang-Wiesenfeld sandpile — 2D, conditions ouvertes (perte bord).

Règle :
  - z(i,j) = nombre grains site (i,j)
  - drop : z[i,j] += 1 sur site aléatoire
  - toppling : si z(i,j) ≥ 4, z(i,j) -= 4, z(voisins NESW) += 1 (perte bord)
  - répéter toppling jusqu'à stable
  - avalanche size = nb topplings total dans une drop

Auto-organisé criticalité (SOC) : converge vers état critique sans param contrôle.
P(avalanche size s) ~ s^-τ avec τ ≈ 1.0 (2D BTW universel).
"""
import numpy as np


def relax(z):
    """Topple jusqu'à stable. Retourne (z_stabilisé, nb_topplings)."""
    L = z.shape[0]
    n_top = 0
    while True:
        unstable = z >= 4
        if not unstable.any():
            break
        n_top += int(unstable.sum())
        # distribue : voisins gagnent +1 (sauf bord = perte)
        z = z - 4 * unstable.astype(z.dtype)
        # voisins NESW (no wrap = perte bord)
        z[1:, :] += unstable[:-1, :]
        z[:-1, :] += unstable[1:, :]
        z[:, 1:] += unstable[:, :-1]
        z[:, :-1] += unstable[:, 1:]
    return z, n_top


def simulate(L=32, n_drops=2000, n_warmup=500, seed=0):
    """Simule sandpile BTW. Retourne (final_state, avalanche_sizes, snapshots)."""
    rng = np.random.default_rng(seed)
    z = np.zeros((L, L), dtype=np.int32)
    # warmup pour atteindre SOC (le système se règle tout seul)
    for _ in range(n_warmup):
        i, j = rng.integers(0, L, size=2)
        z[i, j] += 1
        z, _ = relax(z)
    # mesure
    avalanche_sizes = []
    snapshots = []
    for k in range(n_drops):
        i, j = rng.integers(0, L, size=2)
        z[i, j] += 1
        z, n_top = relax(z)
        avalanche_sizes.append(n_top)
        if k % max(1, n_drops // 200) == 0:
            snapshots.append(z.copy())
    return z, np.array(avalanche_sizes), snapshots


def fit_powerlaw_exponent(sizes, smin=4, smax=None):
    """Fit τ : P(s) ~ s^-τ. MLE (Clauset-Shalizi-Newman 2009).
    τ = 1 + n / Σ log(s_i / smin).
    """
    s = np.asarray(sizes)
    s = s[s >= smin]
    if smax is not None:
        s = s[s <= smax]
    if len(s) < 10:
        return np.nan
    n = len(s)
    tau = 1 + n / np.sum(np.log(s / smin))
    return tau
