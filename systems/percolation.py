"""Site percolation 2D sur grille L×L.

Chaque site occupé indépendamment avec probabilité p. Pas de dynamique
temporelle — μ = produit de Bernoulli(p). Transition géométrique à
p_c ≈ 0.5927 (cluster spannant infinit en L→∞).

Order parameter : P_∞ = taille du plus grand cluster / N.
Susceptibilité : χ = ⟨s²⟩ / ⟨s⟩ moyenne pondérée des tailles de clusters
finis (excluant le spannant).

Classe universalité ≠ Ising 2D — fournit test transversalité χ_info.
"""
import numpy as np
from scipy.ndimage import label


def make_config(L, p, seed=0):
    """Configuration aléatoire site percolation. Retourne (L,L) bool."""
    rng = np.random.default_rng(seed)
    return (rng.random((L, L)) < p)


def cluster_stats(config):
    """Labels les clusters connectés (4-neighbours). Retourne :
       - labels (L,L)
       - sizes (n_clusters,) tailles
       - P_inf : fraction du plus grand cluster (order param)
       - chi : susceptibilité (moments tailles clusters finis)
    """
    labels, n = label(config)
    if n == 0:
        return labels, np.array([]), 0.0, 0.0
    sizes = np.bincount(labels.flatten())[1:]  # skip background label 0
    N = config.size
    largest = sizes.max()
    P_inf = largest / N
    # χ_perco : ⟨s²⟩/⟨s⟩ des clusters FINIS = tous sauf le plus grand TOUJOURS
    if len(sizes) > 1:
        sorted_desc = np.sort(sizes)[::-1]
        finite_sizes = sorted_desc[1:]
        chi = (finite_sizes ** 2).sum() / max(finite_sizes.sum(), 1)
    else:
        chi = 0.0
    return labels, sizes, P_inf, chi


def sample_ensemble(L, p, n_samples, seed=0):
    """Retourne array (n_samples, L, L) int8 ∈ {0,1}."""
    rng = np.random.default_rng(seed)
    return (rng.random((n_samples, L, L)) < p).astype(np.int8)
