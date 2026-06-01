import numpy as np


def make_rbn(N, K, p=0.5, seed=0):
    """Réseau booléen aléatoire Kauffman N-K.
    inputs : (N, K) indices des K entrées de chaque nœud.
    tables : (N, 2^K) table de vérité par nœud, biais p de sortie=1.
    """
    rng = np.random.default_rng(seed)
    inputs = np.empty((N, K), dtype=np.int64)
    for i in range(N):
        inputs[i] = rng.choice(N, size=K, replace=False)
    tables = (rng.random((N, 2 ** K)) < p).astype(np.int8)
    return inputs, tables


def step_batch(states, inputs, tables, K):
    """Update synchrone vectorisé sur M conditions initiales en parallèle.
    states : (M, N) int8 ∈ {0,1}. Retourne (M, N)."""
    M, N = states.shape
    idx = np.zeros((M, N), dtype=np.int64)
    for j in range(K):
        col = inputs[:, j]                       # (N,)
        idx = (idx << 1) | states[:, col].astype(np.int64)  # (M, N)
    node_idx = np.arange(N)[None, :]             # (1, N)
    return tables[node_idx, idx].astype(np.int8)  # (M, N)


def derrida_slope(inputs, tables, K, N, n_pairs=2000, seed=0):
    """Paramètre de Derrida : pente Hamming(t+1) vs Hamming(t) sur 1 pas.
    >1 chaos, <1 ordre, =1 edge of chaos. Annealed K_c : 2Kp(1-p)=1."""
    rng = np.random.default_rng(seed)
    s1 = rng.integers(0, 2, size=(n_pairs, N)).astype(np.int8)
    s2 = s1.copy()
    # perturbe une fraction aléatoire de bits par paire
    for m in range(n_pairs):
        nflip = rng.integers(1, max(2, N // 4))
        idx = rng.choice(N, nflip, replace=False)
        s2[m, idx] ^= 1
    din = (s1 != s2).mean(axis=1)
    o1 = step_batch(s1, inputs, tables, K)
    o2 = step_batch(s2, inputs, tables, K)
    dout = (o1 != o2).mean(axis=1)
    slope = np.polyfit(din, dout, 1)[0]
    return slope


def attractor_ensemble(inputs, tables, K, N, n_init=3000, t_equil=300, seed=0):
    """Échantillonne la mesure stationnaire μ : état final depuis n_init CI aléatoires.
    Retourne (n_init, N) ∈ {0,1}."""
    rng = np.random.default_rng(seed)
    states = rng.integers(0, 2, size=(n_init, N)).astype(np.int8)
    for _ in range(t_equil):
        states = step_batch(states, inputs, tables, K)
    return states


def frozen_fraction(ensemble):
    """Fraction de nœuds constants sur tout l'ensemble (cœur gelé)."""
    same_as_first = (ensemble == ensemble[0]).all(axis=0)
    return same_as_first.mean()


def activity(ensemble):
    """Fraction moyenne de 1."""
    return ensemble.mean()
