import numpy as np


def _make_checkerboard(L):
    ii, jj = np.indices((L, L))
    even = ((ii + jj) % 2 == 0)
    return even, ~even


def checkerboard_sweep(state, T, beta_inv=None):
    """Une passe complète : update sub-lattice paire puis impaire (numpy vectorisé).
    Convention β = 1/T (J = 1, k_B = 1)."""
    L = state.shape[0]
    even, odd = _make_checkerboard(L)
    for mask in (even, odd):
        nb = (
            np.roll(state, 1, 0) + np.roll(state, -1, 0)
            + np.roll(state, 1, 1) + np.roll(state, -1, 1)
        )
        dE = 2.0 * state * nb
        rand = np.random.random(state.shape)
        accept = (dE <= 0) | (rand < np.exp(-dE / T))
        flip = accept & mask
        state = np.where(flip, -state, state)
    return state


def simulate(L=32, T=2.269, n_equil=2000, n_sample=300, sample_gap=5, seed=0):
    rng = np.random.default_rng(seed)
    state = rng.choice([-1, 1], size=(L, L)).astype(np.int8)
    np.random.seed(seed)
    for _ in range(n_equil):
        state = checkerboard_sweep(state, T)
    snapshots = []
    for _ in range(n_sample):
        for _ in range(sample_gap):
            state = checkerboard_sweep(state, T)
        snapshots.append(state.copy())
    return snapshots
