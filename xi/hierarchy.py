import numpy as np


def rg_blocks(state_grid, block_size=2):
    """Partition LxL grid en (L/b)^2 blocs aplatis (b*b,).
    Retourne array (num_blocks, b*b)."""
    L = state_grid.shape[0]
    b = block_size
    assert L % b == 0, f"L={L} non divisible par b={b}"
    n_blocks = (L // b) ** 2
    out = np.empty((n_blocks, b * b), dtype=state_grid.dtype)
    k = 0
    for i in range(0, L, b):
        for j in range(0, L, b):
            out[k] = state_grid[i:i + b, j:j + b].flatten()
            k += 1
    return out


def coarse_grain(state_grid, block_size=2, rule="majority"):
    """Réduit LxL → (L/b)x(L/b).
    rule="majority"  : spins ±1, vote majoritaire (tie aléatoire)
    rule="mode"      : catégoriel entier (ex. quadrants 0..3), mode du bloc
    rule="mean"      : moyenne réelle
    """
    L = state_grid.shape[0]
    b = block_size
    assert L % b == 0
    Lp = L // b
    out = np.empty((Lp, Lp), dtype=state_grid.dtype)
    for i in range(Lp):
        for j in range(Lp):
            block = state_grid[i * b:(i + 1) * b, j * b:(j + 1) * b]
            if rule == "majority":
                s = block.sum()
                out[i, j] = 1 if s > 0 else (-1 if s < 0 else np.random.choice([-1, 1]))
            elif rule == "mode":
                vals, counts = np.unique(block, return_counts=True)
                out[i, j] = vals[np.argmax(counts)]
            else:
                out[i, j] = block.mean()
    return out


def hierarchical_partition(state_grid, max_level=None, block_size=2):
    """Génère partitions {0: atomique, 1: blocs b, 2: blocs b^2, ...}.
    Retourne dict {k: list_of_block_samples}."""
    L = state_grid.shape[0]
    parts = {}
    current = state_grid
    k = 0
    while current.shape[0] >= block_size and (max_level is None or k <= max_level):
        if k == 0:
            parts[0] = current.flatten()[:, None]
        else:
            parts[k] = rg_blocks(current, block_size)
        if current.shape[0] // block_size < block_size:
            break
        current = coarse_grain(current, block_size)
        k += 1
    return parts


def coarse_grain_stack(state_grid, k, rule="majority"):
    """Applique k coarse-grainings successifs (block_size=2 fixé). Renvoie grille L/2^k."""
    g = state_grid
    for _ in range(k):
        g = coarse_grain(g, block_size=2, rule=rule)
    return g


def blocks_at_scale(snapshots, k, block_size=2, rule="majority"):
    """Pour chaque snapshot, coarse-grain k fois puis extrait 2x2 blocks.
    Retourne array (N_blocks_total, block_size^2)."""
    out = []
    for s in snapshots:
        g = coarse_grain_stack(s, k, rule=rule)
        if g.shape[0] < block_size:
            return None
        out.append(rg_blocks(g, block_size))
    return np.concatenate(out, axis=0)
