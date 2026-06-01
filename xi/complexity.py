"""Lempel-Ziv 76 complexity et variantes — proxy calculable de Kolmogorov K.

LZ76 : nombre de phrases distinctes dans le parse greedy de la séquence.
Normalisation Lempel : c(n) · log_2(n) / n → 1 pour aléatoire, → 0 pour périodique.

Référence : Lempel & Ziv, "On the complexity of finite sequences", IEEE TIT 1976.
"""
import numpy as np


def lz76(s):
    """LZ76 sur séquence d'entiers (ou string). Retourne nb phrases.

    Implémentation rapide via bytes.find (C-implementé). O(n² · k) avec petite
    constante. Pour n=10^5 binaire : ~1s vs minutes pour version pure-python.
    """
    if isinstance(s, np.ndarray):
        s = bytes(s.astype(np.uint8))
    elif isinstance(s, str):
        s = s.encode("latin1")
    elif isinstance(s, list):
        s = bytes(int(x) for x in s)
    n = len(s)
    if n == 0:
        return 0
    i = 0
    c = 0
    while i < n:
        L = 1
        # Étend phrase tant que s[i:i+L] est sous-chaîne de s[:i+L-1]
        while i + L <= n and s[i:i + L] in s[:i + L - 1]:
            L += 1
        c += 1
        i += L
    return c


def lz_normalized(s):
    """LZ normalisé Lempel : c·log_2(n) / n. Tend vers 1 pour aléatoire pur."""
    n = len(s)
    if n < 2:
        return 0.0
    c = lz76(s)
    return c * np.log2(n) / n


def flatten_snapshots_to_bits(snapshots, n_bins=2):
    """Convertit séquence snapshots en string de bits/symboles.

    snapshots : list of arrays (shape variable), valeurs entières ∈ [0, n_bins).
    Retourne array 1D d'entiers concaténés.
    """
    flat = np.concatenate([np.asarray(s).flatten() for s in snapshots])
    return flat.astype(np.int8)


def lz_per_realization(snapshots):
    """LZ normalisé d'une réalisation entière (concaténation snapshots aplatis)."""
    bits = flatten_snapshots_to_bits(snapshots)
    return lz_normalized(bits)
