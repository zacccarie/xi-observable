import numpy as np


def susceptibility(magnetizations, T, N):
    m = np.asarray(magnetizations, dtype=np.float64)
    return N * m.var() / T


def edge_mask(values, epsilon=0.2, smooth=True, ref_max=None):
    """Mask {ω : χ(ω) ≥ χ_ref · (1 − ε)}.

    ref_max=None  → seuil relatif au max LOCAL (sweep self-contained)
    ref_max=float → seuil absolu (utile pour comparer bandes off-critical)
    """
    v = np.asarray(values, dtype=np.float64)
    if smooth and len(v) >= 5:
        kernel = np.ones(3) / 3.0
        v_smooth = np.convolve(v, kernel, mode="same")
    else:
        v_smooth = v
    vmax = ref_max if ref_max is not None else v_smooth.max()
    threshold = vmax * (1.0 - epsilon)
    return v_smooth >= threshold


def edge_band(values, epsilon=0.2):
    mask = edge_mask(values, epsilon)
    idxs = np.where(mask)[0]
    if len(idxs) == 0:
        return None
    return idxs[0], idxs[-1]
