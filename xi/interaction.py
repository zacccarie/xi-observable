import numpy as np


def _entropy_from_counts(counts, base=2):
    counts = np.asarray(counts, dtype=np.float64)
    counts = counts[counts > 0]
    total = counts.sum()
    p = counts / total
    log = np.log2 if base == 2 else np.log
    return -np.sum(p * log(p))


def joint_entropy(samples, base=2):
    """samples (N, d) — entropie jointe par histogramme exact."""
    samples = np.asarray(samples)
    if samples.ndim == 1:
        samples = samples[:, None]
    # Encode chaque ligne en entier unique
    N, d = samples.shape
    cards = np.array([np.max(samples[:, j]) + 1 for j in range(d)])
    encoded = np.zeros(N, dtype=np.int64)
    mult = 1
    for j in range(d):
        encoded += samples[:, j].astype(np.int64) * mult
        mult *= cards[j]
    _, counts = np.unique(encoded, return_counts=True)
    return _entropy_from_counts(counts, base=base)


def total_correlation(samples, base=2):
    """TC = Σ H(X_i) − H(X)."""
    samples = np.asarray(samples)
    N, d = samples.shape
    H_joint = joint_entropy(samples, base=base)
    H_marg = sum(joint_entropy(samples[:, [i]], base=base) for i in range(d))
    return H_marg - H_joint


def dual_total_correlation(samples, base=2):
    """DTC = H(X) − Σ H(X_i | X_{−i}) = H(X) − Σ [H(X) − H(X_{−i})]
              = Σ H(X_{−i}) − (d−1) H(X)."""
    samples = np.asarray(samples)
    N, d = samples.shape
    H_joint = joint_entropy(samples, base=base)
    H_comp = sum(joint_entropy(np.delete(samples, i, axis=1), base=base) for i in range(d))
    return H_comp - (d - 1) * H_joint


def o_information(samples, base=2):
    """Ω = TC − DTC.
    Convention : Ω > 0 → redondance-dominée ; Ω < 0 → synergie-dominée.
    """
    return total_correlation(samples, base=base) - dual_total_correlation(samples, base=base)
