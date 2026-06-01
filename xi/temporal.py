import numpy as np

from .edge import edge_mask
from .interaction import o_information, total_correlation


def autocorr(x, max_lag=None):
    """Autocorrélation normalisée de x via FFT.
    Retourne r(τ) = ⟨(x_t − x̄)(x_{t+τ} − x̄)⟩ / Var(x), τ = 0..max_lag.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if max_lag is None:
        max_lag = n // 2
    xc = x - x.mean()
    var = xc.var()
    if var < 1e-12:
        return np.zeros(max_lag + 1)
    # FFT autocorr (Wiener-Khinchin)
    pad = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(xc, n=pad)
    acf = np.fft.irfft(f * np.conj(f), n=pad)[:n] / (n * var)
    return acf[: max_lag + 1]


def correlation_time(x, method="integrated", threshold=1.0 / np.e, max_lag=None):
    """τ_corr de la série x. Trois méthodes :
       - "integrated" : τ = Σ_{τ≥0} r(τ) jusqu'à premier zero crossing
       - "1_over_e"   : premier τ tel que r(τ) < 1/e
       - "exp_fit"    : fit r(τ) ~ exp(−τ/τ_corr) sur τ ∈ [0, lag_fit]
    """
    acf = autocorr(x, max_lag=max_lag)
    if method == "integrated":
        # somme jusqu'à premier zero crossing (cutoff classique)
        zero_idx = np.where(acf < 0)[0]
        cutoff = zero_idx[0] if len(zero_idx) > 0 else len(acf)
        tau = acf[1:cutoff].sum() + 0.5  # +0.5 = correction Sokal
        return max(tau, 0.0)
    elif method == "1_over_e":
        below = np.where(acf < threshold)[0]
        return float(below[0]) if len(below) > 0 else float(len(acf))
    elif method == "exp_fit":
        acf_pos = acf[acf > 0]
        if len(acf_pos) < 5:
            return np.nan
        xs = np.arange(len(acf_pos))
        log_y = np.log(acf_pos)
        slope, _ = np.polyfit(xs, log_y, 1)
        return -1.0 / slope if slope < 0 else np.inf
    else:
        raise ValueError(method)


def circular_window_mean(phases_series, window):
    """Moyenne circulaire des phases sur fenêtres consécutives de longueur 'window'.
    phases_series : (T, N) angles en radians.
    Retourne (T//window, N) angles moyens dans [0, 2π).
    """
    z = np.exp(1j * phases_series)
    T, N = z.shape
    nw = T // window
    if nw == 0:
        return np.zeros((0, N))
    z_trunc = z[: nw * window].reshape(nw, window, N)
    z_mean = z_trunc.mean(axis=1)
    return np.mod(np.angle(z_mean), 2 * np.pi)


def quadrantize_phases(phases, n_bins=4):
    return np.floor(phases / (2 * np.pi / n_bins)).astype(np.int8)


def tc_random_subsets(samples, subset_size=4, n_subsets=200, rng=None):
    """samples (N_samples, N_var) catégoriel. Tire n_subsets de subset_size variables
    au hasard, calcule TC et |Ω| sur chacun, retourne moyennes."""
    if rng is None:
        rng = np.random.default_rng(0)
    N_samples, N_var = samples.shape
    if subset_size > N_var:
        return np.nan, np.nan
    tcs, abs_os, os = [], [], []
    for _ in range(n_subsets):
        idx = rng.choice(N_var, size=subset_size, replace=False)
        sub = samples[:, idx]
        tc = total_correlation(sub)
        o = o_information(sub)
        tcs.append(tc)
        abs_os.append(abs(o))
        os.append(o)
    return (np.mean(tcs), np.mean(abs_os), np.mean(os),
            np.std(tcs), np.std(abs_os))


def xi_hierarchical_temporal(
    phases_series_per_K,
    K_values,
    N,
    k_max=4,
    n_bins=4,
    subset_size=4,
    n_subsets=200,
    epsilon=0.3,
    chi_provided=None,
    chi_ref_max=None,
    alpha=0.0,
    seed=0,
):
    """Ξ multi-échelle TEMPOREL pour systèmes à couplage global (Kuramoto).

    Pour chaque K, pour chaque k ∈ [0, k_max]:
      window = 2^k
      phases coarsened circulairement sur fenêtres de durée window·sample_gap
      quantize en n_bins
      tire n_subsets sous-ensembles aléatoires de subset_size oscillateurs
      compute TC et |Ω| moyens sur subsets

    Critical slowing down attendu : à K_c, TC reste haut pour tous k.
    Loin de K_c, TC chute exponentiellement avec k (décorrélation rapide).
    """
    K_values = np.asarray(K_values, dtype=np.float64)
    n_K = len(K_values)
    K_levels = k_max + 1

    chi = np.zeros(n_K)
    tc_hier = np.zeros((n_K, K_levels))
    tc_std = np.zeros((n_K, K_levels))
    abs_o_hier = np.zeros((n_K, K_levels))
    o_hier = np.zeros((n_K, K_levels))
    n_snap_at_k = np.zeros((n_K, K_levels), dtype=int)

    rng = np.random.default_rng(seed)
    for i, K in enumerate(K_values):
        phases = phases_series_per_K[float(K)]  # (T_snap, N)
        if chi_provided is not None:
            chi[i] = chi_provided[i]
        for k in range(K_levels):
            window = 2 ** k
            cg = circular_window_mean(phases, window)
            if cg.shape[0] < 32:
                tc_hier[i, k] = np.nan
                abs_o_hier[i, k] = np.nan
                o_hier[i, k] = np.nan
                continue
            samples = quadrantize_phases(cg, n_bins=n_bins)
            tc_mean, abs_o_mean, o_mean, tc_s, _ = tc_random_subsets(
                samples, subset_size=subset_size,
                n_subsets=n_subsets, rng=rng,
            )
            tc_hier[i, k] = tc_mean
            tc_std[i, k] = tc_s
            abs_o_hier[i, k] = abs_o_mean
            o_hier[i, k] = o_mean
            n_snap_at_k[i, k] = samples.shape[0]

    mask = edge_mask(chi, epsilon=epsilon, ref_max=chi_ref_max)

    gamma_T = np.full(n_K, np.nan)
    for i in range(n_K):
        ys = tc_hier[i]
        valid = ~np.isnan(ys) & (ys > 1e-9)
        if valid.sum() >= 2:
            xs = np.arange(K_levels)[valid]
            log_y = np.log(ys[valid])
            slope, _ = np.polyfit(xs, log_y, 1)
            gamma_T[i] = slope

    weights = np.array([(k + 1) ** alpha for k in range(K_levels)])
    weighted = np.zeros(n_K)
    for i in range(n_K):
        valid = ~np.isnan(tc_hier[i])
        if valid.any():
            weighted[i] = np.nansum(tc_hier[i, valid] * weights[valid])
    xi_hier = weighted * mask

    saturation_k = np.full(n_K, np.nan)
    for i in range(n_K):
        ys = tc_hier[i]
        valid = ~np.isnan(ys)
        if valid.sum() >= 2 and ys[valid].max() > 0:
            thresh = ys[valid].max() * 0.1
            for k in range(K_levels):
                if not np.isnan(ys[k]) and ys[k] < thresh:
                    saturation_k[i] = k
                    break
            else:
                saturation_k[i] = K_levels

    return {
        "K": K_values,
        "chi": chi,
        "mask": mask.astype(int),
        "tc_hier": tc_hier,
        "tc_std": tc_std,
        "abs_o_hier": abs_o_hier,
        "o_hier": o_hier,
        "gamma_T": gamma_T,
        "xi_hier": xi_hier,
        "saturation_k": saturation_k,
        "n_snap_at_k": n_snap_at_k,
        "k_max": k_max,
        "alpha": alpha,
    }
