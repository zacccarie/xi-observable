import numpy as np

from .edge import susceptibility, edge_mask
from .interaction import o_information, total_correlation
from .hierarchy import rg_blocks, blocks_at_scale


def xi_observable(
    spin_snapshots_per_T,
    T_values,
    L,
    block_size=2,
    epsilon=0.3,
    beta=0.0,
    chi_ref_max=None,
):
    """Stade A : Ξ_A(T) = mask_χ(T) · interaction(T).

    spin_snapshots_per_T : dict {T: list of (L,L) int arrays in {-1,+1}}.
    Retourne dict de séries indexées par T_values.
    """
    T_values = np.asarray(T_values, dtype=np.float64)
    chi = np.zeros_like(T_values)
    o_info = np.zeros_like(T_values)
    tc = np.zeros_like(T_values)
    n_snap = np.zeros_like(T_values)

    for i, T in enumerate(T_values):
        snaps = spin_snapshots_per_T[float(T)]
        n_snap[i] = len(snaps)
        mags = np.array([s.mean() for s in snaps])
        chi[i] = susceptibility(mags, T, L * L)

        # Échantillonne tous les blocs sur toutes les snapshots, recode en {0,1}
        blocks = np.concatenate([rg_blocks(s, block_size) for s in snaps], axis=0)
        blocks01 = ((blocks + 1) // 2).astype(np.int8)
        o_info[i] = o_information(blocks01)
        tc[i] = total_correlation(blocks01)

    mask = edge_mask(chi, epsilon=epsilon, ref_max=chi_ref_max)
    synergy = -o_info
    abs_o = np.abs(o_info)
    regime = np.sign(o_info)  # +1 redondance, -1 synergie

    xi_abs = abs_o * mask          # intensité interaction, signe-agnostique
    xi_syn = synergy * mask        # signé : positif=synergie, négatif=redondance
    xi_tc = tc * mask
    xi_chi_tc = tc * chi / (chi.max() + 1e-12)

    return {
        "T": T_values,
        "chi": chi,
        "o_info": o_info,
        "abs_o": abs_o,
        "regime": regime,
        "synergy": synergy,
        "tc": tc,
        "mask": mask.astype(int),
        "xi_abs": xi_abs,
        "xi_syn": xi_syn,
        "xi_tc": xi_tc,
        "xi_chi_tc": xi_chi_tc,
        "n_snap": n_snap,
    }


def xi_hierarchical(
    spin_snapshots_per_T,
    T_values,
    L,
    k_max=3,
    block_size=2,
    alpha=0.0,
    epsilon=0.3,
    chi_ref_max=None,
    encode="spin_binary",
    cg_rule="majority",
    chi_provided=None,
):
    """Stade B : Ξ multi-échelle via coarse-graining RG.

    Pour chaque T et chaque k ∈ [0, k_max]:
      - coarse-grain k fois (factor 2)
      - extrait blocs 2x2 du résultat
      - calcule TC(T, k) et Ω(T, k)

    Renvoie:
      tc_hier (n_T, k_max+1), abs_o_hier, gamma_T (exposant scaling log TC(k) ~ γ·k),
      xi_hier_alpha (somme pondérée par (k+1)^α), mask via χ.
    """
    T_values = np.asarray(T_values, dtype=np.float64)
    n_T = len(T_values)
    K = k_max + 1

    chi = np.zeros(n_T)
    tc_hier = np.zeros((n_T, K))
    abs_o_hier = np.zeros((n_T, K))
    o_hier = np.zeros((n_T, K))
    n_blocks_hier = np.zeros((n_T, K), dtype=int)

    for i, T in enumerate(T_values):
        snaps = spin_snapshots_per_T[float(T)]
        if chi_provided is not None:
            chi[i] = chi_provided[i]
        else:
            mags = np.array([s.mean() for s in snaps])
            chi[i] = susceptibility(mags, T, L * L)
        for k in range(K):
            blocks = blocks_at_scale(snaps, k=k, block_size=block_size, rule=cg_rule)
            if blocks is None or blocks.shape[0] < 16:
                tc_hier[i, k] = np.nan
                abs_o_hier[i, k] = np.nan
                o_hier[i, k] = np.nan
                continue
            if encode == "spin_binary":
                blocks_enc = ((blocks + 1) // 2).astype(np.int8)
            elif encode == "categorical":
                blocks_enc = blocks.astype(np.int8)
            else:
                raise ValueError(f"encode={encode} non supporté")
            o = o_information(blocks_enc)
            tc_hier[i, k] = total_correlation(blocks_enc)
            o_hier[i, k] = o
            abs_o_hier[i, k] = abs(o)
            n_blocks_hier[i, k] = blocks.shape[0]

    mask = edge_mask(chi, epsilon=epsilon, ref_max=chi_ref_max)

    # Exposant scaling γ : pente régression linéaire log(TC(k)) vs k
    gamma_T = np.full(n_T, np.nan)
    for i in range(n_T):
        ys = tc_hier[i]
        valid = ~np.isnan(ys) & (ys > 1e-9)
        if valid.sum() >= 2:
            xs = np.arange(K)[valid]
            log_y = np.log(ys[valid])
            slope, _ = np.polyfit(xs, log_y, 1)
            gamma_T[i] = slope

    # Ξ_hier : somme pondérée par h(k, α) = (k+1)^α, gatée par mask
    weights = np.array([(k + 1) ** alpha for k in range(K)])
    weighted_sum = np.zeros(n_T)
    for i in range(n_T):
        valid = ~np.isnan(tc_hier[i])
        if valid.any():
            weighted_sum[i] = np.nansum(tc_hier[i, valid] * weights[valid])
    xi_hier = weighted_sum * mask

    # Saturation : niveau k où TC(k) tombe sous threshold relatif (10% du max)
    saturation_k = np.full(n_T, np.nan)
    for i in range(n_T):
        ys = tc_hier[i]
        valid = ~np.isnan(ys)
        if valid.sum() >= 2 and ys[valid].max() > 0:
            thresh = ys[valid].max() * 0.1
            for k in range(K):
                if not np.isnan(ys[k]) and ys[k] < thresh:
                    saturation_k[i] = k
                    break
            else:
                saturation_k[i] = K  # n'a jamais saturé

    return {
        "T": T_values,
        "chi": chi,
        "mask": mask.astype(int),
        "tc_hier": tc_hier,
        "o_hier": o_hier,
        "abs_o_hier": abs_o_hier,
        "n_blocks_hier": n_blocks_hier,
        "gamma_T": gamma_T,
        "xi_hier": xi_hier,
        "alpha": alpha,
        "k_max": k_max,
        "saturation_k": saturation_k,
        "weights": weights,
    }
