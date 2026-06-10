"""Ξ v2 — synergie nette surrogate-corrigée, gatée par criticalité, pondérée par hiérarchie.

Différences vs observable.py (v1) :
  1. SURROGATE OBLIGATOIRE — méta-leçon du programme : le chaos profond et le
     biais d'estimation fabriquent de la fausse synergie. La synergie nette est
     s_net = max(0, Ω_surrogate − Ω_réel), pas −Ω brut.
  2. SYNERGIE (pas |Ω|, pas TC) — c'est la composante émergence vraie (Ω < 0).
     TC/|Ω| sont gardés comme ABLATIONS pour prouver que la synergie est load-bearing.
  3. COMPOSITION par moyenne gatée (pas produit) — la multiplicativité stricte
     E = Irr × Crit a été réfutée (emergetics finding 003). Forme retenue :
     moyenne dans la fenêtre critique, pondérée par la masse d'interaction par échelle.

Ξ_v2(S) = Σ_g gate(g) · Σ_ℓ p_ℓ(g) · s_net(ℓ, g)  /  Σ_g gate(g)

  gate(g)  ∈ [0,1] : fenêtre edge-of-chaos (douce si tau > 0)
  p_ℓ(g)           : fraction de la masse d'interaction (TC) au niveau ℓ
  s_net(ℓ,g)       : synergie nette surrogate-corrigée au niveau ℓ
"""
import numpy as np

from .interaction import o_information, total_correlation


# ---------------------------------------------------------------- synergie nette

def shuffle_columns(samples, rng):
    """Surrogate : shuffle indépendant par variable. Préserve marginales,
    détruit toutes les corrélations jointes."""
    out = samples.copy()
    for j in range(out.shape[1]):
        rng.shuffle(out[:, j])
    return out


def synergy_delta(samples, n_surrogates=5, seed=0):
    """delta = Ω_surr − Ω_réel, NON flooré. Convention Ω : >0 redondance, <0 synergie.

    PIÈGE STATISTIQUE (découvert killswitch v2.1) : floorer max(0, delta) PAR
    mesure puis moyenner donne E[max(0, bruit)] > 0 — biais positif systématique
    qui fabrique de la fausse synergie nette là où delta vrai = 0.
    → toujours moyenner les deltas D'ABORD, floorer UNE FOIS à la fin.
    """
    samples = np.asarray(samples)
    om_real = o_information(samples)
    rng = np.random.default_rng(seed)
    om_surr = float(np.mean([
        o_information(shuffle_columns(samples, rng)) for _ in range(n_surrogates)
    ]))
    return om_surr - om_real, om_real, om_surr


def net_synergy(samples, n_surrogates=4, seed=0, return_parts=False):
    """s_net = max(0, Ω_surr − Ω_réel) pour une mesure UNIQUE.
    Pour un ensemble de subsets, utiliser synergy_delta + moyenne + floor final."""
    delta, om_real, om_surr = synergy_delta(samples, n_surrogates, seed)
    s_net = max(0.0, delta)
    if return_parts:
        return s_net, om_real, om_surr
    return s_net


# ---------------------------------------------------------------- gate critique

def soft_gate(chi_values, epsilon=0.3, tau=0.0, ref_max=None, smooth=True):
    """Porte edge-of-chaos sur l'axe de contrôle.

    tau = 0      → masque dur (compat edge_mask v1)
    tau > 0      → sigmoïde de raideur 1/(tau·seuil) : appartenance douce,
                   méta-paramètre de netteté de la fenêtre.
    ref_max      → seuil absolu externe (comparaisons inter-bandes).
    """
    v = np.asarray(chi_values, dtype=np.float64)
    if smooth and len(v) >= 5:
        v = np.convolve(v, np.ones(3) / 3.0, mode="same")
    vmax = ref_max if ref_max is not None else v.max()
    thr = vmax * (1.0 - epsilon)
    if tau <= 0:
        return (v >= thr).astype(np.float64)
    return 1.0 / (1.0 + np.exp(-(v - thr) / (tau * thr + 1e-12)))


# ---------------------------------------------------------------- composition Ξ v2

def hierarchy_weights(tc_per_level):
    """p_ℓ = fraction de masse d'interaction par niveau + profondeur effective D_h.

    D_h = exp(−Σ p ln p) : 1 si tout vit à une échelle, ≈ n_levels si réparti."""
    tc = np.asarray(tc_per_level, dtype=np.float64)
    tc = np.where(np.isnan(tc), 0.0, np.clip(tc, 0.0, None))
    total = tc.sum()
    if total <= 1e-12:
        K = len(tc)
        return np.full(K, 1.0 / K), 1.0
    p = tc / total
    nz = p[p > 0]
    d_h = float(np.exp(-np.sum(nz * np.log(nz))))
    return p, d_h


def xi_v2(
    gate,
    syn_net_per_level,           # (n_control, n_levels)
    tc_per_level=None,           # (n_control, n_levels) — pour p_ℓ ; None → plat
    use_gate=True,
    use_hierarchy=True,
    swap_tc=False,               # ABLATION : remplace s_net par TC (détecteur criticalité ?)
):
    """Ξ v2 + variantes d'ablation.

    Retourne dict : xi (scalaire), per_control (n_control,), weights, d_h_mean.
    """
    syn = np.asarray(syn_net_per_level, dtype=np.float64)
    n_control, n_levels = syn.shape
    gate = np.asarray(gate, dtype=np.float64) if use_gate else np.ones(n_control)

    signal = np.asarray(tc_per_level, dtype=np.float64) if swap_tc else syn
    signal = np.where(np.isnan(signal), 0.0, signal)

    per_control = np.zeros(n_control)
    d_h_list = []
    for i in range(n_control):
        if use_hierarchy and tc_per_level is not None:
            p, d_h = hierarchy_weights(tc_per_level[i])
        else:
            p, d_h = np.full(n_levels, 1.0 / n_levels), 1.0
        per_control[i] = float(np.sum(p * signal[i]))
        d_h_list.append(d_h)

    Z = gate.sum()
    xi = float(np.sum(gate * per_control) / Z) if Z > 1e-12 else 0.0
    return {
        "xi": xi,
        "per_control": per_control,
        "gate": gate,
        "d_h_mean": float(np.mean(d_h_list)),
    }
