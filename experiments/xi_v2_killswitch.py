"""Ξ v2 — KILL-SWITCH cross-système (pré-enregistré).

HYPOTHÈSE H : Ξ v2 (synergie nette × gate critique × hiérarchie) sépare
  l'émergence vraie (synergie-à-la-criticalité) de la criticalité-sans-émergence.

PRÉDICTIONS PRÉ-ENREGISTRÉES (avant run) :
  P1. Ξ(sandpile SOC)  ≫ Ξ(Ising)      — sandpile = synergie pure (Ω≈−0.5 connu),
                                          Ising Tc = redondance pure (Ω>0 connu).
      Seuil : Ξ_sandpile ≥ 5× Ξ_ising.
  P2. Ξ(RBN sweep K)   > Ξ(Ising)      — synergie nette pique à l'edge K≈2-3.
  P3. ABLATION swap_tc : Ξ_TC(Ising) devient COMPARABLE à Ξ_TC(sandpile)
      (ratio < 5×) → l'ordre s'écrase → la SYNERGIE est la composante load-bearing,
      pas l'intensité d'interaction. Si l'ordre persiste identique sous TC,
      la synergie n'est pas load-bearing → nuance majeure.
  P4. ABLATION no_gate : sur Ising, Ξ change peu (la synergie est ≈0 partout) ;
      le gate est surtout protecteur sur les systèmes à fenêtre (RBN).

RÈGLE DE DÉCISION (déclarée avant run) :
  KEEP H  si P1 et P2 tiennent (séparation > barres d'erreur) et P3 confirme.
  KILL H  si Ξ(Ising) ≥ 0.5 × Ξ(sandpile) → Ξ = détecteur de criticalité déguisé.

Budget : CPU local, ~3-6 min. Seeds : Ising 3, RBN 8, sandpile 6.
"""
import sys
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from systems.ising2d import simulate as ising_simulate
from systems.sandpile import simulate as sandpile_simulate, fit_powerlaw_exponent
from systems.rbn import make_rbn, attractor_ensemble, derrida_slope
from xi.edge import susceptibility
from xi.hierarchy import blocks_at_scale
from xi.interaction import total_correlation
from xi.observable_v2 import net_synergy, synergy_delta, soft_gate, xi_v2

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xi_v2_killswitch")


# ---------------------------------------------------------------- mesure par niveau

def random_subset_measure(snaps, subset_size=4, n_subsets=120, binarize=None, seed=0):
    """Portée GLOBALE : variables = sites aléatoires distants, samples = snapshots.
    Capture la synergie longue-portée (médiée par avalanches/corrélations critiques)
    que les blocs 2x2 locaux ratent. Conjecture C3 emergetics : le sampler doit
    matcher la portée effective de C. Retourne (s_net moyen, TC moyen, Ω moyen)."""
    flat = np.stack([s.flatten() for s in snaps])           # (n_snap, L*L)
    if binarize == "spin":
        flat = ((flat + 1) // 2).astype(np.int8)
    elif binarize == "z2":
        flat = (flat >= 2).astype(np.int8)
    else:
        flat = flat.astype(np.int8)
    rng = np.random.default_rng(seed)
    N = flat.shape[1]
    deltas, oms, tcs = [], [], []
    for _ in range(n_subsets):
        idx = rng.choice(N, subset_size, replace=False)
        sub = flat[:, idx]
        if sub.std(axis=0).min() < 1e-9:
            continue
        d, om_r, _ = synergy_delta(sub, n_surrogates=5, seed=int(rng.integers(1e6)))
        deltas.append(d)
        oms.append(om_r)
        tcs.append(total_correlation(sub))
    if not deltas:
        return 0.0, 0.0, 0.0
    # moyenne des deltas PUIS floor unique — évite le biais E[max(0, bruit)] > 0
    return max(0.0, float(np.mean(deltas))), float(np.mean(oms)), float(np.mean(tcs))


def levels_from_snapshots(snaps, k_levels=(0, 1, 2), binarize=None, seed=0):
    """Pour chaque échelle k : coarse-grain, blocs 2x2 → (s_net, TC, Ω_réel, Ω_surr)."""
    out = {"s_net": [], "tc": [], "om_real": [], "om_surr": []}
    for k in k_levels:
        blocks = blocks_at_scale(snaps, k=k, block_size=2, rule="majority")
        if blocks is None or blocks.shape[0] < 64:
            for key in out:
                out[key].append(np.nan)
            continue
        if binarize == "spin":
            enc = ((blocks + 1) // 2).astype(np.int8)
        elif binarize == "z2":
            enc = (blocks >= 2).astype(np.int8)
        else:
            enc = blocks.astype(np.int8)
        s_net, om_r, om_s = net_synergy(enc, n_surrogates=4, seed=seed, return_parts=True)
        out["s_net"].append(s_net)
        out["tc"].append(total_correlation(enc))
        out["om_real"].append(om_r)
        out["om_surr"].append(om_s)
    return {key: np.array(val) for key, val in out.items()}


# ---------------------------------------------------------------- Ising

def run_ising(T_grid, L=32, n_seeds=3, n_equil=1200, n_sample=120, gap=5):
    print(f"[ising] L={L} T_grid={list(T_grid)} seeds={n_seeds}")
    per_seed = []
    for seed in range(n_seeds):
        chi = np.zeros(len(T_grid))
        syn = np.zeros((len(T_grid), 3))
        tc = np.zeros((len(T_grid), 3))
        om = np.zeros((len(T_grid), 3))
        syn_g = np.zeros(len(T_grid))
        tc_g = np.zeros(len(T_grid))
        om_g = np.zeros(len(T_grid))
        for i, T in enumerate(T_grid):
            snaps = ising_simulate(L=L, T=T, n_equil=n_equil, n_sample=n_sample,
                                   sample_gap=gap, seed=seed * 100 + i)
            mags = np.array([s.mean() for s in snaps])
            chi[i] = susceptibility(mags, T, L * L)
            lv = levels_from_snapshots(snaps, binarize="spin", seed=seed)
            syn[i], tc[i], om[i] = lv["s_net"], lv["tc"], lv["om_real"]
            syn_g[i], om_g[i], tc_g[i] = random_subset_measure(
                snaps, binarize="spin", seed=seed * 7 + i)
        per_seed.append({"chi": chi, "syn": syn, "tc": tc, "om": om,
                         "syn_g": syn_g, "tc_g": tc_g, "om_g": om_g})
        print(f"  seed {seed} : chi_max={chi.max():.1f} @T={T_grid[np.argmax(chi)]:.3f} "
              f"| Ω@Tc-window={om[np.argmax(chi)]}")
    return per_seed


# ---------------------------------------------------------------- RBN

def omega_subsets(ensemble, subset_size=4, n_subsets=150, seed=0):
    """Ω moyen + TC moyen + s_net moyen sur subsets aléatoires de nœuds."""
    rng = np.random.default_rng(seed)
    N = ensemble.shape[1]
    deltas, oms, tcs = [], [], []
    for _ in range(n_subsets):
        idx = rng.choice(N, subset_size, replace=False)
        sub = ensemble[:, idx]
        if sub.std(axis=0).min() < 1e-9:        # nœud gelé → Ω dégénéré, skip
            continue
        d, om_r, _ = synergy_delta(sub, n_surrogates=5, seed=int(rng.integers(1e6)))
        deltas.append(d)
        oms.append(om_r)
        tcs.append(total_correlation(sub))
    if not deltas:
        return 0.0, 0.0, 0.0
    return max(0.0, float(np.mean(deltas))), float(np.mean(oms)), float(np.mean(tcs))


def run_rbn(K_grid=(1, 2, 3, 4, 5), N=60, n_seeds=8):
    print(f"[rbn] N={N} K_grid={list(K_grid)} seeds={n_seeds}")
    syn = np.zeros((n_seeds, len(K_grid)))
    om = np.zeros((n_seeds, len(K_grid)))
    tc = np.zeros((n_seeds, len(K_grid)))
    derrida = np.zeros((n_seeds, len(K_grid)))
    for s in range(n_seeds):
        for i, K in enumerate(K_grid):
            inputs, tables = make_rbn(N, K, p=0.5, seed=s * 37 + i)
            ens = attractor_ensemble(inputs, tables, K, N, n_init=1500,
                                     t_equil=250, seed=s)
            derrida[s, i] = derrida_slope(inputs, tables, K, N, n_pairs=1000, seed=s)
            syn[s, i], om[s, i], tc[s, i] = omega_subsets(ens, seed=s * 11 + i)
        print(f"  seed {s} : s_net(K)={np.round(syn[s], 4)}")
    return {"syn": syn, "om": om, "tc": tc, "derrida": derrida}


# ---------------------------------------------------------------- Sandpile

def run_sandpile(L=32, n_seeds=6, n_drops=2500):
    print(f"[sandpile] L={L} seeds={n_seeds}")
    rows = []
    for seed in range(n_seeds):
        _, sizes, snaps = sandpile_simulate(L=L, n_drops=n_drops, n_warmup=800, seed=seed)
        tau = fit_powerlaw_exponent(sizes)
        lv = levels_from_snapshots(snaps, binarize="z2", seed=seed)
        syn_g, om_g, tc_g = random_subset_measure(snaps, binarize="z2", seed=seed * 7)
        rows.append({"tau": tau, "syn": lv["s_net"], "tc": lv["tc"], "om": lv["om_real"],
                     "syn_g": syn_g, "tc_g": tc_g, "om_g": om_g})
        print(f"  seed {seed} : tau={tau:.3f} s_net_loc={np.round(lv['s_net'], 4)} "
              f"| GLOBAL s_net={syn_g:.4f} Ω={om_g:.4f}")
    return rows


# ---------------------------------------------------------------- assemblage Ξ

def xi_for_ising(per_seed, variant):
    """Ξ = sup sur portées d'échantillonnage (local blocs / global subsets).
    Incarne le sup_Φ de U(S|F) : l'observable choisit l'opérateur de mesure
    qui révèle le plus de signal net (C3 : sampler doit matcher portée de C)."""
    vals = []
    for ps in per_seed:
        gate = soft_gate(ps["chi"], epsilon=0.3, tau=0.1)
        r_loc = xi_v2(gate, ps["syn"], ps["tc"], **variant)
        r_glo = xi_v2(gate, ps["syn_g"][:, None], ps["tc_g"][:, None], **variant)
        vals.append(max(r_loc["xi"], r_glo["xi"]))
    return vals


def xi_for_rbn(rbn, variant):
    # gate partagé sur l'axe K : χ_info = std inter-seeds de Ω (emergetics C4)
    chi_info_K = rbn["om"].std(axis=0)
    gate = soft_gate(chi_info_K, epsilon=0.3, tau=0.1, smooth=False)
    vals = []
    for s in range(rbn["syn"].shape[0]):
        r = xi_v2(gate, rbn["syn"][s][:, None], rbn["tc"][s][:, None], **variant)
        vals.append(r["xi"])
    return vals, gate, chi_info_K


def xi_for_sandpile(rows, variant):
    vals = []
    for row in rows:
        gate_ok = 1.0 if (1.0 <= row["tau"] <= 2.0) else 0.0   # SOC vérifié par loi puissance
        gate = np.array([gate_ok if variant.get("use_gate", True) else 1.0])
        var = {k: v for k, v in variant.items() if k != "use_gate"}
        r_loc = xi_v2(gate, row["syn"][None, :], row["tc"][None, :], **var)
        r_glo = xi_v2(gate, np.array([[row["syn_g"]]]), np.array([[row["tc_g"]]]), **var)
        vals.append(max(r_loc["xi"], r_glo["xi"]))
    return vals


def main():
    t0 = time.time()
    T_grid = np.array([1.6, 1.9, 2.1, 2.2, 2.27, 2.35, 2.5, 2.8, 3.2])

    ising = run_ising(T_grid)
    rbn = run_rbn()
    sandpile = run_sandpile()

    variants = {
        "full":      dict(use_gate=True, use_hierarchy=True, swap_tc=False),
        "no_gate":   dict(use_gate=False, use_hierarchy=True, swap_tc=False),
        "flat_hier": dict(use_gate=True, use_hierarchy=False, swap_tc=False),
        "swap_tc":   dict(use_gate=True, use_hierarchy=True, swap_tc=True),
    }

    table = {}
    for name, var in variants.items():
        xi_i = xi_for_ising(ising, var)
        xi_r, gate_r, chi_info_K = xi_for_rbn(rbn, var)
        xi_s = xi_for_sandpile(sandpile, var)
        table[name] = {
            "ising":    {"mean": float(np.mean(xi_i)), "std": float(np.std(xi_i)), "vals": xi_i},
            "rbn":      {"mean": float(np.mean(xi_r)), "std": float(np.std(xi_r)), "vals": xi_r},
            "sandpile": {"mean": float(np.mean(xi_s)), "std": float(np.std(xi_s)), "vals": xi_s},
        }

    # -------- verdict pré-enregistré
    full = table["full"]
    eps = 1e-12
    ratio_sand_ising = full["sandpile"]["mean"] / (full["ising"]["mean"] + eps)
    ratio_rbn_ising = full["rbn"]["mean"] / (full["ising"]["mean"] + eps)
    tcv = table["swap_tc"]
    ratio_tc = tcv["sandpile"]["mean"] / (tcv["ising"]["mean"] + eps)

    p1 = ratio_sand_ising >= 5.0
    p2 = full["rbn"]["mean"] > full["ising"]["mean"] + full["ising"]["std"] + full["rbn"]["std"]
    p3 = ratio_tc < 5.0
    kill = full["ising"]["mean"] >= 0.5 * full["sandpile"]["mean"]
    verdict = "KILL" if kill else ("KEEP" if (p1 and p2) else "NUANCE")

    out = {
        "preregistered": {
            "P1_sandpile_ge_5x_ising": bool(p1),
            "P2_rbn_gt_ising_beyond_std": bool(p2),
            "P3_tc_ablation_crushes_separation": bool(p3),
            "ratio_sandpile_over_ising": ratio_sand_ising,
            "ratio_rbn_over_ising": ratio_rbn_ising,
            "ratio_tc_ablation": ratio_tc,
            "verdict": verdict,
        },
        "table": table,
        "rbn_gate_K": gate_r.tolist(),
        "rbn_chi_info_K": chi_info_K.tolist(),
        "rbn_derrida_mean": rbn["derrida"].mean(axis=0).tolist(),
        "T_grid": T_grid.tolist(),
        "runtime_s": time.time() - t0,
    }
    with open(RESULTS + ".json", "w") as f:
        json.dump(out, f, indent=1)

    # -------- figure (fond noir, émissif)
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    systems = ["ising", "rbn", "sandpile"]
    colors = {"ising": "#00e5ff", "rbn": "#b388ff", "sandpile": "#ff6d00"}
    x = np.arange(len(systems))
    for ax, vname, title in [(axes[0], "full", "Ξ v2 (full)"),
                             (axes[1], "swap_tc", "ablation swap → TC")]:
        means = [table[vname][s]["mean"] for s in systems]
        stds = [table[vname][s]["std"] for s in systems]
        ax.bar(x, means, yerr=stds, capsize=5,
               color=[colors[s] for s in systems], alpha=0.9,
               error_kw=dict(ecolor="white", alpha=0.6))
        ax.set_xticks(x)
        ax.set_xticklabels(["Ising 2D\n(critique, redondant)",
                            "RBN sweep K\n(edge of chaos)",
                            "Sandpile BTW\n(SOC, synergique)"])
        ax.set_title(title, color="white")
        ax.set_ylabel("Ξ")
    fig.suptitle(f"Kill-switch Ξ v2 — verdict {verdict} "
                 f"(sandpile/ising = {ratio_sand_ising:.1f}×, TC-ablation = {ratio_tc:.1f}×)",
                 color="white")
    plt.tight_layout()
    plt.savefig(RESULTS + ".png", dpi=130, facecolor="black")

    print("\n" + "=" * 60)
    for name in variants:
        row = table[name]
        print(f"{name:10s} | " + " | ".join(
            f"{s}: {row[s]['mean']:.4f}±{row[s]['std']:.4f}" for s in systems))
    print(f"P1 (sand ≥ 5× ising) : {p1}  [{ratio_sand_ising:.1f}×]")
    print(f"P2 (rbn > ising+σ)   : {p2}  [{ratio_rbn_ising:.1f}×]")
    print(f"P3 (TC écrase sép.)  : {p3}  [{ratio_tc:.1f}×]")
    print(f"VERDICT : {verdict}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
