"""Test ΔK universel — distance peak χ_info vs peak Var(LZ).

Suite test RBN (#17) : ΔK=1 (info à K=2, algo à K=3). Question : est-ce
universel ?

Test sur 3 systèmes spatiaux : Ising 2D, Kuramoto 2D local, percolation 2D.
Pour chaque : pour chaque valeur paramètre contrôle, n_real réalisations,
calculer (a) χ_info = std_realisations(Ω), (b) std_realisations(LZ).
Comparer arg peak. Si ΔK universellement non nul → dualité info/algo
trait fondamental.
"""
import sys
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from systems.ising2d import simulate as ising_sim
from systems.kuramoto_2d import simulate as kura2d_sim, chi_from_r as kura_chi
from systems.percolation import sample_ensemble as perco_sample
from xi.complexity import lz_normalized
from xi.interaction import o_information


def lz_of_snapshots(snapshots, encode="binary"):
    """Concat snapshots aplatis, encode {-1,1} ou catégoriel → LZ_normalized."""
    flat = np.concatenate([np.asarray(s).flatten() for s in snapshots])
    if encode == "binary":
        # spin ±1 → 0,1
        flat = ((flat + 1) // 2).astype(np.uint8)
    else:
        flat = flat.astype(np.uint8)
    return lz_normalized(flat)


def omega_random_subsets_grid(snapshots, subset_size=4, n_subsets=200,
                              encode="binary", rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    L = snapshots[0].shape[0]
    N = L * L
    flat = np.stack([s.flatten() for s in snapshots], axis=0)
    if encode == "binary":
        flat = ((flat + 1) // 2).astype(np.int8)
    else:
        flat = flat.astype(np.int8)
    oms = []
    for _ in range(n_subsets):
        idx = rng.choice(N, size=subset_size, replace=False)
        sub = flat[:, idx]
        oms.append(o_information(sub))
    return np.mean(oms)


def run_ising(L=16, T_values=None, n_real=8, n_sample=120, seed=42):
    if T_values is None:
        T_values = np.linspace(2.0, 2.8, 9)
    Tc = 2.0 / np.log(1.0 + np.sqrt(2.0))
    print(f"\n=== ISING L={L} | {len(T_values)} T | Tc={Tc:.4f} ===")
    om_means = np.zeros((len(T_values), n_real))
    lz_vals = np.zeros((len(T_values), n_real))
    rng_master = np.random.default_rng(seed)
    t0 = time.time()
    for i, T in enumerate(T_values):
        for r in range(n_real):
            s = int(rng_master.integers(0, 2**31))
            snaps = ising_sim(L=L, T=float(T), n_equil=600,
                              n_sample=n_sample, sample_gap=3, seed=s)
            om_means[i, r] = omega_random_subsets_grid(
                snaps, subset_size=4, n_subsets=150, encode="binary",
                rng=np.random.default_rng(s + 7),
            )
            lz_vals[i, r] = lz_of_snapshots(snaps, encode="binary")
        print(f"  T={T:.3f} ⟨Ω⟩={om_means[i].mean():+.4f} "
              f"χ_info={om_means[i].std():.4f} "
              f"⟨LZ⟩={lz_vals[i].mean():.4f} Var(LZ)={lz_vals[i].std():.4f}")
    print(f"  total {time.time()-t0:.1f}s")
    return T_values, om_means, lz_vals, Tc


def run_kura2d(L=12, K_values=None, n_real=6, seed=42):
    if K_values is None:
        K_values = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    Kc_eff = 3.0  # ≈ trouvé task #10
    print(f"\n=== KURAMOTO 2D L={L} | {len(K_values)} K | Kc_eff≈{Kc_eff} ===")
    om_means = np.zeros((len(K_values), n_real))
    lz_vals = np.zeros((len(K_values), n_real))
    rng_master = np.random.default_rng(seed)
    t0 = time.time()
    for i, K in enumerate(K_values):
        for r in range(n_real):
            s = int(rng_master.integers(0, 2**31))
            snaps, _ = kura2d_sim(
                L=L, K=float(K), sigma_omega=1.0,
                dt=0.05, t_equil=50.0, t_sample=80.0, sample_gap=1.0,
                n_bins=4, seed=s,
            )
            om_means[i, r] = omega_random_subsets_grid(
                snaps, subset_size=4, n_subsets=150, encode="cat",
                rng=np.random.default_rng(s + 7),
            )
            lz_vals[i, r] = lz_of_snapshots(snaps, encode="cat")
        print(f"  K={K:.2f} ⟨Ω⟩={om_means[i].mean():+.4f} "
              f"χ_info={om_means[i].std():.4f} "
              f"⟨LZ⟩={lz_vals[i].mean():.4f} Var(LZ)={lz_vals[i].std():.4f}")
    print(f"  total {time.time()-t0:.1f}s")
    return K_values, om_means, lz_vals, Kc_eff


def run_perco(L=20, p_values=None, n_real=8, n_sample=150, seed=42):
    if p_values is None:
        p_values = np.linspace(0.45, 0.75, 11)
    p_c = 0.5927
    print(f"\n=== PERCOLATION L={L} | {len(p_values)} p | p_c={p_c:.4f} ===")
    om_means = np.zeros((len(p_values), n_real))
    lz_vals = np.zeros((len(p_values), n_real))
    rng_master = np.random.default_rng(seed)
    t0 = time.time()
    for i, p in enumerate(p_values):
        for r in range(n_real):
            s = int(rng_master.integers(0, 2**31))
            ens = perco_sample(L, float(p), n_sample, seed=s)
            snaps = list(ens)
            om_means[i, r] = omega_random_subsets_grid(
                snaps, subset_size=4, n_subsets=150, encode="cat",
                rng=np.random.default_rng(s + 7),
            )
            # perco déjà 0/1, encode cat OK
            lz_vals[i, r] = lz_of_snapshots(snaps, encode="cat")
        print(f"  p={p:.4f} ⟨Ω⟩={om_means[i].mean():+.4f} "
              f"χ_info={om_means[i].std():.4f} "
              f"⟨LZ⟩={lz_vals[i].mean():.4f} Var(LZ)={lz_vals[i].std():.4f}")
    print(f"  total {time.time()-t0:.1f}s")
    return p_values, om_means, lz_vals, p_c


def analyse_system(name, x_values, om_means, lz_vals, critical):
    chi_info = om_means.std(axis=1)
    var_lz = lz_vals.std(axis=1)
    mean_lz = lz_vals.mean(axis=1)
    mean_om = om_means.mean(axis=1)
    abs_om = np.abs(mean_om)
    cv_om = chi_info / (abs_om + 1e-9)

    peak_chi_info = x_values[np.argmax(chi_info)]
    peak_var_lz = x_values[np.argmax(var_lz)]
    peak_cv = x_values[np.argmax(cv_om)]
    delta_k = peak_var_lz - peak_chi_info
    pearson = np.corrcoef(chi_info, var_lz)[0, 1] if len(chi_info) >= 3 else np.nan

    return {
        "system": name,
        "x_values": x_values.tolist(),
        "critical": float(critical),
        "chi_info": chi_info.tolist(),
        "var_lz": var_lz.tolist(),
        "mean_lz": mean_lz.tolist(),
        "mean_om": mean_om.tolist(),
        "cv_om": cv_om.tolist(),
        "peak_chi_info": float(peak_chi_info),
        "peak_var_lz": float(peak_var_lz),
        "peak_cv": float(peak_cv),
        "delta_k": float(delta_k),
        "delta_chi_critical": float(peak_chi_info - critical),
        "delta_varlz_critical": float(peak_var_lz - critical),
        "pearson_chiinfo_varlz": float(pearson),
    }


def run(outdir=None):
    if outdir is None:
        outdir = os.path.dirname(os.path.abspath(__file__))

    results = []

    # ISING
    Ts, om_i, lz_i, Tc = run_ising()
    res_i = analyse_system("Ising 2D", Ts, om_i, lz_i, Tc)
    results.append(res_i)

    # KURAMOTO 2D
    Ks, om_k, lz_k, Kc = run_kura2d()
    res_k = analyse_system("Kuramoto 2D", Ks, om_k, lz_k, Kc)
    results.append(res_k)

    # PERCOLATION
    ps, om_p, lz_p, pc = run_perco()
    res_p = analyse_system("Percolation 2D", ps, om_p, lz_p, pc)
    results.append(res_p)

    # Synthèse
    print("\n" + "=" * 75)
    print(f"{'Système':<18} {'critique':>10} {'peak χ_info':>12} "
          f"{'peak Var(LZ)':>13} {'ΔK':>8} {'Pearson':>8}")
    print("-" * 75)
    for r in results:
        print(f"{r['system']:<18} {r['critical']:>10.4f} "
              f"{r['peak_chi_info']:>12.4f} {r['peak_var_lz']:>13.4f} "
              f"{r['delta_k']:>+8.4f} {r['pearson_chiinfo_varlz']:>+8.3f}")
    print("=" * 75)

    # Verdict
    deltas = [r["delta_k"] for r in results]
    all_zero = all(abs(d) < 0.1 for d in deltas)
    all_nonzero = all(abs(d) > 0.05 for d in deltas)
    if all_zero:
        verdict = "Δ_K ≈ 0 universel — UNE seule frontière info=algo"
    elif all_nonzero:
        verdict = "ΔK ≠ 0 universel — DUALITÉ info/algo trait fondamental (conséquence #8 confirmée)"
    else:
        verdict = "ΔK variable — dépend de la classe de C"
    print(f"\nVerdict : {verdict}")

    # Plot
    fig, axes = plt.subplots(3, 2, figsize=(13, 10))
    for row, r in enumerate(results):
        x = np.array(r["x_values"])
        xc = r["critical"]
        ax = axes[row, 0]
        ax.plot(x, r["chi_info"], "o-", color="#00ff88", lw=1.5, ms=5,
                label="χ_info=std(Ω)")
        ax.plot(x, r["var_lz"], "s-", color="magenta", lw=1.5, ms=5,
                label="std(LZ)")
        ax.axvline(xc, color="#ff3355", ls="--", lw=1,
                   label=f"crit={xc:.3f}")
        ax.axvline(r["peak_chi_info"], color="#00ff88", ls=":", lw=1, alpha=0.7)
        ax.axvline(r["peak_var_lz"], color="magenta", ls=":", lw=1, alpha=0.7)
        ax.set_title(f"{r['system']} — ΔK={r['delta_k']:+.3f}",
                     color="white", fontsize=10)
        ax.legend(facecolor="black", labelcolor="white", edgecolor="white",
                  fontsize=7)

        ax = axes[row, 1]
        def norm(v):
            v = np.asarray(v)
            return (v - v.min()) / (v.max() - v.min() + 1e-12)
        ax.plot(x, norm(r["chi_info"]), "o-", color="#00ff88", lw=1.4,
                label="χ_info (normed)")
        ax.plot(x, norm(r["var_lz"]), "s-", color="magenta", lw=1.4,
                label="Var(LZ) (normed)")
        ax.plot(x, norm(r["mean_lz"]), "^--", color="yellow", lw=1.0,
                alpha=0.6, label="⟨LZ⟩")
        ax.axvline(xc, color="#ff3355", ls="--", lw=1)
        ax.set_title(f"Pearson={r['pearson_chiinfo_varlz']:+.3f}",
                     color="white", fontsize=10)
        ax.legend(facecolor="black", labelcolor="white", edgecolor="white",
                  fontsize=7)

    for ax in axes.flat:
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_color("white")

    fig.patch.set_facecolor("black")
    fig.suptitle(f"ΔK = peak Var(LZ) − peak χ_info — test universalité dualité",
                 color="white", y=0.995)
    plt.tight_layout()
    png = os.path.join(outdir, "delta_k_multi.png")
    plt.savefig(png, dpi=130, facecolor="black", bbox_inches="tight")
    print(f"\nsaved {png}")

    out = {"results": results, "verdict": verdict, "deltas": deltas}
    with open(os.path.join(outdir, "delta_k_multi.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {os.path.join(outdir, 'delta_k_multi.json')}")
    return out


if __name__ == "__main__":
    run()
