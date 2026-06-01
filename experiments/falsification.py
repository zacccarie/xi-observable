import sys
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from systems.ising2d import simulate
from xi.observable import xi_observable


def run(outdir=None, L=32, seed=42):
    if outdir is None:
        outdir = os.path.dirname(os.path.abspath(__file__))

    Tc = 2.0 / np.log(1.0 + np.sqrt(2.0))
    # Bandes : critique (autour Tc), warm, hot, very-hot, cold
    bands = {
        "critical":  np.linspace(2.20, 2.40, 11),
        "warm":      np.linspace(3.0, 4.0, 11),
        "hot":       np.linspace(8.0, 12.0, 11),
        "very_hot":  np.linspace(40.0, 60.0, 11),
        "cold":      np.linspace(0.5, 1.5, 11),
    }

    # 1ère passe : computes peak χ sur bande critique → référence absolue pour mask global
    print("=== ref pass : critical band only (for χ_ref_max) ===")
    crit_T = bands["critical"]
    snaps_crit = {float(T): simulate(L=L, T=float(T), n_equil=1200, n_sample=200,
                                     sample_gap=3, seed=seed) for T in crit_T}
    res_crit = xi_observable(snaps_crit, crit_T, L=L, block_size=2, epsilon=0.3)
    chi_ref = float(res_crit["chi"].max())
    print(f"χ_ref_max (critical band) = {chi_ref:.3f}")

    results = {"critical": {
        "T": crit_T.tolist(),
        "chi": res_crit["chi"].tolist(),
        "tc": res_crit["tc"].tolist(),
        "abs_o": res_crit["abs_o"].tolist(),
        "regime": res_crit["regime"].tolist(),
        "mask": res_crit["mask"].tolist(),
        "xi_abs": res_crit["xi_abs"].tolist(),
        "xi_tc": res_crit["xi_tc"].tolist(),
        "peak_chi": float(res_crit["chi"].max()),
        "peak_tc": float(res_crit["tc"].max()),
        "peak_xi_abs": float(res_crit["xi_abs"].max()),
        "peak_xi_tc": float(res_crit["xi_tc"].max()),
        "mask_any": bool(res_crit["mask"].sum() > 0),
    }}

    # 2e passe : autres bandes avec χ_ref_max ABSOLU
    for name, T_values in bands.items():
        if name == "critical":
            continue
        print(f"\n=== band {name}: T ∈ [{T_values.min():.2f}, {T_values.max():.2f}] ===")
        snaps = {}
        t0 = time.time()
        for T in T_values:
            snaps[float(T)] = simulate(
                L=L, T=float(T),
                n_equil=1200, n_sample=200, sample_gap=3, seed=seed,
            )
        res = xi_observable(snaps, T_values, L=L, block_size=2, epsilon=0.3,
                            chi_ref_max=chi_ref)
        peak_xi_abs = res["xi_abs"].max()
        peak_xi_tc = res["xi_tc"].max()
        peak_chi = res["chi"].max()
        peak_tc = res["tc"].max()
        mask_any = res["mask"].sum() > 0
        regime = "REDONDANCE" if (res["regime"] > 0).sum() >= len(T_values) // 2 else "SYNERGIE"
        print(f"  peak χ={peak_chi:.3f}  TC={peak_tc:.3f}  |Ω|={res['abs_o'].max():.4f}")
        print(f"  Ξ_abs={peak_xi_abs:.4f}  Ξ_tc={peak_xi_tc:.4f}  mask_any={mask_any}  regime={regime}")
        results[name] = {
            "T": T_values.tolist(),
            "chi": res["chi"].tolist(),
            "tc": res["tc"].tolist(),
            "abs_o": res["abs_o"].tolist(),
            "regime": res["regime"].tolist(),
            "mask": res["mask"].tolist(),
            "xi_abs": res["xi_abs"].tolist(),
            "xi_tc": res["xi_tc"].tolist(),
            "peak_chi": float(peak_chi),
            "peak_tc": float(peak_tc),
            "peak_xi_abs": float(peak_xi_abs),
            "peak_xi_tc": float(peak_xi_tc),
            "mask_any": bool(mask_any),
            "time_s": time.time() - t0,
        }

    # SNR : critical band vs autres
    crit = results["critical"]
    snr = {}
    for name in bands:
        if name == "critical":
            continue
        snr[name] = {
            "Ξ_tc_ratio_crit_over_band": crit["peak_xi_tc"] / max(results[name]["peak_xi_tc"], 1e-9),
            "TC_ratio_crit_over_band":   crit["peak_tc"] / max(results[name]["peak_tc"], 1e-9),
            "χ_ratio_crit_over_band":    crit["peak_chi"] / max(results[name]["peak_chi"], 1e-9),
        }

    print("\n=== SNR (peak critical / peak off-critical) ===")
    print(f"{'band':>10} | {'Ξ_tc':>10} | {'TC':>10} | {'χ':>10}")
    print("-" * 50)
    for name, vals in snr.items():
        print(f"{name:>10} | {vals['Ξ_tc_ratio_crit_over_band']:>10.2f} | "
              f"{vals['TC_ratio_crit_over_band']:>10.2f} | "
              f"{vals['χ_ratio_crit_over_band']:>10.2f}")

    # Plot synthèse
    fig, axes = plt.subplots(3, 1, figsize=(9, 9))
    band_colors = {"critical": "#ff3355", "warm": "#ffaa00", "hot": "#ff00ff",
                   "very_hot": "#aa00ff", "cold": "#00aaff"}

    for name, data in results.items():
        T = np.array(data["T"])
        axes[0].plot(T, data["chi"], "o-", color=band_colors[name], label=name, lw=1.4, ms=4)
        axes[1].plot(T, data["tc"], "o-", color=band_colors[name], label=name, lw=1.4, ms=4)
        axes[2].plot(T, data["xi_tc"], "o-", color=band_colors[name], label=name, lw=1.4, ms=4)

    for ax, ylab in zip(axes, ["χ susceptibility", "TC total corr.", "Ξ_tc (masked)"]):
        ax.axvline(Tc, color="white", ls="--", lw=0.8, alpha=0.6)
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.set_ylabel(ylab, color="white")
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        for s in ax.spines.values():
            s.set_color("white")
        ax.legend(facecolor="black", edgecolor="white", labelcolor="white", fontsize=8)

    axes[2].set_xlabel("T (log)", color="white")
    fig.patch.set_facecolor("black")
    fig.suptitle("Ξ Falsification : critical vs off-critical bands", color="white", y=0.995)
    plt.tight_layout()

    png = os.path.join(outdir, "ising_falsification.png")
    plt.savefig(png, dpi=130, facecolor="black", bbox_inches="tight")
    print(f"\nsaved {png}")

    jpath = os.path.join(outdir, "ising_falsification.json")
    with open(jpath, "w") as f:
        json.dump({"results": results, "snr": snr, "Tc": Tc}, f, indent=2)
    print(f"saved {jpath}")

    return results, snr


if __name__ == "__main__":
    run()
