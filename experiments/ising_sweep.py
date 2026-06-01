import sys
import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from systems.ising2d import simulate
from xi.observable import xi_observable


def run(
    L=32,
    T_min=1.5,
    T_max=3.5,
    n_T=21,
    n_equil=1500,
    n_sample=250,
    sample_gap=4,
    epsilon=0.3,
    block_size=2,
    seed=42,
    outdir=None,
):
    if outdir is None:
        outdir = os.path.dirname(os.path.abspath(__file__))
    T_values = np.linspace(T_min, T_max, n_T)
    Tc = 2.0 / np.log(1.0 + np.sqrt(2.0))  # ≈ 2.2691

    print(f"L={L} | T ∈ [{T_min},{T_max}] ({n_T} pts) | Tc={Tc:.4f}")
    print(f"n_equil={n_equil} n_sample={n_sample} gap={sample_gap}")

    t0 = time.time()
    snapshots = {}
    for T in T_values:
        ts = time.time()
        snapshots[float(T)] = simulate(
            L=L, T=float(T),
            n_equil=n_equil, n_sample=n_sample, sample_gap=sample_gap,
            seed=seed,
        )
        print(f"  T={T:.3f} | {time.time()-ts:.1f}s")
    print(f"sim total {time.time()-t0:.1f}s")

    res = xi_observable(
        snapshots, T_values, L=L,
        block_size=block_size, epsilon=epsilon,
    )

    peak_chi = T_values[np.argmax(res["chi"])]
    peak_syn = T_values[np.argmax(res["synergy"])]
    peak_tc = T_values[np.argmax(res["tc"])]
    peak_xi_syn = T_values[np.argmax(res["xi_syn"])]
    peak_xi_tc = T_values[np.argmax(res["xi_tc"])]

    print()
    print(f"Tc analytic       = {Tc:.4f}")
    print(f"peak χ            = {peak_chi:.4f}  (Δ={peak_chi-Tc:+.3f})")
    print(f"peak synergy(-Ω)  = {peak_syn:.4f}  (Δ={peak_syn-Tc:+.3f})")
    print(f"peak TC           = {peak_tc:.4f}  (Δ={peak_tc-Tc:+.3f})")
    print(f"peak Ξ_syn        = {peak_xi_syn:.4f}  (Δ={peak_xi_syn-Tc:+.3f})")
    print(f"peak Ξ_tc         = {peak_xi_tc:.4f}  (Δ={peak_xi_tc-Tc:+.3f})")
    print(f"edge band T range = [{T_values[res['mask'].astype(bool)].min():.3f}, "
          f"{T_values[res['mask'].astype(bool)].max():.3f}]")

    # Plot dark-emissive
    fig, axes = plt.subplots(5, 1, figsize=(8, 11), sharex=True)
    colors = ["#00ffff", "#ff00ff", "#ffff00", "#00ff88", "#ff8800"]

    panels = [
        ("χ susceptibility", res["chi"], colors[0]),
        ("Ω O-information", res["o_info"], colors[1]),
        ("−Ω synergy", res["synergy"], colors[1]),
        ("TC total corr.", res["tc"], colors[2]),
        ("Ξ_syn  &  Ξ_tc", None, None),
    ]
    for i, (label, vec, col) in enumerate(panels[:4]):
        ax = axes[i]
        ax.plot(res["T"], vec, "o-", color=col, lw=1.5, ms=4)
        ax.axvline(Tc, color="#ff3355", ls="--", lw=1, label=f"Tc={Tc:.3f}")
        ax.set_ylabel(label, color="white")
        ax.legend(facecolor="black", edgecolor="white", labelcolor="white")
    ax = axes[4]
    ax.plot(res["T"], res["xi_syn"], "o-", color=colors[1], lw=1.5, ms=4, label="Ξ_syn")
    ax.plot(res["T"], res["xi_tc"], "s-", color=colors[2], lw=1.5, ms=4, label="Ξ_tc")
    ax.axvline(Tc, color="#ff3355", ls="--", lw=1)
    ax.set_ylabel("Ξ (masked)", color="white")
    ax.set_xlabel("T", color="white")
    ax.legend(facecolor="black", edgecolor="white", labelcolor="white")

    for ax in axes:
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("white")
        # surligne bande edge
        band_T = res["T"][res["mask"].astype(bool)]
        if len(band_T) > 0:
            ax.axvspan(band_T.min(), band_T.max(), color="#ffffff", alpha=0.08)

    fig.patch.set_facecolor("black")
    fig.suptitle(f"Ξ Stade A — Ising 2D L={L}", color="white", y=0.995)
    plt.tight_layout()

    png = os.path.join(outdir, "ising_xi_stadeA.png")
    plt.savefig(png, dpi=130, facecolor="black", bbox_inches="tight")
    print(f"\nsaved {png}")

    out_json = {
        "L": L, "Tc": Tc, "epsilon": epsilon, "block_size": block_size,
        "T": res["T"].tolist(),
        "chi": res["chi"].tolist(),
        "o_info": res["o_info"].tolist(),
        "synergy": res["synergy"].tolist(),
        "tc": res["tc"].tolist(),
        "mask": res["mask"].tolist(),
        "xi_syn": res["xi_syn"].tolist(),
        "xi_tc": res["xi_tc"].tolist(),
        "peaks": {
            "chi": peak_chi, "synergy": peak_syn, "tc": peak_tc,
            "xi_syn": peak_xi_syn, "xi_tc": peak_xi_tc,
        },
    }
    jpath = os.path.join(outdir, "ising_xi_stadeA.json")
    with open(jpath, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"saved {jpath}")
    return res


if __name__ == "__main__":
    run()
