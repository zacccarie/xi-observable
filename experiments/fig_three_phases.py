"""LA FIGURE DES TROIS TEMPS — version alignée-sur-grok, moyennée.

Aligne chaque run sur sa bascule (t − t_grok), moyenne Γ (formation) et s_net
(consolidation) à travers les seeds. Récit propre : Γ pique AVANT 0, s_net
monte APRÈS 0. C'est le résultat central, sur une timeline.
Données : modadd seeds 0-2, 10-12 (checkpoints sauvés). Pièce paper + portfolio.
"""
import os
import sys
import glob
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.argv.append("--v1")
import xi_gamma_germination as xg
from xi_nn_grokking import measure_snet

RESULTS = os.path.join(HERE, "fig_three_phases")
SEEDS = (0, 1, 2, 10, 11, 12)
GRID = np.arange(-1400, 1000, 50)        # offset / grok, pas de 50 steps


def run_curves(seed):
    rd = os.path.join(xg.CKPT_ROOT, f"run_wd1.0_s{seed}")
    steps, acts, meta = xg.load_run(rd)                    # Frobenius (--v1)
    anchors, gamma, _ = xg.gamma_series(steps, acts, seed=seed)
    files = sorted(glob.glob(os.path.join(rd, "step_*.npz")))
    snet = [measure_snet(np.load(f)["acts_test"].astype(np.float64), seed=seed)[0]
            for f in files]
    tg = meta["t_grok"]
    val = np.interp([s for s in steps], [t["step"] for t in meta["traj"]],
                    [t["val_acc"] for t in meta["traj"]])
    # normalise Γ et s_net par leur propre max (forme, pas amplitude)
    g = np.array(gamma) / (max(gamma) + 1e-12)
    sn = np.array(snet) / (max(snet) + 1e-12)
    return {
        "g": np.interp(GRID, np.array(anchors) - tg, g, left=0, right=0),
        "sn": np.interp(GRID, np.array(steps) - tg, sn, left=0, right=np.nan),
        "val": np.interp(GRID, np.array(steps) - tg, val, left=0, right=1),
    }


def main():
    def smooth(x, k=3):
        ker = np.ones(k) / k
        return np.convolve(np.nan_to_num(x), ker, mode="same")

    C = [run_curves(s) for s in SEEDS]
    g = smooth(np.nanmean([c["g"] for c in C], 0))
    g_sd = smooth(np.nanstd([c["g"] for c in C], 0))
    sn = smooth(np.nanmean([c["sn"] for c in C], 0))
    sn_sd = smooth(np.nanstd([c["sn"] for c in C], 0))
    val = np.nanmean([c["val"] for c in C], 0)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax2 = ax.twinx()

    # zones de phase
    ax.axvspan(-1400, -150, color="#ff6d00", alpha=0.06)
    ax.axvspan(-150, 150, color="#ffffff", alpha=0.05)
    ax.axvspan(150, 1000, color="#e040fb", alpha=0.06)
    ax.axvline(0, color="white", ls="--", lw=1, alpha=0.6)

    ax.plot(GRID, val, color="#00e5ff", lw=2.4, label="généralisation (val_acc)")
    ax2.plot(GRID, g, color="#ff6d00", lw=2.4, label="Γ — germination")
    ax2.fill_between(GRID, g-g_sd, g+g_sd, color="#ff6d00", alpha=0.13)
    ax2.plot(GRID, sn, color="#e040fb", lw=2.4, label="s_net — synergie")
    ax2.fill_between(GRID, sn-sn_sd, sn+sn_sd, color="#e040fb", alpha=0.13)

    for x, lab, col in [(-560, "FORMATION", "#ff6d00"),
                        (0, "BASCULE", "#ffffff"),
                        (480, "CONSOLIDATION", "#e040fb")]:
        ax.annotate(lab, xy=(x, 1.12), ha="center", fontsize=10,
                    color=col, annotation_clip=False, weight="bold")
    ax.annotate("← Γ seul est élevé,\ns_net ≈ 0", xy=(-380, 0.40), fontsize=8,
                color="#ff9e58", ha="center")
    ax.annotate("s_net grimpe\n(synergie qui s'installe) →", xy=(520, 0.40),
                fontsize=8, color="#e879f5", ha="center")

    ax.set_xlabel("step relatif à la bascule  (t − t_grok)")
    ax.set_ylabel("généralisation (cyan)", color="#00e5ff")
    ax2.set_ylabel("Γ germination (orange) · s_net synergie (violet) — normalisés",
                   color="#ff9e58", fontsize=9)
    ax.set_ylim(-0.05, 1.22); ax2.set_ylim(-0.05, 1.22); ax.set_xlim(-1000, 850)
    fig.suptitle("Les trois temps de l'émergence — avant la bascule Γ seul "
                 "entend le germe ; après, s_net signe la consolidation\n"
                 "(addition modulaire, 6 seeds alignés sur la bascule ; "
                 "Γ refire au cleanup — c'est un mètre de réallocation)",
                 color="white", fontsize=9.5)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    ax2.legend(loc="center left", frameon=False, fontsize=8.5)
    plt.tight_layout()
    plt.savefig(RESULTS + ".png", dpi=150, facecolor="black")

    # pics
    g_peak = GRID[np.argmax(g)]
    sn_peak = GRID[np.nanargmax(sn)]
    json.dump({"gamma_peak_offset": int(g_peak),
               "snet_peak_offset": int(sn_peak),
               "grid": GRID.tolist(), "gamma": g.tolist(),
               "snet": sn.tolist(), "val": val.tolist()},
              open(RESULTS + ".json", "w"), indent=1)
    print(f"Γ pic à t−grok = {g_peak}  |  s_net pic à t−grok = {sn_peak}")
    print(f"→ {RESULTS}.png")


if __name__ == "__main__":
    main()
