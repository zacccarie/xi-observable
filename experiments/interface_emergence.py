"""Conjecture INTERFACE : l'émergence (synergie Ω<0) vit aux interfaces, pas dans le bulk.

Réaction-diffusion Gray–Scott → motif de domaines + interfaces. Pour chaque patch
spatial, série temporelle des cellules → Ω(patch). Classer le patch par |∇v| moyen
(interface = gradient fort, bulk = gradient faible). Tester : Ω plus négatif (synergie)
aux interfaces ?

Prédiction : corr(Ω, gradient) < 0 (plus l'interface est forte, plus Ω négatif).
"""
import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from systems.gray_scott import simulate_gray_scott, gradient_magnitude
from xi.interaction import o_information


def patch_cells(L, p, box):
    """Indices (cy,cx) du patch p×p en haut-gauche de box=(y0,x0)."""
    y0, x0 = box
    idx = []
    for dy in range(p):
        for dx in range(p):
            idx.append(((y0 + dy) % L, (x0 + dx) % L))
    return idx


def run(L=128, F=0.037, k=0.06, n_frames=240, frame_gap=12, t_equil=3000,
        patch=2, n_patches=500, n_bins=3, seed=0, outdir=None):
    if outdir is None:
        outdir = os.path.dirname(os.path.abspath(__file__))

    print(f"Conjecture interface | Gray-Scott L={L} F={F} k={k}")
    print(f"frames={n_frames} patch={patch}×{patch}\n")

    # trajectoire temporelle (frames, L, L)
    V = simulate_gray_scott(L=L, F=F, k=k, t_equil=t_equil,
                            n_frames=n_frames, frame_gap=frame_gap, seed=seed)
    print(f"trajectoire collectée : {V.shape}")
    gradmean = gradient_magnitude(V.mean(axis=0))   # |∇| du champ moyen → carte interfaces

    rng = np.random.default_rng(seed)
    omegas, grads, varis = [], [], []
    for _ in range(n_patches):
        y0, x0 = rng.integers(0, L, 2)
        cells = patch_cells(L, patch, (y0, x0))
        # série temporelle des cellules du patch : (n_frames, patch²)
        ts = np.stack([V[:, cy, cx] for (cy, cx) in cells], axis=1)
        if ts.std() < 1e-6:
            continue  # patch totalement gelé (bulk pur) → entropie nulle
        # discrétise chaque cellule en quantiles temporels
        disc = np.empty(ts.shape, dtype=np.int8)
        for j in range(ts.shape[1]):
            q = np.quantile(ts[:, j], np.linspace(0, 1, n_bins + 1)[1:-1])
            disc[:, j] = np.digitize(ts[:, j], q)
        om = o_information(disc)
        g = np.mean([gradmean[cy, cx] for (cy, cx) in cells])
        omegas.append(om); grads.append(g); varis.append(ts.var())

    omegas = np.array(omegas); grads = np.array(grads); varis = np.array(varis)
    # corrélation Ω vs gradient (interface)
    r_g = np.corrcoef(omegas, grads)[0, 1]

    # binning par quartile de gradient : interface (Q4) vs bulk (Q1)
    qs = np.quantile(grads, [0.25, 0.5, 0.75])
    bulk = omegas[grads <= qs[0]]
    inter = omegas[grads >= qs[2]]
    print(f"patchs actifs : {len(omegas)}")
    print(f"corr(Ω, |∇|) = {r_g:+.3f}")
    print(f"Ω bulk  (Q1 gradient) : mean={bulk.mean():+.4f}  frac_syn={np.mean(bulk<0):.2f}")
    print(f"Ω interf(Q4 gradient) : mean={inter.mean():+.4f}  frac_syn={np.mean(inter<0):.2f}")
    delta = inter.mean() - bulk.mean()
    print(f"\nΔ(interface − bulk) = {delta:+.4f}")
    if delta < -1e-3 and r_g < -0.1:
        verdict = "✓ CONJECTURE SUPPORTÉE : synergie aux interfaces (Ω plus négatif, corr<0)"
    elif delta > 1e-3:
        verdict = "✗ inverse : synergie dans le bulk"
    else:
        verdict = "~ pas de localisation nette"
    print(verdict)

    # plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    im = axes[0].imshow(V[-1], cmap="inferno"); axes[0].set_title("champ v (Gray-Scott)", color="white")
    axes[0].set_xticks([]); axes[0].set_yticks([])
    axes[1].imshow(gradmean, cmap="magma"); axes[1].set_title("|∇v| — interfaces", color="white")
    axes[1].set_xticks([]); axes[1].set_yticks([])
    axes[2].scatter(grads, omegas, c=omegas, cmap="coolwarm_r", s=14, alpha=0.7)
    axes[2].axhline(0, color="white", ls=":", lw=0.8)
    axes[2].set_xlabel("|∇| du patch (interface →)", color="white")
    axes[2].set_ylabel("Ω (synergie si <0)", color="white")
    axes[2].set_title(f"Ω vs interface (corr={r_g:+.2f})", color="white")
    for ax in axes:
        ax.set_facecolor("black"); ax.tick_params(colors="white")
        for s in ax.spines.values(): s.set_color("white")
    fig.patch.set_facecolor("#05060a"); plt.tight_layout()
    png = os.path.join(outdir, "interface_emergence.png")
    plt.savefig(png, dpi=120, facecolor="#05060a", bbox_inches="tight")
    print(f"\nsaved {png}")

    out = {"L": L, "F": F, "k": k, "n_patches": int(len(omegas)),
           "corr_omega_grad": float(r_g), "omega_bulk": float(bulk.mean()),
           "omega_interface": float(inter.mean()), "delta": float(delta),
           "frac_syn_bulk": float(np.mean(bulk < 0)), "frac_syn_interface": float(np.mean(inter < 0))}
    with open(os.path.join(outdir, "interface_emergence.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {os.path.join(outdir, 'interface_emergence.json')}")
    return out


if __name__ == "__main__":
    run()
