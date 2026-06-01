"""Émergence dans l'espace des PARAMÈTRES (insight Zac, 2026-06-01).

Les dimensions ne sont pas équivalentes pour l'émergence. Champ spatial diffusif =
redondant (interface_emergence). Ici : la carte observable(r, ε) — la « surface de
réponse paramétrique » — porte-t-elle de la structure/émergence aux interfaces
(edge of chaos = ensemble de bifurcation) ?

Pour chaque (r, ε) : observables λ (Lyapunov), dim_eff (Koopman), ⟨v⟩, std spatial.
On mesure :
1. cartes 2D dans (r, ε) + contour edge (λ=0)
2. densité d'interface paramétrique = |∇ carte| (où le comportement bascule)
3. synergie paramétrique = Ω entre observables sur la grille (irréductibilité jointe)
4. la synergie/structure se concentre-t-elle à l'edge ?
"""
import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from systems.cml import simulate_cml, lyapunov_cml
from xi.interaction import o_information


def dmd_dim(X, energy=0.99):
    X0 = X[:-1].T
    S = np.linalg.svd(X0, compute_uv=False)
    p = S ** 2 / np.sum(S ** 2)
    return int(np.searchsorted(np.cumsum(p), energy) + 1)


def run(L=16, n_r=26, n_eps=16, r_min=3.4, r_max=4.0, eps_min=0.04, eps_max=0.5,
        t_equil=250, T=60, n_bins=4, seed=0, outdir=None):
    if outdir is None:
        outdir = os.path.dirname(os.path.abspath(__file__))
    rs = np.linspace(r_min, r_max, n_r)
    epss = np.linspace(eps_min, eps_max, n_eps)
    N = L * L

    lyap = np.zeros((n_eps, n_r)); dim = np.zeros((n_eps, n_r))
    mean = np.zeros((n_eps, n_r)); sstd = np.zeros((n_eps, n_r))

    print(f"Espace paramètres | grille {n_r}×{n_eps} = {n_r*n_eps} points | CML L={L}\n")
    for ie, e in enumerate(epss):
        for ir, r in enumerate(rs):
            lyap[ie, ir] = lyapunov_cml(N=N, eps=float(e), r=float(r),
                                        t_equil=t_equil, t_meas=200, seed=seed)
            X = simulate_cml(N=N, eps=float(e), r=float(r), T=T, t_equil=t_equil, seed=seed)
            dim[ie, ir] = dmd_dim(X)
            mean[ie, ir] = X[-1].mean()
            sstd[ie, ir] = X[-1].std()
        print(f"  ε={e:.3f} done")

    # interface paramétrique = |∇| de la carte λ (où le comportement bascule)
    gy, gx = np.gradient(lyap)
    grad = np.sqrt(gx ** 2 + gy ** 2)

    # synergie paramétrique : Ω entre observables sur l'ensemble des points
    obs = np.stack([lyap.ravel(), dim.ravel(), mean.ravel(), sstd.ravel()], axis=1)
    disc = np.empty(obs.shape, dtype=np.int8)
    for j in range(obs.shape[1]):
        q = np.quantile(obs[:, j], np.linspace(0, 1, n_bins + 1)[1:-1])
        disc[:, j] = np.digitize(obs[:, j], q)
    omega_param = o_information(disc)

    # structure : la carte paramétrique vs un champ spatial redondant (ref)
    # densité d'interface = fraction de points à fort gradient
    grad_n = grad / (grad.max() + 1e-9)
    interface_frac = float(np.mean(grad_n > 0.3))
    # corrélation : proximité de l'edge (|λ| petit) ↔ fort gradient ?
    near_edge = (np.abs(lyap) < 0.05).ravel()
    r_edge_grad = np.corrcoef((np.abs(lyap)).ravel(), grad.ravel())[0, 1]

    print(f"\nΩ paramétrique (entre observables sur grille) = {omega_param:+.4f}  "
          f"[{'SYNERGIE' if omega_param<0 else 'redondance'}]")
    print(f"densité d'interface paramétrique (|∇λ|>0.3) = {interface_frac:.2f}")
    print(f"corr(|λ|, |∇λ|) = {r_edge_grad:+.3f}  "
          f"({'gradient max à l edge' if r_edge_grad<-0.1 else 'pas concentré'})")
    print(f"edge (|λ|<0.05) : {near_edge.sum()} points sur {len(near_edge)}")

    # plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    ext = [r_min, r_max, eps_min, eps_max]
    def cmap_panel(ax, Z, title, cm="inferno", contour=False):
        im = ax.imshow(Z, origin="lower", extent=ext, aspect="auto", cmap=cm)
        if contour:
            ax.contour(np.linspace(r_min, r_max, n_r), np.linspace(eps_min, eps_max, n_eps),
                       lyap, levels=[0.0], colors="#00eaff", linewidths=1.5)
        ax.set_title(title, color="white"); ax.set_xlabel("r", color="white"); ax.set_ylabel("ε", color="white")
        plt.colorbar(im, ax=ax); ax.tick_params(colors="white")

    cmap_panel(axes[0, 0], lyap, "λ Lyapunov (edge = ligne cyan)", "RdBu_r", contour=True)
    cmap_panel(axes[0, 1], dim, "dim effective Koopman")
    cmap_panel(axes[0, 2], grad, "|∇λ| — interfaces paramétriques", "magma")
    cmap_panel(axes[1, 0], mean, "⟨v⟩")
    cmap_panel(axes[1, 1], sstd, "std spatiale")
    axes[1, 2].scatter(np.abs(lyap).ravel(), grad.ravel(), s=10, alpha=0.6, color="#ff00ff")
    axes[1, 2].set_xlabel("|λ| (→ loin de l'edge)", color="white")
    axes[1, 2].set_ylabel("|∇λ| structure", color="white")
    axes[1, 2].set_title(f"structure vs edge (corr={r_edge_grad:+.2f})", color="white")
    axes[1, 2].set_facecolor("black"); axes[1, 2].tick_params(colors="white")
    for sp in axes[1, 2].spines.values(): sp.set_color("white")

    fig.patch.set_facecolor("#05060a"); plt.tight_layout()
    png = os.path.join(outdir, "parameter_space_emergence.png")
    plt.savefig(png, dpi=120, facecolor="#05060a", bbox_inches="tight")
    print(f"\nsaved {png}")

    out = {"n_r": n_r, "n_eps": n_eps, "omega_param": float(omega_param),
           "interface_frac": interface_frac, "corr_absL_grad": float(r_edge_grad),
           "lyap": lyap.tolist(), "dim": dim.tolist(), "grad": grad.tolist(),
           "rs": rs.tolist(), "epss": epss.tolist()}
    with open(os.path.join(outdir, "parameter_space_emergence.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {os.path.join(outdir, 'parameter_space_emergence.json')}")
    return out


if __name__ == "__main__":
    run()
