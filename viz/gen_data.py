"""Génère les courbes de référence pour la viz : CML sweep r.
Lyapunov (edge detector), synergie Ω, susceptibilité Var(Ω), dim effective Koopman.
Sortie : viz/data.json (consommé par la viz Three.js)."""
import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from systems.cml import simulate_cml, lyapunov_cml
from xi.interaction import o_information
from xi.temporal import tc_random_subsets


def dmd_dim(X, energy=0.99):
    X0, X1 = X[:-1].T, X[1:].T
    S = np.linalg.svd(X0, compute_uv=False)
    p = S ** 2 / np.sum(S ** 2)
    return int(np.searchsorted(np.cumsum(p), energy) + 1)


def run(L=24, eps=0.3, r_min=3.4, r_max=4.0, n_r=46, n_seed=6,
        T=400, t_equil=300, n_bins=3, subset=4, n_subsets=200, seed0=0):
    rs = np.linspace(r_min, r_max, n_r)
    rng = np.random.default_rng(seed0)
    lyap = np.zeros(n_r)
    om = np.zeros(n_r)
    om_var = np.zeros(n_r)     # susceptibilité informationnelle
    dim = np.zeros(n_r)

    for i, r in enumerate(rs):
        ls, os_, dims = [], [], []
        for s in range(n_seed):
            sd = int(rng.integers(0, 2 ** 31))
            ls.append(lyapunov_cml(N=L * L, eps=eps, r=float(r), seed=sd))
            X = simulate_cml(N=L * L, eps=eps, r=float(r), T=T, t_equil=t_equil, seed=sd)
            disc = np.empty(X.shape, dtype=np.int8)
            for j in range(X.shape[1]):
                q = np.quantile(X[:, j], np.linspace(0, 1, n_bins + 1)[1:-1])
                disc[:, j] = np.digitize(X[:, j], q)
            _, _, o, _, _ = tc_random_subsets(disc, subset, n_subsets,
                                              np.random.default_rng(sd))
            os_.append(o)
            dims.append(dmd_dim(X))
        lyap[i] = np.mean(ls)
        om[i] = np.mean(os_)
        om_var[i] = np.var(os_)
        dim[i] = np.mean(dims)
        print(f"  r={r:.3f} | λ={lyap[i]:+.3f} | Ω={om[i]:+.4f} | "
              f"Var(Ω)={om_var[i]:.5f} | dim={dim[i]:.1f}")

    # edge = r où |λ| minimal (passage ordre→chaos)
    edge_idx = int(np.argmin(np.abs(lyap)))
    out = {
        "L": L, "eps": eps, "r": rs.tolist(), "lyapunov": lyap.tolist(),
        "omega": om.tolist(), "omega_var": om_var.tolist(), "dim_eff": dim.tolist(),
        "edge_r": float(rs[edge_idx]),
        "r_min": r_min, "r_max": r_max,
    }
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nedge (|λ|→0) à r={rs[edge_idx]:.3f}")
    print(f"saved {out_path}")
    return out


if __name__ == "__main__":
    run()
