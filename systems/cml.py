import numpy as np


def simulate_cml(N=64, eps=0.3, r=3.9, T=600, t_equil=400, seed=0, ring=True):
    """Coupled Map Lattice : x_i(t+1) = (1-ε)f(x_i) + (ε/2)[f(x_{i-1})+f(x_{i+1})].
    f = carte logistique r·x·(1-x). Chaotique (r→4), structures cohérentes spatiales.
    Retourne trajectoire (T, N) ∈ [0,1]."""
    rng = np.random.default_rng(seed)
    x = rng.random(N)
    for _ in range(t_equil):
        fx = r * x * (1.0 - x)
        x = (1 - eps) * fx + eps * 0.5 * (np.roll(fx, 1) + np.roll(fx, -1))
    traj = np.empty((T, N))
    for t in range(T):
        fx = r * x * (1.0 - x)
        x = (1 - eps) * fx + eps * 0.5 * (np.roll(fx, 1) + np.roll(fx, -1))
        traj[t] = x
    return traj


def lyapunov_cml(N=64, eps=0.3, r=3.9, t_equil=300, t_meas=600, seed=0):
    """Plus grand exposant de Lyapunov (proxy edge of chaos) par croissance
    d'une perturbation infinitésimale renormalisée."""
    rng = np.random.default_rng(seed)
    x = rng.random(N)
    for _ in range(t_equil):
        fx = r * x * (1.0 - x)
        x = (1 - eps) * fx + eps * 0.5 * (np.roll(fx, 1) + np.roll(fx, -1))
    d0 = 1e-8
    y = x + d0 * rng.standard_normal(N)
    s = 0.0
    for _ in range(t_meas):
        fx = r * x * (1.0 - x)
        x = (1 - eps) * fx + eps * 0.5 * (np.roll(fx, 1) + np.roll(fx, -1))
        fy = r * y * (1.0 - y)
        y = (1 - eps) * fy + eps * 0.5 * (np.roll(fy, 1) + np.roll(fy, -1))
        d = np.linalg.norm(y - x)
        if d > 0:
            s += np.log(d / d0)
            y = x + (d0 / d) * (y - x)
    return s / t_meas
