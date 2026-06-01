"""Kuramoto 2D local : oscillateurs sur grille L×L, couplage 4 voisins NESW.

θ̇_i = ω_i + (K/4) Σ_{j ∈ NESW(i)} sin(θ_j − θ_i)

Différent du Kuramoto all-to-all : C local (portée=1), structure spatiale →
CG spatial (rg_blocks) devient légitime. Test direct conséquence #5 dualité CG↔C.

K_c effectif ~ 1-2 (différent K_c~1.6 all-to-all, dépend distribution ω et L).
"""
import numpy as np


def order_parameter(theta):
    return np.abs(np.mean(np.exp(1j * theta)))


def step_rk4_2d(theta, omega, K, dt, L):
    """θ (L,L). Voisins NESW via np.roll, vectorisé."""
    def f(th):
        sin_n = np.sin(np.roll(th, 1, 0) - th)
        sin_s = np.sin(np.roll(th, -1, 0) - th)
        sin_e = np.sin(np.roll(th, 1, 1) - th)
        sin_w = np.sin(np.roll(th, -1, 1) - th)
        return omega + (K / 4.0) * (sin_n + sin_s + sin_e + sin_w)
    k1 = f(theta)
    k2 = f(theta + 0.5 * dt * k1)
    k3 = f(theta + 0.5 * dt * k2)
    k4 = f(theta + dt * k3)
    return theta + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def quadrantize(theta, n_bins=4):
    th = np.mod(theta, 2 * np.pi)
    return np.floor(th / (2 * np.pi / n_bins)).astype(np.int8)


def simulate(L=16, K=1.5, sigma_omega=1.0, dt=0.05,
             t_equil=60.0, t_sample=120.0, sample_gap=1.0,
             n_bins=4, seed=0):
    """Renvoie (snapshots (T_snap, L, L) en {0..n_bins-1}, r_traj)."""
    rng = np.random.default_rng(seed)
    omega = rng.normal(0.0, sigma_omega, size=(L, L))
    theta = rng.uniform(0, 2 * np.pi, size=(L, L))

    n_equil_steps = int(t_equil / dt)
    n_sample_steps = int(t_sample / dt)
    gap_steps = max(1, int(sample_gap / dt))

    for _ in range(n_equil_steps):
        theta = step_rk4_2d(theta, omega, K, dt, L)

    snapshots = []
    r_traj = []
    steps_since = 0
    for _ in range(n_sample_steps):
        theta = step_rk4_2d(theta, omega, K, dt, L)
        steps_since += 1
        if steps_since >= gap_steps:
            steps_since = 0
            snapshots.append(quadrantize(theta, n_bins=n_bins))
            r_traj.append(order_parameter(theta.flatten()))

    return snapshots, np.array(r_traj)


def chi_from_r(r_traj, N):
    return N * r_traj.var()
