import numpy as np


def order_parameter(theta):
    """r e^{iψ} = (1/N) Σ exp(iθ_j) ; renvoie r ∈ [0,1]."""
    return np.abs(np.mean(np.exp(1j * theta)))


def step_rk4(theta, omega, K, dt, N):
    """Pas RK4. dθ/dt = ω + (K/N) Σ_j sin(θ_j − θ_i)."""
    def f(th):
        sin_diff = np.sin(th[None, :] - th[:, None])  # (N,N)
        return omega + (K / N) * sin_diff.sum(axis=1)
    k1 = f(theta)
    k2 = f(theta + 0.5 * dt * k1)
    k3 = f(theta + 0.5 * dt * k2)
    k4 = f(theta + dt * k3)
    return theta + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def quadrantize(theta, n_bins=4):
    """θ mod 2π → bin ∈ {0,..,n_bins-1}."""
    th = np.mod(theta, 2 * np.pi)
    return np.floor(th / (2 * np.pi / n_bins)).astype(np.int8)


def simulate(
    N=64,
    K=1.6,
    sigma_omega=1.0,
    dt=0.05,
    t_equil=80.0,
    t_sample=120.0,
    sample_gap=1.0,
    n_bins=4,
    seed=0,
    grid_shape=None,
    return_phases=False,
):
    """Simule Kuramoto. Renvoie:
       - snapshots_grid : list of (L,L) int arrays en {0..n_bins-1}, où L=sqrt(N)
       - r_traj : list de r(t) sur les samples
    grid_shape par défaut = (sqrt(N), sqrt(N)) ; impose N=L²."""
    if grid_shape is None:
        L = int(np.sqrt(N))
        assert L * L == N, f"N={N} doit être un carré parfait pour grid auto"
        grid_shape = (L, L)
    L = grid_shape[0]

    rng = np.random.default_rng(seed)
    omega = rng.normal(0.0, sigma_omega, size=N)
    theta = rng.uniform(0, 2 * np.pi, size=N)

    n_equil_steps = int(t_equil / dt)
    n_sample_steps = int(t_sample / dt)
    gap_steps = max(1, int(sample_gap / dt))

    for _ in range(n_equil_steps):
        theta = step_rk4(theta, omega, K, dt, N)

    snapshots_grid = []
    phases_series = []  # liste de array(N,) phases θ
    r_traj = []
    steps_since = 0
    for _ in range(n_sample_steps):
        theta = step_rk4(theta, omega, K, dt, N)
        steps_since += 1
        if steps_since >= gap_steps:
            steps_since = 0
            q = quadrantize(theta, n_bins=n_bins)
            snapshots_grid.append(q.reshape(grid_shape))
            if return_phases:
                phases_series.append(theta.copy())
            r_traj.append(order_parameter(theta))

    if return_phases:
        return snapshots_grid, np.array(r_traj), np.array(phases_series)
    return snapshots_grid, np.array(r_traj)


def chi_from_r(r_traj, N):
    """Susceptibilité analogue : χ = N · Var(r)."""
    return N * r_traj.var()
