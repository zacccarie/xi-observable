import numpy as np


def laplacian(Z):
    """Laplacien 9-point (toroïdal), noyau standard reaction-diffusion."""
    return (
        -1.0 * Z
        + 0.20 * (np.roll(Z, 1, 0) + np.roll(Z, -1, 0) + np.roll(Z, 1, 1) + np.roll(Z, -1, 1))
        + 0.05 * (np.roll(np.roll(Z, 1, 0), 1, 1) + np.roll(np.roll(Z, 1, 0), -1, 1)
                  + np.roll(np.roll(Z, -1, 0), 1, 1) + np.roll(np.roll(Z, -1, 0), -1, 1))
    )


def simulate_gray_scott(L=128, F=0.037, k=0.06, Du=0.16, Dv=0.08, dt=1.0,
                        t_equil=2000, n_frames=0, frame_gap=20, seed=0):
    """Gray–Scott : u + 2v → 3v, v → P.
    du/dt = Du∇²u − uv² + F(1−u) ; dv/dt = Dv∇²v + uv² − (F+k)v.
    Retourne l'état final v (L,L), ou (n_frames, L, L) si n_frames>0."""
    rng = np.random.default_rng(seed)
    u = np.ones((L, L), dtype=np.float64)
    v = np.zeros((L, L), dtype=np.float64)
    # graines centrales perturbées
    n_seed = max(1, L // 16)
    for _ in range(n_seed * n_seed):
        cy, cx = rng.integers(0, L, 2)
        u[cy-2:cy+3, cx-2:cx+3] = 0.50
        v[cy-2:cy+3, cx-2:cx+3] = 0.25
    u += 0.02 * rng.standard_normal((L, L))
    v += 0.02 * rng.standard_normal((L, L))

    frames = []
    total = t_equil + (n_frames * frame_gap if n_frames else 0)
    for t in range(total):
        uvv = u * v * v
        u += dt * (Du * laplacian(u) - uvv + F * (1.0 - u))
        v += dt * (Dv * laplacian(v) + uvv - (F + k) * v)
        np.clip(u, 0, 1, out=u); np.clip(v, 0, 1, out=v)
        if n_frames and t >= t_equil and (t - t_equil) % frame_gap == 0:
            frames.append(v.copy())
    if n_frames:
        return np.array(frames)
    return v


def gradient_magnitude(Z):
    """|∇Z| par différences centrées toroïdales — détecte les interfaces."""
    gy = 0.5 * (np.roll(Z, -1, 0) - np.roll(Z, 1, 0))
    gx = 0.5 * (np.roll(Z, -1, 1) - np.roll(Z, 1, 1))
    return np.sqrt(gx * gx + gy * gy)
