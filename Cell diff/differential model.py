import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import os

# ====================== PARAMETERS ======================
# Grid
L = 1.0                # Domain size: from -L to L
N_grid = 200           # Grid resolution (N x N)
dx = 2 * L / N_grid
x = np.linspace(-L, L, N_grid)
y = np.linspace(-L, L, N_grid)
X, Y = np.meshgrid(x, y)

# Time
T_max = 50.0            # Total time (arbitrary units)
dt = 0.001             # Time step (must be small)
steps = int(T_max / dt)
output_every = 500     # Save every N steps


# Model parameters
D_N = 0.01      # Nutrient diffusion
D_H = 0.001     # Health smoothing
gamma = 0.05    # Nutrient consumption rate
alpha = 0.3     # Proliferation rate
beta = 0.2      # Death rate

'''
D_N    = 0.002     # 🔻 Slower nutrient diffusion
gamma  = 0.2       # 🔺 Faster nutrient consumption
alpha  = 0.3       # ⏸️  Keep proliferation the same
beta   = 0.8  
'''
# Initial conditions
N_max = 1.0
R0 = 0.2             # Initial spheroid radius
H_init = 1.0         # Initial health level inside spheroid

# ====================== INITIALIZATION ======================
N = N_max * np.ones((N_grid, N_grid))
H = H_init * np.exp(-(X**2 + Y**2) / (2 * (R0 / 2)**2))

# Create output directory
os.makedirs("output", exist_ok=True)

# ====================== CORE UPDATE ======================
@njit
def laplacian(Z, dx):
    Z_top    = Z[:-2, 1:-1]
    Z_bottom = Z[2:, 1:-1]
    Z_left   = Z[1:-1, :-2]
    Z_right  = Z[1:-1, 2:]
    Z_center = Z[1:-1, 1:-1]

    lap = (Z_top + Z_bottom + Z_left + Z_right - 4 * Z_center) / dx**2
    return lap

@njit
def step(N, H, D_N, D_H, gamma, alpha, beta, dx, dt):
    lap_N = laplacian(N, dx)
    lap_H = laplacian(H, dx)

    # Update interior only
    N_new = N.copy()
    H_new = H.copy()

    N_new[1:-1, 1:-1] += dt * (D_N * lap_N - gamma * H[1:-1, 1:-1])
    H_new[1:-1, 1:-1] += dt * (
    D_H * lap_H
    + alpha * N[1:-1, 1:-1] * H[1:-1, 1:-1] * (1 - H[1:-1, 1:-1])
    - beta * (1 - N[1:-1, 1:-1]) * H[1:-1, 1:-1]
    )

    # Dirichlet BC for N
    N_new[0, :] = N_new[-1, :] = N_new[:, 0] = N_new[:, -1] = 1.0

    # Neumann BC for H
    H_new[0, :]  = H_new[1, :]
    H_new[-1, :] = H_new[-2, :]
    H_new[:, 0]  = H_new[:, 1]
    H_new[:, -1] = H_new[:, -2]

    return N_new, H_new

# ====================== TIME LOOP ======================
for t in range(steps):
    N, H = step(N, H, D_N, D_H, gamma, alpha, beta, dx, dt)

    if t % output_every == 0:
        print(f"Step {t}/{steps}")

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(N, extent=[-L, L, -L, L], origin='lower', cmap='Blues', vmin=0, vmax=1.0)
        plt.title(f'Nutrient (t={t*dt:.2f})')
        plt.colorbar(label='Nutrient level')

        plt.subplot(1, 2, 2)
        plt.imshow(H, extent=[-L, L, -L, L], origin='lower', cmap='YlOrRd', vmin=0, vmax=1.0)
        plt.title(f'Health (t={t*dt:.2f})')
        plt.colorbar(label='Health level')

        plt.tight_layout()
        plt.savefig(f"output/step_{t:05d}.png")
        plt.close()
