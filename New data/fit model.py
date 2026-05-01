import numpy as np
import pandas as pd
from numba import njit
from scipy.optimize import differential_evolution
from tqdm import tqdm

# ====================== PDE CORE ======================
@njit
def laplacian(Z, dx):
    Z_top    = Z[:-2, 1:-1]
    Z_bottom = Z[2:, 1:-1]
    Z_left   = Z[1:-1, :-2]
    Z_right  = Z[1:-1, 2:]
    Z_center = Z[1:-1, 1:-1]
    return (Z_top + Z_bottom + Z_left + Z_right - 4 * Z_center) / dx**2

@njit
def step(N, H, D_N, D_H, gamma, alpha, beta, dx, dt):
    lap_N = laplacian(N, dx)
    lap_H = laplacian(H, dx)
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

# ====================== RADIUS EXTRACTION ======================
def estimate_radius(H, X, Y, threshold=0.2):
    R = np.sqrt(X**2 + Y**2).flatten()
    Hvals = H.flatten()

    bins = np.linspace(0, R.max(), 200)
    digitized = np.digitize(R, bins)
    radial_mean = [Hvals[digitized == i].mean() for i in range(1, len(bins))]
    radial_mean = np.nan_to_num(radial_mean)

    below = np.where(radial_mean < threshold)[0]
    if len(below) == 0:
        return bins[-1]
    else:
        return bins[below[0]]

# ====================== SIMULATION WRAPPER ======================
def run_simulation(D_N, D_H, gamma, alpha, beta, R0=0.1, days=[1,5,10],
                   L=1.0, N_grid=200, dt=0.001, threshold=0.2,
                   show_bar=True):
    dx = 2 * L / N_grid
    x = np.linspace(-L, L, N_grid)
    y = np.linspace(-L, L, N_grid)
    X, Y = np.meshgrid(x, y)

    N = np.ones((N_grid, N_grid))
    #H = np.exp(-(X**2 + Y**2) / (2 * (R0/2)**2))
    
    R = np.sqrt(X**2 + Y**2)
    H = 1 / (1 + np.exp((R - R0)/0.02))

    max_day = max(days)
    steps = int(max_day / dt)

    target_steps = [int(d / dt) for d in days]
    out_radii = []

    iterator = range(steps+1)
    if show_bar:
        iterator = tqdm(iterator, desc="Simulating", leave=False)

    for t in iterator:
        N, H = step(N, H, D_N, D_H, gamma, alpha, beta, dx, dt)
        if t in target_steps:
            r = estimate_radius(H, X, Y, threshold)
            out_radii.append(r)

    return np.array(out_radii)

# ====================== FITTING ======================
# Example: load one experimental curve
exp = pd.read_csv("train_txt/15625.txt", 
                  comment="#", sep="\t", header=None,
                  names=["Day", "MeanRadius", "StdRadius", "N"])
days = exp["Day"].values
r_exp = exp["MeanRadius"].values

def objective(params):
    D_N, D_H, gamma, alpha, beta = params
    # turn off simulation bar during optimization (too many)
    r_model = run_simulation(D_N, D_H, gamma, alpha, beta, days=days, show_bar=False)
    return np.mean((r_model - r_exp)**2)

bounds = [
    (0.001, 0.1),   # D_N
    (0.0001, 0.01), # D_H
    (0.01, 1.0),    # gamma
    (0.01, 1.0),    # alpha
    (0.01, 1.0)     # beta
]

# ---- Progress bar for optimization ----
max_generations = 20   # set equal to maxiter you want
pbar = tqdm(total=max_generations, desc="Optimization progress")

def callback(xk, convergence):
    pbar.update(1)   # update each generation
    return False     # continue optimization

result = differential_evolution(
    objective, bounds,
    maxiter=max_generations,
    polish=False,
    updating='deferred',
    workers=1,
    callback=callback
)

pbar.close()
print("Best-fit params:", result.x)
print("MSE:", result.fun)

# ---- Run final simulation with visible bar ----
print("\nRunning final simulation with best-fit parameters:")
_ = run_simulation(*result.x, days=days, show_bar=True)
