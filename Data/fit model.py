import numpy as np
import pandas as pd
from numba import njit
from scipy.optimize import differential_evolution
from tqdm import tqdm


# ====================== PDE CORE ======================

@njit
def laplacian(Z, dx):
    Z_top = Z[:-2, 1:-1]
    Z_bottom = Z[2:, 1:-1]
    Z_left = Z[1:-1, :-2]
    Z_right = Z[1:-1, 2:]
    Z_center = Z[1:-1, 1:-1]

    return (Z_top + Z_bottom + Z_left + Z_right - 4.0 * Z_center) / dx**2


@njit
def step(n, h, lambda_, delta, k_n, k_p, k_d, alpha, beta, dx, dt):
    lap_n = laplacian(n, dx)
    lap_h = laplacian(h, dx)

    n_new = n.copy()
    h_new = h.copy()

    n_mid = n[1:-1, 1:-1]
    h_mid = h[1:-1, 1:-1]

    # Saturation terms
    nutrient_consumption = lambda_ * (n_mid / (k_n + n_mid)) * h_mid

    proliferation = alpha * (n_mid / (k_p + n_mid)) * h_mid * (1.0 - h_mid)

    death = beta * (k_d / (k_d + n_mid)) * h_mid

    # PDE update
    n_new[1:-1, 1:-1] += dt * (
        lap_n - nutrient_consumption
    )

    h_new[1:-1, 1:-1] += dt * (
        delta * lap_h + proliferation - death
    )

    # Dirichlet BC for nutrient: n = 1 at boundary
    n_new[0, :] = 1.0
    n_new[-1, :] = 1.0
    n_new[:, 0] = 1.0
    n_new[:, -1] = 1.0

    # Neumann BC for health: zero-flux
    h_new[0, :] = h_new[1, :]
    h_new[-1, :] = h_new[-2, :]
    h_new[:, 0] = h_new[:, 1]
    h_new[:, -1] = h_new[:, -2]

    # Keep fields inside physical interval [0, 1]
    for i in range(n_new.shape[0]):
        for j in range(n_new.shape[1]):
            if n_new[i, j] < 0.0:
                n_new[i, j] = 0.0
            elif n_new[i, j] > 1.0:
                n_new[i, j] = 1.0

            if h_new[i, j] < 0.0:
                h_new[i, j] = 0.0
            elif h_new[i, j] > 1.0:
                h_new[i, j] = 1.0

    return n_new, h_new


# ====================== RADIUS EXTRACTION ======================

def estimate_radius(h, X, Y, threshold=0.2):
    R = np.sqrt(X**2 + Y**2).flatten()
    Hvals = h.flatten()

    bins = np.linspace(0, R.max(), 200)
    digitized = np.digitize(R, bins)

    radial_mean = []
    for i in range(1, len(bins)):
        vals = Hvals[digitized == i]
        if len(vals) == 0:
            radial_mean.append(np.nan)
        else:
            radial_mean.append(vals.mean())

    radial_mean = np.nan_to_num(radial_mean)

    below = np.where(radial_mean < threshold)[0]

    if len(below) == 0:
        return bins[-1]
    else:
        return bins[below[0]]


# ====================== SIMULATION WRAPPER ======================

def run_simulation(
    lambda_,
    delta,
    k_n,
    k_p,
    k_d,
    alpha,
    beta,
    R0=0.1,
    radius_scale=1000.0,
    days=None,
    L=1.0,
    N_grid=200,
    dt=0.001,
    threshold=0.2,
    show_bar=True
):
    if days is None:
        days = [1, 5, 10]

    dx = 2.0 * L / N_grid

    x = np.linspace(-L, L, N_grid)
    y = np.linspace(-L, L, N_grid)
    X, Y = np.meshgrid(x, y)

    # Initial nutrient field
    n = np.ones((N_grid, N_grid))

    # Smooth disk initial health field
    R = np.sqrt(X**2 + Y**2)
    h = 1.0 / (1.0 + np.exp((R - R0) / 0.02))

    max_day = max(days)
    steps = int(max_day / dt)

    target_steps = [int(d / dt) for d in days]
    out_radii = []

    iterator = range(steps + 1)

    if show_bar:
        iterator = tqdm(iterator, desc="Simulating", leave=False)

    for t in iterator:
        if t in target_steps:
            r_model = estimate_radius(h, X, Y, threshold)
            out_radii.append(radius_scale * r_model)

        n, h = step(
            n,
            h,
            lambda_,
            delta,
            k_n,
            k_p,
            k_d,
            alpha,
            beta,
            dx,
            dt
        )

    return np.array(out_radii)


# ====================== LOAD DATA ======================

exp = pd.read_csv(
    "train_txt/15625.txt",
    comment="#",
    sep=r"\s+",
    header=None,
    names=["Day", "MeanRadius", "StdRadius", "N"],
    engine="python"
)

exp = exp.dropna()

days = exp["Day"].values.astype(float)
r_exp = exp["MeanRadius"].values.astype(float)
std_exp = exp["StdRadius"].values.astype(float)
n_repl = exp["N"].values.astype(float)


# ====================== FITTING ======================

def objective(params):
    lambda_, delta, k_n, k_p, k_d, alpha, beta, R0, radius_scale = params

    r_model = run_simulation(
        lambda_,
        delta,
        k_n,
        k_p,
        k_d,
        alpha,
        beta,
        R0=R0,
        radius_scale=radius_scale,
        days=days,
        show_bar=False
    )

    # Weighted residuals
    sigma0 = 10.0
    weights = np.sqrt(n_repl) / np.sqrt(std_exp**2 + sigma0**2)

    residuals = weights * (r_model - r_exp)

    return np.mean(residuals**2)


bounds = [
    (1e-4, 50.0),    # lambda_: nutrient consumption strength
    (1e-5, 1.0),     # delta = D_H / D_N
    (1e-4, 10.0),    # k_n: half-saturation for nutrient consumption
    (1e-4, 10.0),    # k_p: half-saturation for proliferation
    (1e-4, 10.0),    # k_d: saturation constant for death
    (1e-4, 50.0),    # alpha: proliferation strength
    (1e-4, 50.0),    # beta: death strength
    (0.01, 0.5),     # R0 in model units
    (50.0, 2000.0),  # radius_scale: micrometers per model unit
]


# ---- Progress bar for optimization ----

max_generations = 20

pbar = tqdm(total=max_generations, desc="Optimization progress")


def callback(xk, convergence):
    pbar.update(1)
    return False


result = differential_evolution(
    objective,
    bounds,
    maxiter=max_generations,
    polish=False,
    updating="deferred",
    workers=1,
    callback=callback,
    seed=42
)

pbar.close()


# ====================== RESULTS ======================

print("Best-fit params:")
print(f"lambda_      = {result.x[0]}")
print(f"delta        = {result.x[1]}")
print(f"k_n          = {result.x[2]}")
print(f"k_p          = {result.x[3]}")
print(f"k_d          = {result.x[4]}")
print(f"alpha        = {result.x[5]}")
print(f"beta         = {result.x[6]}")
print(f"R0           = {result.x[7]}")
print(f"radius_scale = {result.x[8]}")
print("MSE:", result.fun)


print("\nRunning final simulation with best-fit parameters:")

r_fit = run_simulation(
    *result.x[:7],
    R0=result.x[7],
    radius_scale=result.x[8],
    days=days,
    show_bar=True
)

print("\nObserved radius:")
print(r_exp)

print("\nFitted radius:")
print(r_fit)