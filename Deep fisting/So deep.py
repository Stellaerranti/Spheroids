import numpy as np
import matplotlib.pyplot as plt
import os
import base_functions as bf
import pandas as pd
from scipy.optimize import differential_evolution
import os
import json
import numpy as np
import pandas as pd

def load_empirical_data(path):
    data = pd.read_csv(path, sep="\t", skiprows=1)

    data.columns = [str(c).strip() for c in data.columns]

    if "# Day" in data.columns:
        data = data.rename(columns={"# Day": "Day"})

    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=["Day"])
    data = data.sort_values("Day").reset_index(drop=True)

    required = [
    "Day",
    "MeanRadius",
    "StdRadius",
    "MeanGrayMean",
    "StdGrayMean",
    "MeanIntDen",
    "StdIntDen",
    ]

    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return data

def simulate_model(params, empirical_days, N_grid=80, L=1.0, dt=0.002):
    D_N, D_H, gamma, alpha, beta, R0, time_scale = params

    x = np.linspace(-L, L, N_grid)
    y = np.linspace(-L, L, N_grid)
    dx = x[1] - x[0]


    X, Y = np.meshgrid(x, y)

    N = np.ones((N_grid, N_grid))
    H = np.exp(-(X**2 + Y**2) / (2 * (R0 / 2)**2))

    empirical_days = np.asarray(empirical_days, dtype=float)
    empirical_model_times = empirical_days * time_scale

    T_max = np.max(empirical_model_times)
    steps = int(np.ceil(T_max / dt))

    output_steps = {
        int(round(t / dt)): day
        for t, day in zip(empirical_model_times, empirical_days)
    }

    results = {}

    for t_step in range(steps + 1):
        if t_step in output_steps:
            day = output_steps[t_step]
            results[day] = {
                "H": H.copy(),
                "N": N.copy(),
                "dx": dx,
                "x": x,
                "y": y,
            }

        if t_step < steps:
            N, H = step(N, H, D_N, D_H, gamma, alpha, beta, dx, dt)
        
            N = np.clip(N, 0.0, 1.0)
            H = np.clip(H, 0.0, 1.0)

    return results

def calculate_model_observables(H, dx, pixel_scale=1.0, intensity_scale=255.0, threshold=0.2):
    """
    Calculates ImageJ-like observables from the synthetic H field.

    Parameters
    ----------
    H : 2D array
        Synthetic spheroid density / health field, usually in [0, 1].

    dx : float
        Model grid spacing in dimensionless/model units.

    pixel_scale : float
        Converts model length units to empirical image pixels.

    intensity_scale : float
        Converts model H values to empirical grayscale units.

    threshold : float
        Threshold for defining the outer spheroid mask.

    Returns
    -------
    obs : dict
        Dictionary with synthetic observables comparable to empirical columns.
    """

    mask_area = bf.get_spheroid_mask(H, threshold=threshold)

    if not np.any(mask_area):
        return {
            "Radius": np.nan,
            "Area": np.nan,
            "GrayMean": np.nan,
            "IntDen": np.nan,
            "Circularity": np.nan,
            "Feret": np.nan,
            "Perimeter": np.nan,
        }

    # Geometry in model units
    area_model = bf.area_from_mask(mask_area, dx=dx)
    perimeter_model = bf.perimeter_from_mask(mask_area, dx=dx)
    feret_model = bf.feret_diameter_from_mask(mask_area, dx=dx)

    radius_model = np.sqrt(area_model / np.pi)

    if perimeter_model == 0 or np.isnan(perimeter_model):
        circularity = np.nan
    else:
        circularity = 4 * np.pi * area_model / perimeter_model**2

    # Intensity values
    H_scaled = H * intensity_scale
    values = H_scaled[mask_area]

    gray_mean = np.mean(values)

    # Convert model geometry to empirical pixel units
    radius_pixels = radius_model * pixel_scale
    area_pixels = area_model * pixel_scale**2
    perimeter_pixels = perimeter_model * pixel_scale
    feret_pixels = feret_model * pixel_scale

    # ImageJ-like integrated density:
    # area in pixels * mean gray value
    int_den = area_pixels * gray_mean

    return {
        "Radius": radius_pixels,
        "Area": area_pixels,
        "GrayMean": gray_mean,
        "IntDen": int_den,
        "Circularity": circularity,
        "Feret": feret_pixels,
        "Perimeter": perimeter_pixels,
    }

def objective(params, empirical_data, targets, N_grid=80, L=1.0, dt=0.002, threshold=0.2):
    """
    Objective function for optimization.

    Parameters
    ----------
    params : list or array
        Model and scaling parameters:
        [D_N, D_H, gamma, alpha, beta, R0, time_scale, pixel_scale, intensity_scale]

    empirical_data : DataFrame
        Loaded empirical data table.

    targets : list of str
        Names of observables to fit, for example:
        ["Radius", "GrayMean", "IntDen"]

    Returns
    -------
    error : float
        Sum of squared weighted residuals.
    """

    D_N, D_H, gamma, alpha, beta, R0, time_scale, pixel_scale, intensity_scale = params

    # Basic safety checks
    if D_N <= 0 or D_H <= 0 or gamma < 0 or alpha < 0 or beta < 0:
        return 1e30

    if R0 <= 0 or time_scale <= 0 or pixel_scale <= 0 or intensity_scale <= 0:
        return 1e30

    empirical_days = empirical_data["Day"].values

    try:
        sim_results = simulate_model(
            params=[D_N, D_H, gamma, alpha, beta, R0, time_scale],
            empirical_days=empirical_days,
            N_grid=N_grid,
            L=L,
            dt=dt,
        )
    except Exception:
        return 1e30

    residuals = []

    for _, row in empirical_data.iterrows():
        day = row["Day"]

        if day not in sim_results:
            return 1e30

        H = sim_results[day]["H"]
        dx = sim_results[day]["dx"]

        obs = calculate_model_observables(
            H,
            dx=dx,
            pixel_scale=pixel_scale,
            intensity_scale=intensity_scale,
            threshold=threshold,
        )

        for target in targets:
            mean_col = f"Mean{target}"
            std_col = f"Std{target}"
        
            if mean_col not in empirical_data.columns:
                continue
        
            empirical_mean = row[mean_col]
            model_value = obs[target]
        
            if std_col in empirical_data.columns:
                sigma = row[std_col]
            else:
                sigma = 1.0
        
            if pd.isna(empirical_mean):
                continue
        
            # Critical: failed model prediction is not acceptable
            if pd.isna(model_value):
                return 1e30
        
            if pd.isna(sigma) or sigma <= 0:
                sigma = 1.0
        
            residuals.append((model_value - empirical_mean) / sigma)

    if len(residuals) == 0:
        return 1e30

    residuals = np.array(residuals)

    if not np.all(np.isfinite(residuals)):
        return 1e30

    return np.sum(residuals**2)

def laplacian(Z, dx):
    Z_top    = Z[:-2, 1:-1]
    Z_bottom = Z[2:, 1:-1]
    Z_left   = Z[1:-1, :-2]
    Z_right  = Z[1:-1, 2:]
    Z_center = Z[1:-1, 1:-1]

    lap = (Z_top + Z_bottom + Z_left + Z_right - 4 * Z_center) / dx**2
    return lap

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

    N_new[0, :] = N_new[-1, :] = N_new[:, 0] = N_new[:, -1] = 1.0

    H_new[0, :]  = H_new[1, :]
    H_new[-1, :] = H_new[-2, :]
    H_new[:, 0]  = H_new[:, 1]
    H_new[:, -1] = H_new[:, -2]

    return N_new, H_new 

def evaluate_fit(params, empirical_data, targets, N_grid=80, L=1.0, dt=0.002, threshold=0.2):
    D_N, D_H, gamma, alpha, beta, R0, time_scale, pixel_scale, intensity_scale = params

    sim_results = simulate_model(
        params=[D_N, D_H, gamma, alpha, beta, R0, time_scale],
        empirical_days=empirical_data["Day"].values,
        N_grid=N_grid,
        L=L,
        dt=dt,
    )

    rows = []

    for _, row in empirical_data.iterrows():
        day = row["Day"]

        H = sim_results[day]["H"]
        dx = sim_results[day]["dx"]

        obs = calculate_model_observables(
            H,
            dx=dx,
            pixel_scale=pixel_scale,
            intensity_scale=intensity_scale,
            threshold=threshold,
        )

        out = {"Day": day}

        for target in targets:
            mean_col = f"Mean{target}"
            std_col = f"Std{target}"

            out[f"Empirical{target}"] = row[mean_col]
            out[f"Model{target}"] = obs[target]
            out[f"Std{target}"] = row[std_col]
            if pd.isna(obs[target]):
                out[f"Residual{target}"] = np.nan
            else:
                out[f"Residual{target}"] = (obs[target] - row[mean_col]) / row[std_col]

        rows.append(out)

    return pd.DataFrame(rows)


import os
import json

def save_fit_results(
    result,
    empirical_data,
    targets,
    out_dir="fit_output",
    N_grid=80,
    L=1.0,
    dt=0.002,
    threshold=0.2,
):
    os.makedirs(out_dir, exist_ok=True)

    # Save fit table
    fit_table = evaluate_fit(
        result.x,
        empirical_data,
        targets,
        N_grid=N_grid,
        L=L,
        dt=dt,
        threshold=threshold,
    )

    fit_table.to_csv(
        os.path.join(out_dir, "fit_table.csv"),
        index=False
    )

    # Named parameters
    param_names = [
        "D_N",
        "D_H",
        "gamma",
        "alpha",
        "beta",
        "R0",
        "time_scale",
        "pixel_scale",
        "intensity_scale",
    ]

    fitted_parameters = {
        name: float(value)
        for name, value in zip(param_names, result.x)
    }

    # Save compact summary
    summary = {
        "equations": [
            "N_t = D_N * laplacian(N) - gamma * H",
            "H_t = D_H * laplacian(H) + alpha * N * H * (1 - H) - beta * (1 - N) * H",
        ],
        "boundary_conditions": {
            "N": "Dirichlet N = 1",
            "H": "Neumann zero-flux",
        },
        "targets": targets,
        "fitted_parameters": fitted_parameters,
        "optimizer": {
            "method": "differential_evolution",
            "success": bool(result.success),
            "message": str(result.message),
            "error": float(result.fun),
            "iterations": int(result.nit),
            "function_evaluations": int(result.nfev),
        },
        "simulation": {
            "N_grid": int(N_grid),
            "L": float(L),
            "dt": float(dt),
            "threshold": float(threshold),
        },
        "output_files": {
            "fit_table": "fit_table.csv",
            "fit_summary": "fit_summary.json",
        },
    }

    with open(os.path.join(out_dir, "fit_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return fit_table


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
beta = 0.2  

# Initial conditions
N_max = 1.0
R0 = 0.1             # Initial spheroid radius
H_init = 1.0         # Initial health level inside spheroid

# ====================== INITIALIZATION ======================
N = N_max * np.ones((N_grid, N_grid))
H = H_init * np.exp(-(X**2 + Y**2) / (2 * (R0 / 2)**2))

os.makedirs("output", exist_ok=True)

targets = [
    "Radius",
    "GrayMean",
    "IntDen",
]

fit_uncertainties = [
    "StdRadius",
    "StdGrayMean",
    "StdIntDen",
]


target_columns = {
    "Radius": ("MeanRadius", "StdRadius"),
    "GrayMean": ("MeanGrayMean", "StdGrayMean"),
    "IntDen": ("MeanIntDen", "StdIntDen"),
}

empirical_data = load_empirical_data("500.txt")

bounds = [
    (1e-4, 5e-2),   # D_N
    (1e-5, 1e-2),   # D_H
    (1e-4, 1.0),    # gamma
    (1e-3, 2.0),    # alpha
    (1e-4, 2.0),    # beta
    (0.02, 0.3),    # R0
    (0.1, 5.0),     # time_scale
    (100, 1000),    # pixel_scale
    (1, 255),       # intensity_scale
]



result = differential_evolution(
    objective,
    bounds=bounds,
    args=(empirical_data, targets),
    maxiter=15,
    popsize=6,
    polish=True,
    workers=1,
    updating="immediate",
    seed=None,
)

print("Best error:", result.fun)
print("Best parameters:", result.x)

fit_table = save_fit_results(
    result=result,
    empirical_data=empirical_data,
    targets=targets,
    out_dir="fit_output_simple",
    N_grid=81,
    L=1.0,
    dt=0.002,
    threshold=0.2,
)
print(fit_table)
'''
N, H = step(N, H, D_N, D_H, gamma, alpha, beta, dx, dt)
#plt.imshow(H, extent=[-L, L, -L, L], origin='lower', cmap='YlOrRd', vmin=0, vmax=1.0)
#plt.title('Health')
#plt.colorbar(label='Health level')
cy, cx, r,_,_,_ = bf.make_circle(H)
theta = np.linspace(0, 2*np.pi, 500)

plt.imshow(H, origin="lower", cmap="YlOrRd")
plt.plot(cx + r*np.cos(theta), cy + r*np.sin(theta), linewidth=2)
plt.scatter([cx], [cy], s=20)
plt.show()


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
'''