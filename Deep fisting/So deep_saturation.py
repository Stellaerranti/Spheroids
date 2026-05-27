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
from scipy.optimize import minimize

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

    "MeanFeret",
    "StdFeret",

    "MeanPerimeter",
    "StdPerimeter",
    ]

    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return data

def simulate_model(params, empirical_days, N_grid=80, L=1.0, dt=0.002):
    (
        lambda_,
        delta,
        k_n,
        k_p,
        k_d,
        alpha,
        beta,
        R0,
        time_scale,
    ) = params

    x = np.linspace(-L, L, N_grid)
    y = np.linspace(-L, L, N_grid)
    dx = x[1] - x[0]

    X, Y = np.meshgrid(x, y)

    n = np.ones((N_grid, N_grid))
    h = np.exp(-(X**2 + Y**2) / (2 * (R0 / 2)**2))
    
    #print("Initial h max:", h.max())
    #rint("Initial pixels above threshold:", np.sum(h > 0.2))

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
                "H": h.copy(),
                "N": n.copy(),
                "dx": dx,
                "x": x,
                "y": y,
            }

        if t_step < steps:
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
                dt,
            )

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
    touches_boundary = (
    np.any(mask_area[0, :]) or
    np.any(mask_area[-1, :]) or
    np.any(mask_area[:, 0]) or
    np.any(mask_area[:, -1])
)

    if not np.any(mask_area):
        return {
            "Radius": np.nan,
            "Area": np.nan,
            "GrayMean": np.nan,
            "IntDen": np.nan,
            "Circularity": np.nan,
            "Feret": np.nan,
            "Perimeter": np.nan,
            "TouchesBoundary": False,
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
    "TouchesBoundary": touches_boundary,
    }

def objective(params, empirical_data, targets, N_grid=80, L=1.0, dt=0.002, threshold=0.2):
    (
        lambda_,
        delta,
        k_n,
        k_p,
        k_d,
        alpha,
        beta,
        R0,
        time_scale,
        pixel_scale,
        intensity_scale,
    ) = params

    if (
        lambda_ <= 0 or delta <= 0 or
        k_n <= 0 or k_p <= 0 or k_d <= 0 or
        alpha < 0 or beta < 0 or
        R0 <= 0 or time_scale <= 0 or
        pixel_scale <= 0 or intensity_scale <= 0
    ):
        return 1e30

    empirical_days = empirical_data["Day"].values

    try:
        sim_results = simulate_model(
            params=[
                lambda_,
                delta,
                k_n,
                k_p,
                k_d,
                alpha,
                beta,
                R0,
                time_scale,
            ],
            empirical_days=empirical_days,
            N_grid=N_grid,
            L=L,
            dt=dt,
        )
    except Exception:
        return 1e30

    residuals = []
    model_rows = []

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
        model_rows.append({
    "Day": day,
    "Radius": obs["Radius"],
    "IntDen": obs["IntDen"],
    "Feret": obs["Feret"],
    "Perimeter": obs["Perimeter"],
})
        if obs.get("TouchesBoundary", False):
            return 1e30

        for target in targets:
            mean_col = f"Mean{target}"
            std_col = f"Std{target}"

            if mean_col not in empirical_data.columns:
                return 1e30

            empirical_mean = row[mean_col]
            model_value = obs[target]

            if std_col in empirical_data.columns:
                sigma = row[std_col]
            else:
                sigma = 1.0

            if pd.isna(empirical_mean):
                continue

            if pd.isna(model_value):
                return 1e30

            if pd.isna(sigma) or sigma <= 0:
                sigma = 1.0
                
            

            weight = target_weights.get(target, 1.0)
            residuals.append(np.sqrt(weight) * (model_value - empirical_mean) / sigma)

    if len(residuals) == 0:
        return 1e30

    residuals = np.array(residuals)

    if not np.all(np.isfinite(residuals)):
        return 1e30
    
    base_error = np.sum(residuals**2)

    sat_penalty = late_saturation_penalty(
    model_rows,
    empirical_data,
    start_day=13,
    weight=5.0,
    )
    
    return base_error + sat_penalty

def laplacian(Z, dx):
    Z_top    = Z[:-2, 1:-1]
    Z_bottom = Z[2:, 1:-1]
    Z_left   = Z[1:-1, :-2]
    Z_right  = Z[1:-1, 2:]
    Z_center = Z[1:-1, 1:-1]

    lap = (Z_top + Z_bottom + Z_left + Z_right - 4 * Z_center) / dx**2
    return lap

def step(n, h, lambda_, delta, k_n, k_p, k_d, alpha, beta, dx, dt):
    lap_n = laplacian(n, dx)
    lap_h = laplacian(h, dx)

    n_new = n.copy()
    h_new = h.copy()

    eps = 1e-12

    consumption = lambda_ * n[1:-1, 1:-1] / (k_n + n[1:-1, 1:-1] + eps) * h[1:-1, 1:-1]

    proliferation = (
        alpha
        * n[1:-1, 1:-1] / (k_p + n[1:-1, 1:-1] + eps)
        * h[1:-1, 1:-1]
        * (1 - h[1:-1, 1:-1])
    )

    death = (
        beta
        * k_d / (k_d + n[1:-1, 1:-1] + eps)
        * h[1:-1, 1:-1]
    )

    n_new[1:-1, 1:-1] += dt * (lap_n - consumption)

    h_new[1:-1, 1:-1] += dt * (
        delta * lap_h
        + proliferation
        - death
    )

    # Dirichlet BC for nutrient
    n_new[0, :] = 1.0
    n_new[-1, :] = 1.0
    n_new[:, 0] = 1.0
    n_new[:, -1] = 1.0

    # Neumann BC for h
    h_new[0, :] = h_new[1, :]
    h_new[-1, :] = h_new[-2, :]
    h_new[:, 0] = h_new[:, 1]
    h_new[:, -1] = h_new[:, -2]

    n_new = np.clip(n_new, 0.0, 1.0)
    h_new = np.clip(h_new, 0.0, 1.0)

    return n_new, h_new

def evaluate_fit(params, empirical_data, targets, N_grid=80, L=1.0, dt=0.002, threshold=0.2):
    (
        lambda_,
        delta,
        k_n,
        k_p,
        k_d,
        alpha,
        beta,
        R0,
        time_scale,
        pixel_scale,
        intensity_scale,
    ) = params

    sim_results = simulate_model(
        params=[
            lambda_,
            delta,
            k_n,
            k_p,
            k_d,
            alpha,
            beta,
            R0,
            time_scale,
        ],
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
    optimizer_name="unknown",
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
    "lambda_",
    "delta",
    "k_n",
    "k_p",
    "k_d",
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
    "n_t = laplacian(n) - lambda_ * n / (k_n + n) * h",
    "h_t = delta * laplacian(h) + alpha * n / (k_p + n) * h * (1 - h) - beta * k_d / (k_d + n) * h",
    ],
        "boundary_conditions": {
    "n": "Dirichlet n = 1",
    "h": "Neumann zero-flux"
    },
        "targets": targets,
        "fitted_parameters": fitted_parameters,
        "optimizer": {
            "method": optimizer_name,
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
        "late_saturation_penalty": {
            "enabled": True,
            "targets": ["Radius", "IntDen"],
            "start_day": 13,
            "weight": 50.0,
        },
    }

    with open(os.path.join(out_dir, "fit_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return fit_table

def save_fit_plots(fit_table, out_dir="fit_output"):
    os.makedirs(out_dir, exist_ok=True)

    plot_specs = [
        ("Radius", "Radius"),
        ("GrayMean", "Mean gray value"),
        ("IntDen", "Integrated density"),
        ("Feret", "Feret diameter"),
        ("Perimeter", "Perimeter"),
    ]

    for target, ylabel in plot_specs:
        empirical_col = f"Empirical{target}"
        model_col = f"Model{target}"
        std_col = f"Std{target}"

        if empirical_col not in fit_table.columns or model_col not in fit_table.columns:
            continue

        plt.figure(figsize=(6, 4))

        if std_col in fit_table.columns:
            plt.errorbar(
                fit_table["Day"],
                fit_table[empirical_col],
                yerr=fit_table[std_col],
                fmt="o",
                capsize=3,
                label="Empirical",
            )
        else:
            plt.plot(
                fit_table["Day"],
                fit_table[empirical_col],
                "o",
                label="Empirical",
            )

        plt.plot(
            fit_table["Day"],
            fit_table[model_col],
            "-o",
            label="Model",
        )

        plt.xlabel("Day")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()

        filename = f"{target}_fit.png"
        plt.savefig(os.path.join(out_dir, filename), dpi=300)
        plt.close()

def compute_objective_parts(params, empirical_data, targets, N_grid=81, L=3.0, dt=0.002, threshold=0.2):
    (
        lambda_,
        delta,
        k_n,
        k_p,
        k_d,
        alpha,
        beta,
        R0,
        time_scale,
        pixel_scale,
        intensity_scale,
    ) = params

    sim_results = simulate_model(
        params=[
            lambda_,
            delta,
            k_n,
            k_p,
            k_d,
            alpha,
            beta,
            R0,
            time_scale,
        ],
        empirical_days=empirical_data["Day"].values,
        N_grid=N_grid,
        L=L,
        dt=dt,
    )

    residuals = []
    model_rows = []

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

        model_rows.append({
            "Day": day,
            "Radius": obs["Radius"],
            "IntDen": obs["IntDen"],
            "Feret": obs["Feret"],
            "Perimeter": obs["Perimeter"],
        })

        for target in targets:
            mean_col = f"Mean{target}"
            std_col = f"Std{target}"

            empirical_mean = row[mean_col]
            model_value = obs[target]
            sigma = row[std_col]

            if pd.isna(empirical_mean) or pd.isna(model_value):
                continue

            if pd.isna(sigma) or sigma <= 0:
                sigma = 1.0

            weight = target_weights.get(target, 1.0)
            residuals.append(np.sqrt(weight) * (model_value - empirical_mean) / sigma)

    residuals = np.array(residuals)
    base_error = np.sum(residuals**2)

    sat_penalty = late_saturation_penalty(
        model_rows,
        targets=("Radius", "IntDen"),
        start_day=13,
        weight=50.0,
    )

    return {
        "base_error": base_error,
        "sat_penalty": sat_penalty,
        "total_error": base_error + sat_penalty,
    }

def run_seed_series(
    seeds,
    empirical_data,
    targets,
    bounds,
    out_dir="seed_runs",
    maxiter=50,
    popsize=8,
    N_grid=81,
    L=3.0,
    dt=0.002,
    threshold=0.2,
):
    os.makedirs(out_dir, exist_ok=True)

    all_rows = []

    for seed in seeds:
        print(f"\nRunning seed {seed}...")

        def objective_fixed(params, empirical_data, targets):
            return objective(
                params,
                empirical_data,
                targets,
                N_grid=N_grid,
                L=L,
                dt=dt,
                threshold=threshold,
            )

        result = differential_evolution(
            objective_fixed,
            bounds=bounds,
            args=(empirical_data, targets),
            maxiter=maxiter,
            popsize=popsize,
            polish=True,
            workers=1,
            updating="immediate",
            seed=seed,
        )

        seed_dir = os.path.join(out_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        if result.fun < 1e29:
            fit_table = save_fit_results(
                result=result,
                empirical_data=empirical_data,
                targets=targets,
                out_dir=seed_dir,
                N_grid=N_grid,
                L=L,
                dt=dt,
                threshold=threshold,
            )

            save_fit_plots(fit_table, out_dir=seed_dir)
        else:
            fit_table = None

        param_names = [
            "lambda_",
            "delta",
            "k_n",
            "k_p",
            "k_d",
            "alpha",
            "beta",
            "R0",
            "time_scale",
            "pixel_scale",
            "intensity_scale",
        ]

        row = {
            "seed": seed,
            "error": float(result.fun),
            "success": bool(result.success),
            "message": str(result.message),
            "iterations": int(result.nit),
            "function_evaluations": int(result.nfev),
            "valid_fit": bool(result.fun < 1e29),
        }

        for name, value in zip(param_names, result.x):
            row[name] = float(value)

        all_rows.append(row)

        print(f"Seed {seed}: error = {result.fun}")

    summary_table = pd.DataFrame(all_rows)
    summary_table = summary_table.sort_values("error").reset_index(drop=True)

    summary_path = os.path.join(out_dir, "seed_summary.csv")
    summary_table.to_csv(summary_path, index=False)

    # Save best parameters separately
    best = summary_table.iloc[0].to_dict()

    with open(os.path.join(out_dir, "best_seed_summary.json"), "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    print("\nBest seed:")
    print(summary_table.iloc[0])

    return summary_table

def late_saturation_penalty(model_rows, empirical_data, start_day=13, weight=1.0):
    """
    Penalizes positive late-time growth after start_day,
    scaled by empirical standard deviations.
    """

    penalty = 0.0

    rows = sorted(model_rows, key=lambda r: r["Day"])
    late_rows = [r for r in rows if r["Day"] >= start_day]

    if len(late_rows) < 2:
        return 0.0

    emp = empirical_data.set_index("Day")

    for r0, r1 in zip(late_rows[:-1], late_rows[1:]):
        d0 = r0["Day"]
        d1 = r1["Day"]

        for target in ["Radius", "IntDen"]:
            growth = r1[target] - r0[target]

            if not np.isfinite(growth):
                continue

            if growth <= 0:
                continue

            std_col = f"Std{target}"

            if std_col in emp.columns and d1 in emp.index:
                sigma = emp.loc[d1, std_col]
            else:
                sigma = 1.0

            if pd.isna(sigma) or sigma <= 0:
                sigma = 1.0

            penalty += weight * (growth / sigma) ** 2

    return penalty
        
def objective_L3(params, empirical_data, targets):
    return objective(
        params,
        empirical_data,
        targets,
        N_grid=81,
        L=3.0,
        dt=0.002,
        threshold=0.2,
    )

# Grid
L = 3.0                # Domain size: from -L to L
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


# Initial conditions
N_max = 1.0
R0 = 0.1             # Initial spheroid radius
H_init = 1.0         # Initial health level inside spheroid

# ====================== INITIALIZATION ======================
N = N_max * np.ones((N_grid, N_grid))
H = H_init * np.exp(-(X**2 + Y**2) / (2 * (R0 / 2)**2))

os.makedirs("output", exist_ok=True)


fit_uncertainties = [
    "StdRadius",
    "StdGrayMean",
    "StdIntDen",
]

targets = [
    "Radius",
    "GrayMean",
    "IntDen",
    "Feret",
    "Perimeter",
]

target_weights = {
    "Radius": 1.0,
    "GrayMean": 1.0,
    "IntDen": 1.0,
    "Feret": 0.25,
    "Perimeter": 0.25,
}

target_columns = {
    "Radius": ("MeanRadius", "StdRadius"),
    "GrayMean": ("MeanGrayMean", "StdGrayMean"),
    "IntDen": ("MeanIntDen", "StdIntDen"),
}

empirical_data = load_empirical_data("500.txt")

bounds = [
    (2.0, 10.0),     # lambda_
    (1e-5, 0.04),    # delta
    (0.3, 1.0),      # k_n
    (0.2, 0.7),      # k_p
    (0.05, 0.4),     # k_d
    (4.0, 8.0),      # alpha
    (0.3, 5.0),      # beta
    (0.28, 0.45),    # R0
    (0.03, 0.35),    # time_scale
    (430, 650),      # pixel_scale
    (200, 255),      # intensity_scale
]

with open("seed_runs/best_seed_summary.json", "r", encoding="utf-8") as f:
    best = json.load(f)

best_params = np.array([
    best["lambda_"],
    best["delta"],
    best["k_n"],
    best["k_p"],
    best["k_d"],
    best["alpha"],
    best["beta"],
    best["R0"],
    best["time_scale"],
    best["pixel_scale"],
    best["intensity_scale"],
])


seeds = [1, 2, 3, 4, 5]

seed_summary = run_seed_series(
    seeds=seeds,
    empirical_data=empirical_data,
    targets=targets,
    bounds=bounds,
    out_dir="seed_runs_with_saturation_penalty",
    maxiter=50,
    popsize=8,
    N_grid=81,
    L=3.0,
    dt=0.002,
    threshold=0.2,
)

'''
local_result = minimize(
    objective_L3,
    x0=best_params,
    args=(empirical_data, targets),
    method="Powell",
    bounds=bounds,
    options={
        "maxiter": 1000,
        "xtol": 1e-4,
        "ftol": 1e-4,
        "disp": True,
    },
)

if local_result.fun >= 1e29:
    print("⚠️ Local optimization failed.")
else:
    fit_table = save_fit_results(
    result=local_result,
    empirical_data=empirical_data,
    targets=targets,
    out_dir="local_fit_output",
    N_grid=81,
    L=3.0,
    dt=0.002,
    threshold=0.2,
    optimizer_name="Powell",
    )

    save_fit_plots(fit_table, out_dir="local_fit_output")

    print("Local error:", local_result.fun)
    print("Local parameters:", local_result.x)
    print(fit_table)




seeds = [1, 2, 3, 4, 5]

seed_summary = run_seed_series(
    seeds=seeds,
    empirical_data=empirical_data,
    targets=targets,
    bounds=bounds,
    out_dir="seed_runs",
    maxiter=50,
    popsize=8,
    N_grid=81,
    L=3.0,
    dt=0.002,
    threshold=0.2,
)

print(seed_summary)



result = differential_evolution(
    objective_L3,
    bounds=bounds,
    args=(empirical_data, targets),
    maxiter=50,
    popsize=8,
    polish=True,
    workers=1,
    updating="immediate",
    seed=None,
)

if result.fun >= 1e29:
    print("⚠️ No valid fit found. Best objective is still penalty value.")
    print("Try larger R0 lower bound, lower threshold, or larger N_grid.")
else:
    fit_table = save_fit_results(
        result=result,
        empirical_data=empirical_data,
        targets=targets,
        out_dir="fit_output",
        N_grid=81,
        L=3.0,
        dt=0.002,
        threshold=0.2,
    )
    save_fit_plots(fit_table, out_dir="fit_output")

    print("Best error:", result.fun)
    print("Best parameters:", result.x) 
              
    print(fit_table)
'''