import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import base_functions as bf

from scipy.optimize import differential_evolution, minimize


# ============================================================
# Data loading
# ============================================================

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


# ============================================================
# PDE model
# ============================================================

def laplacian(Z, dx):
    Z_top = Z[:-2, 1:-1]
    Z_bottom = Z[2:, 1:-1]
    Z_left = Z[1:-1, :-2]
    Z_right = Z[1:-1, 2:]
    Z_center = Z[1:-1, 1:-1]

    return (Z_top + Z_bottom + Z_left + Z_right - 4 * Z_center) / dx**2


def step(n, h, lambda_, delta, k_n, k_p, k_d, alpha, beta, mu, dx, dt):
    """
    Saturation model with baseline turnover:

    n_t = laplacian(n) - lambda_ * n/(k_n + n) * h

    h_t = delta * laplacian(h)
          + alpha * n/(k_p + n) * h * (1 - h)
          - beta * k_d/(k_d + n) * h
          - mu * h

    mu is baseline turnover / inverse effective life expectancy.
    """

    lap_n = laplacian(n, dx)
    lap_h = laplacian(h, dx)

    n_new = n.copy()
    h_new = h.copy()

    eps = 1e-12

    n_i = n[1:-1, 1:-1]
    h_i = h[1:-1, 1:-1]

    consumption = lambda_ * n_i / (k_n + n_i + eps) * h_i

    proliferation = (
        alpha
        * n_i / (k_p + n_i + eps)
        * h_i
        * (1 - h_i)
    )

    nutrient_death = (
        beta
        * k_d / (k_d + n_i + eps)
        * h_i
    )

    baseline_turnover = mu * h_i

    n_new[1:-1, 1:-1] += dt * (lap_n - consumption)

    h_new[1:-1, 1:-1] += dt * (
        delta * lap_h
        + proliferation
        - nutrient_death
        - baseline_turnover
    )

    # Dirichlet boundary condition for nutrient
    n_new[0, :] = 1.0
    n_new[-1, :] = 1.0
    n_new[:, 0] = 1.0
    n_new[:, -1] = 1.0

    # Neumann zero-flux boundary condition for h
    h_new[0, :] = h_new[1, :]
    h_new[-1, :] = h_new[-2, :]
    h_new[:, 0] = h_new[:, 1]
    h_new[:, -1] = h_new[:, -2]

    n_new = np.clip(n_new, 0.0, 1.0)
    h_new = np.clip(h_new, 0.0, 1.0)

    return n_new, h_new


def simulate_model(params, empirical_days, N_grid=81, L=3.0, dt=0.002):
    (
        lambda_,
        delta,
        k_n,
        k_p,
        k_d,
        alpha,
        beta,
        mu,
        R0,
        time_scale,
    ) = params

    x = np.linspace(-L, L, N_grid)
    y = np.linspace(-L, L, N_grid)
    dx = x[1] - x[0]

    X, Y = np.meshgrid(x, y)

    n = np.ones((N_grid, N_grid))
    h = np.exp(-(X**2 + Y**2) / (2 * (R0 / 2)**2))

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
                mu,
                dx,
                dt,
            )

    return results


# ============================================================
# ImageJ-like observables
# ============================================================

def calculate_model_observables(H, dx, pixel_scale=1.0, intensity_scale=255.0, threshold=0.2):
    """
    Calculates ImageJ-like observables from the synthetic H field.
    H is interpreted as quasi-density / quasi-probability in [0, 1].
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
            "TotalH": np.nan,
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

    # ImageJ-like integrated density: area in pixels * mean gray value
    int_den = area_pixels * gray_mean

    # Diagnostic: continuous total biomass-like quantity.
    # This is not fitted by default.
    total_h = np.sum(H) * dx**2 * pixel_scale**2

    return {
        "Radius": radius_pixels,
        "Area": area_pixels,
        "GrayMean": gray_mean,
        "IntDen": int_den,
        "Circularity": circularity,
        "Feret": feret_pixels,
        "Perimeter": perimeter_pixels,
        "TotalH": total_h,
        "TouchesBoundary": touches_boundary,
    }


# ============================================================
# Fitting objective
# ============================================================

def objective(params, empirical_data, targets, N_grid=81, L=3.0, dt=0.002, threshold=0.2):
    (
        lambda_,
        delta,
        k_n,
        k_p,
        k_d,
        alpha,
        beta,
        mu,
        R0,
        time_scale,
        pixel_scale,
        intensity_scale,
    ) = params

    if (
        lambda_ <= 0 or delta <= 0 or
        k_n <= 0 or k_p <= 0 or k_d <= 0 or
        alpha < 0 or beta < 0 or mu < 0 or
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
                mu,
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

    # No forced saturation penalty here.
    return np.sum(residuals**2)


def objective_configured(params, empirical_data, targets):
    return objective(
        params,
        empirical_data,
        targets,
        N_grid=N_GRID,
        L=L,
        dt=DT,
        threshold=H_THRESHOLD,
    )


# ============================================================
# Evaluation and saving
# ============================================================

def evaluate_fit(params, empirical_data, targets, N_grid=81, L=3.0, dt=0.002, threshold=0.2):
    (
        lambda_,
        delta,
        k_n,
        k_p,
        k_d,
        alpha,
        beta,
        mu,
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
            mu,
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

        out = {
            "Day": day,
            "ModelTotalH": obs["TotalH"],
            "TouchesBoundary": obs["TouchesBoundary"],
        }

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

    if "ModelTotalH" in fit_table.columns:
        plt.figure(figsize=(6, 4))
        plt.plot(fit_table["Day"], fit_table["ModelTotalH"], "-o", label="Model TotalH")
        plt.xlabel("Day")
        plt.ylabel("Continuous total H")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "TotalH_diagnostic.png"), dpi=300)
        plt.close()


def get_param_names():
    return [
        "lambda_",
        "delta",
        "k_n",
        "k_p",
        "k_d",
        "alpha",
        "beta",
        "mu",
        "R0",
        "time_scale",
        "pixel_scale",
        "intensity_scale",
    ]


def save_fit_results(
    result,
    empirical_data,
    targets,
    out_dir="fit_output",
    N_grid=81,
    L=3.0,
    dt=0.002,
    threshold=0.2,
    optimizer_name="unknown",
):
    os.makedirs(out_dir, exist_ok=True)

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
        index=False,
    )

    param_names = get_param_names()

    fitted_parameters = {
        name: float(value)
        for name, value in zip(param_names, result.x)
    }

    summary = {
        "equations": [
            "n_t = laplacian(n) - lambda_ * n / (k_n + n) * h",
            "h_t = delta * laplacian(h) + alpha * n / (k_p + n) * h * (1 - h) - beta * k_d / (k_d + n) * h - mu * h",
        ],
        "boundary_conditions": {
            "n": "Dirichlet n = 1",
            "h": "Neumann zero-flux",
        },
        "targets": targets,
        "target_weights": target_weights,
        "fitted_parameters": fitted_parameters,
        "optimizer": {
            "method": optimizer_name,
            "success": bool(getattr(result, "success", False)),
            "message": str(getattr(result, "message", "")),
            "error": float(getattr(result, "fun", np.nan)),
            "iterations": int(getattr(result, "nit", -1)),
            "function_evaluations": int(getattr(result, "nfev", -1)),
        },
        "simulation": {
            "N_grid": int(N_grid),
            "L": float(L),
            "dt": float(dt),
            "threshold": float(threshold),
        },
        "model_extension": {
            "baseline_turnover_enabled": True,
            "mu_interpretation": "mu is inverse effective cell life expectancy in model time units",
            "late_saturation_penalty_enabled": False,
        },
        "output_files": {
            "fit_table": "fit_table.csv",
            "fit_summary": "fit_summary.json",
        },
    }

    with open(os.path.join(out_dir, "fit_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return fit_table


# ============================================================
# Multi-seed fitting
# ============================================================

def run_seed_series(
    seeds,
    empirical_data,
    targets,
    bounds,
    out_dir="seed_runs_mu",
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
                optimizer_name="differential_evolution",
            )

            save_fit_plots(fit_table, out_dir=seed_dir)

        row = {
            "seed": seed,
            "error": float(result.fun),
            "success": bool(result.success),
            "message": str(result.message),
            "iterations": int(result.nit),
            "function_evaluations": int(result.nfev),
            "valid_fit": bool(result.fun < 1e29),
        }

        for name, value in zip(get_param_names(), result.x):
            row[name] = float(value)

        all_rows.append(row)

        print(f"Seed {seed}: error = {result.fun}")

    summary_table = pd.DataFrame(all_rows)
    summary_table = summary_table.sort_values("error").reset_index(drop=True)

    summary_path = os.path.join(out_dir, "seed_summary.csv")
    summary_table.to_csv(summary_path, index=False)

    best = summary_table.iloc[0].to_dict()

    with open(os.path.join(out_dir, "best_seed_summary.json"), "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    print("\nBest seed:")
    print(summary_table.iloc[0])

    return summary_table


def load_best_params(path):
    with open(path, "r", encoding="utf-8") as f:
        best = json.load(f)

    params = []
    for name in get_param_names():
        if name == "mu":
            params.append(float(best.get("mu", 0.1)))
        else:
            params.append(float(best[name]))

    return np.array(params, dtype=float)



def run_powell_from_best(
    best_json_path,
    empirical_data,
    targets,
    bounds,
    out_dir="local_fit_mu",
    N_grid=81,
    L=3.0,
    dt=0.002,
    threshold=0.2,
):
    os.makedirs(out_dir, exist_ok=True)

    best_params = load_best_params(best_json_path)

    def local_objective(params):
        return objective(
            params,
            empirical_data,
            targets,
            N_grid=N_grid,
            L=L,
            dt=dt,
            threshold=threshold,
        )

    local_result = minimize(
        local_objective,
        x0=best_params,
        method="Powell",
        bounds=bounds,
        options={
            "maxiter": 1000,
            "xtol": 1e-4,
            "ftol": 1e-4,
            "disp": True,
        },
    )

    param_names = get_param_names()

    local_summary = {
        "optimizer": "Powell",
        "success": bool(local_result.success),
        "message": str(local_result.message),
        "error": float(local_result.fun),
        "iterations": int(getattr(local_result, "nit", -1)),
        "function_evaluations": int(getattr(local_result, "nfev", -1)),
        "valid_fit": bool(local_result.fun < 1e29),
        "start_from": best_json_path,
        "parameters": {
            name: float(value)
            for name, value in zip(param_names, local_result.x)
        },
        "settings": {
            "N_grid": int(N_grid),
            "L": float(L),
            "dt": float(dt),
            "threshold": float(threshold),
            "targets": targets,
        },
    }

    local_summary_path = os.path.join(out_dir, "powell_result_summary.json")

    with open(local_summary_path, "w", encoding="utf-8") as f:
        json.dump(local_summary, f, indent=2)

    if local_result.fun >= 1e29:
        print("⚠️ Local optimization failed.")
        print(f"Powell summary saved to: {local_summary_path}")
        return local_result, None

    fit_table = save_fit_results(
        result=local_result,
        empirical_data=empirical_data,
        targets=targets,
        out_dir=out_dir,
        N_grid=N_grid,
        L=L,
        dt=dt,
        threshold=threshold,
        optimizer_name="Powell",
    )

    save_fit_plots(fit_table, out_dir=out_dir)

    print("Local error:", local_result.fun)
    print("Local parameters:", local_result.x)
    print(f"Powell summary saved to: {local_summary_path}")
    print(fit_table)

    return local_result, fit_table


# ============================================================
# Configuration
# ============================================================

L = 3.0
N_GRID = 81
DT = 0.002
H_THRESHOLD = 0.2

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

bounds = [
    (2.0, 10.0),     # lambda_
    (1e-5, 0.04),    # delta
    (0.3, 1.0),      # k_n
    (0.2, 0.7),      # k_p
    (0.05, 0.4),     # k_d
    (4.0, 8.0),      # alpha
    (0.3, 5.0),      # beta
    (0.0, 3.0),      # mu, baseline turnover / inverse effective life expectancy
    (0.28, 0.45),    # R0
    (0.03, 0.35),    # time_scale
    (430, 650),      # pixel_scale
    (200, 255),      # intensity_scale
]


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    empirical_data = load_empirical_data("2000.txt")
    '''


'''
    seeds = [1,2,3,4,5]

    seed_summary = run_seed_series(
        seeds=seeds,
        empirical_data=empirical_data,
        targets=targets,
        bounds=bounds,
        out_dir="seed_runs_mu_2000",
        maxiter=150,
        popsize=20,
        N_grid=N_GRID,
        L=L,
        dt=DT,
        threshold=H_THRESHOLD,
    )

    print(seed_summary)

    # Optional local polishing from the best seed:
    local_result, local_fit_table = run_powell_from_best(
         best_json_path="seed_runs_mu_2000/best_seed_summary.json",
         empirical_data=empirical_data,
         targets=targets,
         bounds=bounds,
         out_dir="local_fit_mu_2000",
         N_grid=N_GRID,
         L=L,
         dt=DT,
         threshold=H_THRESHOLD,)
