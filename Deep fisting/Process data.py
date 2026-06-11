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

def load_fitted_params(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)

    param_names = get_param_names()

    # Case 1: your local_result_summary.json format
    if "parameters" in d:
        p = d["parameters"]

    # Case 2: your fit_summary.json format from save_fit_results()
    elif "fitted_parameters" in d:
        p = d["fitted_parameters"]

    else:
        raise KeyError(
            "Could not find 'parameters' or 'fitted_parameters' in the JSON file."
        )

    # Convert dictionary to ordered numeric list
    if isinstance(p, dict):
        missing = [name for name in param_names if name not in p]
        if missing:
            raise KeyError(f"Missing parameters in JSON: {missing}")

        return [float(p[name]) for name in param_names]

    # If it is already a list/array, just convert to floats
    return [float(x) for x in p]

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

def compare_two_parameter_sets(
    params_1000,
    params_2000,
    data_1000,
    data_2000,
    targets,
    N_grid=81,
    L=3.0,
    dt=0.002,
    threshold=0.2,
):
    errors = {
        "1000_params_on_1000_data": objective(
            params_1000, data_1000, targets,
            N_grid=N_grid, L=L, dt=dt, threshold=threshold
        ),
        "1000_params_on_2000_data": objective(
            params_1000, data_2000, targets,
            N_grid=N_grid, L=L, dt=dt, threshold=threshold
        ),
        "2000_params_on_1000_data": objective(
            params_2000, data_1000, targets,
            N_grid=N_grid, L=L, dt=dt, threshold=threshold
        ),
        "2000_params_on_2000_data": objective(
            params_2000, data_2000, targets,
            N_grid=N_grid, L=L, dt=dt, threshold=threshold
        ),
    }

    return pd.DataFrame(
        [{"test": key, "error": value} for key, value in errors.items()]
    )

def unpack_joint_params(joint_params, dataset_names):
    """
    Joint parameter structure:

    shared biological / global:
        lambda_, delta, k_n, k_p, k_d, alpha, beta, mu,
        time_scale, pixel_scale

    dataset-specific:
        R0_i, intensity_scale_i for each dataset
    """

    joint_params = np.asarray(joint_params, dtype=float)

    lambda_ = joint_params[0]
    delta = joint_params[1]
    k_n = joint_params[2]
    k_p = joint_params[3]
    k_d = joint_params[4]
    alpha = joint_params[5]
    beta = joint_params[6]
    mu = joint_params[7]
    time_scale = joint_params[8]
    pixel_scale = joint_params[9]

    dataset_params = {}

    idx = 10
    for name in dataset_names:
        R0 = joint_params[idx]
        intensity_scale = joint_params[idx + 1]
        idx += 2

        dataset_params[name] = [
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
        ]

    return dataset_params

def joint_objective_shared_biology(
    joint_params,
    datasets,
    targets,
    N_grid=81,
    L=3.0,
    dt=0.002,
    threshold=0.2,
):
    """
    datasets should be a dictionary:

    datasets = {
        "1000": data_1000,
        "2000": data_2000,
        "3375": data_3375,
    }
    """

    dataset_names = list(datasets.keys())
    params_by_dataset = unpack_joint_params(joint_params, dataset_names)

    total_error = 0.0

    for name, empirical_data in datasets.items():
        params = params_by_dataset[name]

        err = objective(
            params,
            empirical_data,
            targets,
            N_grid=N_grid,
            L=L,
            dt=dt,
            threshold=threshold,
        )

        total_error += err

    return total_error

def split_joint_errors(
    joint_params,
    datasets,
    targets,
    N_grid=81,
    L=3.0,
    dt=0.002,
    threshold=0.2,
):
    dataset_names = list(datasets.keys())
    params_by_dataset = unpack_joint_params(joint_params, dataset_names)

    rows = []

    for name, empirical_data in datasets.items():
        params = params_by_dataset[name]

        err = objective(
            params,
            empirical_data,
            targets,
            N_grid=N_grid,
            L=L,
            dt=dt,
            threshold=threshold,
        )

        rows.append({
            "dataset": name,
            "error": err,
        })

    table = pd.DataFrame(rows)
    table.loc[len(table)] = {
        "dataset": "TOTAL",
        "error": table["error"].sum(),
    }

    return table

def build_joint_initial_guess(param_dict):
    """
    param_dict example:

    param_dict = {
        "1000": params_1000,
        "2000": params_2000,
        "3375": params_3375,
    }

    Each params_* is the ordinary 12-parameter vector.
    """

    dataset_names = list(param_dict.keys())
    P = np.array([param_dict[name] for name in dataset_names], dtype=float)

    # Ordinary parameter order:
    # 0 lambda_
    # 1 delta
    # 2 k_n
    # 3 k_p
    # 4 k_d
    # 5 alpha
    # 6 beta
    # 7 mu
    # 8 R0
    # 9 time_scale
    # 10 pixel_scale
    # 11 intensity_scale

    shared = np.mean(P, axis=0)

    x0 = [
        shared[0],   # lambda_
        shared[1],   # delta
        shared[2],   # k_n
        shared[3],   # k_p
        shared[4],   # k_d
        shared[5],   # alpha
        shared[6],   # beta
        shared[7],   # mu
        shared[9],   # time_scale
        shared[10],  # pixel_scale
    ]

    for name in dataset_names:
        p = np.asarray(param_dict[name], dtype=float)
        x0.append(p[8])    # R0 for this dataset
        x0.append(p[11])   # intensity_scale for this dataset

    return np.array(x0, dtype=float)

def independent_error_sum(param_dict, datasets, targets, N_grid=81, L=3.0, dt=0.002, threshold=0.2):
    total = 0.0
    rows = []

    for name in datasets:
        err = objective(
            param_dict[name],
            datasets[name],
            targets,
            N_grid=N_grid,
            L=L,
            dt=dt,
            threshold=threshold,
        )

        rows.append({
            "dataset": name,
            "independent_error": err,
        })

        total += err

    table = pd.DataFrame(rows)
    table.loc[len(table)] = {
        "dataset": "TOTAL",
        "independent_error": total,
    }

    return table

def unpack_joint_params_dataset_time(joint_params, dataset_names):
    """
    Joint parameter structure with dataset-specific time_scale.

    Shared:
        lambda_, delta, k_n, k_p, k_d, alpha, beta, mu, pixel_scale

    Dataset-specific:
        R0_i, time_scale_i, intensity_scale_i
    """

    joint_params = np.asarray(joint_params, dtype=float)

    lambda_ = joint_params[0]
    delta = joint_params[1]
    k_n = joint_params[2]
    k_p = joint_params[3]
    k_d = joint_params[4]
    alpha = joint_params[5]
    beta = joint_params[6]
    mu = joint_params[7]
    pixel_scale = joint_params[8]

    dataset_params = {}

    idx = 9
    for name in dataset_names:
        R0 = joint_params[idx]
        time_scale = joint_params[idx + 1]
        intensity_scale = joint_params[idx + 2]
        idx += 3

        dataset_params[name] = [
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
        ]

    return dataset_params

def joint_objective_dataset_time(
    joint_params,
    datasets,
    targets,
    N_grid=81,
    L=3.0,
    dt=0.002,
    threshold=0.2,
):
    dataset_names = list(datasets.keys())
    params_by_dataset = unpack_joint_params_dataset_time(
        joint_params,
        dataset_names,
    )

    total_error = 0.0

    for name, empirical_data in datasets.items():
        params = params_by_dataset[name]

        err = objective(
            params,
            empirical_data,
            targets,
            N_grid=N_grid,
            L=L,
            dt=dt,
            threshold=threshold,
        )

        total_error += err

    return total_error

def split_joint_errors_dataset_time(
    joint_params,
    datasets,
    targets,
    N_grid=81,
    L=3.0,
    dt=0.002,
    threshold=0.2,
):
    dataset_names = list(datasets.keys())
    params_by_dataset = unpack_joint_params_dataset_time(
        joint_params,
        dataset_names,
    )

    rows = []

    for name, empirical_data in datasets.items():
        params = params_by_dataset[name]

        err = objective(
            params,
            empirical_data,
            targets,
            N_grid=N_grid,
            L=L,
            dt=dt,
            threshold=threshold,
        )

        rows.append({
            "dataset": name,
            "error": err,
            "R0": params[8],
            "time_scale": params[9],
            "intensity_scale": params[11],
        })

    table = pd.DataFrame(rows)
    table.loc[len(table)] = {
        "dataset": "TOTAL",
        "error": table["error"].sum(),
        "R0": np.nan,
        "time_scale": np.nan,
        "intensity_scale": np.nan,
    }

    return table

def build_joint_initial_guess_dataset_time(param_dict):
    """
    Builds initial guess for model with dataset-specific time_scale.

    Ordinary single-dataset parameter order:
        0 lambda_
        1 delta
        2 k_n
        3 k_p
        4 k_d
        5 alpha
        6 beta
        7 mu
        8 R0
        9 time_scale
        10 pixel_scale
        11 intensity_scale

    Joint order:
        shared lambda_, delta, k_n, k_p, k_d, alpha, beta, mu, pixel_scale,
        then for each dataset:
            R0_i, time_scale_i, intensity_scale_i
    """

    dataset_names = list(param_dict.keys())
    P = np.array([param_dict[name] for name in dataset_names], dtype=float)

    shared = np.mean(P, axis=0)

    x0 = [
        shared[0],   # lambda_
        shared[1],   # delta
        shared[2],   # k_n
        shared[3],   # k_p
        shared[4],   # k_d
        shared[5],   # alpha
        shared[6],   # beta
        shared[7],   # mu
        shared[10],  # pixel_scale
    ]

    for name in dataset_names:
        p = np.asarray(param_dict[name], dtype=float)
        x0.append(p[8])    # R0_i
        x0.append(p[9])    # time_scale_i
        x0.append(p[11])   # intensity_scale_i

    return np.array(x0, dtype=float)



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

data_1000 = load_empirical_data("1000.txt")
data_2000 = load_empirical_data("2000.txt")
data_3375 = load_empirical_data("3375.txt")

params_1000 = load_fitted_params("local_fit_mu_1000_final/local_result_summary.json")
params_2000 = load_fitted_params("local_fit_mu_2000_final/local_result_summary.json")
params_3375 = load_fitted_params("local_fit_mu_3375_final/local_result_summary.json")

#err = compare_two_parameter_sets(params_1000,params_2000,data_1000,data_2000,targets)

#print(err)

p1000 = np.array(params_1000, dtype=float)
p2000 = np.array(params_2000, dtype=float)

# parameter order:
# lambda_, delta, k_n, k_p, k_d, alpha, beta, mu,
# R0, time_scale, pixel_scale, intensity_scale

# ============================================================
# Joint 1000 + 2000 + 8000
# ============================================================

data_8000 = load_empirical_data("8000.txt")
params_8000 = load_fitted_params("local_fit_mu_8000_final/local_result_summary.json")

datasets_1000_2000_8000 = {
    "1000": data_1000,
    "2000": data_2000,
    "8000": data_8000,
}

param_dict_1000_2000_8000 = {
    "1000": params_1000,
    "2000": params_2000,
    "8000": params_8000,
}

x0_joint_1000_2000_8000 = build_joint_initial_guess(
    param_dict_1000_2000_8000
)

result_joint_1000_2000_8000 = minimize(
    joint_objective_shared_biology,
    x0=x0_joint_1000_2000_8000,
    args=(datasets_1000_2000_8000, targets),
    method="Nelder-Mead",
    options={
        "maxiter": 2500,
        "maxfev": 3000,
        "xatol": 1e-5,
        "fatol": 1e-4,
        "adaptive": True,
        "disp": True,
    },
)

print("\n=== Joint 1000 + 2000 + 8000 ===")
print("Success:", result_joint_1000_2000_8000.success)
print("Message:", result_joint_1000_2000_8000.message)
print("Iterations:", result_joint_1000_2000_8000.nit)
print("Function evaluations:", result_joint_1000_2000_8000.nfev)
print("Joint error:", result_joint_1000_2000_8000.fun)
print("Joint parameters:")
print(result_joint_1000_2000_8000.x)

independent_1000_2000_8000 = independent_error_sum(
    param_dict_1000_2000_8000,
    datasets_1000_2000_8000,
    targets,
)

shared_1000_2000_8000 = split_joint_errors(
    result_joint_1000_2000_8000.x,
    datasets_1000_2000_8000,
    targets,
)

print("\nIndependent baseline:")
print(independent_1000_2000_8000)

print("\nShared biological fit:")
print(shared_1000_2000_8000)

E_independent_1000_2000_8000 = independent_1000_2000_8000[
    "independent_error"
].iloc[-1]

E_joint_1000_2000_8000 = result_joint_1000_2000_8000.fun

print(
    "\nJoint / independent ratio:",
    E_joint_1000_2000_8000 / E_independent_1000_2000_8000
)

# ============================================================
# Joint 1000 + 2000 + 15625
# ============================================================

data_15625 = load_empirical_data("15625.txt")
params_15625 = load_fitted_params("local_fit_mu_15625_final/local_result_summary.json")

datasets_1000_2000_15625 = {
    "1000": data_1000,
    "2000": data_2000,
    "15625": data_15625,
}

param_dict_1000_2000_15625 = {
    "1000": params_1000,
    "2000": params_2000,
    "15625": params_15625,
}

x0_joint_1000_2000_15625 = build_joint_initial_guess(
    param_dict_1000_2000_15625
)

result_joint_1000_2000_15625 = minimize(
    joint_objective_shared_biology,
    x0=x0_joint_1000_2000_15625,
    args=(datasets_1000_2000_15625, targets),
    method="Nelder-Mead",
    options={
        "maxiter": 2500,
        "maxfev": 3000,
        "xatol": 1e-5,
        "fatol": 1e-4,
        "adaptive": True,
        "disp": True,
    },
)

print("\n=== Joint 1000 + 2000 + 15625 ===")
print("Success:", result_joint_1000_2000_15625.success)
print("Message:", result_joint_1000_2000_15625.message)
print("Iterations:", result_joint_1000_2000_15625.nit)
print("Function evaluations:", result_joint_1000_2000_15625.nfev)
print("Joint error:", result_joint_1000_2000_15625.fun)
print("Joint parameters:")
print(result_joint_1000_2000_15625.x)

independent_1000_2000_15625 = independent_error_sum(
    param_dict_1000_2000_15625,
    datasets_1000_2000_15625,
    targets,
)

shared_1000_2000_15625 = split_joint_errors(
    result_joint_1000_2000_15625.x,
    datasets_1000_2000_15625,
    targets,
)

print("\nIndependent baseline:")
print(independent_1000_2000_15625)

print("\nShared biological fit:")
print(shared_1000_2000_15625)

E_independent_1000_2000_15625 = independent_1000_2000_15625[
    "independent_error"
].iloc[-1]

E_joint_1000_2000_15625 = result_joint_1000_2000_15625.fun

print(
    "\nJoint / independent ratio:",
    E_joint_1000_2000_15625 / E_independent_1000_2000_15625
)

# ============================================================
# Joint 1000 + 2000 + 3375 + 8000 + 15625
# ============================================================

datasets_all_large_test = {
    "1000": data_1000,
    "2000": data_2000,
    "3375": data_3375,
    "8000": data_8000,
    "15625": data_15625,
}

param_dict_all_large_test = {
    "1000": params_1000,
    "2000": params_2000,
    "3375": params_3375,
    "8000": params_8000,
    "15625": params_15625,
}

x0_joint_all_large_test = build_joint_initial_guess(
    param_dict_all_large_test
)

result_joint_all_large_test = minimize(
    joint_objective_shared_biology,
    x0=x0_joint_all_large_test,
    args=(datasets_all_large_test, targets),
    method="Nelder-Mead",
    options={
        "maxiter": 3500,
        "maxfev": 5000,
        "xatol": 1e-5,
        "fatol": 1e-4,
        "adaptive": True,
        "disp": True,
    },
)

print("\n=== Joint 1000 + 2000 + 3375 + 8000 + 15625 ===")
print("Success:", result_joint_all_large_test.success)
print("Message:", result_joint_all_large_test.message)
print("Iterations:", result_joint_all_large_test.nit)
print("Function evaluations:", result_joint_all_large_test.nfev)
print("Joint error:", result_joint_all_large_test.fun)
print("Joint parameters:")
print(result_joint_all_large_test.x)

independent_all_large_test = independent_error_sum(
    param_dict_all_large_test,
    datasets_all_large_test,
    targets,
)

shared_all_large_test = split_joint_errors(
    result_joint_all_large_test.x,
    datasets_all_large_test,
    targets,
)

print("\nIndependent baseline:")
print(independent_all_large_test)

print("\nShared biological fit:")
print(shared_all_large_test)

E_independent_all_large_test = independent_all_large_test[
    "independent_error"
].iloc[-1]

E_joint_all_large_test = result_joint_all_large_test.fun

print(
    "\nJoint / independent ratio:",
    E_joint_all_large_test / E_independent_all_large_test
)