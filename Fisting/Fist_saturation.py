# fit_spheroid_pde_saturation.py
# Joint fitting of a 2D nutrient–health PDE spheroid model to multiple datasets (radius vs day).
# Run directly in Spyder. Requires: numpy, scipy, pandas, matplotlib, tqdm (optional), numba (optional).
#
# Nondimensional model:
#   n_t = ∇²n - lambda * n/(k_n + n) * h
#   h_t = delta ∇²h + alpha * n/(k_p + n) * h * (1 - h) - beta * k_d/(k_d + n) * h
#
# BC:
#   n = 1 at boundary
#   ∂h/∂normal = 0 at boundary
#
# Radius extraction:
#   R_sim(t) from area where h >= c, then R_phys = s * sqrt(A/pi)

import os
import json
import glob
import math
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import least_squares, differential_evolution

try:
    from tqdm import tqdm
    TQDM = True
except Exception:
    TQDM = False

try:
    from numba import njit
    NUMBA = True
except Exception:
    NUMBA = False


# =========================
# ===== User config =======
# =========================

OUT_DIR = "fit_output_saturation"
DATA_GLOB = "*.txt"

SAVE_FIGS = True
SAVE_JSON = True

# Grid/domain in model units.
L = 50.0
N_GRID = 128
CFL_SAFETY = 0.24

# Radius extraction
H_THRESH = 0.8
USE_CONTINUOUS_PROXY = False
KAPPA = 1.5
SMOOTH_DISK_EDGE = 2.0

# Robust fitting
SIGMA0 = 10.0
ROBUST_LOSS = "soft_l1"
ROBUST_FSCALE = 20.0

# Parameter bounds in real space. We optimize in log-space.
BOUNDS = {
    "lambda_": (1e-4, 50.0),   # nutrient consumption strength
    "delta":   (1e-5, 1.0),    # D_H / D_N
    "k_n":     (1e-4, 10.0),   # nutrient half-saturation for consumption
    "k_p":     (1e-4, 10.0),   # nutrient half-saturation for proliferation
    "k_d":     (1e-4, 10.0),   # death saturation constant
    "alpha":   (1e-4, 50.0),   # proliferation strength
    "beta":    (1e-4, 50.0),   # death strength
    "s":       (0.1, 20.0),    # micrometers per model unit
}

USE_GLOBAL_SEARCH = True
DE_MAXITER = 40
DE_POPSIZE = 10
DE_TOL = 1e-3
DE_SEED = 42

LS_MAX_NFEV = 200

# =========================
# ======= Helpers =========
# =========================

@dataclass
class Dataset:
    name: str
    days: np.ndarray
    R_mean: np.ndarray
    R_std: np.ndarray
    N_repl: np.ndarray


def load_datasets(pattern: str) -> List[Dataset]:
    files = sorted(glob.glob(pattern))
    datasets: List[Dataset] = []

    if not files:
        raise FileNotFoundError(f"No data files found by pattern: {pattern}")

    for path in files:
        df = pd.read_csv(
            path,
            comment="#",
            sep=r"\s+",
            header=None,
            names=["Day", "MeanRadius", "StdRadius", "N"],
            engine="python",
        )

        for col in ["Day", "MeanRadius", "StdRadius", "N"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["Day", "MeanRadius", "StdRadius", "N"])
        df = df.sort_values("Day").reset_index(drop=True)

        if df.empty:
            raise ValueError(f"File {path} contained no valid data rows.")

        ds = Dataset(
            name=os.path.splitext(os.path.basename(path))[0],
            days=df["Day"].to_numpy(dtype=float),
            R_mean=df["MeanRadius"].to_numpy(dtype=float),
            R_std=df["StdRadius"].to_numpy(dtype=float),
            N_repl=df["N"].to_numpy(dtype=float),
        )
        datasets.append(ds)

    return datasets


def initial_R0_guesses(datasets: List[Dataset], s0: float) -> List[float]:
    R0s = []

    for ds in datasets:
        if len(ds.R_mean) == 0:
            R0s.append(5.0 / s0)
        else:
            R0s.append(max(1e-3, ds.R_mean[0] / s0))

    return R0s


def build_weights(ds: Dataset, sigma0: float) -> np.ndarray:
    denom = np.sqrt(np.maximum(1e-12, ds.R_std**2 + sigma0**2))
    return np.sqrt(np.maximum(1.0, ds.N_repl)) / denom


# =========================
# ====== PDE kernel =======
# =========================

def _laplacian_numpy(Z: np.ndarray, dx: float) -> np.ndarray:
    return (
        Z[:-2, 1:-1] +
        Z[2:, 1:-1] +
        Z[1:-1, :-2] +
        Z[1:-1, 2:] -
        4.0 * Z[1:-1, 1:-1]
    ) / (dx * dx)


if NUMBA:
    @njit(cache=True, fastmath=True)
    def _laplacian_numba(Z: np.ndarray, dx: float) -> np.ndarray:
        return (
            Z[:-2, 1:-1] +
            Z[2:, 1:-1] +
            Z[1:-1, :-2] +
            Z[1:-1, 2:] -
            4.0 * Z[1:-1, 1:-1]
        ) / (dx * dx)

    LAPL = _laplacian_numba
else:
    LAPL = _laplacian_numpy


def make_initial_H_smooth_disk(X: np.ndarray, Y: np.ndarray, R0: float, edge_cells: float) -> np.ndarray:
    dx = X[0, 1] - X[0, 0]
    r = np.sqrt(X * X + Y * Y)
    w = max(1e-6, edge_cells * dx)
    H = 1.0 / (1.0 + np.exp((r - R0) / w))
    return H


def extract_radius(H: np.ndarray, dx: float, c: float, use_continuous: bool, kappa: float) -> float:
    if use_continuous:
        A = np.sum(H**kappa) * (dx * dx)
    else:
        A = float(np.count_nonzero(H >= c)) * (dx * dx)

    return np.sqrt(max(1e-12, A) / math.pi)


def simulate_radius_times(
    days: np.ndarray,
    lambda_: float,
    delta: float,
    k_n: float,
    k_p: float,
    k_d: float,
    alpha: float,
    beta: float,
    R0: float,
    s: float,
    L: float,
    N_grid: int,
    cfl_safety: float,
    H_thresh: float,
    use_continuous: bool,
    kappa: float,
    progress: bool = False
) -> np.ndarray:
    """
    Run one nondimensional PDE simulation up to max(days),
    returning predicted physical radii in micrometers.
    """

    x = np.linspace(-L, L, N_grid)
    y = np.linspace(-L, L, N_grid)
    X, Y = np.meshgrid(x, y)
    dx = x[1] - x[0]

    # For dimensionless model, the largest diffusion coefficient is max(1, delta).
    Dmax = max(1.0, delta)
    dt_max = cfl_safety * dx * dx / (4.0 * Dmax)

    # Cap dt to resolve reactions as well.
    reaction_scale = max(lambda_, alpha, beta, 1e-12)
    dt_react = 0.1 / reaction_scale

    dt = min(0.5, dt_max, dt_react)
    if dt <= 0:
        dt = 1e-4

    N = np.ones((N_grid, N_grid), dtype=np.float64)
    H = make_initial_H_smooth_disk(X, Y, R0, SMOOTH_DISK_EDGE)

    t_end = float(np.max(days))
    n_steps = int(math.ceil(t_end / dt))

    target_times = np.array(days, dtype=float)
    next_idx = 0
    R_phys = np.full_like(target_times, np.nan, dtype=float)

    iterator = range(n_steps)
    if progress and TQDM:
        iterator = tqdm(iterator, total=n_steps, desc="Simulating", leave=False)

    t = 0.0

    eps = 1e-12

    for _ in iterator:
        lapN = LAPL(N, dx)
        lapH = LAPL(H, dx)

        N_mid = N[1:-1, 1:-1]
        H_mid = H[1:-1, 1:-1]

        consumption = lambda_ * (N_mid / (k_n + N_mid + eps)) * H_mid
        growth = alpha * (N_mid / (k_p + N_mid + eps)) * H_mid * (1.0 - H_mid)
        death = beta * (k_d / (k_d + N_mid + eps)) * H_mid

        N_new = N.copy()
        H_new = H.copy()

        N_new[1:-1, 1:-1] += dt * (lapN - consumption)
        H_new[1:-1, 1:-1] += dt * (delta * lapH + growth - death)

        # Dirichlet for nutrients: nutrient bath at boundary.
        N_new[0, :] = 1.0
        N_new[-1, :] = 1.0
        N_new[:, 0] = 1.0
        N_new[:, -1] = 1.0

        # Neumann for health: copy edges.
        H_new[0, :] = H_new[1, :]
        H_new[-1, :] = H_new[-2, :]
        H_new[:, 0] = H_new[:, 1]
        H_new[:, -1] = H_new[:, -2]

        np.clip(N_new, 0.0, 1.0, out=N_new)
        np.clip(H_new, 0.0, 1.0, out=H_new)

        N, H = N_new, H_new
        t += dt

        while next_idx < len(target_times) and t >= target_times[next_idx] - 0.5 * dt:
            R_model = extract_radius(H, dx, H_thresh, use_continuous, kappa)
            R_phys[next_idx] = s * R_model
            next_idx += 1

        if t >= t_end - 1e-12 and next_idx < len(target_times):
            while next_idx < len(target_times):
                R_model = extract_radius(H, dx, H_thresh, use_continuous, kappa)
                R_phys[next_idx] = s * R_model
                next_idx += 1
            break

    return R_phys


# =========================
# === Optimization glue ===
# =========================

class ParamPack:
    """
    theta =
    [
        log lambda_, log delta, log k_n, log k_p, log k_d,
        log alpha, log beta, log s,
        log R0_1, ..., log R0_m
    ]
    """

    def __init__(
        self,
        datasets: List[Dataset],
        bounds: Dict[str, Tuple[float, float]],
        s0: float,
        R0_guesses: List[float]
    ):
        self.datasets = datasets
        self.m = len(datasets)

        self.idx = {
            "lambda_": 0,
            "delta": 1,
            "k_n": 2,
            "k_p": 3,
            "k_d": 4,
            "alpha": 5,
            "beta": 6,
            "s": 7,
        }

        self.shared_names = ["lambda_", "delta", "k_n", "k_p", "k_d", "alpha", "beta", "s"]

        self.base_len = len(self.shared_names)
        self.R0_start = self.base_len
        self.n_total = self.base_len + self.m

        self.lb = np.full(self.n_total, -np.inf, dtype=float)
        self.ub = np.full(self.n_total, np.inf, dtype=float)

        for name in self.shared_names:
            lo, hi = bounds[name]
            self.lb[self.idx[name]] = math.log(lo)
            self.ub[self.idx[name]] = math.log(hi)

        for j, R0g in enumerate(R0_guesses):
            lo = max(1e-3, 0.5 * R0g)
            hi = 2.0 * max(lo, R0g)
            self.lb[self.R0_start + j] = math.log(lo)
            self.ub[self.R0_start + j] = math.log(hi)

    def to_params(self, theta: np.ndarray) -> Dict:
        params = {}

        for name in self.shared_names:
            params[name] = math.exp(theta[self.idx[name]])

        params["R0s"] = np.exp(theta[self.R0_start:self.R0_start + self.m])

        return params

    def initial_theta(self, s0: float, R0_guesses: List[float]) -> np.ndarray:
        theta = np.zeros(self.n_total, dtype=float)

        for name in self.shared_names:
            if name == "s":
                theta[self.idx[name]] = math.log(s0)
            else:
                lo, hi = BOUNDS[name]
                theta[self.idx[name]] = 0.5 * (math.log(lo) + math.log(hi))

        for j, R0g in enumerate(R0_guesses):
            theta[self.R0_start + j] = math.log(max(1e-3, R0g))

        return theta


def residuals_vector(
    theta: np.ndarray,
    pack: ParamPack,
    datasets: List[Dataset],
    L: float,
    N_grid: int,
    cfl: float,
    H_thresh: float,
    use_cont: bool,
    kappa: float,
    verbose: bool = False
) -> np.ndarray:
    p = pack.to_params(theta)

    res_list = []

    it = enumerate(datasets)
    if verbose and TQDM:
        it = tqdm(list(it), desc="Datasets", leave=False)

    for j, ds in it:
        weights = build_weights(ds, SIGMA0)

        R_sim = simulate_radius_times(
            days=ds.days,
            lambda_=p["lambda_"],
            delta=p["delta"],
            k_n=p["k_n"],
            k_p=p["k_p"],
            k_d=p["k_d"],
            alpha=p["alpha"],
            beta=p["beta"],
            R0=p["R0s"][j],
            s=p["s"],
            L=L,
            N_grid=N_grid,
            cfl_safety=cfl,
            H_thresh=H_thresh,
            use_continuous=use_cont,
            kappa=kappa,
            progress=False
        )

        r = weights * (R_sim - ds.R_mean)
        res_list.append(r)

    return np.concatenate(res_list)


def objective_scalar_for_DE(
    theta: np.ndarray,
    pack: ParamPack,
    datasets: List[Dataset],
    L: float,
    N_grid: int,
    cfl: float,
    H_thresh: float,
    use_cont: bool,
    kappa: float
) -> float:
    r = residuals_vector(theta, pack, datasets, L, N_grid, cfl, H_thresh, use_cont, kappa)
    f = ROBUST_FSCALE

    val = np.sum(2.0 * (np.sqrt(1.0 + (r / f)**2) - 1.0))

    if not np.isfinite(val):
        return 1e12

    return float(val)


def fit_parameters(datasets: List[Dataset]) -> Tuple[Dict, np.ndarray]:
    os.makedirs(OUT_DIR, exist_ok=True)

    s0 = 1.0
    R0_guesses = initial_R0_guesses(datasets, s0=s0)

    pack = ParamPack(datasets, BOUNDS, s0, R0_guesses)
    theta0 = pack.initial_theta(s0, R0_guesses)

    theta_start = theta0.copy()

    if USE_GLOBAL_SEARCH:
        print("[Stage 1/2] Global search (DE) ...")

        bounds_DE = list(zip(pack.lb.tolist(), pack.ub.tolist()))

        result = differential_evolution(
            func=lambda th: objective_scalar_for_DE(
                th, pack, datasets, L, N_GRID, CFL_SAFETY,
                H_THRESH, USE_CONTINUOUS_PROXY, KAPPA
            ),
            bounds=bounds_DE,
            strategy="best1bin",
            maxiter=DE_MAXITER,
            popsize=DE_POPSIZE,
            tol=DE_TOL,
            seed=DE_SEED,
            polish=False,
            updating="deferred"
        )

        print(f"  DE status: {result.message}")
        theta_start = result.x

    print("[Stage 2/2] Local robust least-squares ...")

    ls = least_squares(
        fun=lambda th: residuals_vector(
            th, pack, datasets, L, N_GRID, CFL_SAFETY,
            H_THRESH, USE_CONTINUOUS_PROXY, KAPPA
        ),
        x0=theta_start,
        bounds=(pack.lb, pack.ub),
        loss=ROBUST_LOSS,
        f_scale=ROBUST_FSCALE,
        max_nfev=LS_MAX_NFEV,
        verbose=2
    )

    best = pack.to_params(ls.x)
    best["theta"] = ls.x.tolist()
    best["success"] = bool(ls.success)
    best["message"] = str(ls.message)
    best["cost"] = float(ls.cost)
    best["nfev"] = int(ls.nfev)

    return best, ls.x


def save_summary_and_plots(best: Dict, theta: np.ndarray, datasets: List[Dataset]):
    os.makedirs(OUT_DIR, exist_ok=True)

    if SAVE_JSON:
        summary = {
            "model": {
                "equations": [
                    "n_t = laplacian(n) - lambda_ * n/(k_n + n) * h",
                    "h_t = delta * laplacian(h) + alpha * n/(k_p + n) * h * (1-h) - beta * k_d/(k_d + n) * h"
                ],
                "boundary_conditions": {
                    "nutrient": "Dirichlet n = 1",
                    "health": "Neumann zero-flux"
                }
            },
            "shared_params": {
                "lambda_": best["lambda_"],
                "delta": best["delta"],
                "k_n": best["k_n"],
                "k_p": best["k_p"],
                "k_d": best["k_d"],
                "alpha": best["alpha"],
                "beta": best["beta"],
                "s": best["s"],
            },
            "per_dataset_R0": {
                ds.name: float(best["R0s"][i])
                for i, ds in enumerate(datasets)
            },
            "loss": ROBUST_LOSS,
            "f_scale": ROBUST_FSCALE,
            "sigma0": SIGMA0,
            "H_threshold": H_THRESH,
            "use_continuous_proxy": USE_CONTINUOUS_PROXY,
            "kappa": KAPPA,
            "grid": N_GRID,
            "domain_half_size": L,
            "optimizer": {
                "global_DE": USE_GLOBAL_SEARCH,
                "DE_maxiter": DE_MAXITER,
                "DE_popsize": DE_POPSIZE,
                "LS_max_nfev": LS_MAX_NFEV,
            },
            "solver": {
                "CFL_safety": CFL_SAFETY
            },
            "result": {
                "success": best.get("success", False),
                "message": best.get("message", ""),
                "cost": best.get("cost", None),
                "nfev": best.get("nfev", None),
            }
        }

        with open(os.path.join(OUT_DIR, "fit_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    if SAVE_FIGS:
        for i, ds in enumerate(datasets):
            R_sim = simulate_radius_times(
                days=ds.days,
                lambda_=best["lambda_"],
                delta=best["delta"],
                k_n=best["k_n"],
                k_p=best["k_p"],
                k_d=best["k_d"],
                alpha=best["alpha"],
                beta=best["beta"],
                R0=best["R0s"][i],
                s=best["s"],
                L=L,
                N_grid=N_GRID,
                cfl_safety=CFL_SAFETY,
                H_thresh=H_THRESH,
                use_continuous=USE_CONTINUOUS_PROXY,
                kappa=KAPPA,
                progress=False
            )

            plt.figure(figsize=(6, 4.2))
            plt.errorbar(ds.days, ds.R_mean, yerr=ds.R_std, fmt="o", capsize=3, label="Data")
            plt.plot(ds.days, R_sim, lw=2, label="Model fit")
            plt.xlabel("Day")
            plt.ylabel("Mean radius (µm)")
            plt.title(f"Dataset: {ds.name}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, f"{ds.name}_fit.png"), dpi=160)
            plt.close()

        plt.figure(figsize=(7.5, 4.2))

        for i, ds in enumerate(datasets):
            R_sim = simulate_radius_times(
                days=ds.days,
                lambda_=best["lambda_"],
                delta=best["delta"],
                k_n=best["k_n"],
                k_p=best["k_p"],
                k_d=best["k_d"],
                alpha=best["alpha"],
                beta=best["beta"],
                R0=best["R0s"][i],
                s=best["s"],
                L=L,
                N_grid=N_GRID,
                cfl_safety=CFL_SAFETY,
                H_thresh=H_THRESH,
                use_continuous=USE_CONTINUOUS_PROXY,
                kappa=KAPPA,
                progress=False
            )

            res = R_sim - ds.R_mean
            plt.plot(ds.days, res, marker="o", ls="-", label=ds.name)

        plt.axhline(0, lw=1)
        plt.xlabel("Day")
        plt.ylabel("Residual (µm)")
        plt.title("Residuals: model - data")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "residuals_all.png"), dpi=160)
        plt.close()


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading datasets ...")
    datasets = load_datasets(DATA_GLOB)

    for ds in datasets:
        print(f"  - {ds.name}: {len(ds.days)} points, days ∈ [{ds.days.min()}, {ds.days.max()}]")

    print("Fitting ...")
    t0 = time.time()
    best, theta = fit_parameters(datasets)
    t1 = time.time()

    print(f"Done. Success={best.get('success', False)}; message={best.get('message', '')}")
    print(f"Elapsed: {t1 - t0:.1f} s")

    print("Saving outputs ...")
    save_summary_and_plots(best, theta, datasets)
    print(f"Saved to '{OUT_DIR}/'")

    print("\nBest-fit shared parameters:")
    print(f"  lambda_ = {best['lambda_']:.6g}")
    print(f"  delta   = {best['delta']:.6g}")
    print(f"  k_n     = {best['k_n']:.6g}")
    print(f"  k_p     = {best['k_p']:.6g}")
    print(f"  k_d     = {best['k_d']:.6g}")
    print(f"  alpha   = {best['alpha']:.6g}")
    print(f"  beta    = {best['beta']:.6g}")
    print(f"  s       = {best['s']:.6g} µm/unit")

    print("Per-dataset R0 (model units):")
    for i, ds in enumerate(datasets):
        print(f"  {ds.name:>10s}: R0 = {best['R0s'][i]:.6g}")


if __name__ == "__main__":
    main()
