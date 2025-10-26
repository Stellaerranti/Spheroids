# fit_spheroid_pde.py
# Joint fitting of a 2D nutrient–health PDE spheroid model to multiple datasets (radius vs day).
# Run directly in Spyder. Requires: numpy, scipy, pandas, matplotlib, tqdm (optional), numba (optional).
#
# Model (per your code, with a smooth-disk IC and tunable H-threshold radius extraction):
#   N_t = D_N ∇²N - γ H
#   H_t = D_H ∇²H + α N H (1 - H) - β (1 - N) H
# BC: N = 1 (Dirichlet) at boundary; H Neumann (copy edges)
# Radius extraction: R_sim(t) from area where H >= c (c = H_THRESH), then R = s * sqrt(A/π)
#
# Author: ChatGPT (GPT-5 Thinking) — 2025-10-26

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

# Optional acceleration. The script works without numba; if available it speeds up stepping.
try:
    from numba import njit
    NUMBA = True
except Exception:
    NUMBA = False


# =========================
# ===== User config =======
# =========================
OUT_DIR = "fit_output"        # folder for all outputs
DATA_GLOB = "*.txt"           # pattern for data files (your files like 500.txt, 2000.txt, etc.)
SAVE_FIGS = True
SAVE_JSON = True

# Grid / domain (model units). You can make N_GRID larger for final polish.
L = 50.0                      # half-domain size in model units (domain is [-L, L] × [-L, L])
N_GRID = 128                  # grid resolution (N x N) used during fitting
CFL_SAFETY = 0.24             # CFL safety factor for dt (explicit scheme)

# Radius extraction
H_THRESH = 0.8                # default health threshold c for radius extraction (tunable)
USE_CONTINUOUS_PROXY = False  # if True, uses R ~ sqrt( sum(H**KAPPA) * dx^2 / pi ); otherwise uses threshold area
KAPPA = 1.5                   # exponent for the continuous proxy (if used)
SMOOTH_DISK_EDGE = 2.0        # edge width (in grid cells) of the smooth disk IC

# Robust fitting knobs (your "approximation precision"):
SIGMA0 = 10.0                 # noise floor (µm) added in quadrature to StdRadius: larger -> less overfitting
ROBUST_LOSS = 'soft_l1'       # 'soft_l1' (recommended) or 'huber'
ROBUST_FSCALE = 20.0          # robustness scale (increase to be more tolerant to outliers)

# Parameter bounds (in real space). We optimize in log-space to enforce positivity.
BOUNDS = {
    "D_N":   (1e-4, 1e-1),
    "D_H":   (1e-5, 1e-2),
    "gamma": (1e-3, 2.0),
    "alpha": (1e-3, 1.0),
    "beta":  (1e-3, 2.0),
    "s":     (0.1, 20.0),       # µm per model unit
    # R0_j bounds are built per-dataset around initial guesses
}

# Global search (differential evolution) settings
USE_GLOBAL_SEARCH = True
DE_MAXITER = 40               # keep modest for runtime; increase for more thorough search
DE_POPSIZE = 10
DE_TOL = 1e-3
DE_SEED = 42

# Local polish (least_squares)
LS_MAX_NFEV = 200             # increase if you want more polishing steps


# =========================
# ======= Helpers =========
# =========================
@dataclass
class Dataset:
    name: str
    days: np.ndarray          # (m,) observation times in days
    R_mean: np.ndarray        # (m,) mean radius (µm)
    R_std: np.ndarray         # (m,) stddev (µm)
    N_repl: np.ndarray        # (m,) replicate counts


def load_datasets(pattern: str) -> List[Dataset]:
    files = sorted(glob.glob(pattern))
    datasets: List[Dataset] = []
    if not files:
        raise FileNotFoundError(f"No data files found by pattern: {pattern}")

    for path in files:
        # Many of your files have headers commented out with '#'.
        # Read with NO header, fixed column names, and ignore '#' lines.
        df = pd.read_csv(
            path,
            comment="#",
            sep=r"\s+",
            header=None,
            names=["Day", "MeanRadius", "StdRadius", "N"],
            engine="python",
        )

        # Coerce to numeric and drop any non-data rows just in case
        for col in ["Day", "MeanRadius", "StdRadius", "N"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["Day", "MeanRadius", "StdRadius", "N"]).reset_index(drop=True)

        if df.empty:
            raise ValueError(f"File {path} contained no valid data rows for Day/MeanRadius/StdRadius/N.")

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
    # First observation as a rough proxy
    R0s = []
    for ds in datasets:
        if len(ds.R_mean) == 0:
            R0s.append(5.0 / s0)  # fallback
        else:
            R0s.append(max(1e-3, ds.R_mean[0] / s0))  # model units
    return R0s


def build_weights(ds: Dataset, sigma0: float) -> np.ndarray:
    # w_i = sqrt(N) / sqrt(std^2 + sigma0^2)
    denom = np.sqrt(np.maximum(1e-12, ds.R_std**2 + sigma0**2))
    return np.sqrt(np.maximum(1.0, ds.N_repl)) / denom


# =========================
# ====== PDE kernel =======
# =========================
def _laplacian_numpy(Z: np.ndarray, dx: float) -> np.ndarray:
    # 5-point stencil on interior; returns interior-sized laplacian
    return (Z[:-2, 1:-1] + Z[2:, 1:-1] + Z[1:-1, :-2] + Z[1:-1, 2:] - 4.0 * Z[1:-1, 1:-1]) / (dx * dx)

if NUMBA:
    @njit(cache=True, fastmath=True)
    def _laplacian_numba(Z: np.ndarray, dx: float) -> np.ndarray:
        return (Z[:-2, 1:-1] + Z[2:, 1:-1] + Z[1:-1, :-2] + Z[1:-1, 2:] - 4.0 * Z[1:-1, 1:-1]) / (dx * dx)

    LAPL = _laplacian_numba
else:
    LAPL = _laplacian_numpy


def make_initial_H_smooth_disk(X: np.ndarray, Y: np.ndarray, R0: float, edge_cells: float) -> np.ndarray:
    # Smooth disk: Fermi-like ramp at the edge with width ~ edge_cells * dx
    # H = 1 / (1 + exp((r - R0)/w)), with w ~ edge_cells * dx
    dx = X[0,1] - X[0,0]
    r = np.sqrt(X*X + Y*Y)
    w = max(1e-6, edge_cells * dx)
    H = 1.0 / (1.0 + np.exp((r - R0) / w))
    return H


def extract_radius(H: np.ndarray, dx: float, c: float, use_continuous: bool, kappa: float) -> float:
    if use_continuous:
        A = np.sum(H**kappa) * (dx * dx)
        R = np.sqrt(max(1e-12, A) / math.pi)
    else:
        mask = (H >= c)
        A = float(np.count_nonzero(mask)) * (dx * dx)
        R = np.sqrt(max(1e-12, A) / math.pi)
    return R


def simulate_radius_times(
    days: np.ndarray,
    D_N: float, D_H: float, gamma: float, alpha: float, beta: float,
    R0: float, s: float,
    L: float, N_grid: int, cfl_safety: float,
    H_thresh: float, use_continuous: bool, kappa: float,
    progress: bool = False
) -> np.ndarray:
    """
    Run one PDE simulation up to max(days), returning R_phys[day_i] in microns.
    """
    # Grid
    x = np.linspace(-L, L, N_grid)
    y = np.linspace(-L, L, N_grid)
    X, Y = np.meshgrid(x, y)
    dx = x[1] - x[0]

    # Stable dt for explicit scheme in 2D Laplacian (approx)
    Dmax = max(D_N, D_H)
    if Dmax <= 0:
        Dmax = 1e-12
    dt_max = cfl_safety * dx * dx / (4.0 * Dmax)
    dt = min(0.5, dt_max)  # also cap dt to keep reaction terms resolveable
    if dt <= 0:
        dt = 1e-4

    # Initial conditions
    N = np.ones((N_grid, N_grid), dtype=np.float64)
    H = make_initial_H_smooth_disk(X, Y, R0, SMOOTH_DISK_EDGE)

    # Time integration
    t_end = float(np.max(days))
    n_steps = int(math.ceil(t_end / dt))

    # Prepare query times (integers/real days from data)
    target_times = np.array(days, dtype=float)
    # For robust matching, we keep an index and check |t - target| < dt/2
    next_idx = 0
    R_phys = np.full_like(target_times, np.nan, dtype=float)

    iterator = range(n_steps)
    if progress and TQDM:
        iterator = tqdm(iterator, total=n_steps, desc="Simulating", leave=False)

    t = 0.0
    for step in iterator:
        # Laplacians on interior
        lapN = LAPL(N, dx)
        lapH = LAPL(H, dx)

        # Euler explicit, update interior only
        N_new = N.copy()
        H_new = H.copy()

        N_new[1:-1, 1:-1] += dt * (D_N * lapN - gamma * H[1:-1, 1:-1])
        H_new[1:-1, 1:-1] += dt * (
            D_H * lapH
            + alpha * N[1:-1, 1:-1] * H[1:-1, 1:-1] * (1.0 - H[1:-1, 1:-1])
            - beta  * (1.0 - N[1:-1, 1:-1]) * H[1:-1, 1:-1]
        )

        # Dirichlet for N
        N_new[0, :] = N_new[-1, :] = 1.0
        N_new[:, 0] = N_new[:, -1] = 1.0

        # Neumann for H (copy edges)
        H_new[0, :]  = H_new[1, :]
        H_new[-1, :] = H_new[-2, :]
        H_new[:, 0]  = H_new[:, 1]
        H_new[:, -1] = H_new[:, -2]

        # Clamp to [0,1] for stability
        np.clip(N_new, 0.0, 1.0, out=N_new)
        np.clip(H_new, 0.0, 1.0, out=H_new)

        N, H = N_new, H_new
        t += dt

        # Record any target times we've crossed
        while next_idx < len(target_times) and abs(t - target_times[next_idx]) <= dt/2.0:
            R_model = extract_radius(H, dx, H_thresh, use_continuous, kappa)
            R_phys[next_idx] = s * R_model
            next_idx += 1

        if t >= t_end - 1e-12 and next_idx < len(target_times):
            # final capture if any missed due to step alignment
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
    Packs/unpacks the optimization vector:
      theta = [log D_N, log D_H, log gamma, log alpha, log beta, log s, log R0_1, ..., log R0_m]
    """
    def __init__(self, datasets: List[Dataset], bounds: Dict[str, Tuple[float, float]], s0: float, R0_guesses: List[float]):
        self.datasets = datasets
        self.m = len(datasets)

        self.idx = {
            "D_N": 0, "D_H": 1, "gamma": 2, "alpha": 3, "beta": 4, "s": 5,
        }
        self.base_len = 6
        # per-dataset R0 indices
        self.R0_start = self.base_len
        self.n_total = self.base_len + self.m

        # bounds in log-space
        self.lb = np.full(self.n_total, -np.inf, dtype=float)
        self.ub = np.full(self.n_total,  np.inf, dtype=float)

        def set_log_bounds(name, lo, hi):
            i = self.idx[name]
            self.lb[i] = math.log(lo)
            self.ub[i] = math.log(hi)

        set_log_bounds("D_N", *bounds["D_N"])
        set_log_bounds("D_H", *bounds["D_H"])
        set_log_bounds("gamma", *bounds["gamma"])
        set_log_bounds("alpha", *bounds["alpha"])
        set_log_bounds("beta",  *bounds["beta"])
        set_log_bounds("s",     *bounds["s"])

        # Per-dataset R0 bounds around initial guess (×[0.5, 2.0])
        for j, R0g in enumerate(R0_guesses):
            lo = max(1e-3, 0.5 * R0g)
            hi = 2.0 * max(lo, R0g)
            self.lb[self.R0_start + j] = math.log(lo)
            self.ub[self.R0_start + j] = math.log(hi)

    def to_params(self, theta: np.ndarray) -> Dict:
        D_N   = math.exp(theta[self.idx["D_N"]])
        D_H   = math.exp(theta[self.idx["D_H"]])
        gamma = math.exp(theta[self.idx["gamma"]])
        alpha = math.exp(theta[self.idx["alpha"]])
        beta  = math.exp(theta[self.idx["beta"]])
        s     = math.exp(theta[self.idx["s"]])
        R0s   = np.exp(theta[self.R0_start:self.R0_start + self.m])
        return {"D_N": D_N, "D_H": D_H, "gamma": gamma, "alpha": alpha, "beta": beta, "s": s, "R0s": R0s}

    def initial_theta(self, s0: float, R0_guesses: List[float]) -> np.ndarray:
        # Center of bounds in log-space for shared params; s=s0; R0=guess
        theta = np.zeros(self.n_total, dtype=float)
        for name in ["D_N","D_H","gamma","alpha","beta"]:
            lo, hi = BOUNDS[name]
            theta[self.idx[name]] = 0.5 * (math.log(lo) + math.log(hi))
        theta[self.idx["s"]] = math.log(s0)
        for j, R0g in enumerate(R0_guesses):
            theta[self.R0_start + j] = math.log(max(1e-3, R0g))
        return theta


def residuals_vector(
    theta: np.ndarray, pack: ParamPack, datasets: List[Dataset],
    L: float, N_grid: int, cfl: float, H_thresh: float, use_cont: bool, kappa: float,
    verbose: bool = False
) -> np.ndarray:
    p = pack.to_params(theta)
    D_N, D_H, gamma, alpha, beta, s = p["D_N"], p["D_H"], p["gamma"], p["alpha"], p["beta"], p["s"]
    R0s = p["R0s"]

    res_list = []
    # Simulate each dataset independently (different R0)
    it = enumerate(datasets)
    if verbose and TQDM:
        it = tqdm(list(it), desc="Datasets", leave=False)
    for j, ds in it:
        weights = build_weights(ds, SIGMA0)      # (m,)
        R_sim = simulate_radius_times(
            days=ds.days,
            D_N=D_N, D_H=D_H, gamma=gamma, alpha=alpha, beta=beta,
            R0=R0s[j], s=s,
            L=L, N_grid=N_grid, cfl_safety=cfl,
            H_thresh=H_thresh, use_continuous=use_cont, kappa=kappa,
            progress=False
        )
        # Residuals = weighted difference
        r = weights * (R_sim - ds.R_mean)
        res_list.append(r)

    return np.concatenate(res_list)


def objective_scalar_for_DE(
    theta: np.ndarray, pack: ParamPack, datasets: List[Dataset],
    L: float, N_grid: int, cfl: float, H_thresh: float, use_cont: bool, kappa: float
) -> float:
    # For DE, return a scalar robust objective: use soft-L1 like reduction
    r = residuals_vector(theta, pack, datasets, L, N_grid, cfl, H_thresh, use_cont, kappa, verbose=False)
    # soft-L1 scalarization (approx): sum( f(x) ), f(x)=2*(sqrt(1+(x/f)^2)-1)
    f = ROBUST_FSCALE
    val = np.sum(2.0 * (np.sqrt(1.0 + (r/f)**2) - 1.0))
    # also add a mild penalty if any NaNs
    if not np.isfinite(val):
        return 1e12
    return float(val)


def fit_parameters(datasets: List[Dataset]) -> Tuple[Dict, np.ndarray]:
    os.makedirs(OUT_DIR, exist_ok=True)

    # Initial guesses
    s0 = 1.0  # µm per model unit
    R0_guesses = initial_R0_guesses(datasets, s0=s0)
    pack = ParamPack(datasets, BOUNDS, s0, R0_guesses)
    theta0 = pack.initial_theta(s0, R0_guesses)

    # ===== Global search (optional) =====
    theta_start = theta0.copy()
    if USE_GLOBAL_SEARCH:
        print("[Stage 1/2] Global search (DE) ...")
        # Build DE bounds in real numbers (log-space bounds)
        bounds_DE = list(zip(pack.lb.tolist(), pack.ub.tolist()))

        result = differential_evolution(
            func=lambda th: objective_scalar_for_DE(th, pack, datasets, L, N_GRID, CFL_SAFETY, H_THRESH, USE_CONTINUOUS_PROXY, KAPPA),
            bounds=bounds_DE, strategy='best1bin', maxiter=DE_MAXITER, popsize=DE_POPSIZE,
            tol=DE_TOL, seed=DE_SEED, polish=False, updating='deferred'
        )
        print(f"  DE status: {result.message}")
        theta_start = result.x

    # ===== Local polish =====
    print("[Stage 2/2] Local robust least-squares ...")
    ls = least_squares(
        fun=lambda th: residuals_vector(th, pack, datasets, L, N_GRID, CFL_SAFETY, H_THRESH, USE_CONTINUOUS_PROXY, KAPPA, verbose=False),
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

    # Save JSON
    if SAVE_JSON:
        summary = {
            "shared_params": {
                "D_N": best["D_N"], "D_H": best["D_H"], "gamma": best["gamma"],
                "alpha": best["alpha"], "beta": best["beta"], "s": best["s"]
            },
            "per_dataset_R0": {ds.name: float(best["R0s"][i]) for i, ds in enumerate(datasets)},
            "loss": ROBUST_LOSS, "f_scale": ROBUST_FSCALE, "sigma0": SIGMA0,
            "H_threshold": H_THRESH, "use_continuous_proxy": USE_CONTINUOUS_PROXY, "kappa": KAPPA,
            "grid": N_GRID, "domain_half_size": L,
            "optimizer": {"global_DE": USE_GLOBAL_SEARCH, "DE_maxiter": DE_MAXITER, "DE_popsize": DE_POPSIZE,
                          "LS_max_nfev": LS_MAX_NFEV},
            "solver": {"CFL_safety": CFL_SAFETY},
            "result": {"success": best.get("success", False), "message": best.get("message", ""), "cost": best.get("cost", None),
                       "nfev": best.get("nfev", None)}
        }
        with open(os.path.join(OUT_DIR, "fit_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    # Plots: data vs sim, per dataset
    if SAVE_FIGS:
        for i, ds in enumerate(datasets):
            R_sim = simulate_radius_times(
                days=ds.days,
                D_N=best["D_N"], D_H=best["D_H"], gamma=best["gamma"], alpha=best["alpha"], beta=best["beta"],
                R0=best["R0s"][i], s=best["s"],
                L=L, N_grid=N_GRID, cfl_safety=CFL_SAFETY,
                H_thresh=H_THRESH, use_continuous=USE_CONTINUOUS_PROXY, kappa=KAPPA,
                progress=False
            )

            plt.figure(figsize=(6,4.2))
            plt.errorbar(ds.days, ds.R_mean, yerr=ds.R_std, fmt='o', capsize=3, label="Data")
            plt.plot(ds.days, R_sim, lw=2, label="Model (fit)")
            plt.xlabel("Day")
            plt.ylabel("Mean radius (µm)")
            plt.title(f"Dataset: {ds.name}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, f"{ds.name}_fit.png"), dpi=160)
            plt.close()

        # Residuals across all datasets
        plt.figure(figsize=(7.5,4.2))
        for i, ds in enumerate(datasets):
            R_sim = simulate_radius_times(
                days=ds.days,
                D_N=best["D_N"], D_H=best["D_H"], gamma=best["gamma"], alpha=best["alpha"], beta=best["beta"],
                R0=best["R0s"][i], s=best["s"],
                L=L, N_grid=N_GRID, cfl_safety=CFL_SAFETY,
                H_thresh=H_THRESH, use_continuous=USE_CONTINUOUS_PROXY, kappa=KAPPA,
                progress=False
            )
            res = (R_sim - ds.R_mean)
            plt.plot(ds.days, res, marker='o', ls='-', label=ds.name)
        plt.axhline(0, lw=1)
        plt.xlabel("Day")
        plt.ylabel("Residual (µm)")
        plt.title("Residuals (model - data)")
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

    # Show a few key params in console
    print("\nBest-fit shared parameters:")
    print(f"  D_N   = {best['D_N']:.6g}")
    print(f"  D_H   = {best['D_H']:.6g}")
    print(f"  gamma = {best['gamma']:.6g}")
    print(f"  alpha = {best['alpha']:.6g}")
    print(f"  beta  = {best['beta']:.6g}")
    print(f"  s     = {best['s']:.6g} µm/unit")
    print("Per-dataset R0 (model units):")
    for i, ds in enumerate(load_datasets(DATA_GLOB)):
        print(f"  {ds.name:>10s}: R0 = {best['R0s'][i]:.6g}")

if __name__ == "__main__":
    main()
