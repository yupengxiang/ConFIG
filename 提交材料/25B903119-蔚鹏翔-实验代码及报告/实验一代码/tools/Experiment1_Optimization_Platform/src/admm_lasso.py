"""ADMM solver for a synthetic Lasso problem."""

from __future__ import annotations

import time

import numpy as np


def soft_threshold(v: np.ndarray, kappa: float) -> np.ndarray:
    return np.sign(v) * np.maximum(np.abs(v) - kappa, 0.0)


def make_lasso_problem(m=120, n=50, true_sparsity=8, noise_std=0.01, seed=42) -> dict:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(m, n)) / np.sqrt(m)
    x_true = np.zeros(n)
    support = rng.choice(n, size=true_sparsity, replace=False)
    x_true[support] = rng.normal(loc=0.0, scale=2.0, size=true_sparsity)
    b = a @ x_true + noise_std * rng.normal(size=m)
    return {"A": a, "b": b, "x_true": x_true, "support": support}


def objective(a: np.ndarray, b: np.ndarray, x: np.ndarray, lam: float) -> float:
    residual = a @ x - b
    return float(0.5 * np.dot(residual, residual) + lam * np.linalg.norm(x, ord=1))


def solve_admm_lasso(
    *,
    m=120,
    n=50,
    true_sparsity=8,
    noise_std=0.01,
    lam=0.08,
    rho=1.0,
    iterations=300,
    seed=42,
) -> dict:
    problem = make_lasso_problem(m=m, n=n, true_sparsity=true_sparsity, noise_std=noise_std, seed=seed)
    a = problem["A"]
    b = problem["b"]
    x_true = problem["x_true"]
    x = np.zeros(n)
    z = np.zeros(n)
    u = np.zeros(n)
    lhs = a.T @ a + rho * np.eye(n)
    atb = a.T @ b
    history = []
    t0 = time.perf_counter()

    for _ in range(iterations + 1):
        primal = float(np.linalg.norm(x - z))
        dual = float(rho * np.linalg.norm(z - history[-1]["z"] if history else z))
        history.append(
            {
                "objective": objective(a, b, z, lam),
                "primal_residual": primal,
                "dual_residual": dual,
                "sparsity": float(np.mean(np.abs(z) < 1e-8)),
                "z": z.copy(),
            }
        )
        if len(history) == iterations + 1:
            break
        x = np.linalg.solve(lhs, atb + rho * (z - u))
        z_old = z.copy()
        z = soft_threshold(x + u, lam / rho)
        u = u + x - z
        history[-1]["dual_residual"] = float(rho * np.linalg.norm(z - z_old))

    rel_error = float(np.linalg.norm(z - x_true) / max(np.linalg.norm(x_true), 1e-12))
    return {
        "name": "ADMM-Lasso",
        "settings": {
            "m": m,
            "n": n,
            "true_sparsity": true_sparsity,
            "noise_std": noise_std,
            "lambda": lam,
            "rho": rho,
            "iterations": iterations,
            "seed": seed,
        },
        "history": history,
        "runtime_sec": time.perf_counter() - t0,
        "final_x": z,
        "x_true": x_true,
        "support": problem["support"],
        "final_objective": objective(a, b, z, lam),
        "final_primal_residual": history[-1]["primal_residual"],
        "final_dual_residual": history[-1]["dual_residual"],
        "final_sparsity": float(np.mean(np.abs(z) < 1e-8)),
        "relative_error": rel_error,
    }

