"""Second-order and quasi-Newton optimizers for experiment 1."""

from __future__ import annotations

import time

import numpy as np

from test_functions import armijo_backtracking


def _record(x: np.ndarray, f, grad) -> dict:
    g = grad(x)
    return {"x": x.copy(), "f": f(x), "grad_norm": float(np.linalg.norm(g))}


def _solve_direction(hess_matrix: np.ndarray, grad_vector: np.ndarray) -> np.ndarray:
    try:
        return -np.linalg.solve(hess_matrix, grad_vector)
    except np.linalg.LinAlgError:
        return -np.linalg.pinv(hess_matrix) @ grad_vector


def newton(f, grad, hess, x0, *, max_iter=200, tol=1e-6) -> dict:
    x = np.asarray(x0, dtype=float).copy()
    history = []
    t0 = time.perf_counter()
    converged = False
    for _ in range(max_iter):
        history.append(_record(x, f, grad))
        g = grad(x)
        if np.linalg.norm(g) < tol:
            converged = True
            break
        direction = _solve_direction(hess(x), g)
        x = x + direction
        if not np.all(np.isfinite(x)) or f(x) > 1e30:
            break
    if not converged:
        history.append(_record(x, f, grad))
    return {
        "name": "Newton",
        "history": history,
        "iterations": len(history) - 1,
        "runtime_sec": time.perf_counter() - t0,
        "converged": converged,
        "final_x": x,
        "final_f": f(x),
        "final_grad_norm": float(np.linalg.norm(grad(x))),
    }


def damped_newton(f, grad, hess, x0, *, max_iter=200, tol=1e-6) -> dict:
    x = np.asarray(x0, dtype=float).copy()
    history = []
    t0 = time.perf_counter()
    converged = False
    for _ in range(max_iter):
        history.append(_record(x, f, grad))
        g = grad(x)
        if np.linalg.norm(g) < tol:
            converged = True
            break
        direction = _solve_direction(hess(x), g)
        if float(np.dot(g, direction)) >= 0.0:
            direction = -g
        alpha = armijo_backtracking(f, grad, x, direction)
        x = x + alpha * direction
    if not converged:
        history.append(_record(x, f, grad))
    return {
        "name": "Damped Newton",
        "history": history,
        "iterations": len(history) - 1,
        "runtime_sec": time.perf_counter() - t0,
        "converged": converged,
        "final_x": x,
        "final_f": f(x),
        "final_grad_norm": float(np.linalg.norm(grad(x))),
    }


def bfgs(f, grad, x0, *, max_iter=200, tol=1e-6) -> dict:
    x = np.asarray(x0, dtype=float).copy()
    n = x.size
    h_inv = np.eye(n)
    history = []
    t0 = time.perf_counter()
    converged = False
    for _ in range(max_iter):
        history.append(_record(x, f, grad))
        g = grad(x)
        if np.linalg.norm(g) < tol:
            converged = True
            break
        direction = -h_inv @ g
        if float(np.dot(g, direction)) >= 0.0:
            direction = -g
            h_inv = np.eye(n)
        alpha = armijo_backtracking(f, grad, x, direction)
        s = alpha * direction
        x_new = x + s
        y = grad(x_new) - g
        ys = float(np.dot(y, s))
        if ys > 1e-12:
            rho = 1.0 / ys
            i = np.eye(n)
            h_inv = (i - rho * np.outer(s, y)) @ h_inv @ (i - rho * np.outer(y, s)) + rho * np.outer(s, s)
        x = x_new
    if not converged:
        history.append(_record(x, f, grad))
    return {
        "name": "BFGS",
        "history": history,
        "iterations": len(history) - 1,
        "runtime_sec": time.perf_counter() - t0,
        "converged": converged,
        "final_x": x,
        "final_f": f(x),
        "final_grad_norm": float(np.linalg.norm(grad(x))),
    }

