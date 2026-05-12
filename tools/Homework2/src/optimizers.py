"""Core optimizers used in Homework 2."""

from __future__ import annotations

import time

import numpy as np

from line_search import make_exact_line_search


def _record(x: np.ndarray, f, grad) -> dict:
    g = grad(x)
    return {"x": x.copy(), "f": float(f(x)), "grad_norm": float(np.linalg.norm(g))}


def _finalize(history: list[dict], x: np.ndarray, f, grad, tol: float, converged: bool) -> tuple[list[dict], bool, float]:
    final_grad_norm = float(np.linalg.norm(grad(x)))
    if final_grad_norm < tol:
        converged = True
    if not history or not np.allclose(history[-1]["x"], x):
        history.append(_record(x, f, grad))
    return history, converged, final_grad_norm


def fr_conjugate_gradient(
    f,
    grad,
    x0,
    *,
    line_search=None,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> dict:
    if line_search is None:
        line_search = make_exact_line_search()

    x = np.asarray(x0, dtype=float).copy()
    g = grad(x)
    direction = -g
    history = []
    step_sizes = []
    betas = []
    t0 = time.perf_counter()
    converged = False

    for iteration in range(max_iter):
        history.append(_record(x, f, grad))
        grad_norm = float(np.linalg.norm(g))
        if grad_norm < tol:
            converged = True
            break
        if float(np.dot(g, direction)) >= 0.0:
            direction = -g
        alpha = float(line_search(f, grad, x, direction))
        if not np.isfinite(alpha) or alpha <= 0.0:
            break
        x_new = x + alpha * direction
        g_new = grad(x_new)
        beta = float(np.dot(g_new, g_new) / max(np.dot(g, g), 1e-30))
        direction = -g_new + beta * direction
        x = x_new
        g = g_new
        step_sizes.append(alpha)
        betas.append(beta)
        if (iteration + 1) % x.size == 0:
            direction = -g

    history, converged, final_grad_norm = _finalize(history, x, f, grad, tol, converged)
    return {
        "name": "FR Conjugate Gradient",
        "history": history,
        "step_sizes": step_sizes,
        "betas": betas,
        "iterations": len(history) - 1,
        "runtime_sec": time.perf_counter() - t0,
        "converged": converged,
        "final_x": x,
        "final_f": float(f(x)),
        "final_grad_norm": final_grad_norm,
    }


def dfp(
    f,
    grad,
    x0,
    *,
    line_search=None,
    h0: np.ndarray | None = None,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> dict:
    if line_search is None:
        line_search = make_exact_line_search()

    x = np.asarray(x0, dtype=float).copy()
    n = x.size
    h_inv = np.eye(n) if h0 is None else np.asarray(h0, dtype=float).copy()
    history = []
    step_sizes = []
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
        alpha = float(line_search(f, grad, x, direction))
        if not np.isfinite(alpha) or alpha <= 0.0:
            break
        s = alpha * direction
        x_new = x + s
        g_new = grad(x_new)
        y = g_new - g
        sy = float(np.dot(s, y))
        hy = h_inv @ y
        yhy = float(np.dot(y, hy))
        if sy > 1e-12 and yhy > 1e-12:
            h_inv = h_inv + np.outer(s, s) / sy - np.outer(hy, hy) / yhy
        else:
            h_inv = np.eye(n)
        x = x_new
        step_sizes.append(alpha)

    history, converged, final_grad_norm = _finalize(history, x, f, grad, tol, converged)
    return {
        "name": "DFP",
        "history": history,
        "step_sizes": step_sizes,
        "iterations": len(history) - 1,
        "runtime_sec": time.perf_counter() - t0,
        "converged": converged,
        "final_x": x,
        "final_f": float(f(x)),
        "final_grad_norm": final_grad_norm,
    }


def bfgs(
    f,
    grad,
    x0,
    *,
    line_search=None,
    h0: np.ndarray | None = None,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> dict:
    if line_search is None:
        line_search = make_exact_line_search()

    x = np.asarray(x0, dtype=float).copy()
    n = x.size
    h_inv = np.eye(n) if h0 is None else np.asarray(h0, dtype=float).copy()
    history = []
    step_sizes = []
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
        alpha = float(line_search(f, grad, x, direction))
        if not np.isfinite(alpha) or alpha <= 0.0:
            break
        s = alpha * direction
        x_new = x + s
        g_new = grad(x_new)
        y = g_new - g
        ys = float(np.dot(y, s))
        if ys > 1e-12:
            rho = 1.0 / ys
            identity = np.eye(n)
            h_inv = (identity - rho * np.outer(s, y)) @ h_inv @ (identity - rho * np.outer(y, s)) + rho * np.outer(s, s)
        else:
            h_inv = np.eye(n)
        x = x_new
        step_sizes.append(alpha)

    history, converged, final_grad_norm = _finalize(history, x, f, grad, tol, converged)
    return {
        "name": "BFGS",
        "history": history,
        "step_sizes": step_sizes,
        "iterations": len(history) - 1,
        "runtime_sec": time.perf_counter() - t0,
        "converged": converged,
        "final_x": x,
        "final_f": float(f(x)),
        "final_grad_norm": final_grad_norm,
    }
