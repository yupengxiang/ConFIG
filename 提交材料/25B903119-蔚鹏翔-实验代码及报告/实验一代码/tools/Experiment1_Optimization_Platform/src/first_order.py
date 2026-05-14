"""First-order optimizers for the Rosenbrock experiment."""

from __future__ import annotations

import time

import numpy as np

from test_functions import armijo_backtracking


def _record(x: np.ndarray, f, grad) -> dict:
    g = grad(x)
    return {"x": x.copy(), "f": f(x), "grad_norm": float(np.linalg.norm(g))}


def gradient_descent(f, grad, x0, *, lr=1e-3, max_iter=10000, tol=1e-6) -> dict:
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
        x = x - lr * g
    if not converged:
        history.append(_record(x, f, grad))
    return {
        "name": "Gradient Descent",
        "history": history,
        "iterations": len(history) - 1,
        "runtime_sec": time.perf_counter() - t0,
        "converged": converged,
        "final_x": x,
        "final_f": f(x),
        "final_grad_norm": float(np.linalg.norm(grad(x))),
    }


def armijo_gradient_descent(f, grad, x0, *, max_iter=10000, tol=1e-6) -> dict:
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
        direction = -g
        alpha = armijo_backtracking(f, grad, x, direction)
        x = x + alpha * direction
    if not converged:
        history.append(_record(x, f, grad))
    return {
        "name": "Armijo GD",
        "history": history,
        "iterations": len(history) - 1,
        "runtime_sec": time.perf_counter() - t0,
        "converged": converged,
        "final_x": x,
        "final_f": f(x),
        "final_grad_norm": float(np.linalg.norm(grad(x))),
    }


def adam(f, grad, x0, *, lr=0.02, beta1=0.9, beta2=0.999, eps=1e-8, max_iter=10000, tol=1e-6) -> dict:
    x = np.asarray(x0, dtype=float).copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    history = []
    t0 = time.perf_counter()
    converged = False
    for k in range(1, max_iter + 1):
        history.append(_record(x, f, grad))
        g = grad(x)
        if np.linalg.norm(g) < tol:
            converged = True
            break
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * (g * g)
        m_hat = m / (1.0 - beta1**k)
        v_hat = v / (1.0 - beta2**k)
        x = x - lr * m_hat / (np.sqrt(v_hat) + eps)
    if not converged:
        history.append(_record(x, f, grad))
    return {
        "name": "Adam",
        "history": history,
        "iterations": len(history) - 1,
        "runtime_sec": time.perf_counter() - t0,
        "converged": converged,
        "final_x": x,
        "final_f": f(x),
        "final_grad_norm": float(np.linalg.norm(grad(x))),
    }

