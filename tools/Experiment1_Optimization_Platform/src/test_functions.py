"""Objective functions used in experiment 1."""

from __future__ import annotations

import numpy as np


def rosenbrock(x: np.ndarray) -> float:
    """Two-dimensional Rosenbrock banana function."""
    x = np.asarray(x, dtype=float)
    return float((1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2)


def rosenbrock_grad(x: np.ndarray) -> np.ndarray:
    """Gradient of the two-dimensional Rosenbrock function."""
    x = np.asarray(x, dtype=float)
    return np.array(
        [
            -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] ** 2),
            200.0 * (x[1] - x[0] ** 2),
        ],
        dtype=float,
    )


def rosenbrock_hessian(x: np.ndarray) -> np.ndarray:
    """Hessian of the two-dimensional Rosenbrock function."""
    x = np.asarray(x, dtype=float)
    return np.array(
        [
            [2.0 - 400.0 * x[1] + 1200.0 * x[0] ** 2, -400.0 * x[0]],
            [-400.0 * x[0], 200.0],
        ],
        dtype=float,
    )


def armijo_backtracking(
    f,
    grad,
    x: np.ndarray,
    direction: np.ndarray,
    *,
    alpha0: float = 1.0,
    c1: float = 1e-4,
    beta: float = 0.5,
    min_alpha: float = 1e-12,
) -> float:
    """Return an Armijo step length for a descent direction."""
    alpha = alpha0
    fx = f(x)
    gx = grad(x)
    slope = float(np.dot(gx, direction))
    if slope >= 0.0:
        return min_alpha
    while alpha > min_alpha:
        if f(x + alpha * direction) <= fx + c1 * alpha * slope:
            return alpha
        alpha *= beta
    return min_alpha

