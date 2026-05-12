"""Exact line-search methods for Homework 2."""

from __future__ import annotations

from bisect import bisect_left
from math import sqrt

import numpy as np


def _sorted_interval(a: float, b: float) -> tuple[float, float]:
    if a == b:
        raise ValueError("Interval endpoints must be distinct")
    return (a, b) if a < b else (b, a)


def golden_section_search(f, a: float, b: float, *, tol: float = 1e-6, max_iter: int = 200) -> dict:
    left, right = _sorted_interval(a, b)
    inv_phi = (sqrt(5.0) - 1.0) / 2.0
    c = right - inv_phi * (right - left)
    d = left + inv_phi * (right - left)
    fc = float(f(c))
    fd = float(f(d))
    history = []
    iteration = 0

    while (right - left) > tol and iteration < max_iter:
        iteration += 1
        history.append({"iteration": iteration, "a": left, "b": right, "c": c, "d": d, "f_c": fc, "f_d": fd})
        if fc <= fd:
            right = d
            d = c
            fd = fc
            c = right - inv_phi * (right - left)
            fc = float(f(c))
        else:
            left = c
            c = d
            fc = fd
            d = left + inv_phi * (right - left)
            fd = float(f(d))

    x_best = 0.5 * (left + right)
    return {
        "method": "golden_section",
        "x": x_best,
        "f": float(f(x_best)),
        "interval": [left, right],
        "iterations": iteration,
        "converged": (right - left) <= tol,
        "history": history,
    }


def fibonacci_search(f, a: float, b: float, *, tol: float = 1e-6) -> dict:
    left, right = _sorted_interval(a, b)
    target = 2.0 * (right - left) / tol
    fib = [1, 1]
    while fib[-1] < target:
        fib.append(fib[-1] + fib[-2])
    n = len(fib) - 1
    if n < 2:
        x_best = 0.5 * (left + right)
        return {
            "method": "fibonacci",
            "x": x_best,
            "f": float(f(x_best)),
            "interval": [left, right],
            "iterations": 0,
            "converged": True,
            "history": [],
        }

    history = []
    for k in range(n - 1, 1, -1):
        ratio = fib[k - 2] / fib[k]
        x1 = left + ratio * (right - left)
        x2 = right - ratio * (right - left)
        f1 = float(f(x1))
        f2 = float(f(x2))
        history.append({"iteration": len(history) + 1, "a": left, "b": right, "x1": x1, "x2": x2, "f_x1": f1, "f_x2": f2})
        if f1 > f2:
            left = x1
        else:
            right = x2

    x_best = 0.5 * (left + right)
    return {
        "method": "fibonacci",
        "x": x_best,
        "f": float(f(x_best)),
        "interval": [left, right],
        "iterations": len(history),
        "converged": (right - left) <= tol,
        "history": history,
    }


def bisection_search(derivative, a: float, b: float, *, delta: float = 1e-6, max_iter: int = 200, objective=None) -> dict:
    left, right = _sorted_interval(a, b)
    d_left = float(derivative(left))
    d_right = float(derivative(right))
    if abs(d_left) <= delta:
        return {
            "method": "bisection",
            "x": left,
            "f": None if objective is None else float(objective(left)),
            "interval": [left, left],
            "iterations": 0,
            "converged": True,
            "history": [],
        }
    if abs(d_right) <= delta:
        return {
            "method": "bisection",
            "x": right,
            "f": None if objective is None else float(objective(right)),
            "interval": [right, right],
            "iterations": 0,
            "converged": True,
            "history": [],
        }
    if d_left > 0.0 or d_right < 0.0:
        raise ValueError("Derivative must change sign from negative to positive on the interval")

    history = []
    mid = 0.5 * (left + right)
    d_mid = float(derivative(mid))
    iteration = 0
    while (right - left) > delta and iteration < max_iter:
        iteration += 1
        mid = 0.5 * (left + right)
        d_mid = float(derivative(mid))
        history.append({"iteration": iteration, "a": left, "b": right, "mid": mid, "derivative": d_mid})
        if abs(d_mid) <= delta:
            break
        if d_mid < 0.0:
            left = mid
        else:
            right = mid

    x_best = 0.5 * (left + right)
    return {
        "method": "bisection",
        "x": x_best,
        "f": None if objective is None else float(objective(x_best)),
        "interval": [left, right],
        "iterations": iteration,
        "converged": (right - left) <= delta or abs(d_mid) <= delta,
        "history": history,
    }


def shubert_piyavskii_search(f, a: float, b: float, *, lipschitz: float, tol: float = 1e-6, max_iter: int = 200) -> dict:
    if lipschitz <= 0.0:
        raise ValueError("Lipschitz constant must be positive")

    left, right = _sorted_interval(a, b)
    samples: list[tuple[float, float]] = [(left, float(f(left))), (right, float(f(right)))]
    best_x, best_y = min(samples, key=lambda item: item[1])
    history = []
    best_lower_bound = -np.inf

    def intersection(p: tuple[float, float], q: tuple[float, float]) -> tuple[float, float]:
        x_left, y_left = p
        x_right, y_right = q
        x_new = 0.5 * (x_left + x_right) + 0.5 * (y_left - y_right) / lipschitz
        lower_bound = 0.5 * (y_left + y_right - lipschitz * (x_right - x_left))
        return float(np.clip(x_new, x_left, x_right)), lower_bound

    def insert_sample(x_new: float, y_new: float) -> bool:
        xs = [item[0] for item in samples]
        idx = bisect_left(xs, x_new)
        if idx < len(samples) and abs(samples[idx][0] - x_new) <= 1e-14:
            return False
        if idx > 0 and abs(samples[idx - 1][0] - x_new) <= 1e-14:
            return False
        samples.insert(idx, (x_new, y_new))
        return True

    for iteration in range(1, max_iter + 1):
        candidates = [
            (*intersection(samples[i], samples[i + 1]), samples[i][0], samples[i + 1][0]) for i in range(len(samples) - 1)
        ]
        candidate_x, best_lower_bound, interval_left, interval_right = min(candidates, key=lambda item: item[1])
        candidate_y = float(f(candidate_x))
        history.append(
            {
                "iteration": iteration,
                "x_candidate": candidate_x,
                "lower_bound": best_lower_bound,
                "f_candidate": candidate_y,
                "best_x": best_x,
                "best_f": best_y,
                "interval_left": interval_left,
                "interval_right": interval_right,
            }
        )
        if candidate_y < best_y:
            best_x, best_y = candidate_x, candidate_y
        inserted = insert_sample(candidate_x, candidate_y)
        if not inserted or (interval_right - interval_left) <= tol or (best_y - best_lower_bound) <= lipschitz * tol:
            break

    return {
        "method": "shubert_piyavskii",
        "x": best_x,
        "f": best_y,
        "interval": [left, right],
        "iterations": len(history),
        "converged": bool(history) and (
            (history[-1]["interval_right"] - history[-1]["interval_left"]) <= tol
            or (best_y - best_lower_bound) <= lipschitz * tol
        ),
        "history": history,
        "best_lower_bound": best_lower_bound,
    }


def bracket_minimum(phi, *, start: float = 0.0, initial_step: float = 1.0, expansion: float = 2.0, max_iter: int = 60) -> tuple[float, float]:
    left = start
    step = initial_step
    f_left = float(phi(left))
    right = left + step
    f_right = float(phi(right))
    if f_right > f_left:
        return left, right

    for _ in range(max_iter):
        step *= expansion
        next_right = right + step
        f_next = float(phi(next_right))
        if f_next > f_right:
            return left, next_right
        left, f_left = right, f_right
        right, f_right = next_right, f_next
    return start, right


def bracket_directional_derivative(derivative, *, initial_step: float = 1.0, expansion: float = 2.0, max_iter: int = 60) -> tuple[float, float]:
    left = 0.0
    d_left = float(derivative(left))
    if d_left >= 0.0:
        return 0.0, 0.0

    right = initial_step
    for _ in range(max_iter):
        d_right = float(derivative(right))
        if d_right >= 0.0:
            return left, right
        right *= expansion
    raise ValueError("Failed to bracket a stationary point along the search direction")


def exact_search_along_direction(
    f,
    grad,
    x: np.ndarray,
    direction: np.ndarray,
    *,
    method: str = "bisection",
    tol: float = 1e-8,
    initial_step: float = 1.0,
) -> float:
    direction = np.asarray(direction, dtype=float)
    x = np.asarray(x, dtype=float)
    phi = lambda alpha: f(x + alpha * direction)
    phi_prime = lambda alpha: float(np.dot(grad(x + alpha * direction), direction))

    if method == "bisection":
        left, right = bracket_directional_derivative(phi_prime, initial_step=initial_step)
        if left == right == 0.0:
            return 0.0
        return float(bisection_search(phi_prime, left, right, delta=tol, objective=phi)["x"])
    if method == "golden_section":
        left, right = bracket_minimum(phi, initial_step=initial_step)
        return float(golden_section_search(phi, left, right, tol=tol)["x"])
    if method == "fibonacci":
        left, right = bracket_minimum(phi, initial_step=initial_step)
        return float(fibonacci_search(phi, left, right, tol=tol)["x"])
    if method == "shubert_piyavskii":
        left, right = bracket_minimum(phi, initial_step=initial_step)
        sample_points = np.linspace(left, right, num=33)
        derivative_samples = [abs(phi_prime(point)) for point in sample_points]
        lipschitz = max(max(derivative_samples), 1.0)
        return float(shubert_piyavskii_search(phi, left, right, lipschitz=lipschitz, tol=tol)["x"])
    raise ValueError(f"Unsupported exact line-search method: {method}")


def make_exact_line_search(*, method: str = "bisection", tol: float = 1e-8, initial_step: float = 1.0):
    def _line_search(f, grad, x, direction) -> float:
        return exact_search_along_direction(f, grad, x, direction, method=method, tol=tol, initial_step=initial_step)

    return _line_search


def _directional_setup(f, grad, x: np.ndarray, direction: np.ndarray):
    x = np.asarray(x, dtype=float)
    direction = np.asarray(direction, dtype=float)
    phi = lambda alpha: float(f(x + alpha * direction))
    phi_prime = lambda alpha: float(np.dot(grad(x + alpha * direction), direction))
    phi0 = phi(0.0)
    phi_prime0 = phi_prime(0.0)
    if phi_prime0 >= 0.0:
        raise ValueError("Direction is not a descent direction")
    return phi, phi_prime, phi0, phi_prime0


def armijo_search(f, grad, x: np.ndarray, direction: np.ndarray, *, alpha0: float = 1.0, c1: float = 1e-4, beta: float = 0.5, min_alpha: float = 1e-12, max_iter: int = 60) -> dict:
    phi, _phi_prime, phi0, phi_prime0 = _directional_setup(f, grad, x, direction)
    alpha = alpha0
    history = []

    for iteration in range(1, max_iter + 1):
        phi_alpha = phi(alpha)
        rhs = phi0 + c1 * alpha * phi_prime0
        accepted = phi_alpha <= rhs
        history.append({"iteration": iteration, "alpha": alpha, "phi": phi_alpha, "rhs": rhs, "accepted": accepted})
        if accepted:
            return {"method": "armijo", "alpha": alpha, "phi": phi_alpha, "iterations": iteration, "converged": True, "history": history}
        alpha *= beta
        if alpha < min_alpha:
            break

    phi_alpha = phi(alpha)
    return {"method": "armijo", "alpha": alpha, "phi": phi_alpha, "iterations": len(history), "converged": False, "history": history}


def goldstein_search(f, grad, x: np.ndarray, direction: np.ndarray, *, alpha0: float = 1.0, c: float = 0.2, expand: float = 2.0, max_iter: int = 60) -> dict:
    phi, _phi_prime, phi0, phi_prime0 = _directional_setup(f, grad, x, direction)
    lower = 0.0
    upper = np.inf
    alpha = alpha0
    history = []

    for iteration in range(1, max_iter + 1):
        phi_alpha = phi(alpha)
        lower_bound = phi0 + (1.0 - c) * alpha * phi_prime0
        upper_bound = phi0 + c * alpha * phi_prime0
        accepted = lower_bound <= phi_alpha <= upper_bound
        history.append(
            {
                "iteration": iteration,
                "alpha": alpha,
                "phi": phi_alpha,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "accepted": accepted,
            }
        )
        if accepted:
            return {"method": "goldstein", "alpha": alpha, "phi": phi_alpha, "iterations": iteration, "converged": True, "history": history}
        if phi_alpha > upper_bound:
            upper = alpha
            alpha = 0.5 * (lower + upper)
        else:
            lower = alpha
            alpha = expand * alpha if np.isinf(upper) else 0.5 * (lower + upper)

    phi_alpha = phi(alpha)
    return {"method": "goldstein", "alpha": alpha, "phi": phi_alpha, "iterations": len(history), "converged": False, "history": history}


def wolfe_powell_search(f, grad, x: np.ndarray, direction: np.ndarray, *, alpha0: float = 1.0, c1: float = 1e-4, c2: float = 0.9, beta: float = 0.5, expand: float = 2.0, max_iter: int = 60) -> dict:
    phi, phi_prime, phi0, phi_prime0 = _directional_setup(f, grad, x, direction)
    alpha = alpha0
    lower = 0.0
    upper = np.inf
    history = []

    for iteration in range(1, max_iter + 1):
        phi_alpha = phi(alpha)
        phi_prime_alpha = phi_prime(alpha)
        armijo_ok = phi_alpha <= phi0 + c1 * alpha * phi_prime0
        curvature_ok = phi_prime_alpha >= c2 * phi_prime0
        accepted = armijo_ok and curvature_ok
        history.append(
            {
                "iteration": iteration,
                "alpha": alpha,
                "phi": phi_alpha,
                "phi_prime": phi_prime_alpha,
                "armijo_ok": armijo_ok,
                "curvature_ok": curvature_ok,
                "accepted": accepted,
            }
        )
        if accepted:
            return {"method": "wolfe_powell", "alpha": alpha, "phi": phi_alpha, "iterations": iteration, "converged": True, "history": history}
        if not armijo_ok:
            upper = alpha
            alpha = 0.5 * (lower + upper)
        else:
            lower = alpha
            alpha = expand * alpha if np.isinf(upper) else 0.5 * (lower + upper)
        alpha = max(alpha, beta * alpha0 * 1e-6)

    phi_alpha = phi(alpha)
    return {"method": "wolfe_powell", "alpha": alpha, "phi": phi_alpha, "iterations": len(history), "converged": False, "history": history}


def strong_wolfe_search(f, grad, x: np.ndarray, direction: np.ndarray, *, alpha0: float = 1.0, c1: float = 1e-4, c2: float = 0.9, beta: float = 0.5, expand: float = 2.0, max_iter: int = 60) -> dict:
    phi, phi_prime, phi0, phi_prime0 = _directional_setup(f, grad, x, direction)
    alpha = alpha0
    lower = 0.0
    upper = np.inf
    history = []

    for iteration in range(1, max_iter + 1):
        phi_alpha = phi(alpha)
        phi_prime_alpha = phi_prime(alpha)
        armijo_ok = phi_alpha <= phi0 + c1 * alpha * phi_prime0
        curvature_ok = abs(phi_prime_alpha) <= c2 * abs(phi_prime0)
        accepted = armijo_ok and curvature_ok
        history.append(
            {
                "iteration": iteration,
                "alpha": alpha,
                "phi": phi_alpha,
                "phi_prime": phi_prime_alpha,
                "armijo_ok": armijo_ok,
                "curvature_ok": curvature_ok,
                "accepted": accepted,
            }
        )
        if accepted:
            return {"method": "strong_wolfe", "alpha": alpha, "phi": phi_alpha, "iterations": iteration, "converged": True, "history": history}
        if not armijo_ok or phi_prime_alpha > 0.0:
            upper = alpha
            alpha = 0.5 * (lower + upper)
        else:
            lower = alpha
            alpha = expand * alpha if np.isinf(upper) else 0.5 * (lower + upper)
        alpha = max(alpha, beta * alpha0 * 1e-6)

    phi_alpha = phi(alpha)
    return {"method": "strong_wolfe", "alpha": alpha, "phi": phi_alpha, "iterations": len(history), "converged": False, "history": history}
