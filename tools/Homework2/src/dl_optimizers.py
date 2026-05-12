"""Deterministic deep-learning style first-order optimizers for Homework 2."""

from __future__ import annotations

import time

import numpy as np


def _record(x: np.ndarray, f, grad) -> dict:
    grad_vector = grad(x)
    return {
        "x": x.copy(),
        "f": float(f(x)),
        "grad_norm": float(np.linalg.norm(grad_vector)),
    }


def optimize(
    f,
    grad,
    x0,
    *,
    method: str,
    lr: float,
    steps: int,
    momentum: float = 0.9,
    beta2: float = 0.999,
    rho: float = 0.9,
    eps: float = 1e-8,
) -> dict:
    x = np.asarray(x0, dtype=float).copy()
    velocity = np.zeros_like(x)
    sq_avg = np.zeros_like(x)
    first_moment = np.zeros_like(x)
    second_moment = np.zeros_like(x)
    history = []
    t0 = time.perf_counter()

    for step in range(1, steps + 1):
        history.append(_record(x, f, grad))
        grad_vector = grad(x)

        if method == "SGD":
            update = grad_vector
        elif method == "Momentum":
            velocity = momentum * velocity + grad_vector
            update = velocity
        elif method == "RMSprop":
            sq_avg = rho * sq_avg + (1.0 - rho) * (grad_vector * grad_vector)
            update = grad_vector / (np.sqrt(sq_avg) + eps)
        elif method == "Adam":
            first_moment = momentum * first_moment + (1.0 - momentum) * grad_vector
            second_moment = beta2 * second_moment + (1.0 - beta2) * (grad_vector * grad_vector)
            first_hat = first_moment / (1.0 - momentum**step)
            second_hat = second_moment / (1.0 - beta2**step)
            update = first_hat / (np.sqrt(second_hat) + eps)
        else:
            raise ValueError(f"Unsupported optimizer: {method}")

        x = x - lr * update

    history.append(_record(x, f, grad))
    return {
        "name": method,
        "history": history,
        "iterations": steps,
        "runtime_sec": time.perf_counter() - t0,
        "final_x": x,
        "final_f": float(f(x)),
        "final_grad_norm": float(np.linalg.norm(grad(x))),
    }
