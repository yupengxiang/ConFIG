"""Derivative-free optimizers."""

from __future__ import annotations

import time

import numpy as np


def particle_swarm(
    f,
    *,
    bounds=((-3.0, 3.0), (-1.0, 4.0)),
    particles=40,
    iterations=300,
    w=0.72,
    c1=1.49,
    c2=1.49,
    seed=42,
) -> dict:
    rng = np.random.default_rng(seed)
    lower = np.array([b[0] for b in bounds], dtype=float)
    upper = np.array([b[1] for b in bounds], dtype=float)
    span = upper - lower
    positions = rng.uniform(lower, upper, size=(particles, len(bounds)))
    velocities = rng.uniform(-0.1 * span, 0.1 * span, size=positions.shape)
    personal_best = positions.copy()
    personal_values = np.array([f(p) for p in positions])
    best_index = int(np.argmin(personal_values))
    global_best = personal_best[best_index].copy()
    global_value = float(personal_values[best_index])
    history = []
    t0 = time.perf_counter()

    for k in range(iterations + 1):
        history.append({"x": global_best.copy(), "f": global_value, "grad_norm": float("nan")})
        if k == iterations:
            break
        r1 = rng.random(size=positions.shape)
        r2 = rng.random(size=positions.shape)
        velocities = (
            w * velocities
            + c1 * r1 * (personal_best - positions)
            + c2 * r2 * (global_best - positions)
        )
        positions = np.clip(positions + velocities, lower, upper)
        values = np.array([f(p) for p in positions])
        improved = values < personal_values
        personal_best[improved] = positions[improved]
        personal_values[improved] = values[improved]
        best_index = int(np.argmin(personal_values))
        if personal_values[best_index] < global_value:
            global_value = float(personal_values[best_index])
            global_best = personal_best[best_index].copy()

    return {
        "name": "PSO",
        "history": history,
        "iterations": iterations,
        "runtime_sec": time.perf_counter() - t0,
        "converged": global_value < 1e-6,
        "final_x": global_best,
        "final_f": global_value,
        "final_grad_norm": float("nan"),
        "settings": {
            "particles": particles,
            "iterations": iterations,
            "bounds": bounds,
            "w": w,
            "c1": c1,
            "c2": c2,
            "seed": seed,
        },
    }

