"""Run all experiment-1 algorithms and generate result artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

TOOL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TOOL_ROOT.parents[1]
ASSET_ROOT = PROJECT_ROOT / "report_assets" / "Experiment1_Optimization_Platform"
sys.path.insert(0, str(TOOL_ROOT / "src"))

from admm_lasso import solve_admm_lasso
from derivative_free import particle_swarm
from first_order import adam, armijo_gradient_descent, gradient_descent
from io_utils import write_csv, write_json
from jump_baseline import solve_rosenbrock_baseline
from plotting import (
    plot_admm,
    plot_algorithm_bar,
    plot_rosenbrock_convergence,
    plot_rosenbrock_trajectories,
)
from second_order import bfgs, damped_newton, newton
from test_functions import rosenbrock, rosenbrock_grad, rosenbrock_hessian


RAW = ASSET_ROOT / "raw"
FIGURES = ASSET_ROOT / "figures"
TABLES = ASSET_ROOT / "tables"


def summarize(result: dict, baseline: dict) -> dict:
    bx = np.array(baseline["x"], dtype=float)
    fx = np.array(result["final_x"], dtype=float)
    return {
        "algorithm": result["name"],
        "converged": result["converged"],
        "iterations": result["iterations"],
        "final_x1": f"{fx[0]:.10g}",
        "final_x2": f"{fx[1]:.10g}",
        "final_f": f"{float(result['final_f']):.10e}",
        "final_grad_norm": "" if not np.isfinite(result["final_grad_norm"]) else f"{float(result['final_grad_norm']):.10e}",
        "distance_to_baseline": f"{float(np.linalg.norm(fx - bx)):.10e}",
        "runtime_sec": f"{float(result['runtime_sec']):.6f}",
    }


def main() -> None:
    for path in (RAW, FIGURES, TABLES):
        path.mkdir(parents=True, exist_ok=True)

    baseline_status = solve_rosenbrock_baseline(TOOL_ROOT, RAW)
    write_json(RAW / "jump_baseline_status.json", baseline_status)
    baseline = baseline_status["selected_baseline"]

    x0 = np.array([-1.2, 1.0], dtype=float)
    first_order = [
        gradient_descent(rosenbrock, rosenbrock_grad, x0, lr=1e-3, max_iter=10000),
        armijo_gradient_descent(rosenbrock, rosenbrock_grad, x0, max_iter=10000),
        adam(rosenbrock, rosenbrock_grad, x0, lr=0.02, max_iter=10000),
    ]
    second_order = [
        newton(rosenbrock, rosenbrock_grad, rosenbrock_hessian, x0, max_iter=200),
        damped_newton(rosenbrock, rosenbrock_grad, rosenbrock_hessian, x0, max_iter=200),
        bfgs(rosenbrock, rosenbrock_grad, x0, max_iter=200),
    ]
    pso_result = particle_swarm(rosenbrock)
    all_rosenbrock = first_order + second_order + [pso_result]

    write_json(RAW / "rosenbrock_first_order.json", first_order)
    write_json(RAW / "rosenbrock_second_order.json", second_order)
    write_json(RAW / "rosenbrock_pso.json", pso_result)

    summary_rows = [summarize(result, baseline) for result in all_rosenbrock]
    write_csv(
        TABLES / "rosenbrock_summary.csv",
        summary_rows,
        [
            "algorithm",
            "converged",
            "iterations",
            "final_x1",
            "final_x2",
            "final_f",
            "final_grad_norm",
            "distance_to_baseline",
            "runtime_sec",
        ],
    )

    plot_rosenbrock_trajectories(
        all_rosenbrock,
        baseline,
        FIGURES / "rosenbrock_contours_trajectories.png",
    )
    plot_rosenbrock_convergence(
        all_rosenbrock,
        float(baseline["objective"]),
        FIGURES / "rosenbrock_convergence.png",
    )
    plot_algorithm_bar(summary_rows, FIGURES / "rosenbrock_algorithm_comparison.png")

    admm = solve_admm_lasso()
    write_json(RAW / "admm_lasso.json", admm)
    write_csv(
        TABLES / "admm_lasso_summary.csv",
        [
            {
                "method": admm["name"],
                "iterations": admm["settings"]["iterations"],
                "lambda": admm["settings"]["lambda"],
                "rho": admm["settings"]["rho"],
                "final_objective": f"{admm['final_objective']:.10e}",
                "final_primal_residual": f"{admm['final_primal_residual']:.10e}",
                "final_dual_residual": f"{admm['final_dual_residual']:.10e}",
                "final_sparsity": f"{admm['final_sparsity']:.6f}",
                "relative_error": f"{admm['relative_error']:.10e}",
                "runtime_sec": f"{admm['runtime_sec']:.6f}",
            }
        ],
        [
            "method",
            "iterations",
            "lambda",
            "rho",
            "final_objective",
            "final_primal_residual",
            "final_dual_residual",
            "final_sparsity",
            "relative_error",
            "runtime_sec",
        ],
    )
    plot_admm(
        admm,
        FIGURES / "admm_lasso_objective_residuals.png",
        FIGURES / "admm_lasso_solution_recovery.png",
    )

    print("Experiment 1 completed.")
    print(f"JuMP available: {baseline_status['jump_available']}")
    print(f"Results written to: {ASSET_ROOT}")


if __name__ == "__main__":
    main()
