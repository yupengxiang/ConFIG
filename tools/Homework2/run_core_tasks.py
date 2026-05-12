"""Run the initial Homework 2 programming tasks and export structured artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

TOOL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TOOL_ROOT.parents[1]
ASSET_ROOT = PROJECT_ROOT / "report_assets" / "Homework2"
sys.path.insert(0, str(TOOL_ROOT / "src"))

from io_utils import write_csv, write_json
from line_search import (
    armijo_search,
    bisection_search,
    fibonacci_search,
    golden_section_search,
    goldstein_search,
    make_exact_line_search,
    shubert_piyavskii_search,
    strong_wolfe_search,
    wolfe_powell_search,
)
from optimizers import bfgs, dfp, fr_conjugate_gradient
from problems import Q2_SCALAR_PROBLEMS, get_assignment_problem, rosenbrock, rosenbrock_grad


RAW = ASSET_ROOT / "raw"
TABLES = ASSET_ROOT / "tables"


def _format_point(x: np.ndarray) -> str:
    return "(" + ", ".join(f"{value:.10g}" for value in np.asarray(x, dtype=float)) + ")"


def _vector_summary(problem_name: str, result: dict, exact_solution: np.ndarray, exact_value: float) -> dict:
    final_x = np.asarray(result["final_x"], dtype=float)
    return {
        "problem": problem_name,
        "algorithm": result["name"],
        "converged": result["converged"],
        "iterations": result["iterations"],
        "final_x": _format_point(final_x),
        "final_f": f"{float(result['final_f']):.12g}",
        "solution_error": f"{float(np.linalg.norm(final_x - exact_solution)):.12g}",
        "value_error": f"{abs(float(result['final_f']) - exact_value):.12g}",
        "final_grad_norm": f"{float(result['final_grad_norm']):.12g}",
        "runtime_sec": f"{float(result['runtime_sec']):.6f}",
    }


def _scalar_summary(problem_name: str, result: dict, exact_minimizer: float, exact_value: float) -> dict:
    return {
        "problem": problem_name,
        "method": result["method"],
        "converged": result["converged"],
        "iterations": result["iterations"],
        "x_star": f"{float(result['x']):.12g}",
        "f_star": f"{float(result['f']):.12g}",
        "minimizer_error": f"{abs(float(result['x']) - exact_minimizer):.12g}",
        "value_error": f"{abs(float(result['f']) - exact_value):.12g}",
        "interval_left": f"{float(result['interval'][0]):.12g}",
        "interval_right": f"{float(result['interval'][1]):.12g}",
    }


def run_q2() -> list[dict]:
    all_results = []
    summary_rows = []
    for key, problem in Q2_SCALAR_PROBLEMS.items():
        a, b = problem.interval
        results = [
            golden_section_search(problem.f, a, b, tol=problem.delta),
            fibonacci_search(problem.f, a, b, tol=problem.delta),
            bisection_search(problem.derivative, a, b, delta=problem.delta, objective=problem.f),
            shubert_piyavskii_search(problem.f, a, b, lipschitz=problem.lipschitz, tol=problem.delta),
        ]
        for result in results:
            result["problem"] = problem.name
        all_results.extend(results)
        summary_rows.extend(
            _scalar_summary(problem.name, result, problem.exact_minimizer, problem.exact_value) for result in results
        )
    write_json(RAW / "q2_scalar_search.json", all_results)
    write_csv(
        TABLES / "q2_scalar_search_summary.csv",
        summary_rows,
        [
            "problem",
            "method",
            "converged",
            "iterations",
            "x_star",
            "f_star",
            "minimizer_error",
            "value_error",
            "interval_left",
            "interval_right",
        ],
    )
    return all_results


def run_vector_tasks() -> list[dict]:
    line_search = make_exact_line_search(method="bisection", tol=1e-10, initial_step=1.0)

    q1 = get_assignment_problem("q1")
    q5 = get_assignment_problem("q5")
    q6 = get_assignment_problem("q6")
    q7 = get_assignment_problem("q7")

    results = [
        {
            "problem": q1.name,
            "result": fr_conjugate_gradient(q1.f, q1.grad, np.zeros(2), line_search=line_search, max_iter=50, tol=1e-6),
            "exact_solution": q1.exact_solution,
            "exact_value": q1.exact_value,
        },
        {
            "problem": q5.name,
            "result": dfp(q5.f, q5.grad, np.array([0.1, 1.0], dtype=float), line_search=line_search, h0=np.eye(2), max_iter=50, tol=1e-8),
            "exact_solution": q5.exact_solution,
            "exact_value": q5.exact_value,
        },
        {
            "problem": q6.name,
            "result": bfgs(q6.f, q6.grad, np.zeros(2), line_search=line_search, h0=np.eye(2), max_iter=50, tol=1e-8),
            "exact_solution": q6.exact_solution,
            "exact_value": q6.exact_value,
        },
        {
            "problem": q7.name,
            "result": dfp(q7.f, q7.grad, np.zeros(2), line_search=line_search, h0=np.eye(2), max_iter=50, tol=1e-8),
            "exact_solution": q7.exact_solution,
            "exact_value": q7.exact_value,
        },
        {
            "problem": q7.name,
            "result": bfgs(q7.f, q7.grad, np.zeros(2), line_search=line_search, h0=np.eye(2), max_iter=50, tol=1e-8),
            "exact_solution": q7.exact_solution,
            "exact_value": q7.exact_value,
        },
        {
            "problem": q7.name,
            "result": fr_conjugate_gradient(q7.f, q7.grad, np.zeros(2), line_search=line_search, max_iter=50, tol=1e-6),
            "exact_solution": q7.exact_solution,
            "exact_value": q7.exact_value,
        },
    ]

    raw_payload = []
    summary_rows = []
    for item in results:
        result = item["result"]
        result["problem"] = item["problem"]
        raw_payload.append(result)
        summary_rows.append(_vector_summary(item["problem"], result, item["exact_solution"], item["exact_value"]))

    write_json(RAW / "core_vector_tasks.json", raw_payload)
    write_csv(
        TABLES / "core_vector_tasks_summary.csv",
        summary_rows,
        [
            "problem",
            "algorithm",
            "converged",
            "iterations",
            "final_x",
            "final_f",
            "solution_error",
            "value_error",
            "final_grad_norm",
            "runtime_sec",
        ],
    )
    return raw_payload


def run_q3() -> list[dict]:
    x0 = np.array([-1.0, 1.0], dtype=float)
    direction = np.array([1.0, 1.0], dtype=float)
    methods = [
        goldstein_search(rosenbrock, rosenbrock_grad, x0, direction),
        armijo_search(rosenbrock, rosenbrock_grad, x0, direction),
        wolfe_powell_search(rosenbrock, rosenbrock_grad, x0, direction),
        strong_wolfe_search(rosenbrock, rosenbrock_grad, x0, direction),
    ]
    summary_rows = []
    for result in methods:
        new_point = x0 + result["alpha"] * direction
        summary_rows.append(
            {
                "method": result["method"],
                "converged": result["converged"],
                "iterations": result["iterations"],
                "alpha": f"{float(result['alpha']):.12g}",
                "phi_alpha": f"{float(result['phi']):.12g}",
                "x_new": _format_point(new_point),
                "directional_derivative": f"{float(np.dot(rosenbrock_grad(new_point), direction)):.12g}",
            }
        )
    write_json(RAW / "q3_inexact_line_search.json", methods)
    write_csv(
        TABLES / "q3_inexact_line_search_summary.csv",
        summary_rows,
        ["method", "converged", "iterations", "alpha", "phi_alpha", "x_new", "directional_derivative"],
    )
    return methods


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    q2_results = run_q2()
    vector_results = run_vector_tasks()
    q3_results = run_q3()

    print("Homework 2 core tasks completed.")
    print(f"Q2 methods run: {len(q2_results)}")
    print(f"Vector-task solvers run: {len(vector_results)}")
    print(f"Q3 methods run: {len(q3_results)}")
    print(f"Artifacts written to: {ASSET_ROOT}")


if __name__ == "__main__":
    main()