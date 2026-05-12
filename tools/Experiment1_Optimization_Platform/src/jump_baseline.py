"""JuMP baseline integration with robust Python fallbacks."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from test_functions import rosenbrock, rosenbrock_grad


def solve_rosenbrock_baseline(tool_root: Path, raw_dir: Path) -> dict:
    """Try Julia/JuMP first, then return analytic and SciPy baselines."""
    julia_script = tool_root / "julia" / "jump_rosenbrock.jl"
    output_json = raw_dir / "jump_rosenbrock.json"
    analytic = {"x": [1.0, 1.0], "objective": 0.0, "source": "analytic"}
    scipy_result = minimize(
        rosenbrock,
        np.array([-1.2, 1.0]),
        jac=rosenbrock_grad,
        method="BFGS",
        options={"gtol": 1e-10, "maxiter": 10000},
    )
    status = {
        "jump_available": False,
        "jump_result": None,
        "analytic_result": analytic,
        "scipy_result": {
            "x": scipy_result.x.tolist(),
            "objective": float(scipy_result.fun),
            "success": bool(scipy_result.success),
            "message": str(scipy_result.message),
            "iterations": int(getattr(scipy_result, "nit", -1)),
        },
        "selected_baseline": analytic,
        "notes": "",
    }

    julia = shutil.which("julia")
    if julia is None:
        status["notes"] = "Julia executable was not found; analytic Rosenbrock optimum is used."
        return status

    try:
        completed = subprocess.run(
            [julia, str(julia_script), str(output_json)],
            cwd=str(tool_root),
            check=True,
            text=True,
            capture_output=True,
            timeout=600,
        )
        jump_result = json.loads(output_json.read_text(encoding="utf-8"))
        status["jump_available"] = True
        status["jump_result"] = jump_result
        status["selected_baseline"] = {
            "x": [jump_result["x"], jump_result["y"]],
            "objective": jump_result["objective"],
            "source": "JuMP/Ipopt",
        }
        status["notes"] = completed.stdout.strip()
    except Exception as exc:
        status["notes"] = f"Julia/JuMP execution failed; fallback used. Error: {exc}"
    return status
