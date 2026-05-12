"""Objective functions used in Homework 2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


VectorFunction = Callable[[np.ndarray], float]
VectorGradient = Callable[[np.ndarray], np.ndarray]
VectorHessian = Callable[[np.ndarray], np.ndarray]
ScalarFunction = Callable[[float], float]
ScalarDerivative = Callable[[float], float]


@dataclass(frozen=True)
class ObjectiveDefinition:
    name: str
    dimension: int
    f: VectorFunction
    grad: VectorGradient
    hess: VectorHessian | None = None
    exact_solution: np.ndarray | None = None
    exact_value: float | None = None


@dataclass(frozen=True)
class ScalarOptimizationProblem:
    name: str
    f: ScalarFunction
    derivative: ScalarDerivative
    interval: tuple[float, float]
    delta: float
    lipschitz: float
    exact_minimizer: float
    exact_value: float


def _to_array(x: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    return np.asarray(x, dtype=float)


def q1_function(x: np.ndarray) -> float:
    x = _to_array(x)
    return float(x[0] - x[1] + 2.0 * x[0] ** 2 + 2.0 * x[0] * x[1] + x[1] ** 2)


def q1_gradient(x: np.ndarray) -> np.ndarray:
    x = _to_array(x)
    return np.array([1.0 + 4.0 * x[0] + 2.0 * x[1], -1.0 + 2.0 * x[0] + 2.0 * x[1]], dtype=float)


def q1_hessian(_x: np.ndarray) -> np.ndarray:
    return np.array([[4.0, 2.0], [2.0, 2.0]], dtype=float)


def q5_function(x: np.ndarray) -> float:
    x = _to_array(x)
    return float(10.0 * x[0] ** 2 + x[1] ** 2)


def q5_gradient(x: np.ndarray) -> np.ndarray:
    x = _to_array(x)
    return np.array([20.0 * x[0], 2.0 * x[1]], dtype=float)


def q5_hessian(_x: np.ndarray) -> np.ndarray:
    return np.array([[20.0, 0.0], [0.0, 2.0]], dtype=float)


def q6_function(x: np.ndarray) -> float:
    x = _to_array(x)
    return float(x[0] ** 2 + 4.0 * x[1] ** 2 - 4.0 * x[0] - 8.0 * x[1])


def q6_gradient(x: np.ndarray) -> np.ndarray:
    x = _to_array(x)
    return np.array([2.0 * x[0] - 4.0, 8.0 * x[1] - 8.0], dtype=float)


def q6_hessian(_x: np.ndarray) -> np.ndarray:
    return np.array([[2.0, 0.0], [0.0, 8.0]], dtype=float)


def q7_function(x: np.ndarray) -> float:
    x = _to_array(x)
    return float(x[0] ** 2 + x[1] ** 2 - x[0] * x[1] - 10.0 * x[0] - 4.0 * x[1] + 60.0)


def q7_gradient(x: np.ndarray) -> np.ndarray:
    x = _to_array(x)
    return np.array([2.0 * x[0] - x[1] - 10.0, 2.0 * x[1] - x[0] - 4.0], dtype=float)


def q7_hessian(_x: np.ndarray) -> np.ndarray:
    return np.array([[2.0, -1.0], [-1.0, 2.0]], dtype=float)


def sphere(x: np.ndarray) -> float:
    x = _to_array(x)
    return float(np.dot(x, x))


def sphere_grad(x: np.ndarray) -> np.ndarray:
    x = _to_array(x)
    return 2.0 * x


def rosenbrock(x: np.ndarray) -> float:
    x = _to_array(x)
    return float((1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2)


def rosenbrock_grad(x: np.ndarray) -> np.ndarray:
    x = _to_array(x)
    return np.array(
        [
            -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] ** 2),
            200.0 * (x[1] - x[0] ** 2),
        ],
        dtype=float,
    )


def booth(x: np.ndarray) -> float:
    x = _to_array(x)
    return float((x[0] + 2.0 * x[1] - 7.0) ** 2 + (2.0 * x[0] + x[1] - 5.0) ** 2)


def booth_grad(x: np.ndarray) -> np.ndarray:
    x = _to_array(x)
    return np.array(
        [
            2.0 * (x[0] + 2.0 * x[1] - 7.0) + 4.0 * (2.0 * x[0] + x[1] - 5.0),
            4.0 * (x[0] + 2.0 * x[1] - 7.0) + 2.0 * (2.0 * x[0] + x[1] - 5.0),
        ],
        dtype=float,
    )


def matyas(x: np.ndarray) -> float:
    x = _to_array(x)
    return float(0.26 * (x[0] ** 2 + x[1] ** 2) - 0.48 * x[0] * x[1])


def matyas_grad(x: np.ndarray) -> np.ndarray:
    x = _to_array(x)
    return np.array([0.52 * x[0] - 0.48 * x[1], 0.52 * x[1] - 0.48 * x[0]], dtype=float)


def himmelblau(x: np.ndarray) -> float:
    x = _to_array(x)
    return float((x[0] ** 2 + x[1] - 11.0) ** 2 + (x[0] + x[1] ** 2 - 7.0) ** 2)


def himmelblau_grad(x: np.ndarray) -> np.ndarray:
    x = _to_array(x)
    term1 = x[0] ** 2 + x[1] - 11.0
    term2 = x[0] + x[1] ** 2 - 7.0
    return np.array([4.0 * x[0] * term1 + 2.0 * term2, 2.0 * term1 + 4.0 * x[1] * term2], dtype=float)


def three_hump_camel(x: np.ndarray) -> float:
    x = _to_array(x)
    return float(2.0 * x[0] ** 2 - 1.05 * x[0] ** 4 + (x[0] ** 6) / 6.0 + x[0] * x[1] + x[1] ** 2)


def three_hump_camel_grad(x: np.ndarray) -> np.ndarray:
    x = _to_array(x)
    return np.array([4.0 * x[0] - 4.2 * x[0] ** 3 + x[0] ** 5 + x[1], x[0] + 2.0 * x[1]], dtype=float)


def beale(x: np.ndarray) -> float:
    x = _to_array(x)
    return float(
        (1.5 - x[0] + x[0] * x[1]) ** 2
        + (2.25 - x[0] + x[0] * x[1] ** 2) ** 2
        + (2.625 - x[0] + x[0] * x[1] ** 3) ** 2
    )


def beale_grad(x: np.ndarray) -> np.ndarray:
    x = _to_array(x)
    t1 = 1.5 - x[0] + x[0] * x[1]
    t2 = 2.25 - x[0] + x[0] * x[1] ** 2
    t3 = 2.625 - x[0] + x[0] * x[1] ** 3
    dfdx = 2.0 * t1 * (-1.0 + x[1]) + 2.0 * t2 * (-1.0 + x[1] ** 2) + 2.0 * t3 * (-1.0 + x[1] ** 3)
    dfdy = 2.0 * t1 * x[0] + 4.0 * t2 * x[0] * x[1] + 6.0 * t3 * x[0] * x[1] ** 2
    return np.array([dfdx, dfdy], dtype=float)


def q2a_function(x: float) -> float:
    return 2.0 * x**2 - x - 1.0


def q2a_derivative(x: float) -> float:
    return 4.0 * x - 1.0


def q2b_function(x: float) -> float:
    return 3.0 * x**2 - 21.6 * x - 1.0


def q2b_derivative(x: float) -> float:
    return 6.0 * x - 21.6


ASSIGNMENT_VECTOR_PROBLEMS: dict[str, ObjectiveDefinition] = {
    "q1": ObjectiveDefinition(
        name="Q1 quadratic",
        dimension=2,
        f=q1_function,
        grad=q1_gradient,
        hess=q1_hessian,
        exact_solution=np.array([-1.0, 1.5], dtype=float),
        exact_value=-1.25,
    ),
    "q5": ObjectiveDefinition(
        name="Q5 quadratic",
        dimension=2,
        f=q5_function,
        grad=q5_gradient,
        hess=q5_hessian,
        exact_solution=np.array([0.0, 0.0], dtype=float),
        exact_value=0.0,
    ),
    "q6": ObjectiveDefinition(
        name="Q6 quadratic",
        dimension=2,
        f=q6_function,
        grad=q6_gradient,
        hess=q6_hessian,
        exact_solution=np.array([2.0, 1.0], dtype=float),
        exact_value=-8.0,
    ),
    "q7": ObjectiveDefinition(
        name="Q7 quadratic",
        dimension=2,
        f=q7_function,
        grad=q7_gradient,
        hess=q7_hessian,
        exact_solution=np.array([8.0, 6.0], dtype=float),
        exact_value=8.0,
    ),
}


Q2_SCALAR_PROBLEMS: dict[str, ScalarOptimizationProblem] = {
    "q2a": ScalarOptimizationProblem(
        name="Q2(a)",
        f=q2a_function,
        derivative=q2a_derivative,
        interval=(-1.0, 1.0),
        delta=0.06,
        lipschitz=5.0,
        exact_minimizer=0.25,
        exact_value=-1.125,
    ),
    "q2b": ScalarOptimizationProblem(
        name="Q2(b)",
        f=q2b_function,
        derivative=q2b_derivative,
        interval=(0.0, 25.0),
        delta=0.08,
        lipschitz=128.4,
        exact_minimizer=3.6,
        exact_value=-39.88,
    ),
}


BENCHMARK_FUNCTIONS: dict[str, ObjectiveDefinition] = {
    "sphere": ObjectiveDefinition("Sphere", 2, sphere, sphere_grad, exact_solution=np.zeros(2), exact_value=0.0),
    "rosenbrock": ObjectiveDefinition("Rosenbrock", 2, rosenbrock, rosenbrock_grad, exact_solution=np.array([1.0, 1.0]), exact_value=0.0),
    "booth": ObjectiveDefinition("Booth", 2, booth, booth_grad, exact_solution=np.array([1.0, 3.0]), exact_value=0.0),
    "matyas": ObjectiveDefinition("Matyas", 2, matyas, matyas_grad, exact_solution=np.zeros(2), exact_value=0.0),
    "himmelblau": ObjectiveDefinition("Himmelblau", 2, himmelblau, himmelblau_grad, exact_solution=np.array([3.0, 2.0]), exact_value=0.0),
    "three_hump_camel": ObjectiveDefinition("Three-Hump Camel", 2, three_hump_camel, three_hump_camel_grad, exact_solution=np.zeros(2), exact_value=0.0),
    "beale": ObjectiveDefinition("Beale", 2, beale, beale_grad, exact_solution=np.array([3.0, 0.5]), exact_value=0.0),
    "q7": ASSIGNMENT_VECTOR_PROBLEMS["q7"],
}


def get_assignment_problem(name: str) -> ObjectiveDefinition:
    return ASSIGNMENT_VECTOR_PROBLEMS[name]


def get_q2_problem(name: str) -> ScalarOptimizationProblem:
    return Q2_SCALAR_PROBLEMS[name]
